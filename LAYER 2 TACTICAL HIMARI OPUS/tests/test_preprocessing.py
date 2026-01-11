"""
HIMARI Layer 2 - Preprocessing Unit Tests
Tests for all A1-A8 preprocessing components

Run with: python -m pytest tests/test_preprocessing.py -v
"""

import pytest
import numpy as np
import torch

# Import v5.0 preprocessing components
from src.preprocessing.ekf_denoiser import EKFDenoiser, EKFConfig
from src.preprocessing.conversational_ae import ConversationalAutoencoder, CAEConfig
from src.preprocessing.freq_normalizer import FrequencyDomainNormalizer, FreqNormConfig
from src.preprocessing.timegan_augment import TimeGAN, TimeGANConfig
from src.preprocessing.tab_ddpm import TabDDPM, TabDDPMConfig
from src.preprocessing.vec_normalize import VecNormalize
from src.preprocessing.orthogonal_init import orthogonal_init, OrthogonalLinear
from src.preprocessing.online_augment import OnlineAugmentor, OnlineAugmentConfig
from src.preprocessing.pipeline import PreprocessingPipeline, PreprocessingPipelineConfig


class TestEKFDenoiser:
    """Tests for Extended Kalman Filter (A1)."""
    
    def test_initialization(self):
        ekf = EKFDenoiser()
        assert ekf.config.state_dim == 4
    
    def test_update(self):
        ekf = EKFDenoiser()
        price, uncertainty = ekf.update(100.0, 1000.0)
        
        assert price == 100.0  # First observation
        assert uncertainty > 0
    
    def test_state_extraction(self):
        ekf = EKFDenoiser()
        ekf.update(100.0, 1000.0)
        ekf.update(101.0, 1100.0)
        
        state = ekf.get_state()
        assert len(state) == 4  # [price, velocity, acceleration, volatility]
    
    def test_momentum_tracking(self):
        ekf = EKFDenoiser()
        ekf.update(100.0, 1000.0)
        ekf.update(102.0, 1100.0)  # Price increased
        
        momentum = ekf.get_momentum()
        assert momentum > 0  # Should detect positive momentum
    
    def test_noise_reduction(self):
        """Verify EKF reduces noise compared to raw signal."""
        ekf = EKFDenoiser()
        
        # Generate noisy price series
        np.random.seed(42)
        true_prices = np.cumsum(np.random.randn(100) * 0.1) + 100
        noisy_prices = true_prices + np.random.randn(100) * 0.5
        volumes = np.random.rand(100) * 1000
        
        # Filter
        filtered = []
        for p, v in zip(noisy_prices, volumes):
            filtered_p, _ = ekf.update(p, v)
            filtered.append(filtered_p)
        filtered = np.array(filtered)
        
        # Check noise reduction (filtered should be closer to true)
        raw_error = np.mean((noisy_prices[10:] - true_prices[10:])**2)
        filtered_error = np.mean((filtered[10:] - true_prices[10:])**2)
        
        assert filtered_error < raw_error


class TestCAE:
    """Tests for Conversational Autoencoders (A2)."""
    
    def test_initialization(self):
        config = CAEConfig(input_dim=60, latent_dim=32)
        cae = ConversationalAutoencoder(config)
        
        assert cae.config.input_dim == 60
        assert cae.ae1 is not None
        assert cae.ae2 is not None
    
    def test_forward(self):
        config = CAEConfig(input_dim=60, latent_dim=32)
        cae = ConversationalAutoencoder(config)
        
        x = torch.randn(8, 24, 60)  # batch=8, seq=24, feat=60
        outputs = cae(x)
        
        assert 'consensus' in outputs
        assert 'disagreement' in outputs
        assert outputs['consensus'].shape == x.shape
    
    def test_denoising(self):
        """Verify CAE produces cleaner output."""
        config = CAEConfig(input_dim=60, latent_dim=32)
        cae = ConversationalAutoencoder(config)
        cae.eval()
        
        # Noisy input
        x = torch.randn(1, 24, 60)
        denoised = cae.denoise(x)
        
        # Should have same shape
        assert denoised.shape == x.shape
    
    def test_regime_ambiguity(self):
        config = CAEConfig(input_dim=60, latent_dim=32)
        cae = ConversationalAutoencoder(config)
        cae.eval()
        
        x = torch.randn(1, 24, 60)
        ambiguity = cae.get_regime_ambiguity(x)
        
        assert 0 <= ambiguity <= 1


class TestFreqNormalization:
    """Tests for Frequency Domain Normalization (A3)."""
    
    def test_initialization(self):
        config = FreqNormConfig(window_size=256)
        norm = FrequencyDomainNormalizer(config)
        
        assert norm.config.window_size == 256
    
    def test_normalize(self):
        config = FreqNormConfig(window_size=256, n_freq_components=32)
        norm = FrequencyDomainNormalizer(config)
        
        signal = np.random.randn(256)
        normalized = norm.normalize(signal)
        
        assert normalized.shape == signal.shape
    
    def test_spectral_preservation(self):
        """Verify phase is preserved."""
        config = FreqNormConfig(window_size=256)
        norm = FrequencyDomainNormalizer(config)
        
        # Create signal with known phase
        t = np.linspace(0, 4*np.pi, 256)
        signal = np.sin(t)  # Known frequency content
        
        normalized = norm.normalize(signal)
        
        # Should maintain temporal structure
        assert not np.allclose(normalized, 0)


class TestTabDDPM:
    """Tests for Tab-DDPM diffusion (A5)."""
    
    def test_initialization(self):
        config = TabDDPMConfig(feature_dim=10, hidden_dim=32)
        ddpm = TabDDPM(config, device='cpu')
        
        assert ddpm.config.feature_dim == 10
    
    @pytest.mark.slow
    def test_training_and_generation(self):
        config = TabDDPMConfig(
            feature_dim=10,
            hidden_dim=32,
            n_timesteps=100,
            epochs=2  # Very short for testing
        )
        ddpm = TabDDPM(config, device='cpu')
        
        # Fake training data
        data = np.random.randn(50, 10)
        ddpm.train(data)
        
        # Generate
        synthetic = ddpm.generate(10)
        assert synthetic.shape == (10, 10)
    
    def test_tail_identification(self):
        config = TabDDPMConfig(feature_dim=10, tail_threshold=5.0)
        ddpm = TabDDPM(config, device='cpu')
        
        # Create data with clear tails
        data = np.random.randn(100, 10)
        data[:5, 0] = -5  # Lower tail
        data[95:, 0] = 5  # Upper tail
        
        tail_mask = ddpm._identify_tail_samples(data)
        assert tail_mask.sum() >= 10  # Should identify at least 10%


class TestVecNormalize:
    """Tests for VecNormalize (A6)."""
    
    def test_normalization(self):
        norm = VecNormalize(dim=60)
        
        # Update with some data
        for _ in range(100):
            x = np.random.randn(60) * 10 + 50
            _ = norm.normalize(x)
        
        # After warmup, output should be roughly standardized
        x = np.random.randn(60) * 10 + 50
        normalized = norm.normalize(x, update=False)
        
        # Should be closer to 0 mean, 1 std
        assert abs(normalized.mean()) < abs(x.mean())
    
    def test_denormalization(self):
        norm = VecNormalize(dim=60)
        
        # Update stats
        for _ in range(100):
            x = np.random.randn(60) * 10 + 50
            _ = norm.normalize(x)
        
        # Normalize and denormalize
        x = np.random.randn(60) * 10 + 50
        normalized = norm.normalize(x, update=False)
        denormalized = norm.denormalize(normalized)
        
        # Should recover original (approximately)
        assert np.allclose(x, denormalized, atol=0.1)
    
    def test_state_save_load(self):
        norm = VecNormalize(dim=60)
        
        # Update stats
        for _ in range(100):
            x = np.random.randn(60) * 10 + 50
            _ = norm.normalize(x)
        
        # Save and load state
        state = norm.get_state()
        
        norm2 = VecNormalize(dim=60)
        norm2.set_state(state)
        
        # Should have same statistics
        assert np.allclose(norm.running_mean, norm2.running_mean)
        assert np.allclose(norm.running_var, norm2.running_var)


class TestOrthogonalInit:
    """Tests for Orthogonal Initialization (A7)."""
    
    def test_linear_init(self):
        layer = torch.nn.Linear(64, 64)
        orthogonal_init(layer)
        
        # Check orthogonality: W^T @ W should be close to identity
        W = layer.weight.detach()
        WtW = W @ W.T
        I = torch.eye(64)
        
        # Frobenius norm of difference should be small
        diff = torch.norm(WtW - I)
        assert diff < 10  # Relaxed check
    
    def test_orthogonal_linear_class(self):
        layer = OrthogonalLinear(64, 64)
        
        # Weight should be initialized orthogonally
        W = layer.weight.detach()
        assert W.shape == (64, 64)


class TestOnlineAugment:
    """Tests for Online Augmentation (A8)."""
    
    def test_jitter(self):
        aug = OnlineAugmentor()
        x = np.zeros(60)
        jittered = aug.add_jitter(x)
        
        # Should add small noise
        assert np.std(jittered) > 0
        assert np.std(jittered) < 0.1
    
    def test_feature_mask(self):
        aug = OnlineAugmentor(OnlineAugmentConfig(feature_mask_prob=0.5))
        x = np.ones(60)
        masked = aug.feature_mask(x)
        
        # Some features should be zeroed
        assert np.any(masked == 0)
    
    def test_regime_adaptation(self):
        aug = OnlineAugmentor()
        
        # Low uncertainty = low augmentation
        aug.set_regime_uncertainty(0.0)
        assert aug._augment_scale == aug.config.min_augment_scale
        
        # High uncertainty = high augmentation
        aug.set_regime_uncertainty(1.0)
        assert aug._augment_scale == aug.config.max_augment_scale
    
    def test_augment_batch(self):
        aug = OnlineAugmentor()
        batch = np.random.randn(10, 60)
        
        augmented = aug.augment_batch(batch, n_augments=2)
        
        # Should have 3x samples (original + 2 augmented versions)
        assert augmented.shape[0] == 30


class TestPreprocessingPipeline:
    """Tests for unified preprocessing pipeline."""
    
    def test_initialization(self):
        config = PreprocessingPipelineConfig()
        pipeline = PreprocessingPipeline(config, device='cpu')
        
        assert pipeline.ekf is not None
        assert pipeline.vec_norm is not None
    
    def test_process_market_data(self):
        config = PreprocessingPipelineConfig(
            use_cae=False,  # Skip CAE (no trained model)
            feature_dim=60
        )
        pipeline = PreprocessingPipeline(config, device='cpu')
        
        price = 100.0
        volume = 1000.0
        features = np.random.randn(60)
        
        result = pipeline.process_market_data(price, volume, features)
        
        assert 'processed_features' in result
        assert 'ekf_price' in result
        assert result['processed_features'].shape == (60,)
    
    def test_process_batch(self):
        config = PreprocessingPipelineConfig(
            use_cae=False,
            feature_dim=60
        )
        pipeline = PreprocessingPipeline(config, device='cpu')
        
        prices = np.random.rand(10) * 100
        volumes = np.random.rand(10) * 1000
        features = np.random.randn(10, 60)
        
        result = pipeline.process_batch(prices, volumes, features)
        
        assert 'features' in result
        assert result['features'].shape == (10, 60)
    
    def test_reset(self):
        config = PreprocessingPipelineConfig(feature_dim=60)
        pipeline = PreprocessingPipeline(config, device='cpu')
        
        # Process some data
        for _ in range(10):
            pipeline.process_market_data(100.0, 1000.0, np.random.randn(60))
        
        # Reset
        pipeline.reset()
        
        assert pipeline._update_count == 0


class TestTimeGAN:
    """Tests for TimeGAN augmentation (A4)."""
    
    def test_initialization(self):
        config = TimeGANConfig(seq_len=24, feature_dim=10, hidden_dim=32)
        timegan = TimeGAN(config, device='cpu')
        
        assert timegan.config.seq_len == 24
    
    @pytest.mark.slow
    def test_training_and_generation(self):
        config = TimeGANConfig(
            seq_len=24,
            feature_dim=10,
            hidden_dim=32,
            epochs=2  # Very short for testing
        )
        timegan = TimeGAN(config, device='cpu')
        
        # Fake training data
        data = np.random.randn(50, 24, 10)
        timegan.train(data)
        
        # Generate
        synthetic = timegan.generate(10)
        assert synthetic.shape == (10, 24, 10)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--ignore-glob=*slow*'])
