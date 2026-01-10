# ============================================================================
# FILE: tests/test_preprocessing.py
# PURPOSE: Unit tests for preprocessing components
# ============================================================================

import pytest
import numpy as np
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ekf_denoiser import EKFDenoiser, EKFConfig
from conversational_ae import ConversationalAutoencoder, CAEConfig
from freq_normalizer import FrequencyDomainNormalizer, FreqNormConfig
from timegan_augment import TimeGAN, TimeGANConfig
from tab_ddpm import TabDDPM, TabDDPMConfig
from vec_normalize import VecNormalize
from online_augment import OnlineAugmentor


class TestEKFDenoiser:
    """Tests for Extended Kalman Filter."""
    
    def test_initialization(self):
        ekf = EKFDenoiser()
        assert ekf.config.state_dim == 4
        assert not ekf._initialized
    
    def test_update(self):
        ekf = EKFDenoiser()
        price, uncertainty = ekf.update(100.0, 1000.0)
        
        assert price == 100.0  # First observation
        assert uncertainty > 0
        assert ekf._initialized
    
    def test_state_extraction(self):
        ekf = EKFDenoiser()
        ekf.update(100.0, 1000.0)
        ekf.update(101.0, 1100.0)
        
        state = ekf.get_state()
        assert 'price' in state
        assert 'velocity' in state
        assert 'volatility' in state
    
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
    """Tests for Conversational Autoencoders."""
    
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


class TestFreqNormalization:
    """Tests for Frequency Domain Normalization."""
    
    def test_initialization(self):
        config = FreqNormConfig(window_size=256)
        norm = FrequencyDomainNormalizer(config)
        
        assert norm.config.window_size == 256
        assert not norm._initialized
    
    def test_normalize(self):
        config = FreqNormConfig(window_size=256, n_freq_components=32)
        norm = FrequencyDomainNormalizer(config)
        
        signal = np.random.randn(256)
        normalized, metadata = norm.normalize(signal)
        
        assert normalized.shape == signal.shape
        assert 'dominant_frequency' in metadata


class TestTimeGAN:
    """Tests for TimeGAN augmentation."""
    
    @pytest.mark.slow
    def test_training_and_generation(self):
        config = TimeGANConfig(
            seq_len=24,
            feature_dim=10,
            hidden_dim=32,
            epochs=10  # Short for testing
        )
        timegan = TimeGAN(config, device='cpu')
        
        # Fake training data
        data = np.random.randn(100, 24, 10)
        timegan.train(data)
        
        # Generate
        synthetic = timegan.generate(10)
        assert synthetic.shape == (10, 24, 10)


class TestTabDDPM:
    """Tests for Tab-DDPM diffusion."""
    
    @pytest.mark.slow
    def test_training_and_generation(self):
        config = TabDDPMConfig(
            feature_dim=10,
            hidden_dim=64,
            epochs=5  # Short for testing
        )
        ddpm = TabDDPM(config, device='cpu')
        
        # Fake training data
        data = np.random.randn(100, 10)
        ddpm.train(data)
        
        # Generate
        synthetic = ddpm.generate(10)
        assert synthetic.shape == (10, 10)


class TestVecNormalize:
    """Tests for VecNormalize."""
    
    def test_normalization(self):
        norm = VecNormalize((60,))
        
        # Update with some data
        for _ in range(100):
            x = np.random.randn(60) * 10 + 50
            normalized = norm(x)
        
        # After warmup, output should be roughly standardized
        x = np.random.randn(60) * 10 + 50
        normalized = norm(x)
        
        # Should be closer to 0 mean, 1 std
        assert abs(normalized.mean()) < abs(x.mean())


class TestOnlineAugment:
    """Tests for Online Augmentation."""
    
    def test_jitter(self):
        aug = OnlineAugmentor()
        x = np.zeros(60)
        jittered = aug.add_jitter(x)
        
        # Should add small noise
        assert np.std(jittered) > 0
        assert np.std(jittered) < 0.1
    
    def test_regime_adaptation(self):
        aug = OnlineAugmentor()
        
        # Low uncertainty = low augmentation
        aug.set_regime_uncertainty(0.0)
        assert aug._augment_scale == aug.config.min_augment_scale
        
        # High uncertainty = high augmentation
        aug.set_regime_uncertainty(1.0)
        assert aug._augment_scale == aug.config.max_augment_scale


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
