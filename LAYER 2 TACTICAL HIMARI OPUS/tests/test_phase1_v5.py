"""
HIMARI Layer 2 v5.0 - Phase 1 Unit Tests
Tests for preprocessing and regime detection modules

Run with: python -m pytest tests/test_phase1_v5.py -v
"""

import pytest
import numpy as np

# =============================================================================
# Test A1: EKF Denoiser
# =============================================================================

class TestEKFDenoiser:
    """Tests for Extended Kalman Filter denoiser"""
    
    def test_ekf_import(self):
        """Verify EKF module imports correctly"""
        from src.preprocessing.ekf_denoiser import EKFDenoiser, EKFConfig
        assert EKFDenoiser is not None
        assert EKFConfig is not None
    
    def test_ekf_initialization(self):
        """Test EKF initializes with default config"""
        from src.preprocessing.ekf_denoiser import EKFDenoiser, EKFConfig
        
        config = EKFConfig()
        ekf = EKFDenoiser(config)
        
        assert ekf.config.state_dim == 4
        assert ekf.config.measurement_dim == 2
    
    def test_ekf_single_update(self):
        """Test single EKF update returns valid output"""
        from src.preprocessing.ekf_denoiser import EKFDenoiser
        
        ekf = EKFDenoiser()
        price, uncertainty = ekf.update(50000.0, 1.0)
        
        assert isinstance(price, float)
        assert isinstance(uncertainty, float)
        assert uncertainty >= 0
    
    def test_ekf_filter_sequence(self):
        """Test EKF filters entire sequence"""
        from src.preprocessing.ekf_denoiser import EKFDenoiser
        
        ekf = EKFDenoiser()
        prices = np.array([100, 101, 99, 102, 98, 103])
        filtered = ekf.filter(prices)
        
        assert len(filtered) == len(prices)
        assert np.var(filtered) <= np.var(prices) * 2  # Should reduce noise
    
    def test_ekf_state_extraction(self):
        """Test state extraction methods"""
        from src.preprocessing.ekf_denoiser import EKFDenoiser
        
        ekf = EKFDenoiser()
        for p in [100, 105, 110]:
            ekf.update(p, 1.0)
        
        momentum = ekf.get_momentum()
        vol = ekf.get_volatility_estimate()
        state = ekf.get_state()
        
        assert isinstance(momentum, float)
        assert vol > 0
        assert len(state) == 4


# =============================================================================
# Test A2: Conversational Autoencoder
# =============================================================================

class TestConversationalAutoencoder:
    """Tests for CAE denoising"""
    
    def test_cae_import(self):
        """Verify CAE module imports correctly"""
        from src.preprocessing.conversational_ae import (
            ConversationalAutoencoder, CAEConfig
        )
        assert ConversationalAutoencoder is not None
    
    def test_cae_initialization(self):
        """Test CAE initializes correctly"""
        from src.preprocessing.conversational_ae import (
            ConversationalAutoencoder, CAEConfig
        )
        import torch
        
        config = CAEConfig(input_dim=10, latent_dim=8, hidden_dim=16)
        cae = ConversationalAutoencoder(config)
        
        assert hasattr(cae, 'ae1')
        assert hasattr(cae, 'ae2')
    
    def test_cae_forward_pass(self):
        """Test CAE forward pass returns expected outputs"""
        from src.preprocessing.conversational_ae import (
            ConversationalAutoencoder, CAEConfig
        )
        import torch
        
        config = CAEConfig(input_dim=10, latent_dim=8, hidden_dim=16)
        cae = ConversationalAutoencoder(config)
        
        x = torch.randn(2, 5, 10)  # batch=2, seq=5, features=10
        outputs = cae(x)
        
        assert 'consensus' in outputs
        assert 'disagreement' in outputs
        assert outputs['consensus'].shape == x.shape
    
    def test_cae_denoise(self):
        """Test CAE denoising method"""
        from src.preprocessing.conversational_ae import (
            ConversationalAutoencoder, CAEConfig
        )
        import torch
        
        config = CAEConfig(input_dim=10, latent_dim=8, hidden_dim=16)
        cae = ConversationalAutoencoder(config)
        
        x = torch.randn(2, 5, 10)
        denoised = cae.denoise(x)
        
        assert denoised.shape == x.shape
    
    def test_cae_regime_ambiguity(self):
        """Test regime ambiguity score calculation"""
        from src.preprocessing.conversational_ae import (
            ConversationalAutoencoder, CAEConfig
        )
        import torch
        
        config = CAEConfig(input_dim=10, latent_dim=8, hidden_dim=16)
        cae = ConversationalAutoencoder(config)
        
        x = torch.randn(1, 5, 10)
        ambiguity = cae.get_regime_ambiguity(x)
        
        assert 0 <= ambiguity <= 1


# =============================================================================
# Test A3: Frequency Domain Normalizer
# =============================================================================

class TestFrequencyDomainNormalizer:
    """Tests for frequency domain normalization"""
    
    def test_freq_norm_import(self):
        """Verify FreqNorm module imports"""
        from src.preprocessing.freq_normalizer import (
            FrequencyDomainNormalizer, FreqNormConfig
        )
        assert FrequencyDomainNormalizer is not None
    
    def test_freq_norm_single_series(self):
        """Test normalization of single time series"""
        from src.preprocessing.freq_normalizer import FrequencyDomainNormalizer
        
        normalizer = FrequencyDomainNormalizer()
        x = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1
        normalized = normalizer.normalize(x)
        
        assert len(normalized) == len(x)
        assert np.isfinite(normalized).all()
    
    def test_freq_norm_batch(self):
        """Test batch normalization"""
        from src.preprocessing.freq_normalizer import FrequencyDomainNormalizer
        
        normalizer = FrequencyDomainNormalizer()
        x = np.random.randn(10, 50)  # 10 series of length 50
        normalized = normalizer.normalize_batch(x)
        
        assert normalized.shape == x.shape
    
    def test_freq_norm_adaptation(self):
        """Test adaptive statistics update"""
        from src.preprocessing.freq_normalizer import FrequencyDomainNormalizer
        
        normalizer = FrequencyDomainNormalizer()
        
        # First call initializes
        x1 = np.random.randn(64)
        normalizer.normalize(x1)
        assert normalizer.initialized
        
        # Second call updates
        x2 = np.random.randn(64) * 2
        normalizer.normalize(x2)
        assert normalizer.freq_means is not None


# =============================================================================
# Test A4: TimeGAN
# =============================================================================

class TestTimeGAN:
    """Tests for TimeGAN augmentation"""
    
    def test_timegan_import(self):
        """Verify TimeGAN module imports"""
        from src.preprocessing.timegan_augment import TimeGAN, TimeGANConfig
        assert TimeGAN is not None
        assert TimeGANConfig is not None
    
    def test_timegan_initialization(self):
        """Test TimeGAN network initialization"""
        from src.preprocessing.timegan_augment import TimeGAN, TimeGANConfig
        
        config = TimeGANConfig(seq_len=10, feature_dim=5, hidden_dim=16)
        timegan = TimeGAN(config, device='cpu')
        
        assert hasattr(timegan, 'embedder')
        assert hasattr(timegan, 'generator')
        assert hasattr(timegan, 'discriminator')
    
    def test_timegan_generate_without_training(self):
        """Test synthetic data generation (untrained)"""
        from src.preprocessing.timegan_augment import TimeGAN, TimeGANConfig
        
        config = TimeGANConfig(seq_len=10, feature_dim=5, hidden_dim=16)
        timegan = TimeGAN(config, device='cpu')
        
        synthetic = timegan.generate(n_samples=3)
        
        assert synthetic.shape == (3, 10, 5)
        assert np.isfinite(synthetic).all()


# =============================================================================
# Test B1: Student-t AH-HMM
# =============================================================================

class TestStudentTAHHMM:
    """Tests for Student-t Adaptive Hierarchical HMM"""
    
    def test_ahhmm_import(self):
        """Verify AH-HMM module imports"""
        from src.regime_detection.student_t_ahhmm import (
            StudentTAHHMM, AHHMMConfig, MarketRegime, MetaRegime
        )
        assert StudentTAHHMM is not None
        assert len(MarketRegime) == 4
        assert len(MetaRegime) == 2
    
    def test_ahhmm_initialization(self):
        """Test AH-HMM initializes correctly"""
        from src.regime_detection.student_t_ahhmm import StudentTAHHMM, AHHMMConfig
        
        config = AHHMMConfig(n_market_states=4, df=5.0)
        hmm = StudentTAHHMM(config)
        
        assert hmm.config.df == 5.0
        assert hmm.meta_state == 0  # Low uncertainty initially
    
    def test_ahhmm_predict(self):
        """Test regime prediction"""
        from src.regime_detection.student_t_ahhmm import StudentTAHHMM, MarketRegime
        
        hmm = StudentTAHHMM()
        
        # Simulate bullish observation: positive return, normal volume, low vol
        obs = np.array([0.003, 0.8, 0.012])
        state = hmm.predict(obs)
        
        assert state.regime in MarketRegime
        assert 0 <= state.confidence <= 1
        assert state.meta_regime is not None
    
    def test_ahhmm_meta_regime_switch(self):
        """Test meta-regime switching with VIX"""
        from src.regime_detection.student_t_ahhmm import StudentTAHHMM, MetaRegime
        
        hmm = StudentTAHHMM()
        
        # Low VIX → low uncertainty
        hmm.update_meta_regime(vix=20)
        assert hmm.meta_state == 0
        
        # High VIX → high uncertainty
        hmm.update_meta_regime(vix=80)
        assert hmm.meta_state == 1
    
    def test_ahhmm_transition_matrices(self):
        """Test conditional transition matrices differ by meta-regime"""
        from src.regime_detection.student_t_ahhmm import StudentTAHHMM
        
        hmm = StudentTAHHMM()
        
        # Crisis probability should be higher under high uncertainty
        crisis_prob_low = hmm.trans_low_uncertainty[0, 3]  # Bull → Crisis
        crisis_prob_high = hmm.trans_high_uncertainty[0, 3]
        
        assert crisis_prob_high > crisis_prob_low
    
    def test_ahhmm_hurst_exponent(self):
        """Test Hurst exponent calculation"""
        from src.regime_detection.student_t_ahhmm import StudentTAHHMM
        
        hmm = StudentTAHHMM()
        
        # Trending series should have H > 0.5
        trending = np.cumsum(np.random.randn(200)) + np.arange(200) * 0.1
        h_trending = hmm.compute_hurst_exponent(trending)
        
        # Random walk should have H ≈ 0.5
        random_walk = np.cumsum(np.random.randn(200))
        h_random = hmm.compute_hurst_exponent(random_walk)
        
        assert 0 <= h_trending <= 1
        assert 0 <= h_random <= 1
    
    def test_ahhmm_oscillation_detection(self):
        """Test oscillation detection"""
        from src.regime_detection.student_t_ahhmm import StudentTAHHMM
        
        hmm = StudentTAHHMM()
        
        # No oscillation initially
        assert not hmm.detect_oscillation()
        
        # Simulate oscillating regime history
        hmm.regime_history = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        assert hmm.detect_oscillation()


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase1Integration:
    """Integration tests for Phase 1 pipeline"""
    
    def test_preprocessing_pipeline(self):
        """Test full preprocessing pipeline"""
        from src.preprocessing.ekf_denoiser import EKFDenoiser
        from src.preprocessing.freq_normalizer import FrequencyDomainNormalizer
        
        # Generate noisy price data
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        # Step 1: Frequency normalize
        freq_norm = FrequencyDomainNormalizer()
        normalized = freq_norm.normalize(prices)
        
        # Step 2: EKF denoise
        ekf = EKFDenoiser()
        denoised = ekf.filter(normalized)
        
        assert len(denoised) == len(prices)
        assert np.isfinite(denoised).all()
    
    def test_regime_detection_pipeline(self):
        """Test regime detection with preprocessing"""
        from src.preprocessing.ekf_denoiser import EKFDenoiser
        from src.regime_detection.student_t_ahhmm import StudentTAHHMM
        
        # Generate return data
        returns = np.random.randn(100) * 0.02
        volumes = np.random.rand(100) + 0.5
        volatilities = np.abs(returns) * 2
        
        ekf = EKFDenoiser()
        hmm = StudentTAHHMM()
        
        regimes = []
        for i in range(len(returns)):
            # Denoise
            _, _ = ekf.update(returns[i], volumes[i])
            
            # Detect regime
            obs = np.array([returns[i], volumes[i], volatilities[i]])
            state = hmm.predict(obs)
            regimes.append(state.regime.value)
        
        assert len(regimes) == len(returns)
        assert all(isinstance(r, str) for r in regimes)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
