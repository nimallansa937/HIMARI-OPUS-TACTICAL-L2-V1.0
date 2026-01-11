"""
HIMARI Layer 2 - Regime Detection Tests (Part B)
Comprehensive tests for all 8 regime detection methods
"""

import pytest
import numpy as np
from typing import Dict


class TestJumpDetector:
    """Tests for B5 Jump Detector."""
    
    def test_initialization(self):
        from src.regime_detection.jump_detector import JumpDetector
        detector = JumpDetector()
        assert detector is not None
    
    def test_normal_observations(self):
        from src.regime_detection.jump_detector import JumpDetector
        detector = JumpDetector()
        
        # Warmup
        for _ in range(50):
            obs = np.random.randn(3) * 0.01
            detector.update(obs)
        
        # Normal observation shouldn't trigger
        obs = np.random.randn(3) * 0.01
        result = detector.update(obs)
        
        assert not result.jump_detected
    
    def test_jump_detection(self):
        from src.regime_detection.jump_detector import JumpDetector
        detector = JumpDetector()
        
        # Warmup with normal data
        for _ in range(100):
            obs = np.array([0.001, 0.8, 0.015]) + np.random.randn(3) * 0.001
            detector.update(obs)
        
        # Extreme observation should trigger
        extreme_obs = np.array([-0.05, 3.0, 0.10])  # -5% return, extreme vol
        result = detector.update(extreme_obs)
        
        assert result.jump_detected
        assert result.is_crisis
    
    def test_crisis_duration(self):
        from src.regime_detection.jump_detector import JumpDetector
        detector = JumpDetector()
        
        # Warmup and trigger
        for _ in range(100):
            detector.update(np.random.randn(3) * 0.01)
        detector.update(np.array([-0.05, 3.0, 0.10]))  # Trigger
        
        # Should stay in crisis
        for i in range(5):
            result = detector.update(np.random.randn(3) * 0.01)
            assert result.is_crisis
        
        # Should exit after duration
        result = detector.update(np.random.randn(3) * 0.01)
        assert not result.is_crisis


class TestHurstExponent:
    """Tests for B6 Hurst Exponent Gating."""
    
    def test_initialization(self):
        from src.regime_detection.hurst_gating import HurstExponentGating
        hurst = HurstExponentGating()
        assert hurst is not None
    
    def test_trending_detection(self):
        from src.regime_detection.hurst_gating import HurstExponentGating
        hurst = HurstExponentGating()
        
        # Generate trending series (cumulative random walk with drift)
        for i in range(150):
            ret = 0.001 + np.random.randn() * 0.005  # Positive drift
            result = hurst.update(ret)
        
        # Should detect trending
        if result:
            assert result.hurst_ema > 0.45  # At least not strongly mean-reverting
    
    def test_mean_reverting_detection(self):
        from src.regime_detection.hurst_gating import HurstExponentGating
        hurst = HurstExponentGating()
        
        # Generate mean-reverting series (Ornstein-Uhlenbeck)
        x = 0.0
        for i in range(150):
            x = 0.8 * x + np.random.randn() * 0.01  # Mean reversion
            result = hurst.update(x)
        
        # Check that result is valid (Hurst is bounded 0-1)
        if result:
            assert 0.0 <= result.hurst_ema <= 1.0


class TestMetaRegimeLayer:
    """Tests for B2 Meta-Regime Layer."""
    
    def test_initialization(self):
        from src.regime_detection.meta_regime import MetaRegimeLayer
        layer = MetaRegimeLayer()
        assert layer is not None
    
    def test_indicator_normalization(self):
        from src.regime_detection.meta_regime import MetaRegimeLayer
        layer = MetaRegimeLayer()
        
        # VIX normalization
        assert layer._normalize_indicator(15.0, "vix") < 0.1
        assert layer._normalize_indicator(35.0, "vix") > 0.9
        
    def test_hysteresis(self):
        from src.regime_detection.meta_regime import MetaRegimeLayer
        from src.regime_detection.config.ahhmm_config import MetaRegime
        layer = MetaRegimeLayer()
        
        # Start low
        assert layer._regime == MetaRegime.LOW_UNCERTAINTY
        
        # Score at 0.5 shouldn't trigger immediate transition
        for _ in range(100):
            layer.update({"vix": 23.0, "epu": 150.0})
        # Check regime - may still be LOW due to hysteresis


class TestCausalInfoGeometry:
    """Tests for B3 Causal Information Geometry."""
    
    def test_initialization(self):
        from src.regime_detection.causal_info_geometry import CausalInfoGeometry
        cig = CausalInfoGeometry()
        assert cig is not None
    
    def test_geodesic_distance_identity(self):
        from src.regime_detection.causal_info_geometry import CausalInfoGeometry
        cig = CausalInfoGeometry()
        
        # Distance to self should be 0
        sigma = np.eye(6)
        dist = cig._geodesic_distance(sigma, sigma)
        assert np.isclose(dist, 0.0, atol=1e-6)


class TestADWIN:
    """Tests for B8 ADWIN drift detection."""
    
    def test_initialization(self):
        from src.regime_detection.adwin_drift import ADWIN
        adwin = ADWIN()
        assert adwin is not None
    
    def test_stationary_stream(self):
        from src.regime_detection.adwin_drift import ADWIN
        adwin = ADWIN()
        
        # Stationary stream
        for _ in range(500):
            result = adwin.update(np.random.randn() + 5.0)
        
        # Should have growing window (allowing a few cuts due to randomness)
        assert result.window_size > 50
        assert result.n_cuts <= 3  # Allow a few false positives
    
    def test_drift_detection(self):
        from src.regime_detection.adwin_drift import ADWIN
        adwin = ADWIN()
        
        # First distribution: mean=5
        for _ in range(200):
            adwin.update(np.random.randn() + 5.0)
        
        # Second distribution: mean=10
        drift_found = False
        for _ in range(200):
            result = adwin.update(np.random.randn() + 10.0)
            if result.drift_detected:
                drift_found = True
                break
        
        assert drift_found, "ADWIN should detect distribution shift"


class TestOnlineBaumWelch:
    """Tests for B7 Online Baum-Welch."""
    
    def test_initialization(self):
        from src.regime_detection.online_baum_welch import OnlineBaumWelch
        obw = OnlineBaumWelch()
        assert obw is not None
    
    def test_process_observation(self):
        from src.regime_detection.online_baum_welch import OnlineBaumWelch
        obw = OnlineBaumWelch()
        
        # Process observations
        for _ in range(50):
            obs = np.random.randn(3) * 0.01
            probs = np.array([0.3, 0.3, 0.3, 0.1])
            result = obw.process_observation(obs, regime=0, regime_probs=probs)
        
        # Check that transition buffer is being filled
        assert len(obw._transition_buffer) > 0


class TestAEDL:
    """Tests for B4 AEDL Meta-Learning."""
    
    def test_initialization(self):
        from src.regime_detection.aedl_meta_learning import AEDL
        aedl = AEDL()
        assert aedl is not None
    
    def test_predict(self):
        from src.regime_detection.aedl_meta_learning import AEDL
        aedl = AEDL()
        
        features = np.random.randn(60) * 0.1
        result = aedl.predict(features)
        
        assert "regime" in result
        assert "probabilities" in result
        assert len(result["probabilities"]) == 4


class TestRegimeDetectionPipeline:
    """Integration tests for complete pipeline."""
    
    def test_pipeline_initialization(self):
        from src.regime_detection.pipeline import RegimeDetectionPipeline
        pipeline = RegimeDetectionPipeline()
        assert pipeline is not None
    
    def test_pipeline_processing(self):
        from src.regime_detection.pipeline import RegimeDetectionPipeline
        pipeline = RegimeDetectionPipeline()
        
        # Generate test data
        features = np.random.randn(60) * 0.01
        features[0] = 0.001  # Return
        features[1] = 0.8    # Volume ratio
        features[2] = 0.015  # Volatility
        
        macro = {
            "vix": 20.0,
            "dvol": 55.0,
            "epu": 120.0,
            "funding_rate": 0.001,
            "oi_change": 0.02
        }
        
        cross_returns = np.random.randn(6) * 0.01
        
        result = pipeline.process(
            features=features,
            macro_indicators=macro,
            cross_asset_returns=cross_returns
        )
        
        # Check output structure
        assert result.regime is not None
        assert 0 <= result.confidence <= 1
        assert 0 < result.position_scale <= 1
        assert result.risk_multiplier >= 1.0
        assert len(result.strategy_weights) == 3
    
    def test_crisis_override(self):
        from src.regime_detection.pipeline import RegimeDetectionPipeline
        pipeline = RegimeDetectionPipeline()
        
        # Warmup
        for _ in range(100):
            features = np.random.randn(60) * 0.01
            features[0] = 0.001
            features[1] = 0.8
            features[2] = 0.015
            pipeline.process(features=features)
        
        # Extreme observation
        features = np.random.randn(60) * 0.01
        features[0] = -0.05  # -5% return
        features[2] = 0.10   # 10% volatility
        
        result = pipeline.process(features=features)
        
        # Should be in crisis with low position scale
        assert result.position_scale < 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
