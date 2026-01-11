"""
HIMARI Layer 2 v5.0 - Phase 3 Unit Tests
Tests for Safety and Risk Management modules

Run with: python -m pytest tests/test_phase3_v5.py -v
"""

import pytest
import numpy as np
import torch


# =============================================================================
# Test H1: EVT-GPD Tail Risk
# =============================================================================

class TestEVTGPD:
    """Tests for Extreme Value Theory tail risk"""
    
    def test_evt_import(self):
        """Verify EVT-GPD imports correctly"""
        from src.risk.evt_gpd import EVTGPDRisk, EVTConfig
        assert EVTGPDRisk is not None
    
    def test_evt_initialization(self):
        """Test EVT initializes correctly"""
        from src.risk.evt_gpd import EVTGPDRisk, EVTConfig
        
        config = EVTConfig(threshold_percentile=0.95)
        evt = EVTGPDRisk(config)
        
        assert evt.config.threshold_percentile == 0.95
        assert not evt._fitted
    
    def test_evt_fit(self):
        """Test EVT fitting to data"""
        from src.risk.evt_gpd import EVTGPDRisk
        
        evt = EVTGPDRisk()
        
        # Simulate fat-tailed returns
        np.random.seed(42)
        returns = np.concatenate([
            np.random.randn(900) * 0.02,
            np.random.randn(100) * 0.08  # Tail events
        ])
        
        evt.fit(returns)
        
        assert evt._fitted
        assert evt.threshold is not None
        assert evt.sigma > 0
    
    def test_evt_var(self):
        """Test VaR calculation"""
        from src.risk.evt_gpd import EVTGPDRisk
        
        evt = EVTGPDRisk()
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02
        evt.fit(returns)
        
        var_99 = evt.get_var(0.99)
        var_95 = evt.get_var(0.95)
        
        assert var_99 > 0
        assert var_99 > var_95  # 99% VaR should be higher
    
    def test_evt_cvar(self):
        """Test CVaR calculation"""
        from src.risk.evt_gpd import EVTGPDRisk
        
        evt = EVTGPDRisk()
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02
        evt.fit(returns)
        
        var_99 = evt.get_var(0.99)
        cvar_99 = evt.get_cvar(0.99)
        
        assert cvar_99 >= var_99  # CVaR should be >= VaR
    
    def test_dynamic_kelly(self):
        """Test Dynamic Kelly fraction"""
        from src.risk.evt_gpd import DynamicKellyFraction
        
        kelly = DynamicKellyFraction(max_leverage=3.0)
        
        # Positive expected return, reasonable vol
        fraction = kelly.compute_fraction(0.001, 0.02)
        
        assert 0 <= fraction <= 3.0


# =============================================================================
# Test H3: DCC-GARCH
# =============================================================================

class TestDCCGARCH:
    """Tests for DCC-GARCH correlation"""
    
    def test_dcc_import(self):
        """Verify DCC-GARCH imports correctly"""
        from src.risk.dcc_garch import DCCGARCH, DCCConfig
        assert DCCGARCH is not None
    
    def test_dcc_initialization(self):
        """Test DCC-GARCH initializes correctly"""
        from src.risk.dcc_garch import DCCGARCH, DCCConfig
        
        config = DCCConfig()
        dcc = DCCGARCH(config, n_assets=3)
        
        assert dcc.n_assets == 3
        assert not dcc._fitted
    
    def test_dcc_fit(self):
        """Test DCC-GARCH fitting"""
        from src.risk.dcc_garch import DCCGARCH
        
        dcc = DCCGARCH(n_assets=2)
        
        np.random.seed(42)
        returns = np.random.randn(300, 2) * 0.02
        
        dcc.fit(returns)
        
        assert dcc._fitted
        assert dcc.h is not None
    
    def test_dcc_correlation(self):
        """Test correlation matrix output"""
        from src.risk.dcc_garch import DCCGARCH
        
        dcc = DCCGARCH(n_assets=2)
        np.random.seed(42)
        returns = np.random.randn(300, 2) * 0.02
        dcc.fit(returns)
        
        corr = dcc.get_correlation()
        
        assert corr.shape == (2, 2)
        assert np.allclose(np.diag(corr), 1.0)  # Diagonal = 1
        assert np.allclose(corr, corr.T)  # Symmetric
    
    def test_dcc_covariance(self):
        """Test covariance matrix output"""
        from src.risk.dcc_garch import DCCGARCH
        
        dcc = DCCGARCH(n_assets=2)
        np.random.seed(42)
        returns = np.random.randn(300, 2) * 0.02
        dcc.fit(returns)
        
        cov = dcc.get_covariance()
        
        assert cov.shape == (2, 2)
        assert np.all(np.diag(cov) > 0)  # Positive variances
    
    def test_dcc_portfolio_vol(self):
        """Test portfolio volatility calculation"""
        from src.risk.dcc_garch import DCCGARCH
        
        dcc = DCCGARCH(n_assets=2)
        np.random.seed(42)
        returns = np.random.randn(300, 2) * 0.02
        dcc.fit(returns)
        
        weights = np.array([0.5, 0.5])
        vol = dcc.get_portfolio_volatility(weights)
        
        assert vol > 0


# =============================================================================
# Test I1: Simplex Safety System
# =============================================================================

class TestSimplexSafety:
    """Tests for 4-level Simplex Safety System"""
    
    def test_simplex_import(self):
        """Verify Simplex imports correctly"""
        from src.safety.simplex_safety import (
            SimplexSafetySystem, SafetyLevel, SafetyConfig
        )
        assert SimplexSafetySystem is not None
        assert len(SafetyLevel) == 4
    
    def test_simplex_initialization(self):
        """Test Simplex initializes correctly"""
        from src.safety.simplex_safety import SimplexSafetySystem, SafetyConfig
        
        config = SafetyConfig(max_drawdown=0.05, max_leverage=3.0)
        safety = SimplexSafetySystem(config)
        
        assert safety.config.max_drawdown == 0.05
    
    def test_simplex_check_invariants(self):
        """Test invariant checking"""
        from src.safety.simplex_safety import (
            SimplexSafetySystem, MarketState, SafetyViolation
        )
        
        safety = SimplexSafetySystem()
        
        # Normal state - no violations
        normal_state = MarketState(
            current_position=0.1,
            current_leverage=1.5,
            current_drawdown=0.02,
            current_volatility=0.02
        )
        violations = safety.check_invariants(normal_state)
        assert len(violations) == 0
        
        # Violation state
        risky_state = MarketState(
            current_position=0.5,  # Too large
            current_leverage=5.0,  # Too high
            current_drawdown=0.10,  # Exceeds limit
            current_volatility=0.10
        )
        violations = safety.check_invariants(risky_state)
        assert len(violations) > 0
    
    def test_simplex_level_determination(self):
        """Test safety level determination"""
        from src.safety.simplex_safety import (
            SimplexSafetySystem, MarketState, SafetyLevel
        )
        
        safety = SimplexSafetySystem()
        
        # Low risk - Level 0
        safe_state = MarketState(
            current_drawdown=0.01,
            model_uncertainty=0.1
        )
        result = safety.evaluate(safe_state)
        assert result.current_level == SafetyLevel.LEVEL_0
        
        # High risk - Level 3
        crisis_state = MarketState(
            current_drawdown=0.10,  # Over limit
            current_leverage=5.0,   # Over limit
            model_uncertainty=0.8
        )
        result = safety.evaluate(crisis_state)
        assert result.current_level == SafetyLevel.LEVEL_3
    
    def test_simplex_position_multiplier(self):
        """Test position multiplier by level"""
        from src.safety.simplex_safety import SimplexSafetySystem, MarketState
        
        safety = SimplexSafetySystem()
        
        # Normal state
        state = MarketState(current_drawdown=0.01)
        result = safety.evaluate(state)
        assert result.position_multiplier == 1.0
        
        # Crisis state
        crisis = MarketState(current_drawdown=0.10, current_leverage=5.0)
        result = safety.evaluate(crisis)
        assert result.position_multiplier < 0.5
    
    def test_simplex_action_filter(self):
        """Test action filtering"""
        from src.safety.simplex_safety import (
            SimplexSafetySystem, MarketState, SafetyState, SafetyLevel
        )
        
        safety = SimplexSafetySystem()
        
        # At Level 3, BUY should be blocked
        crisis_state = SafetyState(
            current_level=SafetyLevel.LEVEL_3,
            violations=[],
            risk_score=0.8,
            uncertainty_score=0.7,
            is_safe=False,
            position_multiplier=0.1
        )
        
        action, size = safety.filter_action("BUY", 1.0, crisis_state)
        assert action == "HOLD"
        assert size == 0.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase3Integration:
    """Integration tests for Phase 3 components"""
    
    def test_risk_safety_pipeline(self):
        """Test risk estimation + safety system pipeline"""
        from src.risk.evt_gpd import EVTGPDRisk
        from src.safety.simplex_safety import SimplexSafetySystem, MarketState
        
        # Setup
        evt = EVTGPDRisk()
        safety = SimplexSafetySystem()
        
        # Fit risk model
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02
        evt.fit(returns)
        
        # Get risk metrics
        var_99 = evt.get_var(0.99)
        
        # Create market state with risk info
        state = MarketState(
            current_drawdown=var_99 / 2,  # Half of VaR in drawdown
            current_volatility=np.std(returns)
        )
        
        # Evaluate safety
        result = safety.evaluate(state)
        
        assert result.current_level is not None
        assert 0 <= result.risk_score <= 1
    
    def test_correlation_risk_integration(self):
        """Test correlation-aware risk estimation"""
        from src.risk.dcc_garch import DCCGARCH
        from src.risk.evt_gpd import DynamicKellyFraction
        
        # Setup
        dcc = DCCGARCH(n_assets=2)
        kelly = DynamicKellyFraction()
        
        np.random.seed(42)
        returns = np.random.randn(300, 2) * 0.02
        dcc.fit(returns)
        
        # Portfolio volatility
        weights = np.array([0.5, 0.5])
        port_vol = dcc.get_portfolio_volatility(weights)
        
        # Kelly fraction based on portfolio risk
        fraction = kelly.compute_fraction(0.001, port_vol)
        
        assert fraction > 0
        assert port_vol > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
