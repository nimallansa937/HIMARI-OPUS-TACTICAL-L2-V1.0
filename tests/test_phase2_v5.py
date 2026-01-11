"""
HIMARI Layer 2 v5.0 - Phase 2 Unit Tests
Tests for Decision Engine and Uncertainty modules

Run with: python -m pytest tests/test_phase2_v5.py -v
"""

import pytest
import numpy as np
import torch


# =============================================================================
# Test D1: FLAG-TRADER
# =============================================================================

class TestFLAGTrader:
    """Tests for FLAG-TRADER LLM policy network"""
    
    def test_flag_trader_import(self):
        """Verify FLAG-TRADER imports correctly"""
        from src.decision_engine.flag_trader import (
            FLAGTrader, FLAGTraderConfig, TradeAction
        )
        assert FLAGTrader is not None
        assert TradeAction.HOLD is not None  # Check enum exists
    
    def test_flag_trader_fallback_init(self):
        """Test FLAG-TRADER initializes in fallback mode"""
        from src.decision_engine.flag_trader import FLAGTrader, FLAGTraderConfig
        
        config = FLAGTraderConfig(use_fallback=True)
        trader = FLAGTrader(config, device='cpu')
        
        assert not trader.use_llm
        assert hasattr(trader, 'fallback')
    
    def test_flag_trader_get_action(self):
        """Test get_action returns valid action"""
        from src.decision_engine.flag_trader import FLAGTrader, FLAGTraderConfig, TradeAction
        
        config = FLAGTraderConfig(use_fallback=True)
        trader = FLAGTrader(config, device='cpu')
        
        market_state = {
            'price_change_1h': 0.02,
            'price_change_4h': 0.05,
            'volume_ratio': 1.5,
            'rsi': 65,
            'volatility': 0.02
        }
        
        action, confidence, value = trader.get_action(market_state)
        
        assert action in TradeAction
        assert 0 <= confidence <= 1
        assert isinstance(value, float)
    
    def test_flag_trader_forward_batch(self):
        """Test forward pass with batch of states"""
        from src.decision_engine.flag_trader import FLAGTrader, FLAGTraderConfig
        
        config = FLAGTraderConfig(use_fallback=True, num_actions=3)
        trader = FLAGTrader(config, device='cpu')
        
        states = [{'price_change_1h': 0.01 * i} for i in range(4)]
        logits, values = trader.forward(states)
        
        assert logits.shape == (4, 3)
        assert values.shape == (4, 1)


# =============================================================================
# Test D2: Critic-Guided Decision Transformer
# =============================================================================

class TestCGDT:
    """Tests for Critic-Guided Decision Transformer"""
    
    def test_cgdt_import(self):
        """Verify CGDT imports correctly"""
        from src.decision_engine.cgdt import (
            CriticGuidedDecisionTransformer, CGDTConfig
        )
        assert CriticGuidedDecisionTransformer is not None
    
    def test_cgdt_initialization(self):
        """Test CGDT initializes correctly"""
        from src.decision_engine.cgdt import (
            CriticGuidedDecisionTransformer, CGDTConfig
        )
        
        config = CGDTConfig(
            state_dim=10, action_dim=3, hidden_dim=32, 
            n_layers=2, context_length=20
        )
        cgdt = CriticGuidedDecisionTransformer(config)
        
        assert hasattr(cgdt, 'action_head')
        assert hasattr(cgdt, 'critic_head')
    
    def test_cgdt_forward(self):
        """Test CGDT forward pass"""
        from src.decision_engine.cgdt import (
            CriticGuidedDecisionTransformer, CGDTConfig
        )
        
        config = CGDTConfig(
            state_dim=10, action_dim=3, hidden_dim=32, 
            n_layers=2, context_length=20
        )
        cgdt = CriticGuidedDecisionTransformer(config)
        
        batch_size, seq_len = 2, 5
        states = torch.randn(batch_size, seq_len, 10)
        actions = torch.randint(0, 3, (batch_size, seq_len))
        returns_to_go = torch.randn(batch_size, seq_len, 1)
        
        action_preds, return_preds, q_values = cgdt(states, actions, returns_to_go)
        
        assert action_preds.shape == (batch_size, seq_len, 3)
        assert return_preds.shape == (batch_size, seq_len, 1)
        assert q_values.shape == (batch_size, seq_len, 3)
    
    def test_cgdt_get_action(self):
        """Test CGDT action selection"""
        from src.decision_engine.cgdt import (
            CriticGuidedDecisionTransformer, CGDTConfig
        )
        
        config = CGDTConfig(
            state_dim=10, action_dim=3, hidden_dim=32, 
            n_layers=2, context_length=20
        )
        cgdt = CriticGuidedDecisionTransformer(config)
        
        states = torch.randn(1, 5, 10)
        actions = torch.randint(0, 3, (1, 5))
        returns_to_go = torch.randn(1, 5, 1)
        
        action, confidence = cgdt.get_action(states, actions, returns_to_go)
        
        assert 0 <= action < 3
        assert 0 <= confidence <= 1


# =============================================================================
# Test D3: Conservative Q-Learning
# =============================================================================

class TestCQLAgent:
    """Tests for Conservative Q-Learning agent"""
    
    def test_cql_import(self):
        """Verify CQL imports correctly"""
        from src.decision_engine.cql_agent import CQLAgent, CQLConfig
        assert CQLAgent is not None
    
    def test_cql_initialization(self):
        """Test CQL initializes correctly"""
        from src.decision_engine.cql_agent import CQLAgent, CQLConfig
        
        config = CQLConfig(state_dim=10, action_dim=3, hidden_dim=32)
        agent = CQLAgent(config, device='cpu')
        
        assert hasattr(agent, 'q1')
        assert hasattr(agent, 'q2')
        assert hasattr(agent, 'q1_target')
    
    def test_cql_select_action(self):
        """Test CQL action selection"""
        from src.decision_engine.cql_agent import CQLAgent, CQLConfig
        
        config = CQLConfig(state_dim=10, action_dim=3, hidden_dim=32)
        agent = CQLAgent(config, device='cpu')
        
        state = torch.randn(10)
        action = agent.select_action(state)
        
        assert 0 <= action < 3
    
    def test_cql_action_with_confidence(self):
        """Test CQL action with confidence score"""
        from src.decision_engine.cql_agent import CQLAgent, CQLConfig
        
        config = CQLConfig(state_dim=10, action_dim=3, hidden_dim=32)
        agent = CQLAgent(config, device='cpu')
        
        state = torch.randn(10)
        action, confidence = agent.get_action_with_confidence(state)
        
        assert 0 <= action < 3
        assert 0 <= confidence <= 1
    
    def test_cql_compute_loss(self):
        """Test CQL loss computation"""
        from src.decision_engine.cql_agent import CQLAgent, CQLConfig
        
        config = CQLConfig(state_dim=10, action_dim=3, hidden_dim=32)
        agent = CQLAgent(config, device='cpu')
        
        batch_size = 8
        states = torch.randn(batch_size, 10)
        actions = torch.randint(0, 3, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, 10)
        dones = torch.zeros(batch_size)
        
        losses = agent.compute_cql_loss(states, actions, rewards, next_states, dones)
        
        assert 'q1_loss' in losses
        assert 'cql_loss' in losses
        assert losses['q1_loss'].requires_grad


# =============================================================================
# Test F1: CT-SSF
# =============================================================================

class TestCTSSF:
    """Tests for CT-SSF latent conformal prediction"""
    
    def test_ctssf_import(self):
        """Verify CT-SSF imports correctly"""
        from src.uncertainty.ct_ssf import CTSSF, CTSSFConfig
        assert CTSSF is not None
    
    def test_ctssf_initialization(self):
        """Test CT-SSF initializes correctly"""
        from src.uncertainty.ct_ssf import CTSSF, CTSSFConfig
        
        config = CTSSFConfig(input_dim=10, latent_dim=8, alpha=0.10)
        ctssf = CTSSF(config, device='cpu')
        
        assert hasattr(ctssf, 'encoder')
        assert not ctssf.calibrated
    
    def test_ctssf_calibrate(self):
        """Test CT-SSF calibration"""
        from src.uncertainty.ct_ssf import CTSSF, CTSSFConfig
        
        config = CTSSFConfig(input_dim=10, latent_dim=8, alpha=0.10)
        ctssf = CTSSF(config, device='cpu')
        
        n = 50
        features = torch.randn(n, 10)
        predictions = torch.randn(n)
        targets = predictions + torch.randn(n) * 0.1
        
        ctssf.calibrate(features, predictions, targets)
        
        assert ctssf.calibrated
        assert ctssf.cal_scores is not None
    
    def test_ctssf_predict_interval(self):
        """Test CT-SSF interval prediction"""
        from src.uncertainty.ct_ssf import CTSSF, CTSSFConfig
        
        config = CTSSFConfig(input_dim=10, latent_dim=8, alpha=0.10)
        ctssf = CTSSF(config, device='cpu')
        
        # Calibrate
        n = 50
        cal_features = torch.randn(n, 10)
        cal_predictions = torch.randn(n)
        cal_targets = cal_predictions + torch.randn(n) * 0.1
        ctssf.calibrate(cal_features, cal_predictions, cal_targets)
        
        # Predict
        test_features = torch.randn(5, 10)
        test_predictions = torch.randn(5)
        
        lower, upper = ctssf.predict_interval(test_features, test_predictions)
        
        assert lower.shape == (5,)
        assert upper.shape == (5,)
        assert (upper > lower).all()
    
    def test_ctssf_coverage_rate(self):
        """Test CT-SSF coverage calculation"""
        from src.uncertainty.ct_ssf import CTSSF, CTSSFConfig
        
        config = CTSSFConfig(input_dim=10, latent_dim=8, alpha=0.10)
        ctssf = CTSSF(config, device='cpu')
        
        # Calibrate
        n = 100
        features = torch.randn(n, 10)
        targets = torch.randn(n)
        predictions = targets + torch.randn(n) * 0.05  # Small noise
        
        ctssf.calibrate(features, predictions, targets)
        
        # Test coverage
        test_n = 50
        test_features = torch.randn(test_n, 10)
        test_targets = torch.randn(test_n)
        test_predictions = test_targets + torch.randn(test_n) * 0.05
        
        coverage = ctssf.get_coverage_rate(test_features, test_predictions, test_targets)
        
        assert 0 <= coverage <= 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase2Integration:
    """Integration tests for Phase 2 components"""
    
    def test_decision_engine_pipeline(self):
        """Test decision engine ensemble pipeline"""
        from src.decision_engine.flag_trader import FLAGTrader, FLAGTraderConfig
        from src.decision_engine.cql_agent import CQLAgent, CQLConfig
        
        # Create agents
        flag_config = FLAGTraderConfig(use_fallback=True, num_actions=3)
        flag_trader = FLAGTrader(flag_config, device='cpu')
        
        cql_config = CQLConfig(state_dim=60, action_dim=3, hidden_dim=64)
        cql_agent = CQLAgent(cql_config, device='cpu')
        
        # Get actions from both
        market_state = {'price_change_1h': 0.02, 'rsi': 60}
        state_tensor = torch.randn(60)
        
        flag_action, flag_conf, _ = flag_trader.get_action(market_state)
        cql_action, cql_conf = cql_agent.get_action_with_confidence(state_tensor)
        
        # Both should return valid actions
        assert flag_action is not None
        assert 0 <= cql_action < 3
    
    def test_uncertainty_with_decision(self):
        """Test uncertainty quantification with decision engine"""
        from src.decision_engine.flag_trader import FLAGTrader, FLAGTraderConfig
        from src.uncertainty.ct_ssf import CTSSF, CTSSFConfig
        
        # Setup
        trader_config = FLAGTraderConfig(use_fallback=True)
        trader = FLAGTrader(trader_config, device='cpu')
        
        uq_config = CTSSFConfig(input_dim=60, latent_dim=16, alpha=0.10)
        uq = CTSSF(uq_config, device='cpu')
        
        # Calibrate uncertainty
        n = 50
        features = torch.randn(n, 60)
        predictions = torch.randn(n)
        targets = predictions + torch.randn(n) * 0.1
        uq.calibrate(features, predictions, targets)
        
        # Get uncertainty for decision
        test_features = torch.randn(1, 60)
        uncertainty = uq.get_uncertainty(test_features)
        
        # Use uncertainty for position sizing (conceptual)
        market_state = {'uncertainty': uncertainty.item()}
        action, confidence, _ = trader.get_action(market_state)
        
        assert action is not None
        assert uncertainty.item() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
