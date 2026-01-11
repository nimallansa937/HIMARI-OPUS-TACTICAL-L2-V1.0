"""
HIMARI Layer 2 - Decision Engine Tests (Part D)
Tests for new Part D components
"""

import pytest
import numpy as np
import torch


class TestReturnConditioning:
    """Tests for D9 Return Conditioning."""
    
    def test_initialization(self):
        from src.decision_engine.return_conditioning import ReturnConditioner
        conditioner = ReturnConditioner()
        assert conditioner is not None
    
    def test_regime_targets(self):
        from src.decision_engine.return_conditioning import ReturnConditioner
        conditioner = ReturnConditioner()
        
        # Crisis should have lowest target
        crisis_target = conditioner.get_regime_target(0)
        assert crisis_target == 0.5
        
        # Bullish should have highest target
        bull_target = conditioner.get_regime_target(3)
        assert bull_target == 2.5
    
    def test_volatility_adjustment(self):
        from src.decision_engine.return_conditioning import ReturnConditioner
        conditioner = ReturnConditioner()
        
        # High volatility should reduce target
        conditioner.update_volatility(0.05)
        target = conditioner.get_regime_target(2)
        assert target < 2.0  # Should be reduced


class TestRsLoRA:
    """Tests for D4 rsLoRA utilities."""
    
    def test_module_import(self):
        from src.decision_engine.rslora import apply_rslora, get_lora_params
        assert apply_rslora is not None
        assert get_lora_params is not None
    
    def test_count_trainable_params(self):
        from src.decision_engine.rslora import count_trainable_params
        
        model = torch.nn.Linear(10, 5)
        count = count_trainable_params(model)
        assert count == 10 * 5 + 5  # weights + bias


class TestTrajectoryDataset:
    """Tests for D10 FinRL-DT Pipeline."""
    
    def test_initialization(self):
        from src.decision_engine.finrl_dt_pipeline import TrajectoryDataset
        dataset = TrajectoryDataset()
        assert dataset is not None
        assert len(dataset) == 0
    
    def test_add_trajectory(self):
        from src.decision_engine.finrl_dt_pipeline import TrajectoryDataset, TrajectoryDatasetConfig
        
        config = TrajectoryDatasetConfig(context_length=10)
        dataset = TrajectoryDataset(config)
        
        # Add a trajectory
        states = np.random.randn(100, 64).astype(np.float32)
        actions = np.random.randint(0, 3, 100)
        rewards = np.random.randn(100).astype(np.float32) * 0.01
        
        dataset.add_trajectory(states, actions, rewards)
        
        assert len(dataset.trajectories) == 1
        assert len(dataset) > 0
    
    def test_sample_batch(self):
        from src.decision_engine.finrl_dt_pipeline import TrajectoryDataset, TrajectoryDatasetConfig
        
        config = TrajectoryDatasetConfig(context_length=10)
        dataset = TrajectoryDataset(config)
        
        # Add trajectories
        for _ in range(5):
            states = np.random.randn(50, 64).astype(np.float32)
            actions = np.random.randint(0, 3, 50)
            rewards = np.random.randn(50).astype(np.float32) * 0.01
            dataset.add_trajectory(states, actions, rewards)
        
        # Sample batch
        batch = dataset.sample_batch(16)
        
        assert batch['states'].shape == (16, 10, 64)
        assert batch['actions'].shape == (16, 10)


class TestDisagreementScaler:
    """Tests for D8 Disagreement Scaling."""
    
    def test_initialization(self):
        from src.decision_engine.decision_engine import DisagreementScaler
        scaler = DisagreementScaler()
        assert scaler is not None
    
    def test_low_disagreement(self):
        from src.decision_engine.decision_engine import DisagreementScaler
        scaler = DisagreementScaler()
        
        # Low disagreement should not scale down confidence
        scaled, _ = scaler.scale_confidence(0.8, 0.1, np.array([0.8, 0.1, 0.1]))
        assert scaled == 0.8
    
    def test_high_disagreement(self):
        from src.decision_engine.decision_engine import DisagreementScaler
        scaler = DisagreementScaler()
        
        # High disagreement should scale down confidence
        scaled, _ = scaler.scale_confidence(0.8, 0.8, np.array([0.33, 0.33, 0.34]))
        assert scaled < 0.8


class TestSharpeWeightedEnsemble:
    """Tests for D7 Sharpe-Weighted Voting."""
    
    def test_initialization(self):
        from src.decision_engine.decision_engine import SharpeWeightedEnsemble
        ensemble = SharpeWeightedEnsemble(agent_names=['agent1', 'agent2'])
        assert ensemble is not None
        assert len(ensemble._weights) == 2
    
    def test_vote(self):
        from src.decision_engine.decision_engine import SharpeWeightedEnsemble
        ensemble = SharpeWeightedEnsemble(agent_names=['agent1', 'agent2'])
        
        outputs = {
            'agent1': (np.array([0.7, 0.2, 0.1]), 0.7),
            'agent2': (np.array([0.6, 0.3, 0.1]), 0.6)
        }
        
        voted_action, conf, info = ensemble.vote(outputs)
        
        assert voted_action == 0  # SELL has highest prob
        assert 0 < conf < 1


class TestDecisionEngine:
    """Integration tests for Decision Engine."""
    
    def test_initialization(self):
        from src.decision_engine.decision_engine import DecisionEngine, DecisionEngineConfig
        
        config = DecisionEngineConfig(
            device='cpu',
            feature_dim=64,
            use_flag_trader=False,
            use_cgdt=False,
            use_cql=False,
            use_ppo=False,
            use_sac=False
        )
        
        engine = DecisionEngine(config)
        assert engine is not None
    
    def test_decide_with_no_agents(self):
        from src.decision_engine.decision_engine import DecisionEngine, DecisionEngineConfig
        
        config = DecisionEngineConfig(
            device='cpu',
            feature_dim=64,
            use_flag_trader=False,
            use_cgdt=False,
            use_cql=False,
            use_ppo=False,
            use_sac=False
        )
        
        engine = DecisionEngine(config)
        features = np.random.randn(64).astype(np.float32)
        
        result = engine.decide(features, regime=2)
        
        # Should return HOLD with no agents
        assert result.action is not None
        assert 0 <= result.confidence <= 1
        assert result.latency_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
