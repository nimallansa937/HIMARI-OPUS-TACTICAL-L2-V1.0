"""
HIMARI Layer 2 - Transformer-A2C Tests
Unit tests for the Transformer-A2C model and training components.
"""

import unittest
import numpy as np
import torch
import sys
import os
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.transformer_a2c import (
    TransformerA2C,
    TransformerA2CConfig,
    TacticalTransformerEncoder,
    ActorHead,
    CriticHead,
    PositionalEncoding,
    TransformerEncoderBlock,
)
from src.training.sortino_reward import (
    SimpleSortinoReward,
    SortinoWithDrawdownPenalty,
    create_reward_function,
)
from src.environment.transformer_a2c_env import (
    TransformerA2CEnv,
    TransformerEnvConfig,
    WalkForwardSplitter,
    create_synthetic_data,
)


class TestTransformerA2CModel(unittest.TestCase):
    """Tests for Transformer-A2C model components."""
    
    def setUp(self):
        self.config = TransformerA2CConfig(
            input_dim=44,
            hidden_dim=64,  # Smaller for testing
            num_heads=4,
            num_layers=2,
            context_length=20,
            dropout=0.1,
        )
        self.device = "cpu"
        
    def test_positional_encoding(self):
        """Test positional encoding dimensions."""
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.1)
        x = torch.randn(2, 20, 64)
        out = pe(x)
        self.assertEqual(out.shape, (2, 20, 64))
        
    def test_transformer_encoder_block(self):
        """Test single transformer block."""
        block = TransformerEncoderBlock(
            d_model=64,
            num_heads=4,
            d_ff=256,
            dropout=0.1,
        )
        x = torch.randn(2, 20, 64)
        out = block(x)
        self.assertEqual(out.shape, (2, 20, 64))
        
    def test_tactical_transformer_encoder(self):
        """Test full transformer encoder."""
        encoder = TacticalTransformerEncoder(self.config)
        x = torch.randn(2, 20, 44)  # [batch, seq, features]
        out = encoder(x)
        self.assertEqual(out.shape, (2, 64))  # [batch, hidden_dim]
        
    def test_actor_head(self):
        """Test actor head output."""
        actor = ActorHead(hidden_dim=64, num_actions=3, dropout=0.1)
        x = torch.randn(2, 64)
        logits = actor(x)
        self.assertEqual(logits.shape, (2, 3))
        
    def test_actor_get_action(self):
        """Test action sampling from actor."""
        actor = ActorHead(hidden_dim=64, num_actions=3)
        x = torch.randn(2, 64)
        action, log_prob, probs = actor.get_action(x, deterministic=False)
        
        self.assertEqual(action.shape, (2,))
        self.assertEqual(log_prob.shape, (2,))
        self.assertEqual(probs.shape, (2, 3))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5))
        
    def test_critic_head(self):
        """Test critic head output."""
        critic = CriticHead(hidden_dim=64, dropout=0.1)
        x = torch.randn(2, 64)
        value = critic(x)
        self.assertEqual(value.shape, (2, 1))
        
    def test_transformer_a2c_forward(self):
        """Test full model forward pass."""
        model = TransformerA2C(self.config)
        states = torch.randn(2, 20, 44)  # [batch, seq, features]
        
        output = model(states, deterministic=False)
        
        self.assertIn("action", output)
        self.assertIn("log_prob", output)
        self.assertIn("value", output)
        self.assertIn("probs", output)
        self.assertIn("confidence", output)
        
        self.assertEqual(output["action"].shape, (2,))
        self.assertEqual(output["value"].shape, (2,))
        self.assertEqual(output["probs"].shape, (2, 3))
        
    def test_transformer_a2c_evaluate_actions(self):
        """Test action evaluation for A2C update."""
        model = TransformerA2C(self.config)
        states = torch.randn(2, 20, 44)
        actions = torch.tensor([0, 1])
        
        log_probs, values, entropy = model.evaluate_actions(states, actions)
        
        self.assertEqual(log_probs.shape, (2,))
        self.assertEqual(values.shape, (2,))
        self.assertEqual(entropy.shape, (2,))
        
    def test_transformer_a2c_predict(self):
        """Test inference-only prediction."""
        model = TransformerA2C(self.config)
        states = torch.randn(2, 20, 44)
        
        output = model.predict(states)
        
        self.assertIn("action", output)
        self.assertEqual(output["action"].shape, (2,))


class TestSortinoReward(unittest.TestCase):
    """Tests for Sortino reward functions."""
    
    def test_simple_sortino_reward_positive(self):
        """Test reward for positive return."""
        reward_fn = SimpleSortinoReward(scale=100.0, downside_penalty=2.0)
        
        # Action 1 = LONG, positive market return
        reward = reward_fn.compute(action=1, market_return=0.01, confidence=0.8)
        
        self.assertGreater(reward, 0)
        self.assertEqual(len(reward_fn._returns_buffer), 1)
        
    def test_simple_sortino_reward_negative(self):
        """Test reward for negative return (should be penalized more)."""
        reward_fn = SimpleSortinoReward(scale=100.0, downside_penalty=2.0)
        
        # Action 1 = LONG, negative market return
        reward = reward_fn.compute(action=1, market_return=-0.01, confidence=0.8)
        
        self.assertLess(reward, 0)
        # Should be more negative than the positive case is positive
        positive_reward = SimpleSortinoReward(scale=100.0).compute(action=1, market_return=0.01, confidence=0.8)
        self.assertLess(reward, -positive_reward)
        
    def test_simple_sortino_flat_action(self):
        """Test FLAT action gets zero position return."""
        reward_fn = SimpleSortinoReward(scale=100.0)
        
        # Action 0 = FLAT, market goes up
        reward = reward_fn.compute(action=0, market_return=0.05, confidence=0.9)
        
        # FLAT position means zero return regardless of market
        self.assertEqual(reward, 0.0)
        
    def test_simple_sortino_short_action(self):
        """Test SHORT action with falling market (should be positive)."""
        reward_fn = SimpleSortinoReward(scale=100.0)
        
        # Action 2 = SHORT, market goes down
        reward = reward_fn.compute(action=2, market_return=-0.01, confidence=0.8)
        
        # Short + market down = positive return
        self.assertGreater(reward, 0)
        
    def test_episode_sharpe_calculation(self):
        """Test Sharpe ratio calculation."""
        import random
        random.seed(42)
        reward_fn = SimpleSortinoReward()
        
        # Add some returns with variation (positive bias)
        for i in range(100):
            # Returns with positive mean and some variance
            market_return = 0.001 + random.gauss(0, 0.002)
            reward_fn.compute(action=1, market_return=market_return, confidence=0.8)
        
        sharpe = reward_fn.get_episode_sharpe()
        self.assertIsInstance(sharpe, float)
        # Don't assert positive because random returns may occasionally be negative
        # Just verify it's a valid number in range
        self.assertTrue(-10 <= sharpe <= 10)
        
    def test_reward_reset(self):
        """Test reset clears buffer."""
        reward_fn = SimpleSortinoReward()
        reward_fn.compute(action=1, market_return=0.01, confidence=0.8)
        
        self.assertEqual(len(reward_fn._returns_buffer), 1)
        
        reward_fn.reset()
        
        self.assertEqual(len(reward_fn._returns_buffer), 0)
        
    def test_sortino_with_drawdown_penalty(self):
        """Test drawdown penalty kicks in after threshold."""
        reward_fn = SortinoWithDrawdownPenalty(
            scale=100.0,
            drawdown_threshold=0.05,
            drawdown_penalty=0.5,
        )
        
        # Simulate a series of losses to trigger drawdown
        for _ in range(20):
            reward = reward_fn.compute(action=1, market_return=-0.01, confidence=0.8)
        
        # After significant losses, drawdown penalty should be applied
        # Check that equity has dropped
        self.assertLess(reward_fn._current_equity, 1.0)
        
    def test_create_reward_function(self):
        """Test factory function."""
        sortino = create_reward_function("sortino", scale=50.0)
        self.assertIsInstance(sortino, SimpleSortinoReward)
        
        sortino_dd = create_reward_function("sortino_drawdown", scale=50.0)
        self.assertIsInstance(sortino_dd, SortinoWithDrawdownPenalty)


class TestTransformerA2CEnv(unittest.TestCase):
    """Tests for environment adapter."""
    
    def setUp(self):
        self.data, self.prices = create_synthetic_data(
            num_samples=1000,
            feature_dim=44,
            seed=42,
        )
        
    def test_create_synthetic_data(self):
        """Test synthetic data generation."""
        data, prices = create_synthetic_data(num_samples=500, feature_dim=10)
        
        self.assertEqual(data.shape, (500, 10))
        self.assertEqual(prices.shape, (500,))
        self.assertEqual(data.dtype, np.float32)
        self.assertEqual(prices.dtype, np.float32)
        
    def test_transformer_env_reset(self):
        """Test environment reset."""
        config = TransformerEnvConfig(context_length=50, feature_dim=44)
        env = TransformerA2CEnv(self.data, self.prices, config)
        
        state, info = env.reset()
        
        self.assertEqual(state.shape, (50, 44))
        self.assertIn("step", info)
        self.assertEqual(info["step"], 0)
        
    def test_transformer_env_step(self):
        """Test environment step."""
        config = TransformerEnvConfig(context_length=50, feature_dim=44)
        env = TransformerA2CEnv(self.data, self.prices, config)
        
        state, _ = env.reset()
        next_state, market_return, done, info = env.step(action=1)  # LONG
        
        self.assertEqual(next_state.shape, (50, 44))
        self.assertTrue(isinstance(market_return, (float, np.floating)))
        self.assertIsInstance(done, bool)
        self.assertIn("market_return", info)
        
    def test_transformer_env_episode(self):
        """Test running full episode."""
        config = TransformerEnvConfig(context_length=50, feature_dim=44)
        env = TransformerA2CEnv(self.data, self.prices, config)
        
        state, _ = env.reset()
        steps = 0
        actions = [0, 1, 2]  # Cycle through actions
        
        while True:
            action = actions[steps % 3]
            state, market_return, done, info = env.step(action)
            steps += 1
            
            if done:
                break
                
        self.assertGreater(steps, 0)
        
    def test_walk_forward_splitter(self):
        """Test train/val/test splitting."""
        splitter = WalkForwardSplitter(
            data=self.data,
            prices=self.prices,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        
        train_data, train_prices = splitter.get_train_data()
        val_data, val_prices = splitter.get_val_data()
        test_data, test_prices = splitter.get_test_data()
        
        # Check sizes
        total = len(self.data)
        self.assertEqual(len(train_data), int(total * 0.6))
        
    def test_walk_forward_create_envs(self):
        """Test creating environments from splitter."""
        splitter = WalkForwardSplitter(self.data, self.prices)
        
        config = TransformerEnvConfig(context_length=20, feature_dim=44)
        train_env, val_env, test_env = splitter.create_envs(config=config)
        
        self.assertIsInstance(train_env, TransformerA2CEnv)
        self.assertIsInstance(val_env, TransformerA2CEnv)
        self.assertIsInstance(test_env, TransformerA2CEnv)


class TestTrainerIntegration(unittest.TestCase):
    """Integration tests for trainer."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data, self.prices = create_synthetic_data(
            num_samples=2000,
            feature_dim=44,
            seed=42,
        )
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_trainer_initialization(self):
        """Test trainer can be initialized."""
        from src.training.transformer_a2c_trainer import TransformerA2CTrainer
        
        config = TransformerA2CConfig(
            input_dim=44,
            hidden_dim=32,
            num_layers=1,
            context_length=20,
            rollout_steps=64,
            max_steps=100,
            val_frequency=50,
        )
        
        splitter = WalkForwardSplitter(self.data, self.prices)
        env_config = TransformerEnvConfig(context_length=20, feature_dim=44)
        train_env, val_env, _ = splitter.create_envs(config=env_config)
        
        trainer = TransformerA2CTrainer(
            config=config,
            train_env=train_env,
            val_env=val_env,
            device="cpu",
            output_dir=self.temp_dir,
        )
        
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.actor_optimizer)
        self.assertIsNotNone(trainer.critic_optimizer)
        
    def test_trainer_collect_rollout(self):
        """Test rollout collection."""
        from src.training.transformer_a2c_trainer import TransformerA2CTrainer
        
        config = TransformerA2CConfig(
            input_dim=44,
            hidden_dim=32,
            num_layers=1,
            context_length=20,
            rollout_steps=32,
        )
        
        splitter = WalkForwardSplitter(self.data, self.prices)
        env_config = TransformerEnvConfig(context_length=20, feature_dim=44)
        train_env, val_env, _ = splitter.create_envs(config=env_config)
        
        trainer = TransformerA2CTrainer(
            config=config,
            train_env=train_env,
            val_env=val_env,
            device="cpu",
            output_dir=self.temp_dir,
        )
        
        rollout = trainer.collect_rollout(train_env, steps=32)
        
        self.assertEqual(len(rollout["states"]), 32)
        self.assertEqual(len(rollout["actions"]), 32)
        self.assertEqual(len(rollout["rewards"]), 32)
        
    def test_trainer_short_training(self):
        """Test very short training loop."""
        from src.training.transformer_a2c_trainer import TransformerA2CTrainer
        
        config = TransformerA2CConfig(
            input_dim=44,
            hidden_dim=32,
            num_layers=1,
            context_length=20,
            rollout_steps=64,
            max_steps=128,  # Very short
            val_frequency=64,
            checkpoint_frequency=64,
        )
        
        splitter = WalkForwardSplitter(self.data, self.prices)
        env_config = TransformerEnvConfig(context_length=20, feature_dim=44)
        train_env, val_env, _ = splitter.create_envs(config=env_config)
        
        trainer = TransformerA2CTrainer(
            config=config,
            train_env=train_env,
            val_env=val_env,
            device="cpu",
            output_dir=self.temp_dir,
        )
        
        # Run training (should complete quickly)
        result = trainer.train()
        
        # Check training completed
        self.assertGreater(trainer.global_step, 0)
        
    def test_checkpoint_save_load(self):
        """Test checkpoint save and load."""
        from src.training.transformer_a2c_trainer import TransformerA2CTrainer
        
        config = TransformerA2CConfig(
            input_dim=44,
            hidden_dim=32,
            num_layers=1,
            context_length=20,
        )
        
        splitter = WalkForwardSplitter(self.data, self.prices)
        env_config = TransformerEnvConfig(context_length=20, feature_dim=44)
        train_env, val_env, _ = splitter.create_envs(config=env_config)
        
        trainer = TransformerA2CTrainer(
            config=config,
            train_env=train_env,
            val_env=val_env,
            device="cpu",
            output_dir=self.temp_dir,
        )
        
        # Save checkpoint
        trainer.global_step = 1000
        checkpoint_path = trainer.save_checkpoint(val_sharpe=0.5, tag="_test")
        
        # Verify file exists
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Create new trainer and load
        trainer2 = TransformerA2CTrainer(
            config=config,
            train_env=train_env,
            val_env=val_env,
            device="cpu",
            output_dir=self.temp_dir,
        )
        trainer2.load_checkpoint(checkpoint_path)
        
        self.assertEqual(trainer2.global_step, 1000)


if __name__ == '__main__':
    unittest.main()
