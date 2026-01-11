"""
HIMARI Layer 2 - Complete Package Test
Verify all models, environments, preprocessing, and training components.
"""

import sys
import os
import torch
import numpy as np
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_models():
    """Test all model implementations."""
    logger.info("=" * 80)
    logger.info("Testing Models")
    logger.info("=" * 80)

    state_dim = 65  # 60 features + 5 env features
    batch_size = 32

    # Test BaselineMLP
    try:
        from src.models.baseline_mlp import create_baseline_model
        model = create_baseline_model(input_dim=60, hidden_dims=[128, 64], num_classes=3)
        x = torch.randn(batch_size, 60)
        out = model(x)
        assert out.shape == (batch_size, 3)
        logger.info("[OK] BaselineMLP: Output shape correct")
    except Exception as e:
        logger.error(f"[FAIL] BaselineMLP: {e}")
        return False

    # Test CQL
    try:
        from src.models.cql import create_cql_agent
        agent = create_cql_agent(state_dim=state_dim, action_dim=3, hidden_dim=256)
        states = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, 3, (batch_size,))
        out = agent(states)
        assert out.shape == (batch_size, 3)
        logger.info("[OK] CQL: Forward pass successful")
    except Exception as e:
        logger.error(f"[FAIL] CQL: {e}")
        return False

    # Test PPO-LSTM
    try:
        from src.models.ppo_lstm import create_ppo_lstm_agent
        agent = create_ppo_lstm_agent(state_dim=state_dim, action_dim=3, hidden_dim=128)
        states = torch.randn(batch_size, state_dim)
        action_logits, values, hidden = agent(states)
        assert action_logits.shape == (batch_size, 3)
        assert values.shape == (batch_size, 1)
        logger.info("[OK] PPO-LSTM: Actor-Critic forward pass successful")
    except Exception as e:
        logger.error(f"[FAIL] PPO-LSTM: {e}")
        return False

    # Test CGDT
    try:
        from src.models.cgdt import create_cgdt_agent
        agent = create_cgdt_agent(state_dim=60, action_dim=3, hidden_dim=256, num_layers=6)
        states = torch.randn(batch_size, 32, 60)  # (batch, seq_len, state_dim)
        actions = torch.randint(0, 3, (batch_size, 32))
        returns_to_go = torch.randn(batch_size, 32)
        timesteps = torch.arange(32).unsqueeze(0).expand(batch_size, -1)
        out = agent(states, actions, returns_to_go, timesteps)
        assert out.shape == (batch_size, 32, 3)
        logger.info("[OK] CGDT: Transformer forward pass successful")
    except Exception as e:
        logger.error(f"[FAIL] CGDT: {e}")
        return False

    # Test FLAG-TRADER
    try:
        from src.models.flag_trader import create_flag_trader_agent
        agent = create_flag_trader_agent(state_dim=60, action_dim=3, model_size="135M", lora_rank=16)
        states = torch.randn(batch_size, 64, 60)  # (batch, seq_len, state_dim)
        out = agent(states)
        assert out.shape == (batch_size, 64, 3)
        logger.info("[OK] FLAG-TRADER: LoRA transformer forward pass successful")
    except Exception as e:
        logger.error(f"[FAIL] FLAG-TRADER: {e}")
        return False

    return True


def test_environment():
    """Test trading environment."""
    logger.info("=" * 80)
    logger.info("Testing Trading Environment")
    logger.info("=" * 80)

    try:
        from src.environment.trading_env import TradingEnvironment, VectorizedTradingEnv, TradingConfig

        # Create mock data
        num_samples = 1000
        feature_dim = 60
        data = np.random.randn(num_samples, feature_dim).astype(np.float32)
        prices = np.random.randn(num_samples).cumsum() + 30000

        # Test single environment
        env = TradingEnvironment(data, prices, TradingConfig())
        state = env.reset()
        assert state.shape == (feature_dim + 5,)  # features + env state

        # Test step
        action = 1  # HOLD
        next_state, reward, done, info = env.step(action)
        assert next_state.shape == (feature_dim + 5,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

        logger.info("[OK] TradingEnvironment: Single env test passed")

        # Test vectorized environment
        vec_env = VectorizedTradingEnv(data, prices, num_envs=4)
        states = vec_env.reset()
        assert states.shape == (4, feature_dim + 5)

        actions = np.array([0, 1, 2, 1])
        next_states, rewards, dones, infos = vec_env.step(actions)
        assert next_states.shape == (4, feature_dim + 5)
        assert rewards.shape == (4,)

        logger.info("[OK] VectorizedTradingEnv: Parallel envs test passed")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Trading Environment: {e}")
        return False


def test_data_loading():
    """Test data loading and preprocessing."""
    logger.info("=" * 80)
    logger.info("Testing Data Loading")
    logger.info("=" * 80)

    try:
        from src.data.trajectory_dataset import create_trajectory_dataloader, create_sequence_dataloader

        # Load data
        data_dir = './data'
        if not os.path.exists(data_dir):
            logger.warning("[SKIP] Data directory not found, skipping data loading test")
            return True

        # Load raw numpy arrays directly
        features = np.load(os.path.join(data_dir, 'preprocessed_features.npy'))
        labels = np.load(os.path.join(data_dir, 'labels.npy'))

        logger.info(f"[OK] Loaded {len(features)} samples with {features.shape[1]}D features")

        # Test trajectory dataloader
        train_loader = create_trajectory_dataloader(
            features=features[:1000],  # Subset for speed
            labels=labels[:1000],
            context_length=64,
            batch_size=32
        )

        batch = next(iter(train_loader))
        assert 'states' in batch
        assert 'actions' in batch
        assert 'returns_to_go' in batch

        logger.info("[OK] Trajectory dataloader test passed")

        # Test sequence dataloader
        seq_loader = create_sequence_dataloader(
            features=features[:1000],
            labels=labels[:1000],
            context_length=256,
            batch_size=16
        )

        batch = next(iter(seq_loader))
        assert 'states' in batch
        assert 'actions' in batch

        logger.info("[OK] Sequence dataloader test passed")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Data Loading: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test preprocessing components."""
    logger.info("=" * 80)
    logger.info("Testing Preprocessing")
    logger.info("=" * 80)

    try:
        from src.preprocessing.part_a_preprocessing import create_preprocessor

        # Create sample data
        num_samples = 100
        feature_dim = 60
        data = np.random.randn(num_samples, feature_dim)

        # Create preprocessor
        preprocessor = create_preprocessor({
            'enable_ekf': False,  # Skip EKF for speed
            'enable_freq_norm': True,
            'enable_vec_norm': True,
            'enable_online_aug': True
        })

        # Fit and process
        preprocessor.fit(data)
        processed = preprocessor.process(data, augment=True)

        assert processed.shape == data.shape
        logger.info("[OK] Preprocessing pipeline test passed")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_infrastructure():
    """Test training infrastructure."""
    logger.info("=" * 80)
    logger.info("Testing Training Infrastructure")
    logger.info("=" * 80)

    try:
        from src.training.monitoring import TrainingMonitor, MonitoringConfig
        from src.training.part_k_advanced import PartKTrainer

        # Test monitoring
        config = MonitoringConfig(
            use_wandb=False,  # Disable W&B for testing
            log_interval=10,
            checkpoint_interval=100
        )
        monitor = TrainingMonitor(config)

        monitor.log_metrics({'loss': 0.5, 'accuracy': 0.8}, step=0)
        logger.info("[OK] Training monitoring test passed")

        # Test Part K trainer
        part_k_trainer = PartKTrainer(
            enable_curriculum=True,
            enable_causal_aug=True,
            enable_adversarial=True,
            enable_reward_shaping=True
        )
        logger.info("[OK] Part K trainer initialization passed")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Training Infrastructure: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("HIMARI Layer 2 V1 - Complete Package Test")
    logger.info("=" * 80)

    results = {}

    # Run tests
    results['Models'] = test_models()
    results['Environment'] = test_environment()
    results['Data Loading'] = test_data_loading()
    results['Preprocessing'] = test_preprocessing()
    results['Training Infrastructure'] = test_training_infrastructure()

    # Summary
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)

    all_passed = True
    for component, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        logger.info(f"{status} {component}")
        if not passed:
            all_passed = False

    logger.info("=" * 80)

    if all_passed:
        logger.info("\n[SUCCESS] All tests passed! Package is ready for deployment.")
        return 0
    else:
        logger.error("\n[ERROR] Some tests failed. Fix issues before deployment.")
        return 1


if __name__ == "__main__":
    exit(main())
