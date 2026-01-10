"""
HIMARI Layer 2 - Unified Training Launcher
Train any model: BaselineMLP, CQL, PPO-LSTM, CGDT, or FLAG-TRADER

Usage:
    python train_all_models.py --model baseline
    python train_all_models.py --model cql --epochs 100
    python train_all_models.py --model ppo --env-episodes 1000
    python train_all_models.py --model cgdt --context-length 64
    python train_all_models.py --model flag-trader --lora-rank 16
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.baseline_mlp import create_baseline_model
from src.models.cql import create_cql_agent
from src.models.ppo_lstm import create_ppo_lstm_agent
from src.models.cgdt import create_cgdt_agent
from src.models.flag_trader import create_flag_trader_agent

# from src.data.dataset import load_raw_data  # Not used - using load_raw_data instead
from src.data.trajectory_dataset import create_trajectory_dataloader, create_sequence_dataloader
from src.environment.trading_env import TradingEnvironment, TradingConfig

from src.training.monitoring import TrainingMonitor, MonitoringConfig
from src.preprocessing.part_a_preprocessing import create_preprocessor
from src.training.part_k_advanced import PartKTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_model(model, path):
    """Save model checkpoint. Works for both nn.Module and agents."""
    if hasattr(model, 'save'):
        model.save(str(path))
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model.get_config() if hasattr(model, 'get_config') else {}
        }, str(path))
    logger.info(f"Saved checkpoint: {path}")


def load_raw_data(data_dir):
    """Load raw numpy arrays instead of DataLoaders."""
    import json

    data_path = Path(data_dir)
    features = np.load(data_path / "preprocessed_features.npy")
    labels = np.load(data_path / "labels.npy")

    # Split manually (80/10/10)
    total = len(features)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)

    train_features = features[:train_size]
    train_labels = labels[:train_size]
    val_features = features[train_size:train_size+val_size]
    val_labels = labels[train_size:train_size+val_size]

    # Load metadata
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata.update({
        'feature_dim': features.shape[1],
        'num_samples': total
    })

    logger.info(f"Loaded {total} samples with {features.shape[1]}D features")
    logger.info(f"Split: train={len(train_features)}, val={len(val_features)}")

    return train_features, train_labels, val_features, val_labels, metadata


def train_baseline(args):
    """Train BaselineMLP model."""
    logger.info("=" * 80)
    logger.info("Training BaselineMLP")
    logger.info("=" * 80)

    # Load data
    train_features, train_labels, val_features, val_labels, metadata = load_raw_data(args.data_dir)

    # Create model
    model = create_baseline_model(
        input_dim=metadata['feature_dim'],
        hidden_dims=[128, 64, 32],
        num_classes=3
    ).to(args.device)

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        outputs = model(torch.FloatTensor(train_features).to(args.device))
        loss = criterion(outputs, torch.LongTensor(train_labels).to(args.device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss={loss.item():.4f}")

        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            checkpoint_path = Path(args.checkpoint_dir) / "baseline_best.pt"
            save_model(model, checkpoint_path)

    # Save final model
    final_path = Path(args.checkpoint_dir) / "baseline_final.pt"
    save_model(model, final_path)
    logger.info("BaselineMLP training complete")


def train_cql(args):
    """Train CQL agent."""
    logger.info("=" * 80)
    logger.info("Training CQL (Conservative Q-Learning)")
    logger.info("=" * 80)

    # Load data
    train_features, train_labels, val_features, val_labels, metadata = load_raw_data(args.data_dir)

    # Add position/balance features (5 extra dims)
    state_dim = metadata['feature_dim'] + 5

    # Create CQL agent
    agent = create_cql_agent(
        state_dim=state_dim,
        action_dim=3,
        hidden_dim=256,
        alpha=2.0
    ).to(args.device)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    # Training loop (offline RL)
    best_loss = float('inf')
    logger.info(f"Starting CQL training for {args.epochs} epochs...")

    try:
        for epoch in range(args.epochs):
            # Sample batch
            batch_size = 256
            indices = np.random.choice(len(train_features), batch_size)

            states = torch.FloatTensor(train_features[indices]).to(args.device)
            # Pad with zeros for position/balance features
            states = torch.cat([states, torch.zeros(batch_size, 5).to(args.device)], dim=1)

            actions = torch.LongTensor(train_labels[indices]).to(args.device)
            rewards = torch.FloatTensor(np.random.randn(batch_size) * 0.01).to(args.device)  # Placeholder
            next_states = states  # Placeholder
            dones = torch.zeros(batch_size).to(args.device)

            # CQL update
            info = agent.train_step(states, actions, rewards, next_states, dones, optimizer)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{args.epochs}: Loss={info['loss']:.4f}, CQL Loss={info['cql_loss']:.4f}")

            # Save best model
            if info['loss'] < best_loss:
                best_loss = info['loss']
                checkpoint_path = Path(args.checkpoint_dir) / "cql_best.pt"
                save_model(agent, checkpoint_path)

        logger.info(f"Completed all {args.epochs} epochs successfully!")

    except Exception as e:
        logger.error(f"CQL training failed at epoch {epoch}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    # Save final model
    final_path = Path(args.checkpoint_dir) / "cql_final.pt"
    save_model(agent, final_path)
    logger.info("CQL training complete")


def train_ppo(args):
    """Train PPO-LSTM agent."""
    logger.info("=" * 80)
    logger.info("Training PPO-LSTM (online RL)")
    logger.info("=" * 80)

    # Load data for environment
    train_features, train_labels, val_features, val_labels, metadata = load_raw_data(args.data_dir)

    # Create mock prices for environment
    prices = np.random.randn(len(train_features)).cumsum() + 30000

    # Create environment
    env = TradingEnvironment(
        data=train_features,
        prices=prices,
        config=TradingConfig()
    )

    # Create PPO agent
    agent = create_ppo_lstm_agent(
        state_dim=env.observation_space_dim,
        action_dim=3,
        hidden_dim=128
    ).to(args.device)

    # Collect episodes and train
    best_reward = float('-inf')
    for episode in range(args.env_episodes):
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []

        state = env.reset()
        hidden = agent.ac_network.init_hidden(1, args.device)
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(args.device)
            action, log_prob, value, hidden = agent.select_action(state_tensor, hidden)

            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())

            state = next_state

        total_reward = sum(rewards)

        if episode % 10 == 0:
            logger.info(f"Episode {episode}: Reward={total_reward:.4f}, Steps={len(rewards)}")

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            checkpoint_path = Path(args.checkpoint_dir) / "ppo_best.pt"
            save_model(agent, checkpoint_path)

    # Save final model
    final_path = Path(args.checkpoint_dir) / "ppo_final.pt"
    save_model(agent, final_path)
    logger.info("PPO-LSTM training complete")


def train_cgdt(args):
    """Train CGDT agent."""
    logger.info("=" * 80)
    logger.info("Training CGDT (Critic-Guided Decision Transformer)")
    logger.info("=" * 80)

    # Load data
    train_features, train_labels, val_features, val_labels, metadata = load_raw_data(args.data_dir)

    # Create trajectory dataloader
    train_loader = create_trajectory_dataloader(
        features=train_features,
        labels=train_labels,
        context_length=args.context_length,
        batch_size=args.batch_size
    )

    # Create CGDT agent
    agent = create_cgdt_agent(
        state_dim=metadata['feature_dim'],
        action_dim=3,
        hidden_dim=256,
        num_layers=6
    ).to(args.device)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            states = batch['states'].to(args.device)
            actions = batch['actions'].to(args.device)
            returns_to_go = batch['returns_to_go'].to(args.device)
            timesteps = batch['timesteps'].to(args.device)

            # Compute loss
            loss, info = agent.compute_loss(states, actions, returns_to_go, timesteps, actions)

            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = Path(args.checkpoint_dir) / "cgdt_best.pt"
            save_model(agent, checkpoint_path)

    # Save final model
    final_path = Path(args.checkpoint_dir) / "cgdt_final.pt"
    save_model(agent, final_path)
    logger.info("CGDT training complete")


def train_flag_trader(args):
    """Train FLAG-TRADER agent."""
    logger.info("=" * 80)
    logger.info("Training FLAG-TRADER (Large Transformer with LoRA)")
    logger.info("=" * 80)

    # Load data
    train_features, train_labels, val_features, val_labels, metadata = load_raw_data(args.data_dir)

    # Create sequence dataloader
    train_loader = create_sequence_dataloader(
        features=train_features,
        labels=train_labels,
        context_length=args.context_length,
        batch_size=args.batch_size
    )

    # Create FLAG-TRADER agent
    agent = create_flag_trader_agent(
        state_dim=metadata['feature_dim'],
        action_dim=3,
        model_size="135M",
        lora_rank=args.lora_rank
    ).to(args.device)

    optimizer = optim.AdamW(agent.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for batch in train_loader:
            states = batch['states'].to(args.device)
            actions = batch['actions'].to(args.device)

            # Compute loss
            loss, info = agent.compute_loss(states, actions)

            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += info['accuracy']
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = Path(args.checkpoint_dir) / "flag_trader_best.pt"
            save_model(agent, checkpoint_path)

    # Save final model
    final_path = Path(args.checkpoint_dir) / "flag_trader_final.pt"
    save_model(agent, final_path)
    logger.info("FLAG-TRADER training complete")


def main():
    parser = argparse.ArgumentParser(description="HIMARI Layer 2 - Unified Training Launcher")

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=['baseline', 'cql', 'ppo', 'cgdt', 'flag-trader'],
                       help='Model to train')

    # Data
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')

    # Model-specific
    parser.add_argument('--context-length', type=int, default=64,
                       help='Context length for transformers (CGDT, FLAG-TRADER)')
    parser.add_argument('--lora-rank', type=int, default=16,
                       help='LoRA rank for FLAG-TRADER')
    parser.add_argument('--env-episodes', type=int, default=1000,
                       help='Number of episodes for PPO')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Training device')

    # Checkpoint
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info(f"HIMARI Layer 2 - Training {args.model.upper()}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Data: {args.data_dir}")
    logger.info("=" * 80)

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Train selected model
    if args.model == 'baseline':
        train_baseline(args)
    elif args.model == 'cql':
        train_cql(args)
    elif args.model == 'ppo':
        train_ppo(args)
    elif args.model == 'cgdt':
        train_cgdt(args)
    elif args.model == 'flag-trader':
        train_flag_trader(args)

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
