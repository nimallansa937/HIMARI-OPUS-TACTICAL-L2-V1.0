#!/usr/bin/env python3
"""
HIMARI Layer 2 - Transformer-A2C Training Script
Train the Transformer-A2C model for tactical trading decisions.

Usage:
    # Basic training
    python train_transformer_a2c.py --data ./data/btc_5min_2020_2024.pkl --output ./output/transformer_a2c_v1

    # Resume from checkpoint
    python train_transformer_a2c.py --checkpoint ./output/transformer_a2c_v1/checkpoint_200000_best.pt

    # Quick test with synthetic data
    python train_transformer_a2c.py --synthetic --max_steps 5000 --device cpu
"""

import argparse
import os
import sys
import logging
import pickle
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.transformer_a2c import TransformerA2CConfig
from src.training.transformer_a2c_trainer import TransformerA2CTrainer
from src.environment.transformer_a2c_env import (
    TransformerA2CEnv,
    TransformerEnvConfig,
    WalkForwardSplitter,
    create_synthetic_data,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer-A2C for HIMARI Layer 2")
    
    # Data arguments
    parser.add_argument("--data", type=str, help="Path to training data (pickle file)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--synthetic_samples", type=int, default=50000, help="Number of synthetic samples")
    
    # Model arguments
    parser.add_argument("--input_dim", type=int, default=44, help="Feature dimension per timestep")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Transformer hidden size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--context_length", type=int, default=100, help="Context window length")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--max_steps", type=int, default=500000, help="Maximum training steps")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=3e-4, help="Critic learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--val_frequency", type=int, default=25000, help="Validation frequency")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="./output/transformer_a2c", help="Output directory")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to resume from")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    # Logging arguments
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="himari-layer2-transformer-a2c")
    
    return parser.parse_args()


def load_data(data_path: str):
    """Load training data from pickle file."""
    logger.info(f"Loading data from {data_path}")
    
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)
    
    # Expected format: {"features": np.ndarray, "prices": np.ndarray}
    if isinstance(data_dict, dict):
        features = data_dict.get("features", data_dict.get("data"))
        prices = data_dict.get("prices", data_dict.get("close"))
    else:
        raise ValueError("Data must be a dict with 'features' and 'prices' keys")
    
    logger.info(f"Loaded data: features={features.shape}, prices={prices.shape}")
    
    return features.astype(np.float32), prices.astype(np.float32)


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info("=" * 70)
    logger.info("HIMARI Layer 2 - Transformer-A2C Training")
    logger.info("=" * 70)
    
    # Load or create data
    if args.synthetic:
        logger.info(f"Using synthetic data with {args.synthetic_samples} samples")
        features, prices = create_synthetic_data(
            num_samples=args.synthetic_samples,
            feature_dim=args.input_dim,
        )
    elif args.data:
        features, prices = load_data(args.data)
        args.input_dim = features.shape[1]  # Update from data
    else:
        logger.error("Must specify --data or --synthetic")
        sys.exit(1)
    
    # Create train/val split
    splitter = WalkForwardSplitter(
        data=features,
        prices=prices,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    
    # Create environments
    env_config = TransformerEnvConfig(
        context_length=args.context_length,
        feature_dim=args.input_dim,
    )
    
    train_env, val_env, test_env = splitter.create_envs(config=env_config)
    
    logger.info(f"Train env: {train_env.num_samples} samples")
    logger.info(f"Val env: {val_env.num_samples} samples")
    logger.info(f"Test env: {test_env.num_samples} samples")
    
    # Create model config
    config = TransformerA2CConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        context_length=args.context_length,
        dropout=args.dropout,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
        max_steps=args.max_steps,
        val_frequency=args.val_frequency,
        patience=args.patience,
    )
    
    # Create trainer
    trainer = TransformerA2CTrainer(
        config=config,
        train_env=train_env,
        val_env=val_env,
        device=args.device,
        output_dir=args.output,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    best_checkpoint = trainer.train()
    
    if best_checkpoint:
        logger.info(f"Best checkpoint: {best_checkpoint['path']}")
        logger.info(f"Best validation Sharpe: {best_checkpoint['val_sharpe']:.4f}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
