"""
HIMARI Layer 2 - Walk-Forward Optimization for Transformer-A2C
Based on Layer 3 research: 63-85% OOD failure rate without proper validation.

Walk-Forward Optimization (WFO):
- 6-month training window, 1-month validation window
- Rolling forward by 1 month each iteration
- Warm-start from previous window weights
- Select best windows by validation Sharpe
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import copy

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class WFOConfig:
    """Walk-Forward Optimization configuration."""
    
    # Window settings
    train_months: int = 6
    val_months: int = 1
    step_months: int = 1
    
    # Training settings per window
    steps_per_window: int = 50000
    warmup_steps: int = 10000  # Initial window gets more steps
    fine_tune_lr: float = 1e-5  # Reduced LR for fine-tuning
    
    # Early stopping per window
    patience: int = 3
    min_improvement: float = 0.01
    
    # Success thresholds
    target_sharpe: float = 0.5
    max_drawdown: float = 0.22
    max_ood_failure: float = 0.25
    
    # Ensemble settings
    top_k_windows: int = 3  # Keep best K windows for ensemble
    
    # Output settings
    checkpoint_dir: str = "./wfo_checkpoints"
    save_all_windows: bool = True


@dataclass 
class WFOWindow:
    """Represents a single walk-forward window."""
    window_idx: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    
    # Results (filled after training)
    train_sharpe: float = 0.0
    val_sharpe: float = 0.0
    train_return: float = 0.0
    val_return: float = 0.0
    max_drawdown: float = 0.0
    checkpoint_path: Optional[str] = None
    
    def __repr__(self):
        return (f"Window {self.window_idx}: "
                f"Train[{self.train_start.strftime('%Y-%m')} to {self.train_end.strftime('%Y-%m')}] → "
                f"Val[{self.val_start.strftime('%Y-%m')} to {self.val_end.strftime('%Y-%m')}] | "
                f"Val Sharpe: {self.val_sharpe:.3f}")


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization for Transformer-A2C.
    
    Based on Layer 3 research showing:
    - 63-85% OOD failure rate with static training
    - 18-25% OOD failure with proper WFO
    """
    
    def __init__(
        self,
        config: WFOConfig,
        trainer_class,
        model_config,
        device: str = "cuda",
    ):
        self.config = config
        self.trainer_class = trainer_class
        self.model_config = model_config
        self.device = device
        
        self.windows: List[WFOWindow] = []
        self.best_windows: List[WFOWindow] = []
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[WFOWindow]:
        """
        Generate rolling train/val windows.
        
        Example with 6-month train, 1-month val, 1-month step:
        - Window 0: Train Jan-Jun 2020 → Val Jul 2020
        - Window 1: Train Feb-Jul 2020 → Val Aug 2020
        - Window 2: Train Mar-Aug 2020 → Val Sep 2020
        - ...
        """
        windows = []
        window_idx = 0
        
        current_train_start = start_date
        
        while True:
            # Calculate window boundaries
            train_end = current_train_start + timedelta(days=30 * self.config.train_months)
            val_start = train_end
            val_end = val_start + timedelta(days=30 * self.config.val_months)
            
            # Check if we've exceeded the data range
            if val_end > end_date:
                break
                
            window = WFOWindow(
                window_idx=window_idx,
                train_start=current_train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )
            windows.append(window)
            
            # Step forward
            current_train_start += timedelta(days=30 * self.config.step_months)
            window_idx += 1
            
        logger.info(f"Generated {len(windows)} WFO windows")
        self.windows = windows
        return windows
    
    def get_data_for_window(
        self,
        data: np.ndarray,
        prices: np.ndarray,
        timestamps: np.ndarray,
        window: WFOWindow,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract train and val data for a specific window.
        
        Returns:
            train_data, train_prices, val_data, val_prices
        """
        # Find indices for train period
        train_mask = (timestamps >= window.train_start) & (timestamps < window.train_end)
        val_mask = (timestamps >= window.val_start) & (timestamps < window.val_end)
        
        train_data = data[train_mask]
        train_prices = prices[train_mask]
        val_data = data[val_mask]
        val_prices = prices[val_mask]
        
        logger.info(f"Window {window.window_idx}: Train samples={len(train_data)}, Val samples={len(val_data)}")
        
        return train_data, train_prices, val_data, val_prices
    
    def train_window(
        self,
        window: WFOWindow,
        train_env,
        val_env,
        pretrained_weights: Optional[str] = None,
    ) -> WFOWindow:
        """
        Train a single WFO window.
        
        Args:
            window: Window configuration
            train_env: Training environment
            val_env: Validation environment
            pretrained_weights: Optional path to warm-start weights
            
        Returns:
            Updated window with results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Window {window.window_idx}")
        logger.info(f"{'='*60}")
        
        # Adjust config for this window
        window_config = copy.deepcopy(self.model_config)
        
        # First window gets more steps, subsequent windows fine-tune
        if window.window_idx == 0 and pretrained_weights is None:
            window_config.max_steps = self.config.steps_per_window + self.config.warmup_steps
            window_config.actor_lr = self.model_config.actor_lr
            window_config.critic_lr = self.model_config.critic_lr
        else:
            window_config.max_steps = self.config.steps_per_window
            window_config.actor_lr = self.config.fine_tune_lr
            window_config.critic_lr = self.config.fine_tune_lr * 3  # Maintain ratio
        
        # Create trainer
        output_dir = os.path.join(
            self.config.checkpoint_dir,
            f"window_{window.window_idx:02d}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        trainer = self.trainer_class(
            config=window_config,
            train_env=train_env,
            val_env=val_env,
            device=self.device,
            output_dir=output_dir,
        )
        
        # Load warm-start weights
        if pretrained_weights and os.path.exists(pretrained_weights):
            logger.info(f"Loading warm-start weights from {pretrained_weights}")
            trainer.load_checkpoint(pretrained_weights)
        
        # Train
        best_checkpoint = trainer.train()
        
        # Collect results
        window.train_sharpe = trainer.train_sharpes[-1] if trainer.train_sharpes else 0.0
        window.val_sharpe = trainer.best_val_sharpe
        window.checkpoint_path = best_checkpoint
        
        # Calculate additional metrics from validation
        if hasattr(trainer, 'val_metrics'):
            window.val_return = trainer.val_metrics.get('total_return', 0.0)
            window.max_drawdown = trainer.val_metrics.get('max_drawdown', 0.0)
        
        logger.info(f"Window {window.window_idx} Results:")
        logger.info(f"  Train Sharpe: {window.train_sharpe:.3f}")
        logger.info(f"  Val Sharpe: {window.val_sharpe:.3f}")
        logger.info(f"  Max Drawdown: {window.max_drawdown:.2%}")
        
        return window
    
    def run_wfo(
        self,
        data: np.ndarray,
        prices: np.ndarray,
        timestamps: np.ndarray,
        start_date: datetime,
        end_date: datetime,
        pretrain_checkpoint: Optional[str] = None,
        env_class=None,
        env_config=None,
    ) -> List[WFOWindow]:
        """
        Run full Walk-Forward Optimization.
        
        Args:
            data: Feature array [N, 44]
            prices: Price array [N]
            timestamps: Datetime array [N]
            start_date: Start of WFO period
            end_date: End of WFO period
            pretrain_checkpoint: Optional pre-trained crash-aware weights
            env_class: Environment class for creating train/val envs
            env_config: Environment configuration
            
        Returns:
            List of trained windows with results
        """
        logger.info("="*60)
        logger.info("Starting Walk-Forward Optimization")
        logger.info("="*60)
        
        # Generate windows
        self.generate_windows(start_date, end_date)
        
        if len(self.windows) == 0:
            raise ValueError("No windows generated. Check date range.")
        
        # Track best weights for warm-start
        previous_best_weights = pretrain_checkpoint
        
        # Train each window
        for window in self.windows:
            # Get data for this window
            train_data, train_prices, val_data, val_prices = self.get_data_for_window(
                data, prices, timestamps, window
            )
            
            # Skip if insufficient data
            if len(train_data) < 100 or len(val_data) < 10:
                logger.warning(f"Skipping Window {window.window_idx}: Insufficient data")
                continue
            
            # Create environments
            train_env = env_class(train_data, train_prices, env_config)
            val_env = env_class(val_data, val_prices, env_config)
            
            # Train window
            window = self.train_window(
                window=window,
                train_env=train_env,
                val_env=val_env,
                pretrained_weights=previous_best_weights,
            )
            
            # Update warm-start for next window
            if window.checkpoint_path:
                previous_best_weights = window.checkpoint_path
        
        # Select best windows
        self.best_windows = self.select_best_windows()
        
        # Save WFO summary
        self.save_summary()
        
        return self.windows
    
    def select_best_windows(self) -> List[WFOWindow]:
        """Select top-K windows by validation Sharpe."""
        # Filter valid windows
        valid_windows = [w for w in self.windows if w.val_sharpe > float('-inf')]
        
        # Sort by val Sharpe descending
        sorted_windows = sorted(valid_windows, key=lambda w: w.val_sharpe, reverse=True)
        
        # Take top K
        best = sorted_windows[:self.config.top_k_windows]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Best {len(best)} Windows by Val Sharpe:")
        logger.info("="*60)
        for w in best:
            logger.info(f"  {w}")
            
        return best
    
    def save_summary(self):
        """Save WFO results summary."""
        summary = {
            "config": asdict(self.config),
            "total_windows": len(self.windows),
            "windows": [
                {
                    "window_idx": w.window_idx,
                    "train_period": f"{w.train_start.isoformat()} to {w.train_end.isoformat()}",
                    "val_period": f"{w.val_start.isoformat()} to {w.val_end.isoformat()}",
                    "train_sharpe": w.train_sharpe,
                    "val_sharpe": w.val_sharpe,
                    "max_drawdown": w.max_drawdown,
                    "checkpoint_path": w.checkpoint_path,
                }
                for w in self.windows
            ],
            "best_windows": [w.window_idx for w in self.best_windows],
            "best_val_sharpe": max([w.val_sharpe for w in self.windows]) if self.windows else 0.0,
            "avg_val_sharpe": np.mean([w.val_sharpe for w in self.windows if w.val_sharpe > float('-inf')]),
        }
        
        summary_path = os.path.join(self.config.checkpoint_dir, "wfo_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info(f"WFO summary saved to {summary_path}")
    
    def create_ensemble(self) -> Dict:
        """
        Create ensemble from best windows.
        
        Returns dict with paths to top-K checkpoints for ensemble inference.
        """
        if not self.best_windows:
            raise ValueError("No best windows selected. Run WFO first.")
            
        ensemble = {
            "type": "temporal_ensemble",
            "weights": [1.0 / len(self.best_windows)] * len(self.best_windows),
            "checkpoints": [w.checkpoint_path for w in self.best_windows if w.checkpoint_path],
            "val_sharpes": [w.val_sharpe for w in self.best_windows],
        }
        
        ensemble_path = os.path.join(self.config.checkpoint_dir, "ensemble_config.json")
        with open(ensemble_path, 'w') as f:
            json.dump(ensemble, f, indent=2)
            
        logger.info(f"Ensemble config saved to {ensemble_path}")
        return ensemble


def run_wfo_training(
    data_path: str,
    output_dir: str,
    pretrain_checkpoint: Optional[str] = None,
    device: str = "cuda",
):
    """
    Convenience function to run full WFO training.
    
    Args:
        data_path: Path to training data (pkl or npy)
        output_dir: Directory for checkpoints
        pretrain_checkpoint: Optional crash-aware pre-trained weights
        device: cuda or cpu
    """
    from src.models.transformer_a2c import TransformerA2C, TransformerA2CConfig
    from src.training.transformer_a2c_trainer import TransformerA2CTrainer
    from src.environment.transformer_a2c_env import TransformerA2CEnv, TransformerEnvConfig
    
    # Load data
    import pickle
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    data = data_dict['features']
    prices = data_dict['prices']
    timestamps = data_dict['timestamps']
    
    # Configure WFO
    wfo_config = WFOConfig(
        train_months=6,
        val_months=1,
        step_months=1,
        steps_per_window=50000,
        checkpoint_dir=output_dir,
    )
    
    # Configure model
    model_config = TransformerA2CConfig(
        input_dim=44,
        hidden_dim=256,
        num_layers=4,
        context_length=100,
        rollout_steps=512,
    )
    
    # Configure environment
    env_config = TransformerEnvConfig(
        context_length=100,
        feature_dim=44,
    )
    
    # Run WFO
    optimizer = WalkForwardOptimizer(
        config=wfo_config,
        trainer_class=TransformerA2CTrainer,
        model_config=model_config,
        device=device,
    )
    
    windows = optimizer.run_wfo(
        data=data,
        prices=prices,
        timestamps=timestamps,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2024, 12, 31),
        pretrain_checkpoint=pretrain_checkpoint,
        env_class=TransformerA2CEnv,
        env_config=env_config,
    )
    
    # Create ensemble from best windows
    ensemble = optimizer.create_ensemble()
    
    return windows, ensemble


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization for Transformer-A2C")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--output", type=str, default="./wfo_output", help="Output directory")
    parser.add_argument("--pretrain", type=str, default=None, help="Pre-trained checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    run_wfo_training(
        data_path=args.data,
        output_dir=args.output,
        pretrain_checkpoint=args.pretrain,
        device=args.device,
    )
