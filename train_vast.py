#!/usr/bin/env python3
"""
HIMARI Layer 2 - PPO Training Script for Vast.ai
Downloads dataset from Google Drive and trains regime-conditioned policy.

Usage:
    python train_vast.py

Dataset Format (array-based pickle):
    {
        'train': {'features_raw', 'features_denoised', 'regime_ids', 'prices', 'returns', 'n_samples'},
        'val': {...},
        'test': {...},
        'metadata': {...}
    }
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict
import time

# =============================================================================
# Configuration
# =============================================================================

GDRIVE_FILE_ID = "1DpJAViY1YK_czC3Tfi3R0oXvPg-9Eo87"
DATASET_PATH = "/workspace/btc_1h_2020_2024_enriched_44f.pkl"
CHECKPOINT_DIR = "/workspace/checkpoints"
CONTEXT_LEN = 100
BATCH_SIZE = 64
N_EPOCHS = 50
LEARNING_RATE = 3e-4

# =============================================================================
# Setup
# =============================================================================

def setup_environment():
    """Setup training environment."""
    print("=" * 70)
    print("HIMARI Layer 2 - PPO Training")
    print("=" * 70)

    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        print("WARNING: No GPU detected, using CPU")
        device = torch.device('cpu')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    return device


def download_dataset():
    """Download dataset from Google Drive."""
    if os.path.exists(DATASET_PATH):
        print(f"\nDataset already exists: {DATASET_PATH}")
        return

    print(f"\nDownloading dataset from Google Drive...")

    # Use gdown to download
    import subprocess
    subprocess.run([
        "pip", "install", "-q", "gdown"
    ], check=True)

    import gdown
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, DATASET_PATH, quiet=False)

    print(f"Downloaded to: {DATASET_PATH}")


# =============================================================================
# Dataset
# =============================================================================

def load_enriched_dataset(path: str):
    """Load enriched dataset from pickle file (array-based format)."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # Returns dict with 'train', 'val', 'test' as dicts of arrays, plus 'metadata'
    return data['train'], data['val'], data['test'], data['metadata']


class EnrichedTradingDataset(Dataset):
    """PyTorch Dataset for enriched trading data with context windows.

    Expects array-based format:
        data_dict = {
            'features_raw': np.ndarray (n, feature_dim),
            'features_denoised': np.ndarray (n, feature_dim),
            'regime_ids': np.ndarray (n,),
            'regime_confidences': np.ndarray (n,),
            'prices': np.ndarray (n,),
            'returns': np.ndarray (n,),
            'n_samples': int
        }
    """

    def __init__(
        self,
        data_dict: Dict,
        context_len: int = 100,
        use_denoised: bool = True,
        normalize: bool = True
    ):
        self.context_len = context_len
        self.n_samples = data_dict['n_samples']

        if self.n_samples <= context_len:
            raise ValueError(f"Not enough samples ({self.n_samples}) for context_len ({context_len})")

        # Load arrays directly from dict
        if use_denoised:
            self.features_arr = data_dict['features_denoised'].astype(np.float32)
        else:
            self.features_arr = data_dict['features_raw'].astype(np.float32)

        self.feature_dim = self.features_arr.shape[1]
        self.regime_ids_arr = data_dict['regime_ids'].astype(np.int64)
        self.regime_confs_arr = data_dict['regime_confidences'].astype(np.float32)
        self.prices_arr = data_dict['prices'].astype(np.float32)
        self.returns_arr = data_dict['returns'].astype(np.float32)

        # Normalize features
        if normalize:
            self.feature_mean = self.features_arr.mean(axis=0)
            self.feature_std = self.features_arr.std(axis=0) + 1e-8
            self.features_arr = (self.features_arr - self.feature_mean) / self.feature_std

    def __len__(self):
        return self.n_samples - self.context_len

    def __getitem__(self, idx):
        end_idx = idx + self.context_len
        return {
            'features': torch.from_numpy(self.features_arr[idx:end_idx].copy()),
            'regime_ids': torch.from_numpy(self.regime_ids_arr[idx:end_idx].copy()),
            'regime_confidences': torch.from_numpy(self.regime_confs_arr[idx:end_idx].copy()),
            'prices': torch.from_numpy(self.prices_arr[idx:end_idx].copy()),
            'returns': torch.from_numpy(self.returns_arr[idx:end_idx].copy()),
            'current_idx': end_idx - 1
        }

    def get_feature_dim(self):
        return self.feature_dim


# =============================================================================
# Model
# =============================================================================

class RegimeConditionedPolicy(nn.Module):
    """
    PPO Actor-Critic with regime conditioning.

    Architecture:
        - Transformer encoder for temporal patterns
        - Regime embedding for conditioning
        - Separate actor (policy) and critic (value) heads
    """

    def __init__(
        self,
        feature_dim: int = 49,
        context_len: int = 100,
        n_regimes: int = 4,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        n_actions: int = 3,  # HOLD, LONG, SHORT
        dropout: float = 0.1
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # Regime embedding
        self.regime_embed = nn.Embedding(n_regimes, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, context_len, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_actions)
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features, regime_ids):
        """
        Args:
            features: (B, T, feature_dim)
            regime_ids: (B, T)

        Returns:
            action_logits: (B, n_actions)
            value: (B, 1)
        """
        B, T, _ = features.shape

        # Project features
        x = self.feature_proj(features)  # (B, T, hidden)

        # Add regime embedding
        regime_emb = self.regime_embed(regime_ids)  # (B, T, hidden)
        x = x + regime_emb

        # Add positional encoding
        x = x + self.pos_encoding[:, :T, :]

        # Transformer
        x = self.transformer(x)  # (B, T, hidden)

        # Use last timestep + mean pooling
        last_hidden = x[:, -1, :]  # (B, hidden)
        mean_hidden = x.mean(dim=1)  # (B, hidden)
        combined = torch.cat([last_hidden, mean_hidden], dim=-1)  # (B, hidden*2)

        # Actor and critic heads
        action_logits = self.actor(combined)  # (B, n_actions)
        value = self.critic(combined)  # (B, 1)

        return action_logits, value

    def get_action(self, features, regime_ids, deterministic=False):
        """Sample action from policy."""
        action_logits, value = self.forward(features, regime_ids)

        if deterministic:
            action = action_logits.argmax(dim=-1)
            log_prob = None
        else:
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value.squeeze(-1)


# =============================================================================
# Trainer
# =============================================================================

class PPOTrainer:
    """PPO trainer for regime-conditioned policy."""

    def __init__(
        self,
        policy: RegimeConditionedPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda'
    ):
        self.policy = policy.to(device)
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=N_EPOCHS, eta_min=lr/10
        )

    def train_step(self, batch):
        """Single PPO update step."""
        features = batch['features'].to(self.device)
        regime_ids = batch['regime_ids'].to(self.device)
        returns_data = batch['returns'].to(self.device)

        B = features.shape[0]

        # Get policy outputs
        action_logits, values = self.policy(features, regime_ids)
        values = values.squeeze(-1)

        # Sample actions
        dist = torch.distributions.Categorical(logits=action_logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # Compute regime-aware reward
        # Action: 0=HOLD, 1=LONG, 2=SHORT
        # Regime: 0=LOW_VOL, 1=TRENDING, 2=HIGH_VOL, 3=CRISIS
        final_returns = returns_data[:, -1]  # Last timestep return
        final_regime = regime_ids[:, -1]  # Current regime

        # Position: 1 for LONG, -1 for SHORT, 0 for HOLD
        position = (actions == 1).float() - (actions == 2).float()

        is_hold = (actions == 0).float()
        is_trade = (actions != 0).float()

        # === KEY INSIGHT ===
        # The model needs a BASELINE reward for HOLD that competes with trading
        # Without this, any trading strategy dominates because it has variance

        # Base HOLD reward - small positive baseline for all holds
        hold_baseline = is_hold * 0.02

        # Base trade reward = return * position (scaled)
        trade_pnl = final_returns * position * 100

        # === REGIME-SPECIFIC MODIFIERS ===

        # LOW_VOL (regime 0): Slightly favor HOLD (ranging market)
        low_vol_hold_bonus = is_hold * (final_regime == 0).float() * 0.03
        low_vol_trade_penalty = is_trade * (final_regime == 0).float() * 0.02

        # TRENDING (regime 1): Favor correct trades
        trending_correct_bonus = is_trade * (final_regime == 1).float() * (final_returns * position > 0).float() * 0.08
        trending_wrong_penalty = is_trade * (final_regime == 1).float() * (final_returns * position < 0).float() * 0.04

        # HIGH_VOL (regime 2): Strongly favor HOLD (choppy, unpredictable)
        high_vol_hold_bonus = is_hold * (final_regime == 2).float() * 0.05
        high_vol_trade_penalty = is_trade * (final_regime == 2).float() * 0.04

        # CRISIS (regime 3): Very strongly favor HOLD (extreme risk)
        crisis_hold_bonus = is_hold * (final_regime == 3).float() * 0.08
        crisis_trade_penalty = is_trade * (final_regime == 3).float() * 0.06

        # === UNIVERSAL TRADE COST ===
        trade_cost = is_trade * 0.015

        # === COMBINE ===
        rewards = (
            # Base rewards
            hold_baseline + trade_pnl
            # LOW_VOL modifiers
            + low_vol_hold_bonus - low_vol_trade_penalty
            # TRENDING modifiers
            + trending_correct_bonus - trending_wrong_penalty
            # HIGH_VOL modifiers
            + high_vol_hold_bonus - high_vol_trade_penalty
            # CRISIS modifiers
            + crisis_hold_bonus - crisis_trade_penalty
            # Universal cost
            - trade_cost
        )

        # Compute advantages
        with torch.no_grad():
            advantages = rewards - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO loss
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values, rewards)

        # Entropy bonus
        entropy = dist.entropy().mean()

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'mean_reward': rewards.mean().item()
        }

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.policy.train()
        metrics = []

        for batch in dataloader:
            m = self.train_step(batch)
            metrics.append(m)

        # Average metrics
        avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
        return avg_metrics

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate policy with regime-aware rewards."""
        self.policy.eval()
        total_reward = 0
        total_samples = 0
        action_counts = {0: 0, 1: 0, 2: 0}
        regime_action_counts = {r: {0: 0, 1: 0, 2: 0} for r in range(4)}

        for batch in dataloader:
            features = batch['features'].to(self.device)
            regime_ids = batch['regime_ids'].to(self.device)
            returns_data = batch['returns'].to(self.device)

            actions, _, _ = self.policy.get_action(features, regime_ids, deterministic=True)

            # Compute regime-aware rewards (same as training)
            final_returns = returns_data[:, -1]
            final_regime = regime_ids[:, -1]

            position = (actions == 1).float() - (actions == 2).float()

            is_hold = (actions == 0).float()
            is_trade = (actions != 0).float()

            # Base rewards
            hold_baseline = is_hold * 0.02
            trade_pnl = final_returns * position * 100

            # Regime modifiers
            low_vol_hold_bonus = is_hold * (final_regime == 0).float() * 0.03
            low_vol_trade_penalty = is_trade * (final_regime == 0).float() * 0.02
            trending_correct_bonus = is_trade * (final_regime == 1).float() * (final_returns * position > 0).float() * 0.08
            trending_wrong_penalty = is_trade * (final_regime == 1).float() * (final_returns * position < 0).float() * 0.04
            high_vol_hold_bonus = is_hold * (final_regime == 2).float() * 0.05
            high_vol_trade_penalty = is_trade * (final_regime == 2).float() * 0.04
            crisis_hold_bonus = is_hold * (final_regime == 3).float() * 0.08
            crisis_trade_penalty = is_trade * (final_regime == 3).float() * 0.06
            trade_cost = is_trade * 0.015

            rewards = (hold_baseline + trade_pnl +
                      low_vol_hold_bonus - low_vol_trade_penalty +
                      trending_correct_bonus - trending_wrong_penalty +
                      high_vol_hold_bonus - high_vol_trade_penalty +
                      crisis_hold_bonus - crisis_trade_penalty - trade_cost)

            total_reward += rewards.sum().item()
            total_samples += len(rewards)

            # Track actions per regime
            for a, r in zip(actions.cpu().numpy(), final_regime.cpu().numpy()):
                action_counts[a] += 1
                regime_action_counts[r][a] += 1

        # Compute per-regime action distribution
        regime_action_dist = {}
        regime_names = {0: 'LOW_VOL', 1: 'TRENDING', 2: 'HIGH_VOL', 3: 'CRISIS'}
        for r in range(4):
            total_r = sum(regime_action_counts[r].values())
            if total_r > 0:
                regime_action_dist[regime_names[r]] = {
                    'HOLD': regime_action_counts[r][0] / total_r,
                    'LONG': regime_action_counts[r][1] / total_r,
                    'SHORT': regime_action_counts[r][2] / total_r
                }

        return {
            'mean_reward': total_reward / total_samples,
            'action_distribution': {
                'HOLD': action_counts[0] / total_samples,
                'LONG': action_counts[1] / total_samples,
                'SHORT': action_counts[2] / total_samples
            },
            'regime_actions': regime_action_dist
        }


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    # Setup
    device = setup_environment()

    # Download dataset
    download_dataset()

    # Load data
    print("\nLoading dataset...")
    train_data, val_data, test_data, metadata = load_enriched_dataset(DATASET_PATH)

    print(f"  Train: {train_data['n_samples']} samples")
    print(f"  Val:   {val_data['n_samples']} samples")
    print(f"  Test:  {test_data['n_samples']} samples")
    print(f"  Features: {metadata['feature_dim']}")
    print(f"\nRegime Distribution:")
    for regime, pct in metadata['regime_distribution'].items():
        print(f"  {regime}: {pct*100:.1f}%")

    # Create datasets
    train_dataset = EnrichedTradingDataset(train_data, context_len=CONTEXT_LEN)
    val_dataset = EnrichedTradingDataset(val_data, context_len=CONTEXT_LEN)
    test_dataset = EnrichedTradingDataset(test_data, context_len=CONTEXT_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Feature dim: {train_dataset.get_feature_dim()}")

    # Create model
    FEATURE_DIM = train_dataset.get_feature_dim()

    model = RegimeConditionedPolicy(
        feature_dim=FEATURE_DIM,
        context_len=CONTEXT_LEN,
        n_regimes=4,
        hidden_dim=256,
        n_heads=4,
        n_layers=3,
        n_actions=3,
        dropout=0.1
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    # Higher entropy_coef (0.05) encourages exploration to find good trades
    trainer = PPOTrainer(
        policy=model,
        lr=LEARNING_RATE,
        gamma=0.99,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.05,  # Increased from 0.01 to encourage exploration
        device=device
    )

    # Training loop
    best_val_reward = float('-inf')

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()

        # Train
        train_metrics = trainer.train_epoch(train_loader)

        # Evaluate
        val_metrics = trainer.evaluate(val_loader)

        # Update scheduler
        trainer.scheduler.step()

        # Save best model
        if val_metrics['mean_reward'] > best_val_reward:
            best_val_reward = val_metrics['mean_reward']
            torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/best_model.pt')
            marker = ' *'
        else:
            marker = ''

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Train: {train_metrics['mean_reward']:.4f} | "
              f"Val: {val_metrics['mean_reward']:.4f} | "
              f"Time: {epoch_time:.1f}s{marker}")

        # Print action distribution every 10 epochs
        if (epoch + 1) % 10 == 0:
            ad = val_metrics['action_distribution']
            print(f"         Overall: HOLD={ad['HOLD']:.1%} LONG={ad['LONG']:.1%} SHORT={ad['SHORT']:.1%}")
            # Print per-regime actions
            if 'regime_actions' in val_metrics:
                for regime, actions in val_metrics['regime_actions'].items():
                    print(f"         {regime:8s}: HOLD={actions['HOLD']:.1%} LONG={actions['LONG']:.1%} SHORT={actions['SHORT']:.1%}")

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best val reward: {best_val_reward:.4f}")

    # Test evaluation
    print("\n" + "=" * 70)
    print("Test Set Evaluation")
    print("=" * 70)

    model.load_state_dict(torch.load(f'{CHECKPOINT_DIR}/best_model.pt'))
    test_metrics = trainer.evaluate(test_loader)

    print(f"Mean Reward: {test_metrics['mean_reward']:.4f}")
    print(f"\nOverall Action Distribution:")
    for action, pct in test_metrics['action_distribution'].items():
        print(f"  {action}: {pct:.1%}")

    print(f"\nPer-Regime Action Distribution:")
    if 'regime_actions' in test_metrics:
        for regime, actions in test_metrics['regime_actions'].items():
            print(f"  {regime:8s}: HOLD={actions['HOLD']:.1%} LONG={actions['LONG']:.1%} SHORT={actions['SHORT']:.1%}")

    # Save final checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'feature_dim': FEATURE_DIM,
        'context_len': CONTEXT_LEN,
        'n_regimes': 4,
        'hidden_dim': 256,
        'n_heads': 4,
        'n_layers': 3,
        'n_actions': 3,
        'test_reward': test_metrics['mean_reward'],
        'regime_distribution': metadata['regime_distribution']
    }

    torch.save(checkpoint, f'{CHECKPOINT_DIR}/himari_ppo_final.pt')
    print(f"\nFinal checkpoint saved to: {CHECKPOINT_DIR}/himari_ppo_final.pt")


if __name__ == "__main__":
    main()
