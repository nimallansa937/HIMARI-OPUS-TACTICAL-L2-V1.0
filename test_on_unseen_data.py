#!/usr/bin/env python3
"""
Test trained HIMARI Layer 2 model on unseen 2025-2026 data.

This script:
1. Loads the trained model checkpoint
2. Loads the 2025-2026 test dataset
3. Evaluates regime-conditioned behavior on unseen data
4. Compares to training data results

Run on Vast.ai after uploading btc_1h_2025_2026_test_arrays.pkl
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Configuration
# =============================================================================

CHECKPOINT_PATH = "/workspace/checkpoints/himari_ppo_final.pt"
TEST_DATA_PATH = "/workspace/btc_1h_2025_2026_test_arrays.pkl"
CONTEXT_LEN = 100
BATCH_SIZE = 64

# =============================================================================
# Dataset
# =============================================================================

class TestDataset(Dataset):
    def __init__(self, data_dict, context_len=100):
        self.context_len = context_len
        self.n_samples = data_dict['n_samples']

        self.features = torch.tensor(data_dict['features_denoised'], dtype=torch.float32)
        self.regime_ids = torch.tensor(data_dict['regime_ids'], dtype=torch.long)
        self.returns = torch.tensor(data_dict['returns'], dtype=torch.float32)

        # Normalize features
        self.features = (self.features - self.features.mean(0)) / (self.features.std(0) + 1e-8)

    def __len__(self):
        return self.n_samples - self.context_len

    def __getitem__(self, idx):
        end = idx + self.context_len
        return {
            'features': self.features[idx:end],
            'regime_ids': self.regime_ids[idx:end],
            'returns': self.returns[idx:end]
        }

# =============================================================================
# Model (same architecture as training)
# =============================================================================

class RegimeConditionedPolicy(nn.Module):
    def __init__(
        self,
        feature_dim: int = 49,
        context_len: int = 100,
        n_regimes: int = 4,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        n_actions: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.regime_embed = nn.Embedding(n_regimes, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, context_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_actions)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, regime_ids):
        B, T, F = features.shape

        x = self.feature_proj(features)
        regime_emb = self.regime_embed(regime_ids)
        x = x + regime_emb + self.pos_encoding[:, :T, :]

        x = self.transformer(x)

        last_hidden = x[:, -1, :]
        regime_context = regime_emb[:, -1, :]
        combined = torch.cat([last_hidden, regime_context], dim=-1)

        action_logits = self.actor_head(combined)
        value = self.critic_head(combined)

        return action_logits, value

    def get_action(self, features, regime_ids, deterministic=False):
        action_logits, value = self.forward(features, regime_ids)

        if deterministic:
            actions = action_logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=action_logits)
            actions = dist.sample()

        return actions, action_logits, value

# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate model on test data."""
    model.eval()

    action_counts = {0: 0, 1: 0, 2: 0}
    regime_action_counts = {r: {0: 0, 1: 0, 2: 0} for r in range(4)}

    total_reward = 0
    total_samples = 0

    all_returns = []
    all_positions = []
    all_regimes = []

    for batch in dataloader:
        features = batch['features'].to(device)
        regime_ids = batch['regime_ids'].to(device)
        returns_data = batch['returns'].to(device)

        actions, _, _ = model.get_action(features, regime_ids, deterministic=True)

        final_returns = returns_data[:, -1]
        final_regime = regime_ids[:, -1]

        position = (actions == 1).float() - (actions == 2).float()

        # Store for PnL calculation
        all_returns.extend(final_returns.cpu().numpy())
        all_positions.extend(position.cpu().numpy())
        all_regimes.extend(final_regime.cpu().numpy())

        # Compute reward (same as training)
        is_hold = (actions == 0).float()
        is_trade = (actions != 0).float()

        batch_vol = torch.std(final_returns) + 1e-6
        norm_pnl = final_returns / batch_vol
        trade_pnl = norm_pnl * position

        expected_abs_pnl = torch.abs(norm_pnl).mean()

        base_trade_cost = is_trade * 0.40 * expected_abs_pnl
        highvol_trade_cost = is_trade * (final_regime == 2).float() * 0.40 * expected_abs_pnl
        crisis_trade_cost = is_trade * (final_regime == 3).float() * 0.60 * expected_abs_pnl
        trending_trade_bonus = is_trade * (final_regime == 1).float() * 0.40 * expected_abs_pnl

        trending_hold_cost = is_hold * (final_regime == 1).float() * 1.0 * expected_abs_pnl

        wrong_direction = (position * final_returns < 0).float()
        wrong_penalty = wrong_direction * 0.50 * expected_abs_pnl

        hold_cost = trending_hold_cost
        trade_cost = base_trade_cost + highvol_trade_cost + crisis_trade_cost - trending_trade_bonus

        rewards = trade_pnl - hold_cost - trade_cost - wrong_penalty

        total_reward += rewards.sum().item()
        total_samples += len(rewards)

        # Track actions per regime
        for a, r in zip(actions.cpu().numpy(), final_regime.cpu().numpy()):
            action_counts[a] += 1
            regime_action_counts[r][a] += 1

    # Compute actual PnL
    all_returns = np.array(all_returns)
    all_positions = np.array(all_positions)
    all_regimes = np.array(all_regimes)

    actual_pnl = (all_returns * all_positions).sum()

    # Results
    results = {
        'mean_reward': total_reward / total_samples,
        'actual_pnl': actual_pnl,
        'total_samples': total_samples,
        'action_distribution': {
            'HOLD': action_counts[0] / total_samples * 100,
            'LONG': action_counts[1] / total_samples * 100,
            'SHORT': action_counts[2] / total_samples * 100
        },
        'regime_actions': {}
    }

    regime_names = {0: 'LOW_VOL', 1: 'TRENDING', 2: 'HIGH_VOL', 3: 'CRISIS'}
    for r in range(4):
        total_r = sum(regime_action_counts[r].values())
        if total_r > 0:
            results['regime_actions'][regime_names[r]] = {
                'HOLD': regime_action_counts[r][0] / total_r * 100,
                'LONG': regime_action_counts[r][1] / total_r * 100,
                'SHORT': regime_action_counts[r][2] / total_r * 100,
                'count': total_r
            }

    return results

# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("HIMARI Layer 2 - Testing on Unseen 2025-2026 Data")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load test data
    print(f"\nLoading test data from: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, 'rb') as f:
        test_data = pickle.load(f)

    test_split = test_data['test']
    print(f"Test samples: {test_split['n_samples']}")
    print(f"Features: {test_split['features_denoised'].shape[1]}")

    # Print regime distribution
    regime_ids = test_split['regime_ids']
    regime_names = ['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS']
    print("\nRegime Distribution (2025-2026 data):")
    for r in range(4):
        pct = (regime_ids == r).sum() / len(regime_ids) * 100
        print(f"  {regime_names[r]}: {pct:.1f}%")

    # Create dataset and loader
    dataset = TestDataset(test_split, context_len=CONTEXT_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    print(f"\nLoading model from: {CHECKPOINT_PATH}")
    model = RegimeConditionedPolicy(
        feature_dim=49,
        context_len=CONTEXT_LEN,
        n_regimes=4,
        hidden_dim=256,
        n_heads=4,
        n_layers=3
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded (trained for {checkpoint.get('epoch', 'unknown')} epochs)")

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation Results (Unseen 2025-2026 Data)")
    print("=" * 70)

    results = evaluate_model(model, loader, device)

    print(f"\nMean Reward: {results['mean_reward']:.4f}")
    print(f"Actual PnL: {results['actual_pnl']:.6f}")
    print(f"Total Samples: {results['total_samples']}")

    print("\nOverall Action Distribution:")
    for action, pct in results['action_distribution'].items():
        print(f"  {action}: {pct:.1f}%")

    print("\nPer-Regime Action Distribution:")
    for regime, actions in results['regime_actions'].items():
        print(f"  {regime}: HOLD={actions['HOLD']:.1f}% LONG={actions['LONG']:.1f}% SHORT={actions['SHORT']:.1f}% (n={actions['count']})")

    # Comparison with training results
    print("\n" + "=" * 70)
    print("Comparison: Training vs Unseen Data")
    print("=" * 70)

    training_results = {
        'LOW_VOL': 37.9,
        'TRENDING': 0.0,
        'HIGH_VOL': 49.8,
        'CRISIS': 45.5
    }

    print("\n         Training  | 2025-2026 | Difference")
    print("-" * 50)
    for regime in ['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS']:
        train_hold = training_results[regime]
        test_hold = results['regime_actions'].get(regime, {}).get('HOLD', 0)
        diff = test_hold - train_hold
        print(f"{regime:8s}  {train_hold:5.1f}%    |  {test_hold:5.1f}%   | {diff:+5.1f}%")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
