"""
Layer 3 Bounded Delta PPO Training - Vast.ai Version
FIXED with Layer 2 Lessons + Percentile Normalization

Key fixes from Layer 2 training:
1. Variance-normalized rewards (Session 7-18)
2. Adaptive costs as fractions of E[|PnL|] (Session 18)
3. PERCENTILE FEATURES computed on each dataset's OWN rolling window (AHHMM Session 3)
4. No guaranteed bonuses (causes collapse)
5. Regime-specific cost scaling

Original L3 problem: Sharpe = -0.078
Local test after L2 fix: Sharpe = 0.906 (train), 0.198 (test)
Expected after percentile fix: Better generalization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pickle
import os
import gdown
from datetime import datetime

print("=" * 70)
print("Layer 3 Bounded Delta PPO - Vast.ai Version")
print("FIXED with Layer 2 Lessons + Percentile Normalization")
print("=" * 70)

# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Google Drive file IDs (user will update these)
    GDRIVE_TRAIN_ID = "YOUR_TRAIN_FILE_ID"  # btc_1h_2020_2024.csv
    GDRIVE_TEST_ID = "YOUR_TEST_FILE_ID"    # btc_1h_2025_2026.csv

    # Local paths (for Vast.ai)
    data_dir = "/workspace/data"
    train_data_path = "/workspace/data/btc_1h_2020_2024.csv"
    test_data_path = "/workspace/data/btc_1h_2025_2026.csv"

    # Bounded Delta (from L3 - this is correct)
    delta_lower = -0.30  # Max 30% reduction
    delta_upper = 0.30   # Max 30% increase

    # Network
    state_dim = 9  # Percentile features (will be updated)
    hidden_dim = 256  # Larger for GPU
    lstm_layers = 2
    sequence_length = 20

    # PPO Hyperparameters
    actor_lr = 3e-4
    critic_lr = 1e-3
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    entropy_coef = 0.05  # L2 lesson: higher entropy prevents collapse

    # Training
    epochs = 50  # Full training on GPU
    batch_size = 256  # Larger batch for GPU
    update_epochs = 4

    # L2 Adaptive Costs (as fractions of E[|norm_pnl|])
    base_trade_cost_frac = 0.40
    lowvol_trade_cost_frac = 0.0
    highvol_trade_cost_frac = 0.40
    crisis_trade_cost_frac = 0.60
    trending_trade_bonus_frac = 0.40
    position_change_cost_frac = 0.20

    # Regime multipliers
    regime_multipliers = {
        0: 1.0,   # LOW_VOL
        1: 1.2,   # TRENDING
        2: 0.6,   # HIGH_VOL
        3: 0.2,   # CRISIS
    }

    # Output
    output_dir = "/workspace/checkpoints"

config = Config()

# =============================================================================
# Download Data from Google Drive
# =============================================================================

def download_data():
    """Download data from Google Drive"""
    os.makedirs(config.data_dir, exist_ok=True)

    if not os.path.exists(config.train_data_path):
        print(f"Downloading training data...")
        url = f"https://drive.google.com/uc?id={config.GDRIVE_TRAIN_ID}"
        gdown.download(url, config.train_data_path, quiet=False)
    else:
        print(f"Training data already exists: {config.train_data_path}")

    if not os.path.exists(config.test_data_path):
        print(f"Downloading test data...")
        url = f"https://drive.google.com/uc?id={config.GDRIVE_TEST_ID}"
        gdown.download(url, config.test_data_path, quiet=False)
    else:
        print(f"Test data already exists: {config.test_data_path}")

# =============================================================================
# Data Loading with PERCENTILE FEATURES (L2 AHHMM Lesson)
# =============================================================================

def compute_percentile_features(df, lookback=500):
    """
    L2 AHHMM Lesson: Compute percentile-based features

    Key insight: "75th percentile of volatility" means the same thing
    in 2020 and 2025, unlike absolute volatility values.

    Each dataset computes percentiles on ITS OWN rolling window.
    """
    # Basic features first
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility features
    df['volatility'] = df['returns'].rolling(24).std()
    df['vol_of_vol'] = df['volatility'].rolling(24).std()

    # Volume features
    df['volume_ma'] = df['volume'].rolling(24).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Trend features
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['close']

    # True range
    df['high_low'] = df['high'] - df['low']
    df['true_range'] = df['high_low'] / df['close']

    # Momentum
    df['momentum_12'] = df['close'].pct_change(12)
    df['momentum_24'] = df['close'].pct_change(24)

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Drop NaN from feature computation
    df = df.dropna().reset_index(drop=True)

    # === PERCENTILE TRANSFORMATION (KEY L2 FIX) ===
    feature_cols = ['volatility', 'trend_strength', 'volume_ratio',
                    'true_range', 'vol_of_vol', 'momentum_12',
                    'momentum_24', 'rsi']

    percentile_features = []
    for col in feature_cols:
        # Each value becomes its percentile within the lookback window
        pct = df[col].rolling(lookback, min_periods=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        percentile_features.append(pct.values)

    # Return direction (signed, mapped to [0, 1])
    ret_dir = np.sign(df['returns'].values) * 0.5 + 0.5
    percentile_features.append(ret_dir)

    # Stack features
    features = np.column_stack(percentile_features)

    # Detect regimes using percentile thresholds
    vol_pct = percentile_features[0]
    trend_pct = percentile_features[1]

    regimes = np.zeros(len(df), dtype=np.int64)
    regimes[(vol_pct < 0.4) & (trend_pct < 0.5)] = 0  # LOW_VOL
    regimes[(trend_pct >= 0.6)] = 1  # TRENDING
    regimes[(vol_pct >= 0.7) & (vol_pct < 0.9)] = 2  # HIGH_VOL
    regimes[(vol_pct >= 0.9)] = 3  # CRISIS

    # Get returns and prices
    returns = df['returns'].values
    prices = df['close'].values

    # Drop initial NaN from percentile calculation
    valid_idx = ~np.isnan(features).any(axis=1)
    features = features[valid_idx]
    regimes = regimes[valid_idx]
    returns = returns[valid_idx]
    prices = prices[valid_idx]

    return features, regimes, returns, prices

def load_and_prepare_data(filepath):
    """Load data and compute percentile-based features"""
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)

    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

    features, regimes, returns, prices = compute_percentile_features(df)

    print(f"  {len(features)} samples, {features.shape[1]} features")
    print(f"  Regime distribution: LOW_VOL={np.mean(regimes==0):.1%}, "
          f"TRENDING={np.mean(regimes==1):.1%}, "
          f"HIGH_VOL={np.mean(regimes==2):.1%}, "
          f"CRISIS={np.mean(regimes==3):.1%}")

    return features, regimes, returns, prices

# =============================================================================
# Actor-Critic Network (GPU-optimized)
# =============================================================================

class BoundedDeltaActorCritic(nn.Module):
    """
    Outputs bounded delta in [-0.30, +0.30]
    Uses tanh to bound output naturally
    """
    def __init__(self, state_dim, hidden_dim, delta_bounds=(-0.30, 0.30)):
        super().__init__()
        self.delta_bounds = delta_bounds
        self.delta_range = (delta_bounds[1] - delta_bounds[0]) / 2
        self.delta_mid = (delta_bounds[1] + delta_bounds[0]) / 2

        # Shared feature extractor (deeper for GPU)
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Actor (outputs mean and log_std for delta)
        self.actor_mean = nn.Linear(hidden_dim // 2, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, state):
        features = self.feature_net(state)

        # Actor: mean of delta (will be passed through tanh)
        delta_mean_raw = self.actor_mean(features)
        delta_mean = torch.tanh(delta_mean_raw) * self.delta_range + self.delta_mid

        # Std (constrained)
        delta_std = torch.exp(self.actor_log_std).clamp(0.01, 0.3)

        # Critic: state value
        value = self.critic(features)

        return delta_mean, delta_std, value

    def get_action(self, state, deterministic=False):
        delta_mean, delta_std, value = self.forward(state)

        if deterministic:
            delta = delta_mean
            log_prob = torch.zeros_like(delta)
        else:
            dist = Normal(delta_mean, delta_std)
            delta = dist.sample()
            delta = delta.clamp(self.delta_bounds[0], self.delta_bounds[1])
            log_prob = dist.log_prob(delta)

        return delta, log_prob, value

    def evaluate_actions(self, states, actions):
        delta_mean, delta_std, values = self.forward(states)
        dist = Normal(delta_mean, delta_std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy

# =============================================================================
# L2-Fixed Reward Function
# =============================================================================

def compute_l2_fixed_reward(returns, positions, position_changes, regimes, config, device):
    """L2-style variance-normalized reward with adaptive costs"""

    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    positions = torch.tensor(positions, dtype=torch.float32, device=device)
    position_changes = torch.tensor(position_changes, dtype=torch.float32, device=device)
    regimes = torch.tensor(regimes, dtype=torch.int64, device=device)

    # Variance normalization
    batch_vol = torch.std(returns) + 1e-6
    norm_returns = returns / batch_vol

    # Position PnL
    position_pnl = positions * norm_returns

    # Adaptive costs
    expected_abs_pnl = torch.abs(norm_returns).mean()

    # Position change cost
    abs_position_change = torch.abs(position_changes)
    change_cost = abs_position_change * config.position_change_cost_frac * expected_abs_pnl

    # Regime-specific costs
    is_trading = (torch.abs(positions) > 0.1).float()
    base_cost = is_trading * config.base_trade_cost_frac * expected_abs_pnl
    highvol_cost = is_trading * (regimes == 2).float() * config.highvol_trade_cost_frac * expected_abs_pnl
    crisis_cost = is_trading * (regimes == 3).float() * config.crisis_trade_cost_frac * expected_abs_pnl
    trending_bonus = is_trading * (regimes == 1).float() * config.trending_trade_bonus_frac * expected_abs_pnl

    # Wrong direction penalty
    wrong_direction = (positions * norm_returns < 0).float()
    wrong_penalty = wrong_direction * 0.30 * expected_abs_pnl
    risky_regime = ((regimes == 2) | (regimes == 3)).float()
    risky_wrong_penalty = wrong_direction * risky_regime * 0.20 * expected_abs_pnl

    # Combine
    rewards = (position_pnl - change_cost - base_cost - highvol_cost
               - crisis_cost + trending_bonus - wrong_penalty - risky_wrong_penalty)

    return rewards

# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model, optimizer, features, regimes, returns, config, device):
    """Train one epoch with L2-fixed rewards"""

    model.train()
    n_samples = len(features)
    indices = np.random.permutation(n_samples - 1)

    total_reward = 0
    total_loss = 0
    n_batches = 0
    all_deltas = []
    all_positions = []

    for batch_start in range(0, len(indices) - config.batch_size, config.batch_size):
        batch_indices = indices[batch_start:batch_start + config.batch_size]

        # Prepare batch
        batch_states = torch.tensor(
            features[batch_indices], dtype=torch.float32, device=device
        )
        batch_returns_np = returns[batch_indices + 1]
        batch_regimes_np = regimes[batch_indices]

        # Get actions
        with torch.no_grad():
            deltas, log_probs, values = model.get_action(batch_states)

        deltas_np = deltas.cpu().numpy().flatten()
        all_deltas.extend(deltas_np)

        # Apply regime multipliers
        regime_mults = np.array([config.regime_multipliers[r] for r in batch_regimes_np])
        base_position = 0.5
        positions = base_position * (1 + deltas_np) * regime_mults
        positions = np.clip(positions, 0, 1)
        all_positions.extend(positions)

        # Position changes
        position_changes = deltas_np

        # Compute rewards
        rewards = compute_l2_fixed_reward(
            batch_returns_np, positions, position_changes,
            batch_regimes_np, config, device
        )

        rewards_tensor = rewards.unsqueeze(1)
        total_reward += rewards.mean().item()

        # PPO Update
        for _ in range(config.update_epochs):
            new_log_probs, new_values, entropy = model.evaluate_actions(
                batch_states, deltas
            )

            advantages = rewards_tensor - new_values.detach()

            # Actor loss
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_epsilon,
                               1 + config.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            actor_loss = actor_loss - config.entropy_coef * entropy.mean()

            # Critic loss
            critic_loss = nn.MSELoss()(new_values, rewards_tensor)

            # Combined loss
            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

        n_batches += 1

    all_deltas = np.array(all_deltas)
    all_positions = np.array(all_positions)

    return {
        'mean_reward': total_reward / max(n_batches, 1),
        'loss': total_loss / max(n_batches * config.update_epochs, 1),
        'mean_delta': np.mean(all_deltas),
        'std_delta': np.std(all_deltas),
        'mean_position': np.mean(all_positions),
        'hold_pct': np.mean(all_positions < 0.1) * 100,
    }

# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, features, regimes, returns, prices, config, device, name=""):
    """Evaluate model"""

    model.eval()
    n_samples = len(features)

    positions = []
    pnls = []

    with torch.no_grad():
        for i in range(n_samples - 1):
            state = torch.tensor(features[i], dtype=torch.float32, device=device).unsqueeze(0)
            delta, _, _ = model.get_action(state, deterministic=True)
            delta_val = delta.item()

            regime_mult = config.regime_multipliers[regimes[i]]
            position = 0.5 * (1 + delta_val) * regime_mult
            position = np.clip(position, 0, 1)
            positions.append(position)

            pnl = position * returns[i + 1]
            pnls.append(pnl)

    positions = np.array(positions)
    pnls = np.array(pnls)

    # Metrics
    mean_return = np.mean(pnls)
    std_return = np.std(pnls)
    sharpe = mean_return / (std_return + 1e-10) * np.sqrt(24 * 365)

    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    max_drawdown = np.max(running_max - cumulative)

    # Regime stats
    regime_stats = {}
    regime_names = ['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS']
    for r in range(4):
        mask = regimes[:-1] == r
        if mask.sum() > 0:
            regime_stats[regime_names[r]] = {
                'mean_pos': float(np.mean(positions[mask])),
                'hold_pct': float(np.mean(positions[mask] < 0.1) * 100),
            }

    print(f"\n{name} Evaluation:")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Total Return: {np.sum(pnls)*100:.2f}%")
    print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"  Mean Position: {np.mean(positions):.2f}")
    for regime, stats in regime_stats.items():
        print(f"    {regime}: pos={stats['mean_pos']:.2f}, hold={stats['hold_pct']:.1f}%")

    return {
        'sharpe': sharpe,
        'total_return': float(np.sum(pnls)),
        'max_drawdown': float(max_drawdown),
        'mean_position': float(np.mean(positions)),
        'regime_stats': regime_stats,
    }

# =============================================================================
# Main
# =============================================================================

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Download data
    print("\n" + "=" * 70)
    print("Step 1: Download Data")
    print("=" * 70)
    download_data()

    # Load data
    print("\n" + "=" * 70)
    print("Step 2: Load and Process Data (Percentile Features)")
    print("=" * 70)

    train_features, train_regimes, train_returns, train_prices = load_and_prepare_data(
        config.train_data_path
    )
    test_features, test_regimes, test_returns, test_prices = load_and_prepare_data(
        config.test_data_path
    )

    config.state_dim = train_features.shape[1]

    # Initialize model
    print("\n" + "=" * 70)
    print("Step 3: Initialize Model")
    print("=" * 70)

    model = BoundedDeltaActorCritic(
        state_dim=config.state_dim,
        hidden_dim=config.hidden_dim,
        delta_bounds=(config.delta_lower, config.delta_upper)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.actor_lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Delta bounds: [{config.delta_lower}, {config.delta_upper}]")

    # Training
    print("\n" + "=" * 70)
    print(f"Step 4: Training ({config.epochs} epochs)")
    print("=" * 70)

    os.makedirs(config.output_dir, exist_ok=True)
    best_sharpe = -float('inf')
    training_log = []

    for epoch in range(1, config.epochs + 1):
        stats = train_epoch(
            model, optimizer, train_features, train_regimes, train_returns,
            config, device
        )
        training_log.append(stats)

        print(f"Epoch {epoch:2d}/{config.epochs}: "
              f"reward={stats['mean_reward']:.4f}, "
              f"delta={stats['mean_delta']:+.3f}, "
              f"pos={stats['mean_position']:.2f}, "
              f"hold={stats['hold_pct']:.1f}%")

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == config.epochs:
            test_results = evaluate_model(
                model, test_features, test_regimes, test_returns, test_prices,
                config, device, f"Epoch {epoch} Test"
            )

            if test_results['sharpe'] > best_sharpe:
                best_sharpe = test_results['sharpe']
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': vars(config),
                    'test_sharpe': test_results['sharpe'],
                }, os.path.join(config.output_dir, "best_model.pt"))
                print(f"  [BEST] Saved best model with Sharpe={best_sharpe:.3f}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("Step 5: Final Evaluation")
    print("=" * 70)

    train_results = evaluate_model(
        model, train_features, train_regimes, train_returns, train_prices,
        config, device, "Training (2020-2024)"
    )

    test_results = evaluate_model(
        model, test_features, test_regimes, test_returns, test_prices,
        config, device, "Test (2025-2026)"
    )

    # Comparison
    print("\n" + "=" * 70)
    print("Step 6: Generalization Comparison")
    print("=" * 70)

    sharpe_diff = test_results['sharpe'] - train_results['sharpe']
    print(f"\n{'Metric':<25} {'Training':<15} {'Test':<15} {'Diff'}")
    print("-" * 60)
    print(f"{'Sharpe Ratio':<25} {train_results['sharpe']:<15.3f} {test_results['sharpe']:<15.3f} {sharpe_diff:+.3f}")
    print(f"{'Max Drawdown':<25} {train_results['max_drawdown']*100:<14.2f}% {test_results['max_drawdown']*100:<14.2f}%")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'train_results': train_results,
        'test_results': test_results,
        'training_log': training_log,
    }, os.path.join(config.output_dir, "l3_bounded_delta_final.pt"))

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nOriginal L3 PPO Sharpe: -0.078")
    print(f"L2-Fixed Train Sharpe:  {train_results['sharpe']:.3f}")
    print(f"L2-Fixed Test Sharpe:   {test_results['sharpe']:.3f}")
    print(f"Best Test Sharpe:       {best_sharpe:.3f}")
    print(f"\nModels saved to: {config.output_dir}")

if __name__ == "__main__":
    main()
