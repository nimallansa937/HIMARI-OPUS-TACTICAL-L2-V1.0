#!/usr/bin/env python3
"""
HIMARI Layer 2 V1 - GPU PPO Backtest on Unseen 2025-2026 Data

Runs the trained PPO transformer model on RTX 5070 GPU for fast inference.

Usage:
    python run_ppo_gpu_backtest.py
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Base directory
BASE_DIR = Path(r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1")

# =============================================================================
# Model Architecture (must match training)
# =============================================================================

class RegimeConditionedPolicy(nn.Module):
    """
    Transformer-based PPO policy with regime conditioning.
    Architecture matches the trained checkpoint.
    """

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
        self.context_len = context_len
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
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_actions)
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor, regime_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (batch, seq_len, feature_dim)
            regime_ids: (batch, seq_len)

        Returns:
            action_logits: (batch, n_actions)
            values: (batch, 1)
        """
        batch_size, seq_len, _ = features.shape

        # Project features
        x = self.feature_proj(features)  # (batch, seq_len, hidden_dim)

        # Add regime embedding
        regime_emb = self.regime_embed(regime_ids)  # (batch, seq_len, hidden_dim)
        x = x + regime_emb

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, hidden_dim)

        # Get last timestep and mean pooling
        last_hidden = x[:, -1, :]  # (batch, hidden_dim)
        mean_hidden = x.mean(dim=1)  # (batch, hidden_dim)

        # Concatenate for heads
        combined = torch.cat([last_hidden, mean_hidden], dim=-1)  # (batch, hidden_dim*2)

        # Actor and critic outputs
        action_logits = self.actor(combined)  # (batch, n_actions)
        values = self.critic(combined)  # (batch, 1)

        return action_logits, values

    def get_action(self, features: torch.Tensor, regime_ids: torch.Tensor) -> Tuple[int, float]:
        """Get action and confidence for inference."""
        with torch.no_grad():
            logits, _ = self.forward(features, regime_ids)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, action].item()
        return action - 1, confidence  # Convert 0,1,2 to -1,0,1


# =============================================================================
# Trading Simulator
# =============================================================================

class TradingSimulator:
    """Trading simulator for backtesting."""

    def __init__(self, initial_capital: float = 10000.0, fee_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.reset()

    def reset(self):
        self.capital = self.initial_capital
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.peak_capital = self.initial_capital
        self.trades = []
        self.equity_curve = []

    def step(self, action: int, position_size: float, current_price: float) -> Dict:
        """Execute a trading step."""
        result = {'pnl': 0.0, 'trade': False}

        # Close if changing direction
        if self.position != 0 and action != self.position:
            if self.position == 1:
                pnl = (current_price - self.entry_price) / self.entry_price * self.position_size * self.capital
            else:
                pnl = (self.entry_price - current_price) / self.entry_price * self.position_size * self.capital

            fee = abs(pnl) * self.fee_rate
            self.capital += pnl - fee
            result['pnl'] = pnl - fee
            result['trade'] = True

            self.trades.append({'pnl': pnl - fee})
            self.position = 0
            self.position_size = 0.0

        # Open new position
        if action != 0 and self.position == 0 and position_size > 0:
            self.position = action
            self.position_size = position_size
            self.entry_price = current_price
            result['trade'] = True

        self.equity_curve.append(self.capital)
        self.peak_capital = max(self.peak_capital, self.capital)
        return result

    def get_stats(self) -> Dict:
        """Calculate performance statistics."""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        total_return = (self.capital - self.initial_capital) / self.initial_capital
        max_drawdown = np.max(1 - equity / np.maximum.accumulate(equity))

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365)
        else:
            sharpe = 0.0

        downside = returns[returns < 0]
        sortino = np.mean(returns) / np.std(downside) * np.sqrt(24 * 365) if len(downside) > 0 else sharpe

        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        win_rate = wins / len(self.trades) if self.trades else 0.0

        return {
            'total_return': total_return * 100,
            'max_drawdown': max_drawdown * 100,
            'sharpe': sharpe,
            'sortino': sortino,
            'n_trades': len(self.trades),
            'win_rate': win_rate * 100,
            'final_capital': self.capital
        }


# =============================================================================
# Main Backtest
# =============================================================================

def run_gpu_backtest():
    """Run GPU-accelerated PPO backtest."""
    print("=" * 70)
    print("HIMARI Layer 2 V1 - GPU PPO Backtest (RTX 5070)")
    print("=" * 70)

    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load checkpoint
    print("\n[1/4] Loading PPO checkpoint...")
    checkpoint_path = BASE_DIR / "L2V1 PPO FINAL" / "himari_ppo_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    config = {
        'feature_dim': checkpoint.get('feature_dim', 49),
        'context_len': checkpoint.get('context_len', 100),
        'n_regimes': checkpoint.get('n_regimes', 4),
        'hidden_dim': checkpoint.get('hidden_dim', 256),
        'n_heads': checkpoint.get('n_heads', 4),
        'n_layers': checkpoint.get('n_layers', 3),
        'n_actions': checkpoint.get('n_actions', 3),
    }
    print(f"Model config: {config}")

    # Initialize model
    print("\n[2/4] Initializing model on GPU...")
    model = RegimeConditionedPolicy(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Model size: {n_params * 4 / 1024 / 1024:.2f} MB")

    # Load test data
    print("\n[3/4] Loading test data...")
    data_path = BASE_DIR / "btc_1h_2025_2026_test_arrays.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    test_data = data['test']
    n_samples = test_data['n_samples']
    context_len = config['context_len']

    print(f"Samples: {n_samples}")
    print(f"Date range: {data['metadata']['start_date']} to {data['metadata']['end_date']}")

    # Prepare tensors
    features = torch.tensor(test_data['features_denoised'], dtype=torch.float32, device=device)
    regime_ids = torch.tensor(test_data['regime_ids'], dtype=torch.long, device=device)
    prices = test_data['prices']

    # Normalize features
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)

    # Run backtest
    print("\n[4/4] Running backtest...")
    simulator = TradingSimulator(initial_capital=10000.0)

    actions_taken = []
    confidences = []
    latencies = []

    # Warmup GPU
    with torch.no_grad():
        _ = model(features[:context_len].unsqueeze(0), regime_ids[:context_len].unsqueeze(0))

    start_time = time.time()

    for i in range(context_len, n_samples):
        # Prepare input window
        feat_window = features[i-context_len:i].unsqueeze(0)  # (1, context_len, feature_dim)
        regime_window = regime_ids[i-context_len:i].unsqueeze(0)  # (1, context_len)

        # GPU inference
        t0 = time.perf_counter()
        action, confidence = model.get_action(feat_window, regime_window)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Position sizing (Kelly-based)
        position_size = confidence * 0.25 if action != 0 else 0.0

        # Execute trade
        simulator.step(action, position_size, prices[i])

        actions_taken.append(action)
        confidences.append(confidence)
        latencies.append(latency_ms)

        # Progress
        if (i - context_len + 1) % 2000 == 0:
            elapsed = time.time() - start_time
            progress = (i - context_len + 1) / (n_samples - context_len)
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"  Progress: {i-context_len+1}/{n_samples-context_len} ({progress*100:.1f}%) - ETA: {eta:.0f}s")

    total_time = time.time() - start_time

    # Results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    stats = simulator.get_stats()

    print("\n[PERFORMANCE METRICS]")
    print(f"  Total Return:  {stats['total_return']:+.2f}%")
    print(f"  Max Drawdown:  {stats['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio:  {stats['sharpe']:.3f}")
    print(f"  Sortino Ratio: {stats['sortino']:.3f}")
    print(f"  Win Rate:      {stats['win_rate']:.1f}%")
    print(f"  Total Trades:  {stats['n_trades']}")
    print(f"  Final Capital: ${stats['final_capital']:.2f}")

    print("\n[GPU LATENCY STATISTICS]")
    latencies = np.array(latencies)
    print(f"  Mean:   {np.mean(latencies):.3f} ms")
    print(f"  P50:    {np.percentile(latencies, 50):.3f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.3f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.3f} ms")
    print(f"  Max:    {np.max(latencies):.3f} ms")
    print(f"  Total time: {total_time:.1f}s ({(n_samples-context_len)/total_time:.0f} samples/sec)")

    print("\n[ACTION DISTRIBUTION]")
    actions = np.array(actions_taken)
    print(f"  SELL (-1): {np.sum(actions == -1):5d} ({100*np.mean(actions == -1):.1f}%)")
    print(f"  HOLD (0):  {np.sum(actions == 0):5d} ({100*np.mean(actions == 0):.1f}%)")
    print(f"  BUY (1):   {np.sum(actions == 1):5d} ({100*np.mean(actions == 1):.1f}%)")

    print("\n[CONFIDENCE STATISTICS]")
    confs = np.array(confidences)
    print(f"  Mean:   {np.mean(confs):.3f}")
    print(f"  Min:    {np.min(confs):.3f}")
    print(f"  Max:    {np.max(confs):.3f}")

    # Validation
    print("\n[VALIDATION]")
    sharpe_pass = stats['sharpe'] > 0.3
    dd_pass = stats['max_drawdown'] < 25.0
    latency_pass = np.percentile(latencies, 99) < 50.0

    print(f"  Sharpe > 0.3:         {'PASS' if sharpe_pass else 'FAIL'} ({stats['sharpe']:.3f})")
    print(f"  Max DD < 25%:         {'PASS' if dd_pass else 'FAIL'} ({stats['max_drawdown']:.2f}%)")
    print(f"  Latency P99 < 50ms:   {'PASS' if latency_pass else 'FAIL'} ({np.percentile(latencies, 99):.3f} ms)")

    print("\n" + "=" * 70)
    if sharpe_pass and dd_pass and latency_pass:
        print("[SUCCESS] All criteria met!")
    else:
        print("[INFO] Backtest complete")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    try:
        stats = run_gpu_backtest()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
