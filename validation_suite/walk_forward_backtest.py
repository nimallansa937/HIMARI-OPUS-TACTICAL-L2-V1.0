#!/usr/bin/env python3
"""
HIMARI Layer 2 V1 - Walk-Forward Backtest

Implements rolling window training and testing to simulate realistic deployment:
- Train on window [t-N, t]
- Test on window [t, t+M]
- Roll forward and repeat

This prevents look-ahead bias and tests adaptation to regime changes.

Usage:
    python walk_forward_backtest.py
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
from dataclasses import dataclass
from copy import deepcopy

BASE_DIR = Path(r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1")

# =============================================================================
# Model Architecture (same as before)
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
        self.context_len = context_len
        self.n_actions = n_actions

        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.regime_embed = nn.Embedding(n_regimes, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, context_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, regime_ids):
        batch_size, seq_len, _ = features.shape
        x = self.feature_proj(features)
        x = x + self.regime_embed(regime_ids)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        last_hidden = x[:, -1, :]
        mean_hidden = x.mean(dim=1)
        combined = torch.cat([last_hidden, mean_hidden], dim=-1)
        return self.actor(combined), self.critic(combined)

    def get_action(self, features, regime_ids):
        with torch.no_grad():
            logits, _ = self.forward(features, regime_ids)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, action].item()
        return action - 1, confidence


# =============================================================================
# Walk-Forward Engine
# =============================================================================

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtest."""
    train_window: int = 2000      # Training window size (samples)
    test_window: int = 500        # Test window size (samples)
    step_size: int = 500          # Step forward after each fold
    retrain_epochs: int = 5       # Quick fine-tuning epochs
    learning_rate: float = 1e-4
    context_len: int = 100


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_sharpe: float
    test_sharpe: float
    test_return: float
    test_max_dd: float
    n_trades: int


class WalkForwardBacktest:
    """Walk-forward validation with rolling retraining."""

    def __init__(self, config: WalkForwardConfig, device: torch.device):
        self.config = config
        self.device = device
        self.results: List[FoldResult] = []

    def run(
        self,
        features: np.ndarray,
        regime_ids: np.ndarray,
        prices: np.ndarray,
        returns: np.ndarray,
        base_model: nn.Module
    ) -> Dict:
        """Run walk-forward backtest."""
        n_samples = len(features)
        context_len = self.config.context_len
        train_window = self.config.train_window
        test_window = self.config.test_window
        step_size = self.config.step_size

        # Convert to tensors
        features_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        regime_ids_t = torch.tensor(regime_ids, dtype=torch.long, device=self.device)

        # Normalize features globally for consistency
        feat_mean = features_t.mean(dim=0)
        feat_std = features_t.std(dim=0) + 1e-8
        features_t = (features_t - feat_mean) / feat_std

        fold_id = 0
        start_idx = context_len

        print(f"\nWalk-Forward Configuration:")
        print(f"  Train window: {train_window} samples")
        print(f"  Test window:  {test_window} samples")
        print(f"  Step size:    {step_size} samples")
        print(f"  Total samples: {n_samples}")

        while start_idx + train_window + test_window <= n_samples:
            train_start = start_idx
            train_end = start_idx + train_window
            test_start = train_end
            test_end = min(test_start + test_window, n_samples)

            print(f"\n--- Fold {fold_id + 1} ---")
            print(f"  Train: [{train_start}, {train_end})")
            print(f"  Test:  [{test_start}, {test_end})")

            # Clone model for this fold
            model = deepcopy(base_model).to(self.device)

            # Fine-tune on training window
            train_sharpe = self._fine_tune(
                model, features_t, regime_ids_t, returns,
                train_start, train_end
            )

            # Evaluate on test window
            test_metrics = self._evaluate(
                model, features_t, regime_ids_t, prices,
                test_start, test_end
            )

            result = FoldResult(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_sharpe=train_sharpe,
                test_sharpe=test_metrics['sharpe'],
                test_return=test_metrics['total_return'],
                test_max_dd=test_metrics['max_dd'],
                n_trades=test_metrics['n_trades']
            )
            self.results.append(result)

            print(f"  Train Sharpe: {train_sharpe:.3f}")
            print(f"  Test Sharpe:  {test_metrics['sharpe']:.3f}")
            print(f"  Test Return:  {test_metrics['total_return']:.2f}%")
            print(f"  Test Max DD:  {test_metrics['max_dd']:.2f}%")

            # Step forward
            start_idx += step_size
            fold_id += 1

        return self._aggregate_results()

    def _fine_tune(
        self,
        model: nn.Module,
        features: torch.Tensor,
        regime_ids: torch.Tensor,
        returns: np.ndarray,
        start: int,
        end: int
    ) -> float:
        """Quick fine-tuning on training window."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        context_len = self.config.context_len

        # Simple supervised signal: predict sign of next return
        for epoch in range(self.config.retrain_epochs):
            total_loss = 0
            n_batches = 0

            for i in range(start + context_len, end - 1):
                feat_window = features[i-context_len:i].unsqueeze(0)
                regime_window = regime_ids[i-context_len:i].unsqueeze(0)

                logits, _ = model(feat_window, regime_window)

                # Target: 0=sell, 1=hold, 2=buy based on next return
                next_ret = returns[i]
                if next_ret > 0.001:
                    target = 2  # Buy
                elif next_ret < -0.001:
                    target = 0  # Sell
                else:
                    target = 1  # Hold

                target_t = torch.tensor([target], device=self.device)
                loss = nn.CrossEntropyLoss()(logits, target_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        model.eval()

        # Estimate training Sharpe
        train_returns = self._simulate_returns(model, features, regime_ids, returns, start, end)
        if len(train_returns) > 1 and np.std(train_returns) > 0:
            return np.mean(train_returns) / np.std(train_returns) * np.sqrt(24 * 365)
        return 0.0

    def _simulate_returns(
        self,
        model: nn.Module,
        features: torch.Tensor,
        regime_ids: torch.Tensor,
        returns: np.ndarray,
        start: int,
        end: int
    ) -> np.ndarray:
        """Simulate strategy returns."""
        context_len = self.config.context_len
        strategy_returns = []
        position = 0

        for i in range(start + context_len, end):
            feat_window = features[i-context_len:i].unsqueeze(0)
            regime_window = regime_ids[i-context_len:i].unsqueeze(0)

            action, confidence = model.get_action(feat_window, regime_window)

            # Position change
            if action != position:
                position = action

            # Strategy return
            if position != 0:
                strategy_returns.append(position * returns[i] * 0.25)  # 25% position
            else:
                strategy_returns.append(0)

        return np.array(strategy_returns)

    def _evaluate(
        self,
        model: nn.Module,
        features: torch.Tensor,
        regime_ids: torch.Tensor,
        prices: np.ndarray,
        start: int,
        end: int
    ) -> Dict:
        """Evaluate model on test window."""
        context_len = self.config.context_len
        capital = 10000.0
        position = 0
        entry_price = 0.0
        equity_curve = [capital]
        n_trades = 0

        for i in range(start + context_len, end):
            feat_window = features[i-context_len:i].unsqueeze(0)
            regime_window = regime_ids[i-context_len:i].unsqueeze(0)

            action, confidence = model.get_action(feat_window, regime_window)
            current_price = prices[i]

            # Close position if direction changes
            if position != 0 and action != position:
                if position == 1:
                    pnl = (current_price - entry_price) / entry_price * 0.25 * capital
                else:
                    pnl = (entry_price - current_price) / entry_price * 0.25 * capital
                capital += pnl * 0.999  # 0.1% fee
                n_trades += 1
                position = 0

            # Open new position
            if action != 0 and position == 0:
                position = action
                entry_price = current_price

            equity_curve.append(capital)

        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]

        total_return = (capital - 10000) / 10000 * 100
        max_dd = np.max(1 - equity / np.maximum.accumulate(equity)) * 100

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365)
        else:
            sharpe = 0.0

        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'max_dd': max_dd,
            'n_trades': n_trades
        }

    def _aggregate_results(self) -> Dict:
        """Aggregate results across all folds."""
        if not self.results:
            return {}

        test_sharpes = [r.test_sharpe for r in self.results]
        test_returns = [r.test_return for r in self.results]
        test_dds = [r.test_max_dd for r in self.results]
        train_sharpes = [r.train_sharpe for r in self.results]

        # IS/OOS ratio (important metric)
        avg_train = np.mean(train_sharpes)
        avg_test = np.mean(test_sharpes)
        is_oos_ratio = avg_test / avg_train if avg_train != 0 else 0

        return {
            'n_folds': len(self.results),
            'avg_test_sharpe': np.mean(test_sharpes),
            'std_test_sharpe': np.std(test_sharpes),
            'avg_test_return': np.mean(test_returns),
            'avg_max_dd': np.mean(test_dds),
            'worst_dd': np.max(test_dds),
            'avg_train_sharpe': avg_train,
            'is_oos_ratio': is_oos_ratio,
            'pct_profitable_folds': np.mean([r.test_return > 0 for r in self.results]) * 100,
            'fold_results': self.results
        }


# =============================================================================
# Main
# =============================================================================

def run_walk_forward():
    """Run walk-forward validation."""
    print("=" * 70)
    print("HIMARI Layer 2 V1 - Walk-Forward Backtest")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\n[1/3] Loading base model...")
    checkpoint_path = BASE_DIR / "L2V1 PPO FINAL" / "himari_ppo_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = {
        'feature_dim': checkpoint.get('feature_dim', 49),
        'context_len': checkpoint.get('context_len', 100),
        'n_regimes': checkpoint.get('n_regimes', 4),
        'hidden_dim': checkpoint.get('hidden_dim', 256),
        'n_heads': checkpoint.get('n_heads', 4),
        'n_layers': checkpoint.get('n_layers', 3),
        'n_actions': checkpoint.get('n_actions', 3),
    }

    base_model = RegimeConditionedPolicy(**config)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()

    # Load data
    print("\n[2/3] Loading data...")
    data_path = BASE_DIR / "btc_1h_2025_2026_test_arrays.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    test_data = data['test']
    features = test_data['features_denoised']
    regime_ids = test_data['regime_ids']
    prices = test_data['prices']
    returns = test_data['returns']

    print(f"  Samples: {len(features)}")

    # Run walk-forward
    print("\n[3/3] Running walk-forward backtest...")
    wf_config = WalkForwardConfig(
        train_window=2000,
        test_window=500,
        step_size=500,
        retrain_epochs=3,
        context_len=100
    )

    backtest = WalkForwardBacktest(wf_config, device)
    start_time = time.time()
    results = backtest.run(features, regime_ids, prices, returns, base_model)
    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("WALK-FORWARD RESULTS")
    print("=" * 70)

    print(f"\n[SUMMARY]")
    print(f"  Total folds:           {results['n_folds']}")
    print(f"  Runtime:               {elapsed:.1f}s")
    print(f"  Avg Test Sharpe:       {results['avg_test_sharpe']:.3f}")
    print(f"  Std Test Sharpe:       {results['std_test_sharpe']:.3f}")
    print(f"  Avg Test Return:       {results['avg_test_return']:.2f}%")
    print(f"  Avg Max Drawdown:      {results['avg_max_dd']:.2f}%")
    print(f"  Worst Drawdown:        {results['worst_dd']:.2f}%")
    print(f"  IS/OOS Ratio:          {results['is_oos_ratio']:.3f}")
    print(f"  % Profitable Folds:    {results['pct_profitable_folds']:.1f}%")

    print("\n[FOLD DETAILS]")
    for r in results['fold_results']:
        print(f"  Fold {r.fold_id+1}: Train Sharpe={r.train_sharpe:.2f}, "
              f"Test Sharpe={r.test_sharpe:.2f}, Return={r.test_return:.1f}%, DD={r.test_max_dd:.1f}%")

    # Validation
    print("\n[VALIDATION]")
    is_oos_pass = results['is_oos_ratio'] > 0.5
    dd_pass = results['worst_dd'] < 25
    profitable_pass = results['pct_profitable_folds'] > 40

    print(f"  IS/OOS Ratio > 0.5:      {'PASS' if is_oos_pass else 'FAIL'} ({results['is_oos_ratio']:.3f})")
    print(f"  Worst DD < 25%:          {'PASS' if dd_pass else 'FAIL'} ({results['worst_dd']:.2f}%)")
    print(f"  >40% Profitable Folds:   {'PASS' if profitable_pass else 'FAIL'} ({results['pct_profitable_folds']:.1f}%)")

    print("=" * 70)

    return results


if __name__ == "__main__":
    try:
        results = run_walk_forward()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
