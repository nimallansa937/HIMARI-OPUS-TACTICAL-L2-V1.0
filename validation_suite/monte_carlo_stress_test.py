#!/usr/bin/env python3
"""
HIMARI Layer 2 V1 - Monte Carlo Permutation Stress Test

Generates thousands of alternative market histories by permuting returns
to test strategy robustness. A robust strategy should be profitable
on most permutations, not just the actual historical path.

Usage:
    python monte_carlo_stress_test.py
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1")

# =============================================================================
# Model (same architecture)
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


# =============================================================================
# Monte Carlo Engine
# =============================================================================

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo stress test."""
    n_permutations: int = 1000      # Number of random permutations
    block_size: int = 20            # Block size for block bootstrap
    context_len: int = 100
    position_size: float = 0.25
    fee_rate: float = 0.001


@dataclass
class PermutationResult:
    """Results from a single permutation."""
    perm_id: int
    total_return: float
    max_drawdown: float
    sharpe: float
    n_trades: int


class MonteCarloStressTest:
    """Monte Carlo permutation stress test."""

    def __init__(self, config: MonteCarloConfig, device: torch.device):
        self.config = config
        self.device = device

    def run(
        self,
        model: nn.Module,
        features: np.ndarray,
        regime_ids: np.ndarray,
        returns: np.ndarray
    ) -> Dict:
        """Run Monte Carlo stress test."""
        n_samples = len(features)
        context_len = self.config.context_len

        # Get model decisions for all timesteps (deterministic)
        print("  Computing model decisions...")
        decisions = self._get_all_decisions(model, features, regime_ids)

        # Run actual historical path
        print("  Running historical baseline...")
        actual_result = self._simulate_path(decisions, returns)
        print(f"    Historical: Return={actual_result['total_return']:.2f}%, "
              f"Sharpe={actual_result['sharpe']:.3f}")

        # Run permutations
        print(f"  Running {self.config.n_permutations} permutations...")
        results: List[PermutationResult] = []

        start_time = time.time()

        for perm_id in range(self.config.n_permutations):
            # Block bootstrap permutation
            perm_returns = self._block_bootstrap(returns)

            # Simulate with permuted returns
            result = self._simulate_path(decisions, perm_returns)

            results.append(PermutationResult(
                perm_id=perm_id,
                total_return=result['total_return'],
                max_drawdown=result['max_drawdown'],
                sharpe=result['sharpe'],
                n_trades=result['n_trades']
            ))

            # Progress
            if (perm_id + 1) % 200 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (perm_id + 1) * (self.config.n_permutations - perm_id - 1)
                print(f"    Progress: {perm_id + 1}/{self.config.n_permutations} - ETA: {eta:.0f}s")

        return self._aggregate_results(results, actual_result)

    def _get_all_decisions(
        self,
        model: nn.Module,
        features: np.ndarray,
        regime_ids: np.ndarray
    ) -> np.ndarray:
        """Get model decisions for all timesteps."""
        model.eval()
        context_len = self.config.context_len
        n_samples = len(features)

        features_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        regime_ids_t = torch.tensor(regime_ids, dtype=torch.long, device=self.device)

        # Normalize
        features_t = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0) + 1e-8)

        decisions = np.zeros(n_samples, dtype=np.int32)

        with torch.no_grad():
            for i in range(context_len, n_samples):
                feat_window = features_t[i-context_len:i].unsqueeze(0)
                regime_window = regime_ids_t[i-context_len:i].unsqueeze(0)

                logits, _ = model(feat_window, regime_window)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item() - 1  # -1, 0, 1

                decisions[i] = action

        return decisions

    def _block_bootstrap(self, returns: np.ndarray) -> np.ndarray:
        """Block bootstrap to preserve autocorrelation."""
        n = len(returns)
        block_size = self.config.block_size

        # Generate blocks
        n_blocks = (n // block_size) + 1
        block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)

        # Concatenate blocks
        permuted = []
        for start in block_starts:
            permuted.extend(returns[start:start + block_size])

        return np.array(permuted[:n])

    def _simulate_path(self, decisions: np.ndarray, returns: np.ndarray) -> Dict:
        """Simulate trading on a return path."""
        context_len = self.config.context_len
        position_size = self.config.position_size
        fee_rate = self.config.fee_rate

        capital = 10000.0
        position = 0
        equity_curve = [capital]
        n_trades = 0

        for i in range(context_len, len(returns)):
            action = decisions[i]
            ret = returns[i]

            # Position change
            if action != position:
                n_trades += 1
                position = action

            # Apply return
            if position != 0:
                pnl = position * ret * position_size * capital
                fee = abs(pnl) * fee_rate
                capital += pnl - fee

            equity_curve.append(capital)

        equity = np.array(equity_curve)
        period_returns = np.diff(equity) / equity[:-1]

        total_return = (capital - 10000) / 10000 * 100
        max_dd = np.max(1 - equity / np.maximum.accumulate(equity)) * 100

        if len(period_returns) > 1 and np.std(period_returns) > 0:
            sharpe = np.mean(period_returns) / np.std(period_returns) * np.sqrt(24 * 365)
        else:
            sharpe = 0.0

        return {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'n_trades': n_trades
        }

    def _aggregate_results(self, results: List[PermutationResult], actual: Dict) -> Dict:
        """Aggregate Monte Carlo results."""
        sharpes = [r.sharpe for r in results]
        returns = [r.total_return for r in results]
        drawdowns = [r.max_drawdown for r in results]

        # Percentiles
        sharpe_5 = np.percentile(sharpes, 5)
        sharpe_50 = np.percentile(sharpes, 50)
        sharpe_95 = np.percentile(sharpes, 95)

        return_5 = np.percentile(returns, 5)
        return_95 = np.percentile(returns, 95)

        dd_95 = np.percentile(drawdowns, 95)

        # Probability of profit
        prob_profit = np.mean([r > 0 for r in returns]) * 100

        # Probability of beating random
        prob_beat_zero = np.mean([s > 0 for s in sharpes]) * 100

        # Historical vs Monte Carlo
        actual_percentile = np.mean([s < actual['sharpe'] for s in sharpes]) * 100

        return {
            'n_permutations': len(results),
            'actual_sharpe': actual['sharpe'],
            'actual_return': actual['total_return'],
            'actual_max_dd': actual['max_drawdown'],
            'mc_sharpe_5th': sharpe_5,
            'mc_sharpe_50th': sharpe_50,
            'mc_sharpe_95th': sharpe_95,
            'mc_return_5th': return_5,
            'mc_return_95th': return_95,
            'mc_dd_95th': dd_95,
            'prob_profit': prob_profit,
            'prob_positive_sharpe': prob_beat_zero,
            'actual_percentile': actual_percentile,
            'all_sharpes': sharpes,
            'all_returns': returns
        }


# =============================================================================
# Main
# =============================================================================

def run_monte_carlo():
    """Run Monte Carlo stress test."""
    print("=" * 70)
    print("HIMARI Layer 2 V1 - Monte Carlo Stress Test")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\n[1/3] Loading model...")
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

    model = RegimeConditionedPolicy(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data
    print("\n[2/3] Loading data...")
    data_path = BASE_DIR / "btc_1h_2025_2026_test_arrays.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    test_data = data['test']
    features = test_data['features_denoised']
    regime_ids = test_data['regime_ids']
    returns = test_data['returns']

    print(f"  Samples: {len(features)}")

    # Run Monte Carlo
    print("\n[3/3] Running Monte Carlo stress test...")
    mc_config = MonteCarloConfig(
        n_permutations=1000,
        block_size=20,
        context_len=100
    )

    stress_test = MonteCarloStressTest(mc_config, device)
    start_time = time.time()
    results = stress_test.run(model, features, regime_ids, returns)
    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("MONTE CARLO RESULTS")
    print("=" * 70)

    print(f"\n[ACTUAL HISTORICAL PATH]")
    print(f"  Sharpe:      {results['actual_sharpe']:.3f}")
    print(f"  Return:      {results['actual_return']:.2f}%")
    print(f"  Max DD:      {results['actual_max_dd']:.2f}%")

    print(f"\n[MONTE CARLO DISTRIBUTION ({results['n_permutations']} permutations)]")
    print(f"  Sharpe 5th percentile:   {results['mc_sharpe_5th']:.3f}")
    print(f"  Sharpe 50th percentile:  {results['mc_sharpe_50th']:.3f}")
    print(f"  Sharpe 95th percentile:  {results['mc_sharpe_95th']:.3f}")
    print(f"  Return 5th percentile:   {results['mc_return_5th']:.2f}%")
    print(f"  Return 95th percentile:  {results['mc_return_95th']:.2f}%")
    print(f"  Max DD 95th percentile:  {results['mc_dd_95th']:.2f}%")

    print(f"\n[ROBUSTNESS METRICS]")
    print(f"  Probability of profit:       {results['prob_profit']:.1f}%")
    print(f"  Probability of Sharpe > 0:   {results['prob_positive_sharpe']:.1f}%")
    print(f"  Actual vs MC percentile:     {results['actual_percentile']:.1f}th")
    print(f"  Runtime:                     {elapsed:.1f}s")

    # Validation
    print("\n[VALIDATION]")
    prob_profit_pass = results['prob_profit'] > 50
    sharpe_5th_pass = results['mc_sharpe_5th'] > -1.0
    dd_95th_pass = results['mc_dd_95th'] < 30

    print(f"  >50% Prob Profit:       {'PASS' if prob_profit_pass else 'FAIL'} ({results['prob_profit']:.1f}%)")
    print(f"  5th Sharpe > -1.0:      {'PASS' if sharpe_5th_pass else 'FAIL'} ({results['mc_sharpe_5th']:.3f})")
    print(f"  95th DD < 30%:          {'PASS' if dd_95th_pass else 'FAIL'} ({results['mc_dd_95th']:.2f}%)")

    print("=" * 70)

    return results


if __name__ == "__main__":
    try:
        results = run_monte_carlo()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
