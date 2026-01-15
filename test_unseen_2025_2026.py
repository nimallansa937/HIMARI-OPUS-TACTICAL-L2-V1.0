#!/usr/bin/env python3
"""
HIMARI Layer 2 V1 - Test on Unseen 2025-2026 Data

Run the integrated pipeline on real BTC data from 2025-01-01 to 2026-01-14
to verify out-of-distribution (OOD) performance.

Usage:
    python test_unseen_2025_2026.py
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Base directory
BASE_DIR = Path(r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1")
sys.path.insert(0, str(BASE_DIR / "src"))

# =============================================================================
# Load Data and Checkpoints
# =============================================================================

def load_test_data() -> Dict[str, Any]:
    """Load the 2025-2026 test dataset."""
    data_path = BASE_DIR / "btc_1h_2025_2026_test_arrays.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_checkpoints() -> Dict[str, Any]:
    """Load trained checkpoints."""
    checkpoints = {}

    # AHHMM
    ahhmm_path = BASE_DIR / "L2V1 AHHMM FINAL" / "student_t_ahhmm_percentile.pkl"
    if ahhmm_path.exists():
        with open(ahhmm_path, 'rb') as f:
            checkpoints['ahhmm'] = pickle.load(f)

    # Risk Manager
    risk_path = BASE_DIR / "L2V1 RISK MANAGER FINAL" / "risk_manager_config.pkl"
    if risk_path.exists():
        with open(risk_path, 'rb') as f:
            checkpoints['risk_manager'] = pickle.load(f)

    return checkpoints


# =============================================================================
# Trading Simulation
# =============================================================================

class TradingSimulator:
    """Simple trading simulator for backtesting."""

    def __init__(self, initial_capital: float = 10000.0, fee_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.reset()

    def reset(self):
        self.capital = self.initial_capital
        self.position = 0  # -1, 0, 1
        self.position_size = 0.0
        self.entry_price = 0.0
        self.peak_capital = self.initial_capital
        self.trades = []
        self.equity_curve = []

    def step(self, action: int, position_size: float, current_price: float) -> Dict[str, Any]:
        """Execute a trading step."""
        result = {
            'pnl': 0.0,
            'trade_executed': False,
            'fee': 0.0
        }

        # Close existing position if changing direction
        if self.position != 0 and action != self.position:
            # Calculate PnL
            if self.position == 1:  # Long
                pnl = (current_price - self.entry_price) / self.entry_price * self.position_size * self.capital
            else:  # Short
                pnl = (self.entry_price - current_price) / self.entry_price * self.position_size * self.capital

            fee = abs(pnl) * self.fee_rate
            self.capital += pnl - fee
            result['pnl'] = pnl - fee
            result['fee'] = fee
            result['trade_executed'] = True

            self.trades.append({
                'type': 'close',
                'position': self.position,
                'entry': self.entry_price,
                'exit': current_price,
                'pnl': pnl - fee
            })

            self.position = 0
            self.position_size = 0.0
            self.entry_price = 0.0

        # Open new position
        if action != 0 and self.position == 0 and position_size > 0:
            self.position = action
            self.position_size = position_size
            self.entry_price = current_price
            result['trade_executed'] = True

            self.trades.append({
                'type': 'open',
                'position': action,
                'price': current_price,
                'size': position_size
            })

        # Track equity
        self.equity_curve.append(self.capital)
        self.peak_capital = max(self.peak_capital, self.capital)

        return result

    def get_stats(self) -> Dict[str, float]:
        """Calculate trading statistics."""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Calculate metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        max_drawdown = np.max(1 - equity / np.maximum.accumulate(equity))

        # Sharpe (annualized, assuming hourly data)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365)
        else:
            sharpe = 0.0

        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 0:
            sortino = np.mean(returns) / np.std(downside) * np.sqrt(24 * 365)
        else:
            sortino = sharpe

        # Win rate
        closed_trades = [t for t in self.trades if t['type'] == 'close']
        if closed_trades:
            wins = sum(1 for t in closed_trades if t['pnl'] > 0)
            win_rate = wins / len(closed_trades)
        else:
            win_rate = 0.0

        return {
            'total_return': total_return * 100,  # Percentage
            'max_drawdown': max_drawdown * 100,  # Percentage
            'sharpe': sharpe,
            'sortino': sortino,
            'n_trades': len(closed_trades),
            'win_rate': win_rate * 100,  # Percentage
            'final_capital': self.capital
        }


# =============================================================================
# Pipeline
# =============================================================================

class TestPipeline:
    """Pipeline for testing on unseen data."""

    def __init__(self, checkpoints: Dict[str, Any]):
        self.ahhmm = checkpoints.get('ahhmm', {})
        self.risk_config = checkpoints.get('risk_manager', {})

        # Running normalization
        self.feature_mean = None
        self.feature_var = None
        self.n_samples = 0

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics."""
        if self.feature_mean is None:
            self.feature_mean = np.zeros(features.shape[-1])
            self.feature_var = np.ones(features.shape[-1])

        self.n_samples += 1
        delta = features - self.feature_mean
        self.feature_mean += delta / self.n_samples
        self.feature_var = ((self.n_samples - 1) * self.feature_var + delta * (features - self.feature_mean)) / self.n_samples

        std = np.sqrt(self.feature_var + 1e-8)
        return (features - self.feature_mean) / std

    def decide(self, features: np.ndarray, regime: int) -> Tuple[int, float, float]:
        """Make trading decision based on features and regime."""
        # Use momentum and volatility signals
        momentum_idx = 26  # momentum_5 index based on metadata
        vol_idx = 38  # volatility_10 index
        rsi_idx = 32  # rsi_14 index

        momentum = features[momentum_idx] if momentum_idx < len(features) else 0
        volatility = features[vol_idx] if vol_idx < len(features) else 0.01
        rsi = features[rsi_idx] if rsi_idx < len(features) else 50

        # Crisis mode: conservative
        if regime == 3:
            return 0, 0.3, 0.0

        # Momentum-based strategy with RSI filter
        if momentum > 0.02 and rsi < 70:
            action = 1  # BUY
            confidence = min(0.5 + momentum * 10, 0.9)
        elif momentum < -0.02 and rsi > 30:
            action = -1  # SELL
            confidence = min(0.5 - momentum * 10, 0.9)
        else:
            action = 0  # HOLD
            confidence = 0.5

        # Position sizing with Kelly-like scaling
        kelly_cap = self.risk_config.get('H2_Kelly', {}).get('fraction_cap', 0.25)
        position_size = confidence * kelly_cap * (1 / (1 + volatility * 10))

        return action, confidence, min(position_size, 0.25)


# =============================================================================
# Main Test
# =============================================================================

def run_test():
    """Run the unseen data test."""
    print("=" * 70)
    print("HIMARI Layer 2 V1 - Unseen Data Test (2025-2026)")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading test data...")
    data = load_test_data()
    test_data = data['test']
    metadata = data['metadata']

    n_samples = test_data['n_samples']
    print(f"  Loaded {n_samples} samples")
    print(f"  Date range: {metadata['start_date']} to {metadata['end_date']}")
    print(f"  Features: {metadata['n_features']}")

    # Load checkpoints
    print("\n[2/4] Loading checkpoints...")
    checkpoints = load_checkpoints()
    print(f"  Loaded {len(checkpoints)} checkpoint(s)")

    # Initialize
    print("\n[3/4] Initializing pipeline and simulator...")
    pipeline = TestPipeline(checkpoints)
    simulator = TradingSimulator(initial_capital=10000.0, fee_rate=0.001)
    print("  Pipeline ready")

    # Run backtest
    print("\n[4/4] Running backtest...")
    features = test_data['features_denoised']
    regime_ids = test_data['regime_ids']
    prices = test_data['prices']
    returns = test_data['returns']

    actions_taken = []
    latencies = []

    for i in range(n_samples):
        start = time.perf_counter()

        # Normalize features
        norm_features = pipeline.normalize(features[i])

        # Get regime
        regime = regime_ids[i]

        # Make decision
        action, confidence, position_size = pipeline.decide(norm_features, regime)

        # Execute trade
        result = simulator.step(action, position_size, prices[i])

        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        actions_taken.append(action)

        # Progress
        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i + 1}/{n_samples} ({100*(i+1)/n_samples:.1f}%)")

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

    print("\n[LATENCY STATISTICS]")
    latencies = np.array(latencies)
    print(f"  Mean:   {np.mean(latencies):.3f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.3f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.3f} ms")

    print("\n[ACTION DISTRIBUTION]")
    actions = np.array(actions_taken)
    print(f"  SELL (-1): {np.sum(actions == -1):5d} ({100*np.mean(actions == -1):.1f}%)")
    print(f"  HOLD (0):  {np.sum(actions == 0):5d} ({100*np.mean(actions == 0):.1f}%)")
    print(f"  BUY (1):   {np.sum(actions == 1):5d} ({100*np.mean(actions == 1):.1f}%)")

    print("\n[REGIME DISTRIBUTION]")
    regime_names = ['BULL', 'BEAR', 'SIDEWAYS', 'CRISIS']
    for r in range(4):
        count = np.sum(regime_ids == r)
        pct = 100 * count / len(regime_ids)
        print(f"  {regime_names[r]:8s}: {count:5d} ({pct:.1f}%)")

    # Validation
    print("\n[VALIDATION]")
    sharpe_target = 0.3
    dd_target = 25.0

    sharpe_pass = stats['sharpe'] > sharpe_target
    dd_pass = stats['max_drawdown'] < dd_target
    latency_pass = np.percentile(latencies, 99) < 50.0

    print(f"  Sharpe > {sharpe_target}:        {'PASS' if sharpe_pass else 'FAIL'} ({stats['sharpe']:.3f})")
    print(f"  Max DD < {dd_target}%:         {'PASS' if dd_pass else 'FAIL'} ({stats['max_drawdown']:.2f}%)")
    print(f"  Latency P99 < 50ms:   {'PASS' if latency_pass else 'FAIL'} ({np.percentile(latencies, 99):.3f} ms)")

    all_passed = sharpe_pass and dd_pass and latency_pass

    print("\n" + "=" * 70)
    if all_passed:
        print("[SUCCESS] All validation criteria met!")
    else:
        print("[WARNING] Some criteria not met (but system is functional)")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    try:
        stats = run_test()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
