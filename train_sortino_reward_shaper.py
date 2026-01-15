#!/usr/bin/env python3
"""
Calibrate Sortino Reward Shaper for HIMARI Layer 2.

The Sortino Reward Shaper transforms raw PnL into shaped rewards that:
1. Penalize downside volatility (not all volatility like Sharpe)
2. Penalize excessive trading
3. Account for drawdowns
4. Scale with regime

Calibration Approach:
- Grid search over key parameters (trade_cost, drawdown_weight, etc.)
- Evaluate using: Sortino ratio improvement, stability, regime adaptation
- Test on unseen 2025-2026 data

Key Lessons Applied:
- Variance-normalized parameters
- Test on unseen data
- Percentile-based thresholds where possible
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = r"C:\Users\chari\OneDrive\Documents\BTC DATA SETS"
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "btc_1h_2020_2024.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "btc_1h_2025_2026.csv")

OUTPUT_DIR = r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1\L2V1 SORTINO FINAL"
MODEL_PATH = os.path.join(OUTPUT_DIR, "sortino_config_calibrated.pkl")

# =============================================================================
# Sortino Reward Shaper Implementation
# =============================================================================

@dataclass
class SortinoConfig:
    """Sortino Reward Shaper configuration."""
    target_return: float = 0.0          # Minimum acceptable return (MAR)
    lookback_window: int = 50           # Window for downside deviation
    trade_cost: float = 0.001           # Penalty per trade
    drawdown_weight: float = 0.5        # Drawdown penalty weight
    reward_scale: float = 100.0         # Final reward scaling
    regime_scaling: bool = True         # Enable regime-based scaling


@dataclass
class RewardComponents:
    """Breakdown of shaped reward."""
    raw_return: float
    sortino_bonus: float
    trade_penalty: float
    drawdown_penalty: float
    total_reward: float


class SortinoRewardShaper:
    """Shapes rewards to optimize for Sortino ratio."""

    def __init__(self, config: Optional[SortinoConfig] = None):
        self.config = config or SortinoConfig()
        self.reset()

    def reset(self):
        """Reset shaper state."""
        self.returns_buffer = deque(maxlen=self.config.lookback_window)
        self.equity_buffer = deque(maxlen=self.config.lookback_window * 2)
        self.prev_position = None
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.n_updates = 0
        self.total_trades = 0
        self.total_reward = 0.0

    def shape(self, price_return: float, position: int,
              regime_id: int = 0, regime_confidence: float = 1.0) -> RewardComponents:
        """Shape raw return into training reward."""
        self.n_updates += 1

        # 1. Base reward: position * return
        raw_return = position * price_return

        # Update equity
        self.current_equity *= (1 + raw_return)
        self.equity_buffer.append(self.current_equity)
        self.returns_buffer.append(raw_return)

        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        # 2. Sortino bonus
        sortino_bonus = self._compute_sortino_bonus()

        # 3. Trade penalty
        trade_penalty = 0.0
        if self.prev_position is not None and position != self.prev_position:
            trade_penalty = self.config.trade_cost
            self.total_trades += 1
        self.prev_position = position

        # 4. Drawdown penalty
        drawdown_penalty = self._compute_drawdown_penalty()

        # 5. Regime scaling
        if self.config.regime_scaling:
            regime_scale = self._get_regime_scale(regime_id, regime_confidence)
        else:
            regime_scale = 1.0

        # Combine
        total_reward = (
            raw_return * self.config.reward_scale
            + sortino_bonus
            - trade_penalty * self.config.reward_scale
            - drawdown_penalty
        ) * regime_scale

        self.total_reward += total_reward

        return RewardComponents(
            raw_return=raw_return,
            sortino_bonus=sortino_bonus,
            trade_penalty=trade_penalty,
            drawdown_penalty=drawdown_penalty,
            total_reward=total_reward
        )

    def _compute_sortino_bonus(self) -> float:
        """Compute Sortino-based bonus."""
        if len(self.returns_buffer) < 10:
            return 0.0

        returns = np.array(list(self.returns_buffer))
        downside = returns[returns < self.config.target_return]

        if len(downside) == 0:
            return 0.1  # No downside = bonus

        downside_std = np.std(downside)

        if downside_std < 0.005:
            return 0.05
        elif downside_std < 0.01:
            return 0.0
        elif downside_std < 0.02:
            return -0.05
        else:
            return -0.1

    def _compute_drawdown_penalty(self) -> float:
        """Compute drawdown penalty."""
        if self.peak_equity <= 0:
            return 0.0

        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity

        if drawdown <= 0.01:
            return 0.0
        elif drawdown < 0.05:
            return drawdown * self.config.drawdown_weight * 0.5
        elif drawdown < 0.10:
            return drawdown * self.config.drawdown_weight
        else:
            return drawdown * self.config.drawdown_weight * 2.0

    def _get_regime_scale(self, regime_id: int, confidence: float) -> float:
        """Scale reward based on regime."""
        regime_scales = {
            0: 1.0,    # LOW_VOL
            1: 1.1,    # TRENDING
            2: 0.9,    # HIGH_VOL
            3: 0.7     # CRISIS
        }
        base_scale = regime_scales.get(regime_id, 1.0)
        return base_scale * confidence + 1.0 * (1 - confidence)

    def compute_sortino_ratio(self) -> float:
        """Compute Sortino ratio for episode."""
        if len(self.returns_buffer) < 20:
            return 0.0

        returns = np.array(list(self.returns_buffer))
        excess = returns - self.config.target_return
        mean_return = np.mean(excess)

        downside = excess[excess < 0]
        if len(downside) == 0:
            return 10.0

        downside_std = np.std(downside)
        if downside_std < 1e-8:
            return 10.0

        # Annualize for hourly data
        ann = np.sqrt(365 * 24)
        sortino = (mean_return * ann) / (downside_std * ann)

        return float(np.clip(sortino, -10, 10))

    def get_statistics(self) -> dict:
        """Get shaping statistics."""
        returns = np.array(list(self.returns_buffer)) if self.returns_buffer else np.array([0])
        min_eq = min(self.equity_buffer) if self.equity_buffer else 1.0

        return {
            'n_updates': self.n_updates,
            'total_trades': self.total_trades,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.n_updates, 1),
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'max_drawdown': (self.peak_equity - min_eq) / self.peak_equity,
            'sortino_ratio': self.compute_sortino_ratio(),
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns))
        }

# =============================================================================
# Evaluation
# =============================================================================

def simulate_trading(prices: np.ndarray, returns: np.ndarray,
                     config: SortinoConfig) -> Dict:
    """
    Simulate trading with momentum strategy and evaluate reward shaping.
    """
    n = len(returns)

    # Simple momentum strategy (sign of 5-bar MA of returns)
    ma_returns = pd.Series(returns).rolling(5).mean().values
    positions = np.sign(ma_returns)
    positions = np.nan_to_num(positions, nan=0).astype(int)

    # Create regime labels (based on volatility percentiles)
    vol_20 = pd.Series(returns).rolling(20).std().values
    vol_percentiles = pd.Series(vol_20).rank(pct=True).values

    regimes = np.zeros(n, dtype=int)
    regimes[vol_percentiles > 0.90] = 3  # CRISIS
    regimes[(vol_percentiles > 0.67) & (vol_percentiles <= 0.90)] = 2  # HIGH_VOL
    regimes[vol_percentiles <= 0.33] = 0  # LOW_VOL
    # Rest is TRENDING (1) - detected by trend strength
    trend_20 = pd.Series(returns).rolling(20).sum().abs().values
    trend_pct = pd.Series(trend_20).rank(pct=True).values
    regimes[(trend_pct > 0.60) & (vol_percentiles <= 0.67)] = 1

    # Run reward shaper
    shaper = SortinoRewardShaper(config)
    rewards = []

    for i in range(n):
        result = shaper.shape(
            price_return=returns[i],
            position=positions[i],
            regime_id=regimes[i]
        )
        rewards.append(result.total_reward)

    rewards = np.array(rewards)
    stats = shaper.get_statistics()

    # Compute additional metrics
    # Raw strategy performance (no shaping)
    raw_pnl = positions * returns
    raw_equity = np.cumprod(1 + raw_pnl)

    # Shaped vs raw correlation
    reward_return_corr = np.corrcoef(rewards[20:], raw_pnl[20:])[0, 1]

    return {
        'sortino_ratio': stats['sortino_ratio'],
        'total_reward': stats['total_reward'],
        'avg_reward': stats['avg_reward'],
        'max_drawdown': stats['max_drawdown'],
        'total_trades': stats['total_trades'],
        'final_equity': stats['current_equity'],
        'reward_return_corr': reward_return_corr if not np.isnan(reward_return_corr) else 0.0,
        'reward_std': np.std(rewards),
        'reward_mean': np.mean(rewards)
    }

def score_config(metrics: Dict) -> float:
    """Score configuration quality."""
    # Higher Sortino is better
    sortino_score = min(metrics['sortino_ratio'] / 2.0, 1.0)  # Cap at 2.0

    # Lower drawdown is better
    dd_score = max(0, 1 - metrics['max_drawdown'] * 5)  # 20% DD = 0 score

    # Higher correlation between reward and actual return is better
    corr_score = (metrics['reward_return_corr'] + 1) / 2  # Map [-1,1] to [0,1]

    # Reasonable trade frequency
    trade_ratio = metrics['total_trades'] / metrics.get('n_bars', 1000)
    trade_score = 1.0 if 0.1 < trade_ratio < 0.5 else 0.5

    score = (0.4 * sortino_score +
             0.3 * dd_score +
             0.2 * corr_score +
             0.1 * trade_score)

    return score

# =============================================================================
# Grid Search
# =============================================================================

def grid_search_calibration(prices: np.ndarray, returns: np.ndarray,
                            verbose: bool = True) -> Tuple[SortinoConfig, Dict]:
    """Grid search for optimal parameters."""

    trade_cost_values = [0.0005, 0.001, 0.002, 0.005]
    drawdown_weight_values = [0.25, 0.5, 1.0]
    reward_scale_values = [50, 100, 200]

    best_config = None
    best_score = -np.inf
    best_metrics = None
    all_results = []

    total = len(trade_cost_values) * len(drawdown_weight_values) * len(reward_scale_values)
    current = 0

    if verbose:
        print(f"\nGrid search: {total} configurations")
        print("-" * 70)

    for tc in trade_cost_values:
        for dw in drawdown_weight_values:
            for rs in reward_scale_values:
                current += 1

                config = SortinoConfig(
                    trade_cost=tc,
                    drawdown_weight=dw,
                    reward_scale=rs
                )

                metrics = simulate_trading(prices, returns, config)
                metrics['n_bars'] = len(returns)
                score = score_config(metrics)

                all_results.append({
                    'trade_cost': tc,
                    'drawdown_weight': dw,
                    'reward_scale': rs,
                    'score': score,
                    'metrics': metrics
                })

                if verbose:
                    print(f"  [{current}/{total}] tc={tc:.4f}, dw={dw:.2f}, rs={rs} -> "
                          f"Score={score:.4f} (Sortino={metrics['sortino_ratio']:.2f}, "
                          f"DD={metrics['max_drawdown']:.2%})")

                if score > best_score:
                    best_score = score
                    best_config = config
                    best_metrics = metrics

    return best_config, {
        'best_score': best_score,
        'best_metrics': best_metrics,
        'all_results': all_results
    }

# =============================================================================
# Data Loading
# =============================================================================

def load_btc_data(csv_path: str) -> pd.DataFrame:
    """Load BTC data."""
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"  {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df

# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Sortino Reward Shaper Calibration")
    print("=" * 70)
    print("\nKey objectives:")
    print("  - Optimize for Sortino ratio (penalize downside only)")
    print("  - Balance trade costs vs opportunity")
    print("  - Adapt to regimes")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Step 1: Load Training Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading Training Data (2020-2024)")
    print("=" * 70)

    train_df = load_btc_data(TRAIN_DATA_PATH)
    train_prices = train_df['close'].values
    train_returns = np.diff(train_prices) / train_prices[:-1]
    train_returns = np.concatenate([[0], train_returns])

    print(f"  Return std: {np.std(train_returns):.4f}")
    print(f"  Return range: [{train_returns.min():.4f}, {train_returns.max():.4f}]")

    # =========================================================================
    # Step 2: Grid Search
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Grid Search Calibration")
    print("=" * 70)

    best_config, calibration_results = grid_search_calibration(
        train_prices, train_returns, verbose=True
    )

    print(f"\nBest Configuration:")
    print(f"  trade_cost: {best_config.trade_cost}")
    print(f"  drawdown_weight: {best_config.drawdown_weight}")
    print(f"  reward_scale: {best_config.reward_scale}")
    print(f"  Score: {calibration_results['best_score']:.4f}")

    # =========================================================================
    # Step 3: Training Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Training Data Results")
    print("=" * 70)

    train_metrics = simulate_trading(train_prices, train_returns, best_config)

    print(f"\nTraining Performance:")
    print(f"  Sortino Ratio: {train_metrics['sortino_ratio']:.4f}")
    print(f"  Max Drawdown: {train_metrics['max_drawdown']:.2%}")
    print(f"  Total Trades: {train_metrics['total_trades']}")
    print(f"  Final Equity: {train_metrics['final_equity']:.4f}")
    print(f"  Reward-Return Corr: {train_metrics['reward_return_corr']:.4f}")

    # =========================================================================
    # Step 4: Test on Unseen Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Testing on Unseen Data (2025-2026)")
    print("=" * 70)

    test_df = load_btc_data(TEST_DATA_PATH)
    test_prices = test_df['close'].values
    test_returns = np.diff(test_prices) / test_prices[:-1]
    test_returns = np.concatenate([[0], test_returns])

    test_metrics = simulate_trading(test_prices, test_returns, best_config)

    print(f"\nTest Performance:")
    print(f"  Sortino Ratio: {test_metrics['sortino_ratio']:.4f}")
    print(f"  Max Drawdown: {test_metrics['max_drawdown']:.2%}")
    print(f"  Total Trades: {test_metrics['total_trades']}")
    print(f"  Final Equity: {test_metrics['final_equity']:.4f}")
    print(f"  Reward-Return Corr: {test_metrics['reward_return_corr']:.4f}")

    # =========================================================================
    # Step 5: Generalization Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Generalization Comparison")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Training':>12} {'Test':>12} {'Diff':>10}")
    print("-" * 59)

    for metric in ['sortino_ratio', 'max_drawdown', 'reward_return_corr', 'final_equity']:
        train_val = train_metrics[metric]
        test_val = test_metrics[metric]
        diff = test_val - train_val

        if metric == 'max_drawdown':
            print(f"{metric:<25} {train_val:>11.2%} {test_val:>11.2%} {diff:>+9.2%}")
        else:
            print(f"{metric:<25} {train_val:>12.4f} {test_val:>12.4f} {diff:>+10.4f}")

    # =========================================================================
    # Step 6: Save Config
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 6: Saving Calibrated Config")
    print("=" * 70)

    save_data = {
        'config': {
            'target_return': best_config.target_return,
            'lookback_window': best_config.lookback_window,
            'trade_cost': best_config.trade_cost,
            'drawdown_weight': best_config.drawdown_weight,
            'reward_scale': best_config.reward_scale,
            'regime_scaling': best_config.regime_scaling
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'calibration_results': {
            'best_score': calibration_results['best_score'],
            'grid_search_results': calibration_results['all_results']
        },
        'created': datetime.now().isoformat()
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Config saved to: {MODEL_PATH}")

    # Save log
    log_path = os.path.join(OUTPUT_DIR, "calibration_log.txt")
    with open(log_path, 'w') as f:
        f.write("Sortino Reward Shaper Calibration Log\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        f.write("Best Configuration:\n")
        f.write(f"  trade_cost: {best_config.trade_cost}\n")
        f.write(f"  drawdown_weight: {best_config.drawdown_weight}\n")
        f.write(f"  reward_scale: {best_config.reward_scale}\n\n")
        f.write("Training Metrics:\n")
        for k, v in train_metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTest Metrics:\n")
        for k, v in test_metrics.items():
            f.write(f"  {k}: {v}\n")

    print(f"Log saved to: {log_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Calibration Complete!")
    print("=" * 70)

    train_score = score_config({**train_metrics, 'n_bars': len(train_returns)})
    test_score = score_config({**test_metrics, 'n_bars': len(test_returns)})
    score_diff = test_score - train_score

    print(f"\nOverall Score: Train={train_score:.4f}, Test={test_score:.4f}, Diff={score_diff:+.4f}")

    if abs(score_diff) < 0.1:
        print("\n[OK] Sortino Reward Shaper generalizes well")
    else:
        print(f"\n[WARN] Generalization concern: score drift = {score_diff:.4f}")

    print(f"\nOptimal Config:")
    print(f"  trade_cost: {best_config.trade_cost}")
    print(f"  drawdown_weight: {best_config.drawdown_weight}")
    print(f"  reward_scale: {best_config.reward_scale}")

if __name__ == "__main__":
    main()
