#!/usr/bin/env python3
"""
Train/Calibrate EKF Denoiser for HIMARI Layer 2.

EKF (Extended Kalman Filter) is not trained in the ML sense - it's calibrated.
We find optimal process_noise (Q) and measurement_noise (R) parameters.

Key Lessons from PPO/AHHMM Training Applied:
1. Use percentile/relative metrics (not absolute values)
2. Test on unseen 2025-2026 data to verify generalization
3. Adaptive parameters that scale with data characteristics

What EKF Does:
- Denoises price series (removes market microstructure noise)
- Extracts momentum (velocity) and acceleration
- Estimates volatility as a state variable
- Outputs cleaner features for PPO policy

Calibration Approach:
- Grid search over Q and R parameters
- Evaluate using: smoothness, lag, signal preservation
- Pick parameters that balance noise reduction vs signal lag
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = r"C:\Users\chari\OneDrive\Documents\BTC DATA SETS"
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "btc_1h_2020_2024.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "btc_1h_2025_2026.csv")

OUTPUT_DIR = r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1\L2V1 EKF FINAL"
MODEL_PATH = os.path.join(OUTPUT_DIR, "ekf_config_calibrated.pkl")

# =============================================================================
# EKF Implementation (Simplified, no filterpy dependency)
# =============================================================================

@dataclass
class EKFConfig:
    """EKF configuration parameters."""
    process_noise: float = 0.0001      # Q scaling
    measurement_noise: float = 0.001    # R scaling
    vol_mean_reversion: float = 0.05    # Volatility mean reversion speed
    vol_long_run: float = 0.02          # Long-run volatility
    use_faux_riccati: bool = True       # Stability enhancement
    dt: float = 1.0                     # Time step


class EKFDenoiser:
    """
    Extended Kalman Filter for price denoising.

    State: [price, velocity, acceleration, volatility]
    Observation: [price, volume_normalized]
    """

    def __init__(self, config: Optional[EKFConfig] = None):
        self.config = config or EKFConfig()
        self._initialize()

    def _initialize(self):
        """Initialize filter state and covariance."""
        # State: [price, velocity, acceleration, volatility]
        self.x = np.array([0.0, 0.0, 0.0, self.config.vol_long_run])

        # Covariance
        self.P = np.diag([1.0, 0.1, 0.01, 0.001])

        # Process noise Q
        q = self.config.process_noise
        dt = self.config.dt
        self.Q = np.diag([q * dt**2, q * dt, q, q * 0.1])

        # Measurement noise R
        r = self.config.measurement_noise
        self.R = np.diag([r, r * 10])

        # Observation matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        self._initialized = False

    def _state_transition(self, x: np.ndarray) -> np.ndarray:
        """Non-linear state transition."""
        dt = self.config.dt
        kappa = self.config.vol_mean_reversion
        theta = self.config.vol_long_run

        price, vel, accel, vol = x

        new_price = price + vel * dt + 0.5 * accel * dt**2
        new_vel = vel + accel * dt
        new_accel = accel * 0.9  # Mean reversion
        new_vol = vol + kappa * (theta - vol) * dt
        new_vol = max(new_vol, 1e-6)

        return np.array([new_price, new_vel, new_accel, new_vol])

    def _jacobian_F(self) -> np.ndarray:
        """State transition Jacobian."""
        dt = self.config.dt
        kappa = self.config.vol_mean_reversion

        return np.array([
            [1, dt, 0.5*dt**2, 0],
            [0, 1, dt, 0],
            [0, 0, 0.9, 0],
            [0, 0, 0, 1 - kappa*dt]
        ])

    def _apply_faux_riccati(self):
        """Stability enhancement."""
        if not self.config.use_faux_riccati:
            return

        P_steady = np.diag([0.01, 0.001, 0.0001, 0.001])
        alpha = 0.05
        self.P = (1 - alpha) * self.P + alpha * P_steady

        # Clip covariance
        np.fill_diagonal(self.P, np.clip(np.diag(self.P), 1e-8, 1e4))

    def update(self, price: float, volume_norm: float) -> Tuple[float, float]:
        """
        Process observation and return filtered price.

        Returns:
            filtered_price: Denoised price
            uncertainty: State uncertainty
        """
        if not self._initialized:
            self.x[0] = price
            self._initialized = True
            return price, np.trace(self.P)

        # Observation
        z = np.array([price, volume_norm * self.config.vol_long_run])

        # Predict
        x_pred = self._state_transition(self.x)
        F = self._jacobian_F()
        P_pred = F @ self.P @ F.T + self.Q

        # Update
        z_pred = self.H @ x_pred
        y = z - z_pred  # Innovation
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        self._apply_faux_riccati()

        return float(self.x[0]), float(np.trace(self.P))

    def get_momentum(self) -> float:
        return float(self.x[1])

    def get_acceleration(self) -> float:
        return float(self.x[2])

    def get_volatility(self) -> float:
        return float(self.x[3])

    def reset(self):
        self._initialize()

# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_denoising(raw_prices: np.ndarray, filtered_prices: np.ndarray,
                       returns: np.ndarray) -> Dict[str, float]:
    """
    Evaluate denoising quality.

    Metrics:
    1. Smoothness: reduction in high-frequency noise
    2. Lag: delay between raw and filtered (lower is better)
    3. Signal Preservation: correlation with trend
    4. Noise Reduction Ratio: variance reduction
    """
    n = len(raw_prices)

    # 1. Smoothness (2nd derivative magnitude)
    raw_d2 = np.diff(raw_prices, 2)
    filtered_d2 = np.diff(filtered_prices, 2)
    smoothness_improvement = np.std(raw_d2) / (np.std(filtered_d2) + 1e-10)

    # 2. Lag (cross-correlation peak offset)
    raw_norm = (raw_prices - np.mean(raw_prices)) / (np.std(raw_prices) + 1e-10)
    filt_norm = (filtered_prices - np.mean(filtered_prices)) / (np.std(filtered_prices) + 1e-10)

    # Find lag via cross-correlation
    max_lag = 10
    best_lag = 0
    best_corr = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = np.corrcoef(raw_norm[lag:], filt_norm[:n-lag])[0, 1]
        else:
            corr = np.corrcoef(raw_norm[:n+lag], filt_norm[-lag:])[0, 1]
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag

    # 3. Signal Preservation (correlation with raw returns direction)
    filtered_returns = np.diff(filtered_prices) / (filtered_prices[:-1] + 1e-10)
    raw_returns = returns[1:len(filtered_returns)+1]

    # Same direction ratio
    same_direction = (np.sign(filtered_returns) == np.sign(raw_returns)).mean()

    # 4. Noise Reduction Ratio
    raw_noise = np.std(np.diff(raw_prices))
    filtered_noise = np.std(np.diff(filtered_prices))
    noise_reduction = raw_noise / (filtered_noise + 1e-10)

    # 5. Trend following accuracy (does filtered price follow trend?)
    trend_20 = pd.Series(raw_prices).rolling(20).mean().values
    trend_corr = np.corrcoef(filtered_prices[20:], trend_20[20:])[0, 1]

    return {
        'smoothness_improvement': smoothness_improvement,
        'lag_bars': abs(best_lag),
        'signal_correlation': best_corr,
        'same_direction_ratio': same_direction,
        'noise_reduction': noise_reduction,
        'trend_correlation': trend_corr if not np.isnan(trend_corr) else 0.0
    }

def score_config(metrics: Dict[str, float]) -> float:
    """
    Compute overall score for EKF config.

    Higher is better.
    """
    # Weights
    w_smooth = 0.2
    w_lag = 0.3        # Penalize lag heavily
    w_signal = 0.2
    w_direction = 0.2
    w_trend = 0.1

    # Normalize metrics to [0, 1] range
    smooth_score = min(metrics['smoothness_improvement'] / 3.0, 1.0)  # Cap at 3x improvement
    lag_score = max(0, 1 - metrics['lag_bars'] / 5.0)  # 0 lag = 1.0, 5+ lag = 0.0
    signal_score = metrics['signal_correlation']
    direction_score = metrics['same_direction_ratio']
    trend_score = metrics['trend_correlation']

    score = (w_smooth * smooth_score +
             w_lag * lag_score +
             w_signal * signal_score +
             w_direction * direction_score +
             w_trend * trend_score)

    return score

# =============================================================================
# Grid Search Calibration
# =============================================================================

def run_ekf_on_data(prices: np.ndarray, volumes: np.ndarray,
                    config: EKFConfig) -> np.ndarray:
    """Run EKF on price/volume data."""
    n = len(prices)

    # Normalize volume
    vol_mean = np.mean(volumes)
    volumes_norm = volumes / (vol_mean + 1e-10)

    filtered_prices = np.zeros(n)
    ekf = EKFDenoiser(config)

    for i in range(n):
        filtered_prices[i], _ = ekf.update(prices[i], volumes_norm[i])

    return filtered_prices

def grid_search_calibration(prices: np.ndarray, volumes: np.ndarray,
                            returns: np.ndarray, verbose: bool = True) -> Tuple[EKFConfig, Dict]:
    """
    Grid search to find optimal Q and R parameters.
    """
    # Parameter grid
    process_noise_values = [0.00001, 0.0001, 0.001, 0.01]
    measurement_noise_values = [0.0001, 0.001, 0.01, 0.1]

    best_config = None
    best_score = -np.inf
    best_metrics = None
    all_results = []

    total = len(process_noise_values) * len(measurement_noise_values)
    current = 0

    if verbose:
        print(f"\nGrid search: {total} configurations")
        print("-" * 60)

    for q in process_noise_values:
        for r in measurement_noise_values:
            current += 1

            config = EKFConfig(
                process_noise=q,
                measurement_noise=r,
                use_faux_riccati=True
            )

            # Run EKF
            filtered = run_ekf_on_data(prices, volumes, config)

            # Evaluate
            metrics = evaluate_denoising(prices, filtered, returns)
            score = score_config(metrics)

            all_results.append({
                'process_noise': q,
                'measurement_noise': r,
                'score': score,
                'metrics': metrics
            })

            if verbose:
                print(f"  [{current}/{total}] Q={q:.5f}, R={r:.4f} -> "
                      f"Score={score:.4f} (smooth={metrics['smoothness_improvement']:.2f}x, "
                      f"lag={metrics['lag_bars']}, dir={metrics['same_direction_ratio']:.2%})")

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
    print("EKF Denoiser Calibration")
    print("=" * 70)
    print("\nApplying lessons from PPO/AHHMM training:")
    print("  - Test on unseen data (2025-2026)")
    print("  - Balance smoothness vs lag")
    print("  - Preserve trading signals")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Step 1: Load Training Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading Training Data (2020-2024)")
    print("=" * 70)

    train_df = load_btc_data(TRAIN_DATA_PATH)
    train_prices = train_df['close'].values
    train_volumes = train_df['volume'].values
    train_returns = np.diff(train_prices) / train_prices[:-1]
    train_returns = np.concatenate([[0], train_returns])

    print(f"  Price range: [{train_prices.min():.2f}, {train_prices.max():.2f}]")
    print(f"  Return std: {np.std(train_returns):.4f}")

    # =========================================================================
    # Step 2: Grid Search Calibration
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Grid Search Calibration")
    print("=" * 70)

    best_config, calibration_results = grid_search_calibration(
        train_prices, train_volumes, train_returns, verbose=True
    )

    print(f"\nBest Configuration Found:")
    print(f"  process_noise (Q): {best_config.process_noise}")
    print(f"  measurement_noise (R): {best_config.measurement_noise}")
    print(f"  Score: {calibration_results['best_score']:.4f}")

    print(f"\nBest Metrics:")
    for k, v in calibration_results['best_metrics'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # =========================================================================
    # Step 3: Evaluate on Training Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Training Data Results")
    print("=" * 70)

    train_filtered = run_ekf_on_data(train_prices, train_volumes, best_config)
    train_metrics = evaluate_denoising(train_prices, train_filtered, train_returns)

    print(f"\nTraining Set Performance:")
    print(f"  Smoothness improvement: {train_metrics['smoothness_improvement']:.2f}x")
    print(f"  Lag (bars): {train_metrics['lag_bars']}")
    print(f"  Signal correlation: {train_metrics['signal_correlation']:.4f}")
    print(f"  Same direction ratio: {train_metrics['same_direction_ratio']:.2%}")
    print(f"  Noise reduction: {train_metrics['noise_reduction']:.2f}x")
    print(f"  Trend correlation: {train_metrics['trend_correlation']:.4f}")

    # =========================================================================
    # Step 4: Test on Unseen Data (2025-2026)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Testing on Unseen Data (2025-2026)")
    print("=" * 70)

    test_df = load_btc_data(TEST_DATA_PATH)
    test_prices = test_df['close'].values
    test_volumes = test_df['volume'].values
    test_returns = np.diff(test_prices) / test_prices[:-1]
    test_returns = np.concatenate([[0], test_returns])

    test_filtered = run_ekf_on_data(test_prices, test_volumes, best_config)
    test_metrics = evaluate_denoising(test_prices, test_filtered, test_returns)

    print(f"\nTest Set Performance:")
    print(f"  Smoothness improvement: {test_metrics['smoothness_improvement']:.2f}x")
    print(f"  Lag (bars): {test_metrics['lag_bars']}")
    print(f"  Signal correlation: {test_metrics['signal_correlation']:.4f}")
    print(f"  Same direction ratio: {test_metrics['same_direction_ratio']:.2%}")
    print(f"  Noise reduction: {test_metrics['noise_reduction']:.2f}x")
    print(f"  Trend correlation: {test_metrics['trend_correlation']:.4f}")

    # =========================================================================
    # Step 5: Generalization Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Generalization Comparison")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Training':>12} {'Test':>12} {'Diff':>10}")
    print("-" * 59)

    for metric in ['smoothness_improvement', 'lag_bars', 'signal_correlation',
                   'same_direction_ratio', 'noise_reduction', 'trend_correlation']:
        train_val = train_metrics[metric]
        test_val = test_metrics[metric]
        diff = test_val - train_val

        if metric == 'same_direction_ratio':
            print(f"{metric:<25} {train_val:>11.1%} {test_val:>11.1%} {diff:>+9.1%}")
        elif metric == 'lag_bars':
            print(f"{metric:<25} {train_val:>12.0f} {test_val:>12.0f} {diff:>+10.0f}")
        else:
            print(f"{metric:<25} {train_val:>12.4f} {test_val:>12.4f} {diff:>+10.4f}")

    # =========================================================================
    # Step 6: Save Calibrated Config
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 6: Saving Calibrated Config")
    print("=" * 70)

    save_data = {
        'config': {
            'process_noise': best_config.process_noise,
            'measurement_noise': best_config.measurement_noise,
            'vol_mean_reversion': best_config.vol_mean_reversion,
            'vol_long_run': best_config.vol_long_run,
            'use_faux_riccati': best_config.use_faux_riccati,
            'dt': best_config.dt
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

    # Save training log
    log_path = os.path.join(OUTPUT_DIR, "calibration_log.txt")
    with open(log_path, 'w') as f:
        f.write("EKF Denoiser Calibration Log\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        f.write("Best Configuration:\n")
        f.write(f"  process_noise (Q): {best_config.process_noise}\n")
        f.write(f"  measurement_noise (R): {best_config.measurement_noise}\n\n")
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

    # Check generalization
    train_score = score_config(train_metrics)
    test_score = score_config(test_metrics)
    score_diff = test_score - train_score

    print(f"\nOverall Score: Train={train_score:.4f}, Test={test_score:.4f}, Diff={score_diff:+.4f}")

    if abs(score_diff) < 0.1:
        print("\n[OK] EKF generalizes well (score drift < 0.1)")
    else:
        print(f"\n[WARN] Generalization concern: score drift = {score_diff:.4f}")

    print(f"\nOptimal EKF Config:")
    print(f"  Q (process_noise): {best_config.process_noise}")
    print(f"  R (measurement_noise): {best_config.measurement_noise}")

if __name__ == "__main__":
    main()
