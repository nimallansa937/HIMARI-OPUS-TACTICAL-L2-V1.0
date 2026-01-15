"""
Layer 2 Risk Manager Calibration

Calibrates the 8-method RSS (Responsibility-Sensitive Safety) Risk Manager:
  H1: EVT + GPD Tail Risk - Fit shape/scale parameters
  H2: Kelly Criterion - Calibrate win rate and payoff ratio
  H3: Volatility Targeting - Calibrate target volatility
  H4: Drawdown Brake - Optimize threshold levels
  H5-H8: Use sensible defaults (rule-based)

Key L2 Lessons Applied:
  - Test on unseen 2025-2026 data
  - Use percentile-based thresholds for generalization
  - Grid search for optimal parameters
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import pickle
import os
from datetime import datetime

print("=" * 70)
print("Layer 2 Risk Manager Calibration")
print("=" * 70)
print()
print("Components to calibrate:")
print("  H1: EVT + GPD Tail Risk (fit xi, sigma)")
print("  H2: Kelly Criterion (win_rate, payoff_ratio)")
print("  H3: Volatility Targeting (target_vol)")
print("  H4: Drawdown Brake (thresholds)")
print()

# =============================================================================
# Configuration
# =============================================================================

TRAIN_DATA_PATH = r"C:\Users\chari\OneDrive\Documents\BTC DATA SETS\btc_1h_2020_2024.csv"
TEST_DATA_PATH = r"C:\Users\chari\OneDrive\Documents\BTC DATA SETS\btc_1h_2025_2026.csv"
OUTPUT_DIR = r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1\L2V1 RISK MANAGER FINAL"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Data Loading
# =============================================================================

def load_data(filepath):
    """Load BTC data and compute returns"""
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna().reset_index(drop=True)

    print(f"  {len(df)} samples")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df

# =============================================================================
# H1: EVT + GPD Tail Risk Calibration
# =============================================================================

def fit_gpd(losses, threshold_percentile=95):
    """
    Fit Generalized Pareto Distribution to tail losses
    using Peaks Over Threshold (POT) method
    """
    # Get threshold
    threshold = np.percentile(losses, threshold_percentile)

    # Get exceedances (losses above threshold)
    exceedances = losses[losses > threshold] - threshold

    if len(exceedances) < 30:
        print(f"  [WARN] Only {len(exceedances)} exceedances, using defaults")
        return {'xi': 0.2, 'sigma': 0.02, 'threshold': threshold, 'n_exceedances': len(exceedances)}

    # MLE fit using scipy
    # GPD parameterization: shape (xi), scale (sigma)
    try:
        # Fit GPD
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

        return {
            'xi': shape,           # Shape parameter (tail heaviness)
            'sigma': scale,        # Scale parameter
            'threshold': threshold,
            'n_exceedances': len(exceedances),
            'mean_exceedance': np.mean(exceedances),
        }
    except Exception as e:
        print(f"  [WARN] GPD fit failed: {e}, using defaults")
        return {'xi': 0.2, 'sigma': 0.02, 'threshold': threshold, 'n_exceedances': len(exceedances)}

def compute_var_es(gpd_params, confidence=0.99):
    """Compute VaR and Expected Shortfall from GPD parameters"""
    xi = gpd_params['xi']
    sigma = gpd_params['sigma']
    threshold = gpd_params['threshold']

    # VaR at confidence level
    p = 1 - confidence
    if xi != 0:
        var = threshold + (sigma / xi) * ((p ** (-xi)) - 1)
    else:
        var = threshold - sigma * np.log(p)

    # Expected Shortfall (CVaR)
    if xi < 1:
        es = var / (1 - xi) + (sigma - xi * threshold) / (1 - xi)
    else:
        es = var * 1.5  # Approximation for heavy tails

    return {'VaR': var, 'ES': es}

def calibrate_evt(returns):
    """Calibrate EVT parameters on loss distribution"""
    print("\n--- H1: EVT + GPD Tail Risk Calibration ---")

    # Convert to losses (positive values)
    losses = -returns[returns < 0]

    print(f"  Total samples: {len(returns)}")
    print(f"  Negative returns: {len(losses)} ({len(losses)/len(returns)*100:.1f}%)")

    # Fit GPD at different thresholds
    results = {}
    for pct in [90, 95, 99]:
        gpd_params = fit_gpd(losses, threshold_percentile=pct)
        var_es = compute_var_es(gpd_params)

        results[pct] = {
            'gpd_params': gpd_params,
            'VaR_99': var_es['VaR'],
            'ES_99': var_es['ES'],
        }

        print(f"  Threshold {pct}%: xi={gpd_params['xi']:.4f}, sigma={gpd_params['sigma']:.4f}")
        print(f"    VaR_99={var_es['VaR']*100:.2f}%, ES_99={var_es['ES']*100:.2f}%")

    # Use 95th percentile as default
    return results[95]

# =============================================================================
# H2: Kelly Criterion Calibration
# =============================================================================

def simulate_trades(returns, lookback=24):
    """Simulate momentum trades to get win rate and payoff ratio"""
    trades = []

    # Simple momentum strategy: go long if last N hours positive
    for i in range(lookback, len(returns)):
        momentum = returns[i-lookback:i].sum()

        if momentum > 0:
            # Long trade
            pnl = returns[i]
            trades.append(pnl)
        elif momentum < 0:
            # Short trade
            pnl = -returns[i]
            trades.append(pnl)

    return np.array(trades)

def calibrate_kelly(returns):
    """Calibrate Kelly criterion parameters"""
    print("\n--- H2: Kelly Criterion Calibration ---")

    # Simulate trades
    trades = simulate_trades(returns)

    wins = trades[trades > 0]
    losses = trades[trades < 0]

    win_rate = len(wins) / len(trades) if len(trades) > 0 else 0.5
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.abs(np.mean(losses)) if len(losses) > 0 else 0.01

    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

    # Kelly fraction: f* = p - (1-p)/b
    kelly_full = win_rate - (1 - win_rate) / payoff_ratio if payoff_ratio > 0 else 0
    kelly_half = kelly_full / 2  # Half-Kelly for safety
    kelly_quarter = kelly_full / 4  # Quarter-Kelly (more conservative)

    print(f"  Total trades: {len(trades)}")
    print(f"  Win rate: {win_rate*100:.1f}%")
    print(f"  Avg win: {avg_win*100:.3f}%")
    print(f"  Avg loss: {avg_loss*100:.3f}%")
    print(f"  Payoff ratio: {payoff_ratio:.2f}")
    print(f"  Full Kelly: {kelly_full*100:.1f}%")
    print(f"  Half Kelly: {kelly_half*100:.1f}%")
    print(f"  Quarter Kelly: {kelly_quarter*100:.1f}%")

    return {
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'payoff_ratio': payoff_ratio,
        'kelly_full': kelly_full,
        'kelly_half': kelly_half,
        'kelly_quarter': kelly_quarter,
        'n_trades': len(trades),
    }

# =============================================================================
# H3: Volatility Targeting Calibration
# =============================================================================

def calibrate_volatility(returns):
    """Calibrate volatility targeting parameters"""
    print("\n--- H3: Volatility Targeting Calibration ---")

    # Hourly returns
    hourly_vol = np.std(returns)
    daily_vol = hourly_vol * np.sqrt(24)
    annual_vol = hourly_vol * np.sqrt(24 * 365)

    # Rolling volatility
    rolling_vol = pd.Series(returns).rolling(24).std()
    vol_of_vol = rolling_vol.std()

    # Percentiles
    vol_25 = np.nanpercentile(rolling_vol, 25) * np.sqrt(24 * 365)
    vol_50 = np.nanpercentile(rolling_vol, 50) * np.sqrt(24 * 365)
    vol_75 = np.nanpercentile(rolling_vol, 75) * np.sqrt(24 * 365)
    vol_95 = np.nanpercentile(rolling_vol, 95) * np.sqrt(24 * 365)

    # Target vol recommendation: 60% of median
    target_vol = vol_50 * 0.6

    print(f"  Annualized volatility: {annual_vol*100:.1f}%")
    print(f"  Vol percentiles: 25th={vol_25*100:.1f}%, 50th={vol_50*100:.1f}%, 75th={vol_75*100:.1f}%, 95th={vol_95*100:.1f}%")
    print(f"  Recommended target: {target_vol*100:.1f}% (60% of median)")

    return {
        'hourly_vol': hourly_vol,
        'daily_vol': daily_vol,
        'annual_vol': annual_vol,
        'vol_of_vol': vol_of_vol,
        'vol_25': vol_25,
        'vol_50': vol_50,
        'vol_75': vol_75,
        'vol_95': vol_95,
        'target_vol': target_vol,
    }

# =============================================================================
# H4: Drawdown Brake Calibration
# =============================================================================

def simulate_drawdown_brake(returns, thresholds, reductions):
    """Simulate trading with drawdown brake and return metrics"""
    equity = 1.0
    peak = 1.0
    positions = []
    pnls = []

    for ret in returns:
        # Calculate current drawdown
        drawdown = (peak - equity) / peak if peak > 0 else 0

        # Determine position based on drawdown
        position = 1.0
        for thresh, reduction in zip(thresholds, reductions):
            if drawdown >= thresh:
                position = 1.0 - reduction

        # Apply position
        pnl = ret * position
        equity = equity * (1 + pnl)
        peak = max(peak, equity)

        positions.append(position)
        pnls.append(pnl)

    pnls = np.array(pnls)
    positions = np.array(positions)

    # Calculate metrics
    total_return = (equity - 1)
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(24 * 365)

    # Max drawdown
    cumulative = np.cumprod(1 + pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    max_dd = np.max(drawdowns)

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'avg_position': np.mean(positions),
        'time_reduced': np.mean(positions < 1.0),
    }

def calibrate_drawdown_brake(returns):
    """Grid search for optimal drawdown brake thresholds"""
    print("\n--- H4: Drawdown Brake Calibration ---")

    # Grid search over threshold configurations
    # Focus on DRAWDOWN PROTECTION (minimize max DD)
    configs = [
        # (thresholds, reductions)
        ([0.05, 0.10, 0.15], [0.25, 0.50, 0.90]),  # Default
        ([0.03, 0.06, 0.10], [0.30, 0.60, 0.95]),  # Aggressive protection
        ([0.05, 0.08, 0.12], [0.25, 0.50, 0.90]),  # Moderate
        ([0.04, 0.08, 0.12], [0.30, 0.60, 1.00]),  # Full halt at 12%
        ([0.05, 0.10, 0.15], [0.30, 0.60, 1.00]),  # Full halt at 15%
    ]

    best_config = None
    best_score = -float('inf')

    print("  Testing configurations...")
    for i, (thresholds, reductions) in enumerate(configs):
        result = simulate_drawdown_brake(returns, thresholds, reductions)

        # Score: Prioritize drawdown protection over returns
        # Lower max DD is better, accept some Sharpe reduction
        score = -result['max_drawdown'] + 0.5 * max(result['sharpe'], 0)

        print(f"  Config {i+1}: thresh={thresholds}, Sharpe={result['sharpe']:.3f}, MaxDD={result['max_drawdown']*100:.1f}%, Score={score:.3f}")

        if score > best_score:
            best_score = score
            best_config = {
                'thresholds': thresholds,
                'reductions': reductions,
                'result': result,
                'score': score,
            }

    print(f"\n  Best config: thresholds={best_config['thresholds']}")
    print(f"    Sharpe: {best_config['result']['sharpe']:.3f}")
    print(f"    Max Drawdown: {best_config['result']['max_drawdown']*100:.1f}%")
    print(f"    Avg Position: {best_config['result']['avg_position']:.2f}")

    return best_config

# =============================================================================
# Main Calibration
# =============================================================================

def main():
    print("=" * 70)
    print("Step 1: Load Training Data (2020-2024)")
    print("=" * 70)

    train_df = load_data(TRAIN_DATA_PATH)
    train_returns = train_df['returns'].values

    print()
    print("=" * 70)
    print("Step 2: Calibrate Risk Manager Components")
    print("=" * 70)

    # H1: EVT Tail Risk
    evt_params = calibrate_evt(train_returns)

    # H2: Kelly Criterion
    kelly_params = calibrate_kelly(train_returns)

    # H3: Volatility Targeting
    vol_params = calibrate_volatility(train_returns)

    # H4: Drawdown Brake
    dd_params = calibrate_drawdown_brake(train_returns)

    # Combine all parameters
    risk_config = {
        'H1_EVT': {
            'xi': evt_params['gpd_params']['xi'],
            'sigma': evt_params['gpd_params']['sigma'],
            'threshold': evt_params['gpd_params']['threshold'],
            'VaR_99': evt_params['VaR_99'],
            'ES_99': evt_params['ES_99'],
        },
        'H2_Kelly': {
            'kelly_fraction': kelly_params['kelly_quarter'],  # Use quarter-Kelly
            'win_rate': kelly_params['win_rate'],
            'payoff_ratio': kelly_params['payoff_ratio'],
        },
        'H3_Volatility': {
            'target_vol': vol_params['target_vol'],
            'annual_vol': vol_params['annual_vol'],
        },
        'H4_DrawdownBrake': {
            'thresholds': dd_params['thresholds'],
            'reductions': dd_params['reductions'],
        },
        'H5_PortfolioVaR': {
            'confidence': 0.99,
            'lookback': 100,
        },
        'H6_SafeMargin': {
            'k_sigma': 2.0,
            'execution_cost': 0.002,
        },
        'H7_LeverageController': {
            'max_leverage': 3.0,
            'decay_start_size': 0.1,
        },
        'H8_RiskBudget': {
            'initial_budget': 1.0,
            'gain_factor': 1.01,
            'loss_factor': 0.99,
            'min_budget': 0.5,
            'max_budget': 1.2,
        },
    }

    print()
    print("=" * 70)
    print("Step 3: Test on Unseen Data (2025-2026)")
    print("=" * 70)

    test_df = load_data(TEST_DATA_PATH)
    test_returns = test_df['returns'].values

    # Test EVT on new data
    print("\n--- Testing H1: EVT on unseen data ---")
    test_evt = calibrate_evt(test_returns)

    # Test drawdown brake on new data
    print("\n--- Testing H4: Drawdown Brake on unseen data ---")
    test_dd = simulate_drawdown_brake(
        test_returns,
        dd_params['thresholds'],
        dd_params['reductions']
    )
    print(f"  Test Sharpe: {test_dd['sharpe']:.3f}")
    print(f"  Test Max DD: {test_dd['max_drawdown']*100:.1f}%")

    print()
    print("=" * 70)
    print("Step 4: Generalization Comparison")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Training':<15} {'Test':<15} {'Diff'}")
    print("-" * 60)

    # EVT comparison
    train_var = evt_params['VaR_99']
    test_var = test_evt['VaR_99']
    print(f"{'VaR_99':<25} {train_var*100:<14.2f}% {test_var*100:<14.2f}% {(test_var-train_var)*100:+.2f}%")

    # Drawdown brake comparison
    train_sharpe = dd_params['result']['sharpe']
    test_sharpe = test_dd['sharpe']
    print(f"{'DD Brake Sharpe':<25} {train_sharpe:<15.3f} {test_sharpe:<15.3f} {test_sharpe-train_sharpe:+.3f}")

    train_maxdd = dd_params['result']['max_drawdown']
    test_maxdd = test_dd['max_drawdown']
    print(f"{'Max Drawdown':<25} {train_maxdd*100:<14.1f}% {test_maxdd*100:<14.1f}% {(test_maxdd-train_maxdd)*100:+.1f}%")

    # Check generalization
    sharpe_diff = test_sharpe - train_sharpe
    if sharpe_diff > -0.2:
        print(f"\n[OK] Risk Manager generalizes well! (Sharpe diff = {sharpe_diff:+.3f})")
    else:
        print(f"\n[WARN] Generalization concern: Sharpe dropped by {-sharpe_diff:.3f}")

    print()
    print("=" * 70)
    print("Step 5: Save Calibrated Config")
    print("=" * 70)

    # Save config
    config_path = os.path.join(OUTPUT_DIR, "risk_manager_config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump(risk_config, f)
    print(f"Config saved to: {config_path}")

    # Save readable version
    log_path = os.path.join(OUTPUT_DIR, "calibration_log.txt")
    with open(log_path, 'w') as f:
        f.write("HIMARI Layer 2 Risk Manager Calibration\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")

        f.write("H1: EVT + GPD Tail Risk\n")
        f.write(f"  xi (shape): {risk_config['H1_EVT']['xi']:.4f}\n")
        f.write(f"  sigma (scale): {risk_config['H1_EVT']['sigma']:.4f}\n")
        f.write(f"  threshold: {risk_config['H1_EVT']['threshold']*100:.2f}%\n")
        f.write(f"  VaR_99: {risk_config['H1_EVT']['VaR_99']*100:.2f}%\n")
        f.write(f"  ES_99: {risk_config['H1_EVT']['ES_99']*100:.2f}%\n\n")

        f.write("H2: Kelly Criterion\n")
        f.write(f"  kelly_fraction: {risk_config['H2_Kelly']['kelly_fraction']*100:.1f}%\n")
        f.write(f"  win_rate: {risk_config['H2_Kelly']['win_rate']*100:.1f}%\n")
        f.write(f"  payoff_ratio: {risk_config['H2_Kelly']['payoff_ratio']:.2f}\n\n")

        f.write("H3: Volatility Targeting\n")
        f.write(f"  target_vol: {risk_config['H3_Volatility']['target_vol']*100:.1f}%\n")
        f.write(f"  annual_vol: {risk_config['H3_Volatility']['annual_vol']*100:.1f}%\n\n")

        f.write("H4: Drawdown Brake\n")
        f.write(f"  thresholds: {risk_config['H4_DrawdownBrake']['thresholds']}\n")
        f.write(f"  reductions: {risk_config['H4_DrawdownBrake']['reductions']}\n\n")

        f.write("H5-H8: Using defaults\n")

    print(f"Log saved to: {log_path}")

    print()
    print("=" * 70)
    print("Calibration Complete!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  H1 EVT: VaR_99 = {risk_config['H1_EVT']['VaR_99']*100:.2f}%")
    print(f"  H2 Kelly: fraction = {risk_config['H2_Kelly']['kelly_fraction']*100:.1f}%")
    print(f"  H3 Vol Target: {risk_config['H3_Volatility']['target_vol']*100:.1f}%")
    print(f"  H4 DD Brake: {risk_config['H4_DrawdownBrake']['thresholds']}")
    print()
    print(f"Test Sharpe: {test_sharpe:.3f}")
    print(f"Test Max DD: {test_maxdd*100:.1f}%")

    return risk_config

if __name__ == "__main__":
    main()
