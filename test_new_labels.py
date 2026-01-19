"""
Test the new Sharpe-based label generation to verify distribution.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))
from train_flagtrader import generate_labels

# Load data
data_path = Path("L2 POSTION FINAL MODELS/orkspace/data/btc_1h_2020_2024.csv")
if not data_path.exists():
    print(f"ERROR: Data file not found at {data_path}")
    print("Please update the path in this script.")
    sys.exit(1)

print("Loading data...")
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} samples")
print()

# Generate labels with new Sharpe-based approach
print("Generating labels with new Sharpe-based approach...")
print("Parameters:")
print("  - Lookahead window: 24 hours")
print("  - Sharpe threshold: 1.0 (stricter)")
print("  - Return threshold: 3% (stricter)")
print("  - Stop-loss: 2.5%")
print()

labels = generate_labels(
    close=df['close'].values,
    high=df['high'].values,
    low=df['low'].values
)

# Show distribution
print("="*80)
print("NEW LABEL DISTRIBUTION")
print("="*80)

unique, counts = np.unique(labels, return_counts=True)
action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

for label, count in zip(unique, counts):
    action = action_map[label]
    pct = count/len(labels)*100
    bar = '#' * int(pct)
    print(f"{action:>4}: {count:>6} ({pct:>5.1f}%) {bar}")

print("="*80)
print()

# Compare with old approach (simple 1% threshold, 6h lookahead)
print("Comparison with old approach (1% threshold, 6h lookahead):")
print()

def old_generate_labels(close, threshold=0.01, lookahead=6):
    """Old simple threshold approach."""
    future_returns = np.zeros_like(close)
    future_returns[:-lookahead] = (close[lookahead:] - close[:-lookahead]) / close[:-lookahead]

    labels = np.ones(len(close), dtype=np.int64)  # Default: HOLD
    labels[future_returns > threshold] = 2  # BUY
    labels[future_returns < -threshold] = 0  # SELL
    return labels

old_labels = old_generate_labels(df['close'].values)
unique_old, counts_old = np.unique(old_labels, return_counts=True)

print("OLD LABEL DISTRIBUTION (for comparison)")
for label, count in zip(unique_old, counts_old):
    action = action_map[label]
    pct = count/len(old_labels)*100
    bar = '#' * int(pct)
    print(f"{action:>4}: {count:>6} ({pct:>5.1f}%) {bar}")

print()
print("="*80)
print("ANALYSIS")
print("="*80)

# Check if distribution is reasonable
sell_pct = counts[0] / len(labels) * 100 if 0 in unique else 0
buy_pct = counts[2] / len(labels) * 100 if 2 in unique else 0
hold_pct = counts[1] / len(labels) * 100 if 1 in unique else 0

print(f"Signal frequency: {100-hold_pct:.1f}% (BUY + SELL)")
print(f"Signal balance: BUY={buy_pct:.1f}% vs SELL={sell_pct:.1f}%")
print()

if sell_pct < 5 or buy_pct < 5:
    print("[WARNING] Less than 5% signals for BUY or SELL")
    print("   Thresholds might be too strict. Consider:")
    print("   - Reduce sharpe_threshold from 0.5 to 0.3")
    print("   - Reduce return_threshold from 0.02 to 0.015")
elif sell_pct > 20 or buy_pct > 20:
    print("[WARNING] More than 20% signals for BUY or SELL")
    print("   Thresholds might be too loose. Consider:")
    print("   - Increase sharpe_threshold from 0.5 to 0.7")
    print("   - Increase return_threshold from 0.02 to 0.025")
elif abs(buy_pct - sell_pct) > 5:
    print("[WARNING] BUY/SELL imbalance > 5%")
    print("   This might cause bias. Check if market has directional trend.")
else:
    print("[OK] Label distribution looks good!")
    print("   Signal frequency is reasonable (10-20%)")
    print("   BUY/SELL are balanced")
    print()
    print("Ready to train on Vast.ai with new labels.")

print()
print("Next steps:")
print("1. If distribution looks good:")
print("   git add train_flagtrader.py IMPROVED_LABELING.md test_new_labels.py")
print("   git commit -m 'Implement Sharpe-based label generation'")
print("   git push")
print()
print("2. On Vast.ai:")
print("   rm data/btc_1h_2020_2024_processed.pkl  # Force label regeneration")
print("   wget https://raw.githubusercontent.com/nimallansa937/HIMARI-OPUS-TACTICAL-L2-V1.0/main/vast_ai_setup.sh && bash vast_ai_setup.sh")
