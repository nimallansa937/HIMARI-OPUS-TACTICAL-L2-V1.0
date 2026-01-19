# Improved Label Generation - January 19, 2026

## Problem with Previous Approach

All three training attempts (no weights, aggressive weights, sqrt weights) resulted in model collapse because **the labels themselves were flawed**, not the model or training approach.

### Previous Label Logic (FLAWED)
```python
# Simple threshold-based approach
lookahead = 6  # 6 hours
future_return = (price[t+6h] - price[t]) / price[t]

if future_return > 0.01:   # 1%
    label = BUY
elif future_return < -0.01:
    label = SELL
else:
    label = HOLD
```

### Why This Failed
1. **Too noisy**: 1% moves in 6 hours are common noise in crypto
2. **No risk adjustment**: Doesn't consider volatility or Sharpe ratio
3. **Look-ahead bias**: Uses exact future prices
4. **No stop-loss**: Doesn't account for drawdown risk
5. **Too short horizon**: 6 hours is insufficient for meaningful signals

**Result**: 88% validation accuracy but 0% trading returns

---

## New Improved Approach

### Sharpe-Based Risk-Adjusted Labels

```python
lookahead_window = 24  # 24 hours (4x longer)
sharpe_threshold = 0.5  # Minimum Sharpe for signals
return_threshold = 0.02  # 2% minimum return (2x higher)
stop_loss_pct = 0.03  # 3% stop-loss threshold
```

### Algorithm

For each timestep `t`:

1. **Calculate forward returns over 24-hour window**:
   ```python
   future_prices = close[t+1:t+25]
   returns = (future_prices - close[t]) / close[t]
   ```

2. **Compute Sharpe ratio** (risk-adjusted return):
   ```python
   sharpe = mean(returns) / std(returns)
   ```

3. **Check stop-loss condition**:
   ```python
   max_drawdown = (close[t] - min(future_low)) / close[t]
   if max_drawdown > 3%:
       stop_loss_triggered = True
   ```

4. **Generate label**:
   ```python
   if sharpe > 0.5 AND final_return > 2% AND NOT stop_loss_triggered:
       label = BUY  # Strong risk-adjusted upside
   elif sharpe < -0.5 AND final_return < -2%:
       label = SELL  # Strong risk-adjusted downside
   else:
       label = HOLD  # Insufficient signal or too risky
   ```

---

## Key Improvements

### 1. Risk-Adjusted Signals
- **Old**: Raw return > 1% → BUY
- **New**: Sharpe > 0.5 AND return > 2% → BUY

**Benefit**: Filters noise, only signals when risk-adjusted return is positive

### 2. Longer Lookahead Window
- **Old**: 6 hours
- **New**: 24 hours

**Benefit**: Reduces noise, captures more meaningful price movements

### 3. Higher Return Threshold
- **Old**: 1%
- **New**: 2%

**Benefit**: More actionable signals, filters small moves

### 4. Stop-Loss Logic
- **Old**: No consideration of drawdown
- **New**: Checks if max drawdown > 3% during window

**Benefit**: Avoids signals that would hit stop-loss

### 5. Combines Multiple Criteria
- **Old**: Single threshold (return)
- **New**: Multiple checks (Sharpe + return + stop-loss)

**Benefit**: More robust, reliable signals

---

## Expected Impact

### Label Distribution
- **Old distribution**: ~16% SELL, 66% HOLD, 18% BUY
- **Expected new**: ~5-10% SELL, 80-90% HOLD, 5-10% BUY

More selective signals (fewer but higher quality)

### Model Behavior
- **Old**: Model collapse (always predicts one class)
- **Expected new**: Balanced predictions on quality signals

### Trading Performance
- **Old**: Sharpe 0.000, 0% returns
- **Expected new**: Sharpe 0.3-0.8, positive returns

---

## Implementation Details

### Modified Functions

**train_flagtrader.py:188-255**
```python
def generate_labels(close: np.ndarray, high: np.ndarray = None, low: np.ndarray = None,
                   threshold: float = 0.01) -> np.ndarray:
    """
    Generate trading labels based on risk-adjusted forward returns (Sharpe ratio).

    Improvements:
    - Uses Sharpe ratio to measure risk-adjusted returns
    - Longer lookahead window (24h) to filter noise
    - Higher return threshold (2%) for more actionable signals
    - Considers drawdown via high/low prices for stop-loss logic
    """
```

**train_flagtrader.py:72-77**
```python
# Generate labels using improved Sharpe-based approach with stop-loss
labels = generate_labels(
    close=df['close'].values,
    high=df['high'].values,
    low=df['low'].values
)
```

---

## Training Instructions

### Local Testing (Verify Label Quality)

Before training on Vast.ai, verify the new labels locally:

```bash
cd "C:\Users\chari\OneDrive\Documents\HIMARI OPUS TESTING\LAYER 2 V1 - Copy"

# Create test script to check label distribution
python -c "
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, '.')
from train_flagtrader import generate_labels

# Load data
df = pd.read_csv('L2 POSTION FINAL MODELS/orkspace/data/btc_1h_2020_2024.csv')

# Generate labels with new approach
labels = generate_labels(
    close=df['close'].values,
    high=df['high'].values,
    low=df['low'].values
)

# Show distribution
unique, counts = np.unique(labels, return_counts=True)
print('New Label Distribution:')
for label, count in zip(unique, counts):
    action = ['SELL', 'HOLD', 'BUY'][label]
    print(f'  {action}: {count} ({count/len(labels)*100:.1f}%)')
"
```

### Expected Output
```
New Label Distribution:
  SELL: 3,500-5,000 (8-11%)
  HOLD: 35,000-38,000 (80-87%)
  BUY: 3,500-5,000 (8-11%)
```

If BUY/SELL are < 5%, the thresholds might be too strict. Adjust parameters:
- Reduce `sharpe_threshold` from 0.5 to 0.3
- Reduce `return_threshold` from 0.02 to 0.015

---

### Training on Vast.ai

Once label distribution looks good:

```bash
# 1. Commit and push changes
git add train_flagtrader.py IMPROVED_LABELING.md
git commit -m "Implement Sharpe-based risk-adjusted label generation"
git push

# 2. On Vast.ai, run training
wget https://raw.githubusercontent.com/nimallansa937/HIMARI-OPUS-TACTICAL-L2-V1.0/main/vast_ai_setup.sh && bash vast_ai_setup.sh
```

**Note**: Delete old preprocessed pickle to force label regeneration:
```bash
rm data/btc_1h_2020_2024_processed.pkl
```

---

## Validation Checklist

After training with new labels, verify:

1. **Label distribution** (should be ~10% BUY, 80% HOLD, 10% SELL)
2. **Prediction diversity** (should see > 5% for each class)
3. **Backtest Sharpe** (should be > 0.3)
4. **Trade frequency** (should trade 10-20% of timesteps)

If model still collapses:
- Check label distribution (might be too imbalanced)
- Adjust Sharpe/return thresholds
- Consider even longer lookahead (48h)

---

## Comparison: Old vs New

| Metric | Old Approach | New Approach |
|--------|-------------|--------------|
| Lookahead | 6 hours | 24 hours |
| Return threshold | 1% | 2% |
| Risk adjustment | None | Sharpe ratio |
| Stop-loss | None | 3% drawdown check |
| Validation accuracy | 88% | Expected: 75-85% |
| Trading Sharpe | 0.000 | Expected: 0.3-0.8 |
| Trade diversity | 0% (collapsed) | Expected: 10-20% |

Lower validation accuracy is expected because the task is now harder (filtering noise), but trading performance should improve significantly.

---

## Next Steps

1. ✅ **Update train_flagtrader.py** (DONE)
2. ⏳ **Test label distribution locally**
3. ⏳ **Commit and push to GitHub**
4. ⏳ **Retrain on Vast.ai with new labels**
5. ⏳ **Validate prediction diversity and backtest results**

If successful, this should solve the model collapse issue by providing higher-quality, more actionable trading signals.

---

**Date**: January 19, 2026
**Author**: Claude (HIMARI Project)
**Status**: Ready for testing
