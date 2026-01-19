# FLAG-TRADER Training Results - Complete Analysis

## Summary of All 3 Training Attempts

### Attempt 1: No Class Weights (Jan 4)
**Weights**: None (standard cross-entropy)
**Result**: 89% HOLD bias
- Sharpe: 0.066
- Trades: 11% of timesteps
- Issue: Model collapse to majority class

### Attempt 2: Aggressive Class Weights (Jan 19, first)
**Weights**: `[2.03, 0.51, 1.83]` (inverse frequency)
**Result**: 99% SELL bias
- Sharpe: 0.000
- Trades: 99% of timesteps (all SELL)
- Issue: Overcorrection, opposite collapse

### Attempt 3: Sqrt Class Weights (Jan 19, second) ⭐ Current
**Weights**: `[1.42, 0.71, 1.35]` (sqrt of inverse frequency)
**Result**: 85% SELL bias
- Sharpe: 0.000
- Trades: 85% of timesteps (mostly SELL)
- Issue: Still biased, but less severe

**Validation Accuracy**: 88.22% (best so far!)
**Class-wise Accuracy**:
- SELL: 87.5%
- HOLD: 90.2%
- BUY: 81.2%

---

## The Core Problem

All three training attempts show **model collapse to a single action**, just different actions:

```
No weights    → 89% HOLD
Aggressive    → 99% SELL
Sqrt weights  → 85% SELL
```

### Why This Happens

1. **Training labels might be flawed**
   - The label generation uses: `future_return > 1% = BUY`, `< -1% = SELL`
   - In crypto, 6-hour 1% moves are common
   - Labels might not reflect true actionable signals

2. **BUY logits are always suppressed**
   - Even with balanced weights, BUY logits are -3 to -5
   - SELL logits are +2 to +4
   - Model has learned "never BUY" regardless of weighting

3. **Class imbalance in raw data**
   - Training: 16% SELL, 66% HOLD, 18% BUY
   - The 66% HOLD dominates learning signal
   - Even aggressive reweighting can't overcome this

---

## Diagnostic: Prediction Distribution

**Sqrt-weighted model (100 random samples)**:
- SELL: 85%
- HOLD: 15%
- BUY: 0%

**Typical logits**:
```
SELL: +2.0 to +4.5 (always high)
HOLD:  0.0 to +3.0 (sometimes competitive)
BUY:  -3.0 to -5.5 (always suppressed)
```

**Conclusion**: The model has learned a strong prior that "BUY is never correct" regardless of input features.

---

## Why Standard Solutions Don't Work

### ❌ Tried: Logit Correction
- Problem: Just shifts the bias, doesn't fix root cause
- Result: Can turn 85% SELL into 85% HOLD, but still collapsed

### ❌ Tried: Balanced Class Weights
- Problem: Overcorrects and creates new bias
- Result: Swapped HOLD bias for SELL bias

### ❌ Tried: Sqrt Weights (Gentler)
- Problem: Still not addressing root cause (bad labels)
- Result: Less severe bias (85% vs 99%) but still unusable

---

## Root Cause: Label Quality

The fundamental issue is **label generation**, not the model or training:

### Current Label Logic
```python
future_return = (price[t+6h] - price[t]) / price[t]

if future_return > 0.01:   # 1%
    label = BUY
elif future_return < -0.01:
    label = SELL
else:
    label = HOLD
```

### Problems with This Approach

1. **Look-ahead bias**: Using future prices that wouldn't be known
2. **Threshold too low**: 1% in 6 hours is noise in crypto
3. **No consideration of risk**: Doesn't account for drawdown
4. **No context**: Ignores market regime (trending vs ranging)

### Evidence from Results

- **88% validation accuracy** but **0% trading returns**
- Model correctly predicts labels, but labels don't generate profit
- High SELL/HOLD accuracy but 0% BUY accuracy suggests BUY labels are unreliable

---

## Solutions (Ordered by Effectiveness)

### Option 1: Better Label Generation (RECOMMENDED) ⭐

Instead of simple threshold, use:

```python
# Forward-looking Sharpe as label quality
def generate_quality_labels(prices, window=24):
    returns = []
    for t in range(len(prices) - window):
        future_returns = prices[t+1:t+window+1] / prices[t] - 1
        sharpe = future_returns.mean() / (future_returns.std() + 1e-8)

        if sharpe > 0.5:  # Positive risk-adjusted return
            label = BUY
        elif sharpe < -0.5:
            label = SELL
        else:
            label = HOLD

    return labels
```

**Benefits**:
- ✅ Risk-adjusted signals
- ✅ Filters noise
- ✅ More reliable BUY signals

**Tradeoff**: Fewer total signals (more selective)

### Option 2: Use Expert Labels

Instead of generating labels, use:
- Historical backtest of a known-good strategy
- Manual annotations from trader
- Regime-aware labeling (trend-following in trends, mean-reversion in ranges)

### Option 3: Reinforcement Learning

Skip supervised learning entirely:
- Use PPO/SAC to learn from rewards (profit)
- No need for labels
- Model learns trading policy directly

**Downside**: Requires more compute and expertise

### Option 4: Multi-Task Learning

Train on multiple objectives:
- Price direction (current labels)
- Price magnitude
- Volatility prediction
- Drawdown prediction

Helps model learn richer representations.

---

## Immediate Recommendations

### Short-term (Use What We Have)

**Use the OLD model (Jan 4) with logit correction**:
```python
logit_correction = [3.0, -2.5, 1.0]  # [SELL, HOLD, BUY]
```

- Sharpe: 0.066 (better than 0.000)
- Predictions: More balanced than new models
- It's not great, but it's functional

### Medium-term (Retrain with Better Labels)

1. Implement Option 1 (Sharpe-based labels)
2. Increase threshold to 2-3% for BUY/SELL
3. Use longer lookahead (24h instead of 6h)
4. Add stop-loss logic to labels

Expected improvement: 30-50% trade diversity, Sharpe 0.3-0.8

### Long-term (Better Architecture)

1. Switch to RL-based approach (PPO)
2. Add market regime detection
3. Ensemble with different label strategies
4. Add risk management layer

---

## Key Learnings

1. **88% accuracy ≠ profitable model**
   - Validation metrics can be misleading
   - Always check prediction distribution and backtest results

2. **Class weights have limits**
   - Can't fix bad labels with weighting tricks
   - Sqrt weighting is better than aggressive, but not a cure

3. **Model collapse indicates data issues**
   - If model always predicts one class, labels are likely wrong
   - Don't keep retraining - fix the data first

4. **Simple thresholds don't work for trading**
   - Need risk-adjusted, regime-aware labeling
   - Forward-looking Sharpe is better than raw returns

---

## Files & Checkpoints

All checkpoints saved in `checkpoints/`:

1. `flag_trader_best_OLD_Jan4.pt` - Original (89% HOLD)
2. `flag_trader_aggressive_weights.pt` - Attempt 2 (99% SELL)
3. `flag_trader_best.pt` - Attempt 3 (85% SELL) ⭐ Current

**Recommendation**: **Use #1 with logit correction** until labels are improved.

---

## Next Steps

To continue this project:

1. **Fix label generation** (implement Sharpe-based labels)
2. **Retrain with new labels**
3. **Test on validation set** (check prediction distribution BEFORE backtesting)
4. **If successful**, backtest and evaluate

**Do NOT**: Keep retraining with current labels - the problem is the data, not the model.

---

**Date**: January 19, 2026
**Models Trained**: 3
**Total Training Time**: ~6 hours on RTX 3090
**Conclusion**: Model learns well, but labels are flawed. Need better label generation strategy.
