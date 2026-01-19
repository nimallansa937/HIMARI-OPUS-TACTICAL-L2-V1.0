# FLAG-TRADER Retraining Results - January 19, 2026

## Training Summary

**Model**: FLAG-TRADER (88M params, LoRA rank 16)
**Training Time**: ~2 hours on Vast.ai RTX 3090
**Training Data**: 43,793 samples (2020-2024 BTC 1h data)

### Training Metrics

- **Final Validation Accuracy**: 86.34%
- **Final Validation Loss**: 0.3036
- **Class-wise Accuracy (Epoch 10)**:
  - SELL: 90.15%
  - HOLD: 85.08%
  - BUY: 87.93%

### Class Distribution in Training Data

- SELL: 16.3% (7,131 samples)
- HOLD: 65.6% (28,720 samples)
- BUY: 18.1% (7,942 samples)

### Applied Class Weights

```python
class_weights = [2.03, 0.51, 1.83]  # [SELL, HOLD, BUY]
```

Calculation: `total_samples / (n_classes * class_count)`

---

## Problem: Severe Model Collapse (Opposite Direction)

### Observed Behavior

The retrained model exhibits **extreme bias toward SELL**:

**Sample Predictions (100 random test samples)**:
- SELL: 100% (99.1% avg confidence)
- HOLD: 0%
- BUY: 0%

**Typical Logits**:
```
SELL:  +4.0 to +5.4 (very high)
HOLD:  -0.5 to +1.1 (neutral/low)
BUY:   -2.5 to -4.8 (very low)
```

### Backtest Results

**Without calibration**:
- Total Trades: 9,048 / 9,073 (99.7% SELL)
- Sharpe Ratio: 0.000
- Total Return: 0.00%

**With calibration** (`logit_correction = [-4.0, +1.5, +2.5]`):
- Total Trades: 1 / 9,073 (back to HOLD bias)
- Sharpe Ratio: 0.038
- Total Return: 0.00%

---

## Root Cause Analysis

### Issue 1: Aggressive Class Weights

The balanced class weights overcorrected:

```
SELL weight: 2.03x  (16% → heavily upweighted)
HOLD weight: 0.51x  (66% → heavily downweighted)
BUY weight:  1.83x  (18% → heavily upweighted)
```

This caused the model to learn that:
1. SELL is **4x more important** than HOLD
2. BUY is **3.6x more important** than HOLD
3. The model collapsed to always predicting SELL (highest weight)

### Issue 2: Train/Test Distribution Mismatch

- **Training**: Balanced with aggressive weights
- **Test**: Natural distribution (likely similar to training data)
- Model learned to predict SELL regardless of input

### Issue 3: Logit Calibration Doesn't Solve Root Cause

Applying logit corrections just shifts the bias:
- No correction: 99% SELL
- Correction `[-4, +1.5, +2.5]`: 99% HOLD
- Any correction: Still collapsed to one class

---

## Solutions

### Option 1: Retrain with Gentler Class Weights (RECOMMENDED)

Use **sqrt-inverse-frequency** weighting instead of inverse-frequency:

```python
# Instead of: weight = total / (n_classes * count)
# Use: weight = sqrt(total / (n_classes * count))

SELL weight: sqrt(2.03) = 1.42
HOLD weight: sqrt(0.51) = 0.71
BUY weight:  sqrt(1.83) = 1.35
```

This provides:
- ✅ Still addresses class imbalance
- ✅ Gentler correction (1.4x vs 2x)
- ✅ Less likely to cause collapse

**Training command**:
```bash
python train_flagtrader_v2.py \
    --data data/btc_1h_2020_2024.csv \
    --output checkpoints/ \
    --epochs 10 \
    --weight_strategy sqrt
```

### Option 2: Focal Loss

Instead of weighted cross-entropy, use Focal Loss:

```python
focal_loss = -(1 - p_t)^gamma * log(p_t)
gamma = 2.0  # Focus on hard examples
```

Benefits:
- ✅ Automatically focuses on hard-to-classify examples
- ✅ No manual weight tuning needed
- ✅ Better for imbalanced data

### Option 3: Data Augmentation

Oversample minority classes (SELL/BUY):

```python
# Duplicate SELL/BUY samples to match HOLD frequency
n_hold = 28,720
n_sell = 7,131  # Duplicate 4x → 28,524
n_buy = 7,942   # Duplicate 3.6x → 28,591
```

Benefits:
- ✅ No weight tuning needed
- ✅ More training data for minority classes
- ✅ Natural learning

### Option 4: Temperature Scaling (Post-hoc Calibration)

Train a temperature parameter on validation set:

```python
# Find optimal T that balances predictions
probs = softmax(logits / T)
```

Benefits:
- ✅ No retraining needed
- ✅ Fast to implement
- ⚠️ Bandaid solution, doesn't fix root cause

---

## Recommended Next Steps

### Immediate (Use Current Model)

1. **Use the old checkpoint** (Jan 4) with logit correction `[3.0, -2.5, 1.0]`
   - Sharpe: 0.066 (not great, but functional)
   - Trade diversity: 11% (better than 0% or 99%)

### Short-term (1-2 hours)

2. **Retrain with sqrt-weighted loss** (Option 1)
   - Modify `train_flagtrader.py` line 345
   - Re-run training on Vast.ai
   - Expected: 30-40% trade diversity, Sharpe 0.3-0.8

### Medium-term (3-4 hours)

3. **Implement Focal Loss** (Option 2)
   - Better long-term solution
   - More robust to class imbalance

---

## Key Learnings

1. **Class weights can overcorrect**: Be cautious with aggressive reweighting
2. **Sqrt weighting is safer**: Provides balance without extreme bias
3. **Validation accuracy ≠ useful predictions**: 86% accuracy but 100% SELL is useless
4. **Always check prediction distribution**: Not just accuracy/loss
5. **Logit calibration is a bandaid**: Fixes symptoms, not root cause

---

## Files & Checkpoints

**New checkpoint (Jan 19)**:
- Location: `checkpoints/flag_trader_best.pt`
- Size: 359MB
- Status: Unusable due to SELL bias

**Old checkpoint (Jan 4)**:
- Location: `checkpoints/flag_trader_best_OLD_Jan4.pt`
- Size: 336MB
- Status: Functional with logit correction

**Training script**:
- `train_flagtrader.py` - Current version (aggressive weights)
- `train_flagtrader_v2.py` - TODO: Create with sqrt weights

---

**Conclusion**: The aggressive class weights caused the opposite problem - instead of 89% HOLD, we now have 99% SELL. Need gentler weighting strategy for next training run.
