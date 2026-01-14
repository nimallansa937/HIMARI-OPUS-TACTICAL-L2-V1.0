# HIMARI Layer 2 - PPO Regime-Conditioned Training Log

**Training Date:** January 14, 2026
**Dataset:** BTC 1H 2020-2024 (44 features, balanced regimes)

---

## Training Session 1: 0% HOLD Issue

### Configuration
- Model: Transformer-based PPO with regime conditioning
- Features: 44 technical indicators (denoised via EKF)
- Context: 100 timesteps
- Epochs: 50
- Batch Size: 64

### Results
```
Overall: HOLD=0.0% LONG=51.9% SHORT=48.1%
LOW_VOL : HOLD=0.0% LONG=52.1% SHORT=47.9%
TRENDING: HOLD=0.0% LONG=53.2% SHORT=46.8%
HIGH_VOL: HOLD=0.0% LONG=49.8% SHORT=50.2%
CRISIS  : HOLD=0.0% LONG=48.5% SHORT=51.5%
```

### Issue
Model learned to ALWAYS trade, never hold. This is problematic because:
- In HIGH_VOL/CRISIS regimes, holding is often safer
- No capital preservation during uncertain periods
- Transaction costs accumulate

---

## Training Session 2: 100% HOLD Issue (Over-correction)

### Reward Changes Applied
```python
# Added HOLD incentives:
- LOW_VOL HOLD bonus: +0.05
- HIGH_VOL HOLD bonus: +0.08
- CRISIS HOLD bonus: +0.10
- Near-zero returns HOLD bonus: +0.03

# Added trading penalties:
- Trade cost: -0.02
- Wrong direction: -0.05
- Risky regime trade: -0.03
```

### Results
```
Overall: HOLD=100.0% LONG=0.0% SHORT=0.0%
LOW_VOL : HOLD=100.0% LONG=0.0% SHORT=0.0%
TRENDING: HOLD=100.0% LONG=0.0% SHORT=0.0%
HIGH_VOL: HOLD=100.0% LONG=0.0% SHORT=0.0%
CRISIS  : HOLD=100.0% LONG=0.0% SHORT=0.0%
```

### Issue
Model collapsed to ALWAYS HOLD because:
- HOLD bonuses were guaranteed (~0.05-0.10)
- Trading returns were uncertain with penalties
- Model learned that holding always beats trading in expectation

---

## Training Session 3: Balanced Reward Function (Current)

### Reward Changes Applied
```python
# Regime-based incentives (Goal: HOLD in risky, TRADE in trending)
- HIGH_VOL HOLD bonus: +0.02 (reduced)
- CRISIS HOLD bonus: +0.03 (reduced)
- TRENDING HOLD penalty: -0.04 (NEW - penalize not trading)
- LOW_VOL: Neutral (no bonus/penalty)

# Trading incentives
- Correct trade bonus: +0.08 (NEW)
- Correct trend trade bonus: +0.06 (extra for trending regime)
- Trending trade bonus: +0.02 (just for activity in trending)

# Trading penalties (reduced)
- Trade cost: -0.01 (was 0.02)
- Wrong direction: -0.03 (was 0.05)
- Crisis trade penalty: -0.02

# Other changes
- Base reward multiplier: 200 (was 100)
- Entropy coefficient: 0.05 (was 0.01, encourages exploration)
```

### Expected Results
```
Target Distribution:
TRENDING: ~60-80% trading (LONG/SHORT)
LOW_VOL : ~40-60% HOLD
HIGH_VOL: ~60-70% HOLD
CRISIS  : ~70-80% HOLD
```

---

## Key Learnings

1. **Guaranteed rewards dominate uncertain rewards**
   - Even small constant bonuses (0.05) can overwhelm uncertain trading returns
   - Model exploits any guaranteed reward source

2. **Must balance bonuses and penalties**
   - HOLD bonuses need corresponding trade bonuses
   - Penalties alone don't encourage action, just avoidance

3. **Entropy coefficient matters**
   - Low entropy (0.01) allows exploitation of easy strategies
   - Higher entropy (0.05) forces exploration of trading strategies

4. **Regime-specific behavior requires regime-specific incentives**
   - Can't just globally encourage/discourage actions
   - Must create different incentive landscapes per regime

---

## Dataset Statistics

- Total samples: ~43,500 (after warmup)
- Train: 26,100 (60%)
- Val: 8,700 (20%)
- Test: 8,700 (20%)

### Regime Distribution (Balanced)
- LOW_VOL: 45.3%
- TRENDING: 26.2%
- HIGH_VOL: 16.9%
- CRISIS: 11.5%

---

## Model Architecture

```
RegimeConditionedPolicy:
  - Feature projection: 44 -> 256
  - Regime embedding: 4 regimes -> 256
  - Positional encoding: 100 timesteps
  - Transformer encoder: 3 layers, 4 heads
  - Actor head: 512 -> 256 -> 3 (HOLD/LONG/SHORT)
  - Critic head: 512 -> 256 -> 1 (value)

Total parameters: ~2.5M
```

---

## Files

- `train_vast.py` - Main training script (updated with balanced rewards)
- `src/pipeline/regime_detector.py` - Balanced 4-regime detector
- `src/pipeline/feature_engineer.py` - 44 technical features
- `src/pipeline/ekf_denoiser.py` - Extended Kalman Filter

---

## Pickle Compatibility Fix

### Issue
When running on Vast.ai, got `ModuleNotFoundError: No module named 'src'` because the dataset pickle contained `EnrichedSample` dataclass objects from `src.pipeline.dataset_generator`.

### Solution
1. Created `convert_dataset_to_arrays.py` to convert dataclass objects → numpy arrays
2. Updated `train_vast.py` to load the new array-based format
3. No more module dependency in the pickle file

### New Dataset Format
```python
{
    'train': {
        'features_raw': np.ndarray (n, 44),
        'features_denoised': np.ndarray (n, 44),
        'regime_ids': np.ndarray (n,),
        'regime_confidences': np.ndarray (n,),
        'prices': np.ndarray (n,),
        'returns': np.ndarray (n,),
        'n_samples': int
    },
    'val': {...},
    'test': {...},
    'metadata': {...}
}
```

### Steps to Fix
1. Run `python convert_dataset_to_arrays.py` locally ✅
2. Upload `btc_1h_2020_2024_enriched_44f_arrays.pkl` to Google Drive ✅
3. Update `GDRIVE_FILE_ID` in `train_vast.py` with new file ID ✅
   - New ID: `1DpJAViY1YK_czC3Tfi3R0oXvPg-9Eo87`
4. Run training on Vast.ai ⏳

### Vast.ai Setup Command
```bash
cd /workspace && rm -rf HIMARI-OPUS-TACTICAL-L2-V1.0 && git clone https://github.com/nimallansa937/HIMARI-OPUS-TACTICAL-L2-V1.0.git && cd HIMARI-OPUS-TACTICAL-L2-V1.0 && python train_vast.py
```

---

## Training Session 4: Simplest Reward (PnL - Cost)

### Reward Structure
```python
# SIMPLEST REWARD: PnL + COST
# HOLD: Gets 0 reward (neutral)
# Trade: Gets PnL - cost

trade_pnl = final_returns * position * 100
trade_cost = is_trade * 0.03  # 0.03 per trade

rewards = trade_pnl - trade_cost
```

### Results
```
Training: 50 epochs, 11.1 minutes
Best Val Reward: 0.2462

Epoch 10:  HOLD=0.8%  LONG=51.7% SHORT=47.6%
Epoch 20:  HOLD=3.6%  LONG=49.2% SHORT=47.1%
Epoch 30:  HOLD=4.3%  LONG=48.4% SHORT=47.4%
Epoch 40:  HOLD=5.1%  LONG=48.4% SHORT=46.5%
Epoch 50:  HOLD=4.9%  LONG=48.7% SHORT=46.3%

Test Set (Final):
Overall:   HOLD=6.5%  LONG=47.6% SHORT=45.9%
LOW_VOL :  HOLD=8.2%  LONG=45.5% SHORT=46.3%
TRENDING:  HOLD=6.0%  LONG=48.2% SHORT=45.8%
HIGH_VOL:  HOLD=4.4%  LONG=50.4% SHORT=45.2%
CRISIS  :  HOLD=4.7%  LONG=49.9% SHORT=45.5%
```

### Analysis

**Progress:**
- HOLD increased from 0.8% → 6.5% over training
- First time model learned HOLD is a valid option
- Training stable (no collapse)

**Problems:**
1. **HOLD still too low** - Need 30-50%, got 6.5%
2. **No regime differentiation** - CRISIS (4.7%) has LESS HOLD than LOW_VOL (8.2%)
   - Should be opposite: CRISIS ~70-80% HOLD
3. **Trade cost too weak** - 0.03 is negligible vs ±1-3% PnL variance
   - Model thinks: "Why hold for 0 when I might win big?"

### Root Cause
The simplest reward doesn't create enough incentive for HOLD because:
- BTC hourly returns have high variance (±1-3% common)
- 0.03 cost is tiny compared to potential ±3 reward
- No regime-specific penalties to discourage risky trades

---

## Training Session 5: Anti-Overtrading Reward (Current)

### Reward Structure (Based on sortino_anti_overtrade.py)
```python
# Base: PnL for trades
trade_pnl = final_returns * position * 100

# 1. PERSISTENCE BONUS - Makes HOLD competitive
#    Small guaranteed reward for staying out of market
hold_bonus = is_hold * 0.05

# 2. BASE TRADE COST - Higher than before
trade_cost = is_trade * 0.08

# 3. REGIME-SPECIFIC PENALTIES - Heavy cost in risky regimes
#    HIGH_VOL (regime 2): Extra penalty for trading
high_vol_penalty = is_trade * (final_regime == 2).float() * 0.10
#    CRISIS (regime 3): Even heavier penalty
crisis_penalty = is_trade * (final_regime == 3).float() * 0.15

# 4. TRENDING BONUS - Reward trading in good conditions
trending_bonus = is_trade * (final_regime == 1).float() * 0.05

# COMBINE
rewards = trade_pnl + hold_bonus - trade_cost - high_vol_penalty - crisis_penalty + trending_bonus
```

### Key Design Principles
1. **Persistence bonus (0.05)** - Gives HOLD a small guaranteed reward
2. **Higher trade cost (0.08)** - Makes trading less attractive for small moves
3. **Regime penalties** - Heavy cost for trading in HIGH_VOL/CRISIS
4. **Trending bonus** - Encourage trading when conditions are favorable

### Expected Results
```
Target Distribution:
TRENDING: ~60-80% trading (LONG/SHORT)
LOW_VOL : ~40-60% HOLD
HIGH_VOL: ~60-70% HOLD
CRISIS  : ~70-80% HOLD
```

### Status
⏳ Ready for training...

---

## Key Learnings (Updated)

1. **Guaranteed rewards dominate uncertain rewards**
   - Even small constant bonuses (0.05) can overwhelm uncertain trading returns
   - Model exploits any guaranteed reward source

2. **Must balance bonuses and penalties**
   - HOLD bonuses need corresponding trade bonuses
   - Penalties alone don't encourage action, just avoidance

3. **Entropy coefficient matters**
   - Low entropy (0.01) allows exploitation of easy strategies
   - Higher entropy (0.05) forces exploration of trading strategies

4. **Regime-specific behavior requires regime-specific incentives**
   - Can't just globally encourage/discourage actions
   - Must create different incentive landscapes per regime

5. **Trade cost must be significant relative to PnL variance**
   - 0.03 is too small when PnL can be ±3
   - Need 0.08-0.15 to make HOLD attractive

6. **Reference: sortino_anti_overtrade.py approach**
   - Cooldown penalty, persistence bonus, carry cost
   - Forces deliberate trading decisions

---

*Last Updated: January 14, 2026*
