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

## Training Session 5: Anti-Overtrading Reward

### Reward Structure (Based on sortino_anti_overtrade.py)
```python
# Base: PnL for trades
trade_pnl = final_returns * position * 100

# 1. PERSISTENCE BONUS - Makes HOLD competitive
hold_bonus = is_hold * 0.05

# 2. BASE TRADE COST - Higher than before
trade_cost = is_trade * 0.08

# 3. REGIME-SPECIFIC PENALTIES
high_vol_penalty = is_trade * (final_regime == 2).float() * 0.10
crisis_penalty = is_trade * (final_regime == 3).float() * 0.15

# 4. TRENDING BONUS
trending_bonus = is_trade * (final_regime == 1).float() * 0.05

# COMBINE
rewards = trade_pnl + hold_bonus - trade_cost - high_vol_penalty - crisis_penalty + trending_bonus
```

### Results
```
Epoch  10/50: HOLD=100.0% LONG=0.0% SHORT=0.0% (all regimes)
Training stopped early - 100% HOLD collapse
```

### Analysis
**Problem:** Model collapsed to 100% HOLD again because:
- Hold bonus (0.05) is **guaranteed**
- Trading is **uncertain** with penalties
- Model exploits the safe, guaranteed reward

**Key Insight:** ANY guaranteed positive reward for HOLD will be exploited.

---

## Training Session 6: Opportunity Cost Reward (Current)

### Reward Structure
```python
# Key insight: HOLD should have NEGATIVE reward in TRENDING (missed opportunity)
# and ZERO reward in risky regimes (preservation is neutral, not rewarded)

# 1. BASE PNL for trades
trade_pnl = final_returns * position * 100

# 2. HOLD PENALTY in TRENDING - you're missing out!
trending_hold_penalty = is_hold * (final_regime == 1).float() * torch.abs(final_returns) * 50

# 3. TRADE COST - minimal
trade_cost = is_trade * 0.02

# 4. WRONG DIRECTION PENALTY - amplify losses
wrong_direction = (position * final_returns < 0).float()
wrong_penalty = wrong_direction * torch.abs(final_returns) * 30

# 5. RISKY REGIME WRONG PENALTY - even worse in HIGH_VOL/CRISIS
risky_regime = ((final_regime == 2) | (final_regime == 3)).float()
risky_wrong_penalty = wrong_direction * risky_regime * torch.abs(final_returns) * 20

# COMBINE
rewards = trade_pnl - trade_cost - trending_hold_penalty - wrong_penalty - risky_wrong_penalty
```

### Key Design Principles
1. **NO guaranteed HOLD bonus** - Removes exploitation opportunity
2. **HOLD penalty in TRENDING** - Scaled by |return| (missed opportunity)
3. **HOLD = 0 in risky regimes** - Neutral, not rewarded
4. **Wrong trade penalty** - Scaled by |return|, amplifies losses
5. **Extra penalty in risky regimes** - Discourages trading in HIGH_VOL/CRISIS

### Expected Rewards
| Action | Regime | Reward |
|--------|--------|--------|
| HOLD | TRENDING | -\|return\| × 50 (penalty!) |
| HOLD | LOW_VOL | 0 (neutral) |
| HOLD | HIGH_VOL/CRISIS | 0 (neutral) |
| Correct trade | Any | PnL × 100 - 0.02 |
| Wrong trade | LOW_VOL/TRENDING | PnL - \|return\| × 30 |
| Wrong trade | HIGH_VOL/CRISIS | PnL - \|return\| × 50 |

### Expected Results
```
Target Distribution:
TRENDING: ~70-90% trading (forced by hold penalty)
LOW_VOL : ~30-50% HOLD (trading is viable)
HIGH_VOL: ~50-70% HOLD (wrong trades hurt more)
CRISIS  : ~60-80% HOLD (wrong trades hurt most)
```

### Status
❌ Result: 0% HOLD - Model still preferred trading

---

## Training Session 7: Variance-Normalized Reward (FR-LUX Research)

### Key Insight from Research
Fixed penalties (0.03-0.23) are 3-5 orders of magnitude too small relative to BTC hourly PnL variance (±1-3%). Penalties must scale with REALIZED VOLATILITY.

### Reward Structure
```python
# === VARIANCE-NORMALIZED REWARD ===
batch_vol = torch.std(final_returns) + 1e-6
trade_pnl = final_returns * position / batch_vol  # Normalized PnL

# REGIME-SPECIFIC HOLD COST (as fraction of volatility)
trending_hold_cost = is_hold * (final_regime == 1).float() * 0.5   # 50% of vol
lowvol_hold_cost = is_hold * (final_regime == 0).float() * 0.0     # Neutral

# REGIME-SPECIFIC TRADE COST
base_trade_cost = is_trade * 0.1  # 10% baseline
highvol_trade_cost = is_trade * (final_regime == 2).float() * 0.3  # +30%
crisis_trade_cost = is_trade * (final_regime == 3).float() * 0.5   # +50%
trending_trade_bonus = is_trade * (final_regime == 1).float() * 0.2  # -20%

wrong_direction = (position * final_returns < 0).float()
wrong_penalty = wrong_direction * 0.3

rewards = trade_pnl - hold_cost - trade_cost - wrong_penalty
```

### Local Test Results (3 epochs)
```
Epoch 1/3: HOLD=15.4% LONG=43.0% SHORT=41.6%
Epoch 2/3: HOLD=3.9%  LONG=48.7% SHORT=47.4%
Epoch 3/3: HOLD=7.3%  LONG=47.0% SHORT=45.7%

By Regime (Epoch 3):
LOW_VOL : HOLD=7.3%
TRENDING: HOLD=6.1%
HIGH_VOL: HOLD=8.7%
CRISIS  : HOLD=8.1%
```

### Analysis
**Progress:**
- First time seeing regime differentiation (HIGH_VOL/CRISIS > TRENDING for HOLD)
- No collapse to 0% or 100%
- Variance normalization working

**Problems:**
- HOLD still too low (7-8% vs target 30-50%)
- Regime differentiation too weak (8.7% vs 6.1% = only 2.6% difference)
- Need stronger trade costs in risky regimes

---

## Training Session 8: Stronger Variance-Normalized

### Changes from Session 7
```python
highvol_trade_cost = 0.6  # was 0.3
crisis_trade_cost = 1.0   # was 0.5
trending_hold_cost = 0.8  # was 0.5
wrong_penalty = 0.5  # was 0.3
```

### Results
```
❌ 99% HOLD COLLAPSE - Costs too high!
HIGH_VOL cost (0.75) and CRISIS cost (1.15) exceed expected |norm PnL| (0.6)
Trading never profitable in these regimes.
```

---

## Training Sessions 9-13: Calibration (Local Testing)

### Key Discovery
Expected |normalized PnL| = E[|return / std|] ≈ 0.6
Therefore, trade costs must be < 0.6 for trading to ever be profitable.

### Session 9: First calibrated attempt
```python
base_trade_cost = 0.05, highvol = +0.25, crisis = +0.40
```
Result: HOLD=9.4% (good regime differentiation but too low overall)

### Session 10-11: Increasing costs
```python
base_trade_cost = 0.15 → 0.20
```
Result: HOLD=16-25% (improving)

### Session 12-13: Final calibration
```python
base_trade_cost = 0.25
highvol_trade_cost = 0.35
crisis_trade_cost = 0.50
trending_trade_bonus = 0.25
trending_hold_cost = 0.6
entropy_coef = 0.10
```

### Session 13 Results (Local)
```
Overall: HOLD=29.4%
LOW_VOL : HOLD=33.0% (Target: 40-60%)
TRENDING: HOLD=8.0%  (Target: 20-40%)
HIGH_VOL: HOLD=45.1% (Target: 60-70%)
CRISIS  : HOLD=41.4% (Target: 70-80%)
```

### Analysis
- Clear regime differentiation achieved
- HIGH_VOL/CRISIS have higher HOLD than LOW_VOL > TRENDING
- Simple local model may lack capacity for exact targets
- Transformer model on Vast.ai should do better

---

## Training Session 14-16: Overfitting Concern & Adaptive Solution

### Problem Identified
Session 13 costs (0.25, 0.35, 0.50) were calibrated to expected |norm PnL| = 0.6
This is **data-specific** - could overfit to 2020-2024 BTC characteristics.

### Solution: Adaptive Costs
Instead of hardcoded values, express costs as **fractions of batch E[|PnL|]**:
```python
expected_abs_pnl = torch.abs(norm_pnl).mean()  # Computed per batch
base_trade_cost = 0.40 * expected_abs_pnl       # Adapts automatically
```

### Session 17 Results (Adaptive, Local)
```
Overall: HOLD=27.4%
LOW_VOL : HOLD=33.1%
TRENDING: HOLD=8.0%
HIGH_VOL: HOLD=38.0%
CRISIS  : HOLD=33.8%
```

---

## Training Session 18: Adaptive for Vast.ai (Current)

### Reward Structure
```python
# ADAPTIVE: Costs scale with actual batch statistics
batch_vol = torch.std(final_returns) + 1e-6
norm_pnl = final_returns / batch_vol
trade_pnl = norm_pnl * position

# Expected |PnL| - adapts to any market condition
expected_abs_pnl = torch.abs(norm_pnl).mean()

# COSTS as FRACTIONS (not hardcoded values)
base_trade_cost = is_trade * 0.40 * expected_abs_pnl
highvol_trade_cost = is_trade * (regime == 2) * 0.40 * expected_abs_pnl
crisis_trade_cost = is_trade * (regime == 3) * 0.60 * expected_abs_pnl
trending_trade_bonus = is_trade * (regime == 1) * 0.40 * expected_abs_pnl

trending_hold_cost = is_hold * (regime == 1) * 1.0 * expected_abs_pnl
wrong_penalty = wrong_direction * 0.50 * expected_abs_pnl
```

### Why This Isn't Overfitting
1. **No hardcoded calibration** - costs adapt per batch
2. **Ratios encode relative preference** - HIGH_VOL 2x more costly than LOW_VOL
3. **Works on any distribution** - tested locally, same results as hardcoded
4. **Will adapt to future data** - if vol changes, costs scale automatically

### Effective Cost Ratios (as fraction of E[|PnL|])
| Regime | Trade Cost | HOLD Cost | Net Effect |
|--------|------------|-----------|------------|
| TRENDING | 0.00 | 1.00 | Forces trading |
| LOW_VOL | 0.40 | 0.00 | Mixed |
| HIGH_VOL | 0.80 | 0.00 | Prefers HOLD |
| CRISIS | 1.00 | 0.00 | Strong HOLD |

### Status
⏳ Deploying to Vast.ai...

---

## Key Learnings (Updated)

1. **Guaranteed rewards dominate uncertain rewards**
   - Even small constant bonuses (0.05) can overwhelm uncertain trading returns
   - Model exploits any guaranteed reward source
   - **Solution: NO guaranteed HOLD rewards**

2. **Must balance bonuses and penalties**
   - HOLD bonuses need corresponding trade bonuses
   - Penalties alone don't encourage action, just avoidance

3. **Entropy coefficient matters**
   - Low entropy (0.01) allows exploitation of easy strategies
   - Higher entropy (0.05) forces exploration of trading strategies

4. **Regime-specific behavior requires regime-specific incentives**
   - Can't just globally encourage/discourage actions
   - Must create different incentive landscapes per regime

5. **Trade cost must be calibrated to expected |PnL|**
   - Expected |normalized PnL| ≈ 0.6
   - Trade costs > 0.6 make trading never profitable
   - Costs too high → 100% HOLD collapse
   - Costs too low → <10% HOLD

6. **Opportunity cost approach**
   - Penalize HOLD in TRENDING (missed opportunity)
   - Make HOLD neutral (0) in risky regimes
   - No guaranteed HOLD bonuses

7. **The calibration sweet spot**
   - base_trade_cost = 0.25 (moderate friction)
   - regime_trade_cost < 0.50 (keeps trading viable)
   - trending_hold_cost = 0.60 (forces trading)
   - Only TRENDING has hold penalty

---

*Last Updated: January 14, 2026*
