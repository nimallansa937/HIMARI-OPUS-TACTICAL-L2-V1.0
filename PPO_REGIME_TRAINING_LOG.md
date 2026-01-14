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

*Last Updated: January 14, 2026*
