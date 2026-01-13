# Transformer A2C/PPO Training Log

## Overview

This document logs all training experiments conducted for the HIMARI Layer 2 Transformer-based trading agent, including problems encountered, solutions attempted, and outcomes.

**Date Range:** January 11-12, 2026  
**Data:** BTC 5-min OHLCV, 527,000 samples (2020-2024)  
**Model:** Transformer-A2C/PPO with 3.3M parameters

---

## Problem Categories (Initial A2C Issues)

### Category 1: Look-Ahead Bias ❌ → ✅ FIXED

- **Bug:** `transformer_a2c_env.py:147` used `self.data[self._current_step]` instead of `self.data[self._current_step - 1]`
- **Symptom:** Unrealistic Sharpe ~210, 8,145% returns
- **Fix:** Fetch previous timestep observation

### Category 2: Reward Calculation Bugs ❌ → ✅ FIXED

- **Bug:** Sharpe clipping masked extreme values
- **Fix:** Added warning logs for suspicious Sharpe values

### Category 3: Exploration Failure ❌ → 🔄 ONGOING

- **Bug:** Model collapses to single action (FLAT or LONG)
- **Symptom:** 100% single action at validation

---

## Training Experiments

### Experiment 1: A2C Baseline (entropy_coef=0.01)

| Metric | Value |
|--------|-------|
| Steps | 100k |
| Validation Sharpe | 0.0000 |
| Action Distribution | 100% FLAT |
| Result | ❌ Policy collapse |

**Problem:** Insufficient exploration, model converges to safe FLAT.

---

### Experiment 2: A2C + Higher Entropy (entropy_coef=0.07)

| Metric | Value |
|--------|-------|
| Steps | 100k |
| Validation Sharpe | 0.0000 |
| Action Distribution | 99.9% FLAT |
| Result | ❌ Still collapsed |

**Problem:** Entropy coefficient still too low for sparse reward.

---

### Experiment 3: A2C + Very High Entropy (entropy_coef=0.20)

| Metric | Value |
|--------|-------|
| Steps | 100k |
| Validation Sharpe | 0.0000 |
| Action Distribution | 100% FLAT |
| Result | ❌ No improvement |

**Problem:** Entropy added to training but validation uses deterministic argmax.

---

### Experiment 4: A2C + Actor Bias Initialization

**Change:** Initialize actor bias to favor LONG/SHORT: `[-0.5, 0.2, 0.2]`

| Metric | Value |
|--------|-------|
| Steps | 100k |
| Best Checkpoint | 26k |
| Validation Sharpe | 37.45 |
| Action Distribution | 85.7% FLAT at 100k |
| Trades | 7,359 |
| Win Rate | 39.3% |
| Result | ⚠️ Partial success - collapsed to FLAT late |

**Problem:** Model initially explored but collapsed mid-training.

---

### Experiment 5: A2C + Transaction Costs (0.15% per trade)

| Metric | Value |
|--------|-------|
| Steps | 120k (early stopped) |
| Best Checkpoint | 20k |
| Validation Sharpe | 8.47 |
| Gross Return | 1538% |
| Net Return | 401% |
| Trades | 7,580 |
| Action Distribution | Started 83% SHORT, ended 100% LONG |
| Result | ❌ Collapsed to LONG |

**Analysis:**

- Transaction costs penalized frequent trading ✅
- Model found degenerate optimum: "stay LONG forever = no trading costs"
- Validation period (2022-2024) had bullish bias, so LONG worked

---

### Experiment 6: PPO with Transaction Costs

**Switch from A2C to PPO for better stability**

| Metric | Step 20k | Step 40k (Best) | Step 120k |
|--------|----------|-----------------|-----------|
| Sharpe | 28.82 | **38.88** | 35.1 |
| Trades | 19,167 | 9,253 | - |
| Net Return | 1403% | **1818%** | - |
| Costs | 2875% | 1388% | - |
| LONG % | 68% | 85.5% | 96% |
| Result | Running | Best | Collapsing |

**PPO Behaviors Observed:**

- KL early stopping active (good - preventing large updates)
- Clip fraction ~0.25 (healthy constraint)
- Still collapsing to LONG but slower than A2C
- Entropy dying: 0.87 → 0.09

**Root Cause:**

1. Validation period (late 2022-2024) was net bullish
2. Model learns "LONG = profit" and exploits this
3. Transaction costs discourage trading → stay LONG

---

### Experiment 6b: PPO with Bear Market Validation (40/40/20 Split)

**Key Change:** Include 2022 bear market in validation period to force model to learn SHORT

**Split:** Train=[0:210800] (2020-2021), Val=[210800:421600] (2021-2023), Test=[421600:527000] (2023-2024)

| Step | Sharpe | Net Return | Trades | Avg Hold | LONG % | SHORT % |
|------|--------|------------|--------|----------|--------|---------|
| 20k | 45.13 | 5370% | 60,490 | 3.5 bars | 51.1% | 48.9% |
| 40k | 53.27 | 6245% | 49,974 | 4.2 bars | 40.4% | 59.6% |
| 61k | 55.12 | 6380% | 39,744 | 5.3 bars | 29.1% | 70.9% |
| 81k | 55.98 | 6451% | 35,976 | 5.9 bars | 26.2% | 73.8% |
| **100k** | **57.44** | **6564%** | 30,194 | 7.0 bars | 21.8% | **78.2%** |
| 120k | 53.68 | 6158% | 31,580 | 6.7 bars | 22.9% | 77.1% |
| 141k | - | - | - | - | 1.5% | 98.5% |

**Key Observations:**

1. ✅ **Model learned to SHORT for the first time!** Bear market validation worked
2. ✅ Net returns improved (6564% vs 1818% with bullish validation)
3. ❌ Still collapsed - but to SHORT instead of LONG
4. ❌ FLAT never used (0% throughout)
5. Best checkpoint at 100k steps

**Conclusion:** Bear market in validation forces SHORT learning, but model still collapses to single dominant action.

---

## Solutions Implemented

### Solution 1: Higher Entropy Coefficient

- Tried: 0.01 → 0.07 → 0.20
- Result: No effect on deterministic validation

### Solution 2: Actor Bias Initialization

- Tried: Bias logits toward LONG/SHORT
- Result: Delayed collapse but didn't prevent it

### Solution 3: Transaction Costs

- Tried: 0.15% per trade
- Result: Model stayed in position longer but collapsed to LONG

### Solution 4: PPO Algorithm

- Tried: Clipped objective, multi-epoch updates
- Result: Slower collapse but still happening

### Solution 5: Carry Cost (NEW - PENDING)

- Add: 0.001% per bar for non-FLAT positions
- Goal: Penalize holding positions indefinitely

### Solution 6: Bear Market Validation (NEW - PENDING)

- Change: 40/40/20 split to include 2022 crash in validation
- Goal: Model must learn SHORT to survive -75% BTC drawdown

---

## Key Files Modified

| File | Changes |
|------|---------|
| `transformer_a2c_env.py` | Fixed look-ahead bias (line 147) |
| `transformer_a2c.py` | Actor bias initialization (line 257-266) |
| `transformer_a2c_trainer.py` | Transaction costs, action logging |
| `sortino_reward.py` | Added `SortinoWithTransactionCosts`, `SortinoWithCarryCost` |
| `transformer_ppo_trainer.py` | NEW: PPO trainer with clipped objective |

---

## Current Status

**Best Known Result:** PPO Step 40k

- Net Return: 1818% (after 0.15% transaction costs)
- Sharpe: 38.88 (unrealistic but directionally correct)
- Still collapsed to ~86% LONG

**Next Experiment:** PPO + Carry Cost + Temperature Sampling + Higher Entropy

---

### Experiment 7: PPO v2 with Carry Cost + Temperature Sampling ✅ BEST SO FAR

**Key Fixes Applied:**

| Fix | What | Why |
|-----|------|-----|
| Carry Cost | 0.002%/bar penalty for holding | Forces model to actively justify positions |
| Temperature Sampling | T=0.5 during validation | Prevents model learning entropy doesn't matter |
| Higher Entropy | 0.10 → 0.02 with 0.2% decay | Maintains exploration longer |
| Higher target_kl | 0.05 instead of 0.03 | Allows more policy change per update |

**Results:**

| Step | Sharpe | Net Return | Trades | FLAT % | LONG % | SHORT % | Entropy |
|------|--------|------------|--------|--------|--------|---------|---------|
| 20k | -32.49 | -1697% | 38,907 | 2.5% | 48.7% | 48.8% | 0.81 |
| 61k | +8.17 | +411% | 30,159 | 1.2% | 52.2% | 46.7% | 0.69 |
| 100k | +17.63 | +873% | 26,801 | 0.4% | 57.7% | 41.9% | 0.58 |
| 141k | **+27.85** | **+1348%** | 23,223 | 0.5% | 65.3% | 34.1% | 0.53 |
| **221k** | **+29.30** | **+1424%** | 25,648 | 0.3% | 54.0% | 45.7% | 0.37 |
| 241k | - | - | - | 2.6% | 42.3% | 55.2% | 0.37 |

**Key Observations:**

1. ✅ **NO COLLAPSE!** Model maintained 40-55% SHORT throughout
2. ✅ **Balanced actions** - First time ever with real LONG/SHORT balance
3. ✅ **Sharpe turned positive** from -32 → +29, consistently improving
4. ✅ **Entropy stable** at 0.35-0.55 (not dying like previous experiments)
5. ⚠️ **Still over-trading** - 23-38k trades, avg hold 3-4 bars (15-20 min)

**Conclusion:** Carry cost + higher entropy WORKS for preventing collapse. But 0.002%/bar carry cost isn't enough to reduce trading frequency.

---

### Experiment 8: Higher Carry Cost (5× Exp 7)

**Config:** `carry_cost=0.0001` (0.01%/bar = 2.9%/day)

| Step | Sharpe | Trades | LONG % | SHORT % |
|------|--------|--------|--------|---------|
| 20k | -72.29 | 43,852 | 64.7% | 20.6% |
| **81k** | **15.44** | 17,689 | 76.4% | 23.5% |
| 120k | 11.14 | 20,994 | 69.0% | 30.7% |
| 180k | 1.77 | 27,198 | 54.6% | 45.0% |

**Result:** Early stopped at 180k. LONG-biased (76/24). Carry cost too high.

---

### Experiment 9: Middle Carry Cost (2.5× Exp 7) ✅ BEST

**Config:** `carry_cost=0.00005` (0.005%/bar = 1.44%/day)

| Step | Sharpe | Net Return | LONG % | SHORT % | Trades |
|------|--------|------------|--------|---------|--------|
| 20k | -8.76 | -433% | 70.5% | 26.6% | 23k |
| 100k | 18.70 | +910% | 65.7% | 33.9% | 20k |
| 161k | 11.65 | +574% | **50.0%** | **49.5%** | 24k |
| 180k | 25.35 | +1224% | 45.0% | 54.5% | 22k |
| 260k | 31.38 | +1502% | 52.3% | 47.6% | 21k |
| 280k | 33.60 | +1597% | 54.0% | 46.0% | 19k |
| **301k** | **35.71** | **+1694%** | **52.6%** | **47.4%** | **19k** |

**Key Observations:**

1. ✅ **Best balance ever:** 53% LONG / 47% SHORT (nearly perfect!)
2. ✅ **Consistent improvement:** Sharpe went 18→25→31→33→35
3. ✅ **Reduced trading:** 23k → 19k trades
4. ✅ **Training completed** without early stopping
5. ⚠️ **FLAT never used** (0% throughout)
6. ⚠️ **Entropy dropped to 0.16** at end (need to verify on test set)

**Best Checkpoint:** `checkpoint_301056_best.pt` (Sharpe 35.71, 53/47 balance)

**Saved to:** `C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1\L2V1 EXPERIMENT 9 MODEL\`

---

## Recommendations for Future Work

1. **Ensemble Methods:** Train multiple agents on different time periods
2. **Regime Detection:** Use regime classifier to switch between LONG/SHORT biased models
3. **Imitation Learning:** Pre-train on profitable trading sequences
4. **Different Reward:** Try direct PnL instead of Sortino
5. **Curriculum Learning:** Start with synthetic trending data

---

## Commands Archive

### Fresh Vast.ai Setup

```bash
git clone https://github.com/nimallansa937/HIMARI-OPUS-TACTICAL-L2-V1.0.git
pip install torch gdown numpy pandas scikit-learn scipy tqdm loguru -q
gdown --id 1_YMRsTCHjfsrqf63RI3xQ4jpehIsEaNW -O "LAYER 2 TACTICAL HIMARI OPUS/data/btc_5min_2020_2024.pkl"
```

### Kill Running Training

```bash
pkill -9 -f python
```

### Download Checkpoints Before Destroying Instance

```bash
cd /workspace/HIMARI-OPUS-TACTICAL-L2-V1.0/"LAYER 2 TACTICAL HIMARI OPUS"
tar -czvf checkpoints.tar.gz ./output/
# Then download via Jupyter file browser
```
