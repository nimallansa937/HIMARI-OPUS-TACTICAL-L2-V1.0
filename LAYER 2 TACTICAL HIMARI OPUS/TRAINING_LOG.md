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

**Next Experiment:** PPO + Carry Cost + Bear Market Validation

- 0.001% per bar carry cost for non-FLAT
- 40/40/20 split (includes 2022 crash)
- Target: Model learns to SHORT during downtrends

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
