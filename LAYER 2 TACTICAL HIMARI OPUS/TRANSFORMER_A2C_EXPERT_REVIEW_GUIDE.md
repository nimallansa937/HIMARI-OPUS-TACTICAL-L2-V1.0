# Expert Review: HIMARI Layer 2 Transformer-A2C Issues

## Executive Summary

Your diagnosis is accurate. You experienced a **cascading failure pattern** where data leakage → false confidence → masked exploration issues. Now that you have clean data and metrics, the fundamental problem is revealed: **entropy collapse in a sparse reward regime**.

---

## Category 1: Data Integrity Issues (Look-Ahead Bias)

### Code Verification

Looking at `transformer_a2c_env.py:144`:

```python
# Current implementation - BUGGY
if self._current_step < self.num_samples:
    new_obs = self.data[self._current_step]  # Line 144
```

**The Bug:** After `self._current_step += 1` on line 135, the code fetches `self.data[self._current_step]` which is the **current bar's features** — the one the model should be predicting. This creates subtle look-ahead bias.

### The Fix

```python
# Corrected - use previous step's observation
new_obs = self.data[self._current_step - 1] if self._current_step > 0 else self.data[0]
```

---

## Category 2: Reward Calculation Bugs

### Issues Found

1. **Sortino Edge Case** in `reward_shaping.py:101-103`:

   ```python
   if len(downside_returns) == 0:
       return 10.0 if mean_return > mar else 0.0  # BAD: arbitrary high value
   ```

2. **Solution:** Your `SimpleSortinoReward` class in `sortino_reward.py` already has improvements:
   - Returns 0.0 when insufficient downside samples (not 10.0)
   - Logs warnings for suspicious Sharpe > 5.0
   - No hardcoded clipping

> [!IMPORTANT]
> Ensure you're using `SimpleSortinoReward` from `sortino_reward.py`, NOT `RewardShaper` from `reward_shaping.py`.

---

## Category 3: Exploration-Exploitation Failure (Policy Collapse)

### Root Cause Analysis

Current config (`transformer_a2c.py:60`):

```python
entropy_coef: float = 0.01    # TOO LOW - causes collapse
```

### Why FLAT Dominates

| Step | What Happens |
|------|--------------|
| 1 | Initial policy outputs ~equal probabilities |
| 2 | FLAT always returns 0 (no risk, no reward) |
| 3 | LONG/SHORT returns sparse, mostly noise |
| 4 | FLAT gets near-zero variance gradients |
| 5 | With `entropy_coef=0.01`: insufficient exploration pressure |
| 6 | Model converges to "safe" FLAT |
| 7 | Critic learns V(s)≈0 everywhere |

This is **bootstrap failure** — the policy never explores enough to discover profitable actions.

---

## Recommended Solutions (Prioritized)

### Priority 1: Increase Entropy Coefficient (Immediate)

Change `transformer_a2c.py:60`:

```python
entropy_coef: float = 0.07    # Was 0.01, increase 7x
```

**Evidence:**

- Research summary recommends 0.05-0.10 for sparse reward financial RL
- SLR shows non-monotonic performance curve; 0.07 is in the "goldilocks zone"

### Priority 2: Staged Entropy Decay

Implement entropy scheduling in the trainer:

```python
# In TransformerA2CTrainer.update()
progress = self.global_step / self.config.max_steps
entropy_coef = self.config.entropy_coef * (1 - 0.7 * progress)
# Decays from 0.07 → 0.021
```

### Priority 3: Directional Alignment Bonus (Optional)

Add to `sortino_reward.py`:

```python
# Add directional alignment bonus (dense signal)
if action == 1 and market_return > 0:  # LONG + up market
    reward += 0.1 * self.scale * abs(market_return)
elif action == 2 and market_return < 0:  # SHORT + down market
    reward += 0.1 * self.scale * abs(market_return)
```

### Priority 4: Consider PPO Fallback

If A2C with increased entropy still fails after 3 runs, switch to PPO:

- Sharpe 1.2695 vs A2C "moderate"
- Lower maximum drawdown (13.44%)
- Better stability through clipped surrogate objective

---

## Algorithm Comparison

| Criterion | A2C (current) | PPO | SAC | TD3 |
|-----------|---------------|-----|-----|-----|
| Sparse reward handling | Poor | Good | Variable | Good |
| Stability | Low | High | Medium | Medium |
| Sample efficiency | Low | Medium | High | High |
| SLR Sharpe | "Moderate" | 1.2695 | 0.4462 | N/A |
| Recommendation | **Fix entropy** | Fallback | Avoid | Avoid |

---

## Implementation Checklist

- [ ] Fix look-ahead bias (`env` line 144): use `self._current_step - 1`
- [ ] Change `entropy_coef`: 0.01 → 0.07
- [ ] Add entropy decay schedule to trainer
- [ ] Verify using `SimpleSortinoReward` (not `RewardShaper`)
- [ ] Add validation logging for action distribution
- [ ] Run 3 training runs with new config
- [ ] If still failing: switch to PPO
- [ ] If PPO failing: implement curriculum learning

---

## Expected Outcomes After Fixes

| Metric | Current | After entropy fix | After full optimization |
|--------|---------|------------------|------------------------|
| Validation Sharpe | 0.0000 | 0.5-0.8 | 0.95-1.05 |
| Action distribution | 100% FLAT | ~40% FLAT, ~30% each L/S | Regime-dependent |
| Returns std | 0.000000 | >0.001 | Normal trading variance |
| Overfitting gap | N/A | <30% | <20% |

> [!TIP]
> The model should start taking LONG/SHORT positions within the first 50k steps once entropy is increased. If it doesn't, the reward signal or environment has other issues.
