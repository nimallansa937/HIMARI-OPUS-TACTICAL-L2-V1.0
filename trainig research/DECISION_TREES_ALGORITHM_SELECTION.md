# Decision Trees: Algorithm & Architecture Selection for DRL Trading

**Use these diagrams to make real-time decisions during implementation.**

---

## 1. ALGORITHM SELECTION FLOW

```
START: Want to build a trading agent?
│
├─ Do you have continuous action space?
│  (e.g., position sizing -1 to +1)
│  │
│  ├─ YES → TD3 or SAC
│  │        (2-3x Sharpe vs PPO)
│  │        │
│  │        ├─ Want max exploration + entropy? → SAC
│  │        │                                      (newer, 2024+)
│  │        │
│  │        └─ Want stability + proven? → TD3
│  │                                      (2017+, battle-tested)
│  │
│  └─ NO (Discrete: BUY/HOLD/SELL) → PPO
│                                      (0.8-1.3 Sharpe)
│
└─ Special case: Ensemble?
   (Combining multiple agents)
   │
   ├─ If high compute budget → PPO + TD3 + SAC (voting)
   └─ If standard compute → PPO + TD3 (majority vote)
```

---

## 2. ACTION SPACE SELECTION

```
Your market: ________

    BTC/USD 5min?
    │
    ├─ YES → Continuous [-1, 1]
    │        (position sizing 0.1, 0.5, 0.9 possible)
    │        → Use TD3/SAC
    │
    ├─ Stock daily?
    │  │
    │  ├─ Need portfolio weighting? → Continuous
    │  │                             → Use DDPG/TD3
    │  │
    │  └─ Simple long/short? → Discrete (3 actions)
    │                         → Use PPO
    │
    └─ News-driven trading?
       (Telegraph, regulatory events)
       │
       └─ Discrete (4-5 actions)
          → Use PPO
```

---

## 3. REWARD FUNCTION CHECKLIST

```
┌─────────────────────────────────────────────────────┐
│ STEP 1: Choose Base Metric                          │
├─────────────────────────────────────────────────────┤
│ ✅ Recommended: Daily Returns % (PnL / initial_capital)
│ ✅ Alternative: Log Returns (better for vol)
│ ❌ Wrong: Absolute PnL ($) → scaling issues
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ STEP 2: Subtract Transaction Costs                  │
├─────────────────────────────────────────────────────┤
│ Formula: cost = abs(action_change) * (maker + taker) * price
│ Binance BTC: 0.0002 (maker) + 0.0004 (taker)
│ Your venue? Check exact fee structure
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ STEP 3: Add Risk Penalty (Optional but Effective)   │
├─────────────────────────────────────────────────────┤
│ Formula: -0.001 * rolling_volatility * position_size
│ Why: Prevents portfolio blow-up in 2022-style crisis
│ Tune coefficient 0.0005 - 0.002
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ STEP 4: Normalize Rewards                           │
├─────────────────────────────────────────────────────┤
│ Per-batch: rewards = (rewards - mean) / (std + eps)
│ NOT global: prevents exploding gradient
│ Critical for convergence
└─────────────────────────────────────────────────────┘
                          │
                          ▼
                    ✅ DONE
```

---

## 4. ENVIRONMENT DESIGN DECISION TREE

```
Do you have your own OHLCV data?

    NO → Get it:
    │    - CCXT (unified exchange API)
    │    - Binance REST (https://api.binance.com)
    │    - Kraken API
    │    - Any exchange with historical data
    │
    YES → Frequency?
          │
          ├─ 5-60 min (intraday)
          │  │
          │  ├─ Transformer useful? → YES, use it
          │  │  (captures intraday patterns, trend clusters)
          │  │  Architecture: 2-3 layers, 4-8 heads, 200-bar lookback
          │  │
          │  └─ Just use MLP? → Faster, 90% of performance
          │                     Start here, graduate to Transformer
          │
          ├─ Daily (swing trading)
          │  │
          │  └─ MLP sufficient
          │     (daily patterns less complex)
          │
          └─ Hourly (medium-term)
             │
             └─ MLP or shallow Transformer
                (less lookback needed, 48-96 bars)
```

---

## 5. VALIDATION STRATEGY DECISION

```
You have trained an agent. Is it good?

    Step 1: On training data (2020-2022)
    │       Sharpe = 1.5, Drawdown = 15%
    │       (OK if market was bull)
    │
    Step 2: On validation data (2023 Q1-Q2, unseen)
    │       │
    │       ├─ Sharpe ≥ 1.0? → ✅ Agent generalized
    │       │  │
    │       │  └─ Proceed to test
    │       │
    │       └─ Sharpe < 0.8? → ❌ Overfit detected
    │          │
    │          ├─ Issue 1: Reward function (prob 40%)
    │          │           → Costs too low, or PnL scaled wrong
    │          │
    │          ├─ Issue 2: Entropy collapsed (prob 30%)
    │          │           → Always LONG (bull market artifact)
    │          │           → Fix: Start entropy 0.3, decay slower
    │          │
    │          └─ Issue 3: Algorithm choice (prob 20%)
    │                      → Try TD3 instead of PPO
    │
    Step 3: On test data (2023 Q3-2024, hold-out)
            │
            ├─ Sharpe ≥ 0.8? → ✅ Production ready
            │
            └─ Sharpe < 0.5? → ❌ Failed on real distribution
                              → Go back to Step 2, debug
```

---

## 6. POLICY COLLAPSE DIAGNOSIS TREE

```
Your agent is always doing the same action (e.g., LONG every step).

Does this happen early (< 50k steps)?
│
├─ YES → Entropy Too Low
│        │
│        ├─ Check: ent_coef in logs
│        │ If ent_coef < 0.05 by step 10k: TOO FAST
│        │
│        └─ Fix: Start with ent_coef = 0.3, decay slower
│                entropy_coeff = 0.3 * (0.01/0.3)^(t/T)
│
└─ NO (Late collapse, after 100k steps) → Reward Function
                                          │
                                          ├─ Check: action_change penalty
                                          │ If not penalizing switches: costs too low
                                          │
                                          ├─ Check: position_size
                                          │ If no cap: agent goes 2.0x leverage, not sellable
                                          │
                                          └─ Fix: Increase transaction_cost coefficient
                                                  Add hard position bounds [-1, 1]
```

---

## 7. HYPERPARAMETER TUNING PRIORITY

```
You have 8 hours to tune one model. What do you change?

Rank      Parameter                Impact    Tune Range
─────────────────────────────────────────────────────
1st       Reward function          ⭐⭐⭐⭐⭐  Cost: 0.0001-0.001
          - cost_coeff             (70%)      Vol: 0.0005-0.002

2nd       Entropy schedule         ⭐⭐⭐⭐   Start: 0.1-0.3
          - initial ent_coef       (20%)      End: 0.001-0.01

3rd       Network size             ⭐⭐⭐    Hidden: 64-256
          - hidden_dim             (5%)       Heads: 4-8

4th       Learning rate            ⭐⭐      LR: 1e-4 to 5e-4
          (PPO default good)       (3%)

5th       Batch size               ⭐        Size: 32-256
          (stable across range)    (2%)

If you've tuned 1-3 and still failing: problem is NOT hyperparams,
problem is environment design or data splits.
```

---

## 8. WHEN TO SWITCH ALGORITHMS

```
Your current algorithm:   PPO
Validation Sharpe:        0.9
Time invested:            1 week

Do you switch to TD3?

    Metric 1: Is action space continuous?
    │
    ├─ YES → SWITCH (TD3 likely 2x Sharpe)
    │
    └─ NO (discrete) → STAY with PPO (TD3 not applicable)

    Metric 2: Do you have compute budget?
    │
    ├─ Limited (laptop) → STAY (TD3 slower)
    │
    └─ Ample (GPU) → SWITCH

    Metric 3: Do you need to deploy fast?
    │
    ├─ Yes (production deadline) → STAY (PPO trained)
    │
    └─ No (research) → SWITCH (empirically better)

DECISION:
  Continuous + GPU + Time? → SWITCH to TD3
  Discrete? → STAY with PPO
  Discrete + Strong GPU? → Try SAC (newest, harder to tune)
```

---

## 9. TRANSACTION COST REALISM CHECKLIST

```
Your backtest Sharpe: 1.8
Expected live Sharpe: ???

Are you modeling:

    [ ] Maker fee
        ├─ Binance: 0.0002 (BTC/USDT)
        └─ Your venue: ________

    [ ] Taker fee
        ├─ Binance: 0.0004
        └─ Your venue: ________

    [ ] Slippage (market impact)
        ├─ Small orders: 0.1-0.5%
        ├─ Medium orders: 0.5-2%
        └─ Your venue/size: ________

    [ ] Funding rate (crypto)
        ├─ Binance: 0.01% every 8h typical
        └─ Your venue: ________

    [ ] Network fee (blockchain)
        ├─ Minimal for spot trading
        └─ Critical for on-chain execution

Estimate true cost per round-trip trade:
  Maker + Taker + Slippage = ______

Impact on Sharpe:
  If you didn't model cost: actual ≈ 70% of backtest
  If you modeled fixed cost: actual ≈ 85% of backtest
  If you modeled realistic: actual ≈ 95% of backtest
```

---

## 10. PRODUCTION DEPLOYMENT DECISION

```
You have a trained model.

Is it ready for live trading?

Check-list:
  [ ] Train Sharpe ≥ 1.5
  [ ] Validation Sharpe ≥ 1.0 (on unseen 2023)
  [ ] Test Sharpe ≥ 0.8 (on hold-out 2023-2024)
  [ ] No OOD signals (val Sharpe > 0.8 * train Sharpe)
  [ ] Win rate 40-60% (not skewed to long-only)
  [ ] Profit factor > 1.3
  [ ] Max drawdown < 25%
  [ ] No look-ahead bias verified
  [ ] Policy deterministic in deployment (argmax, not sample)
  [ ] Position size limits enforced ([-1, 1])
  [ ] Cost model verified against live fees

If all checked: ✅ DEPLOY with position sizing 10-20% of account
If 3+ unchecked: ❌ DO NOT DEPLOY
                    Return to Step 2: Reward function tuning

Monitoring (first month):
  - Daily Sharpe: target > 0.7, alert if < 0.2
  - Max DD: alert if > 40%
  - Refit monthly if live Sharpe < 0.8 for 2+ weeks
```

---

## QUICK REFERENCE: 3-MINUTE DECISION MATRIX

| Scenario | Algorithm | Network | Action Space | Note |
|----------|-----------|---------|--------------|------|
| BTC 5min, new to DRL | PPO | MLP (128) | Discrete (3) | Fast learning |
| BTC 5min, experienced | TD3 | Transformer | Continuous | 2x Sharpe |
| Stock daily | PPO | MLP (64) | Discrete (3) | Simpler |
| Portfolio (multi-asset) | DDPG | MLP (256) | Continuous | Complex |
| News-driven | PPO | MLP (128) | Discrete (4-5) | Event-based |
| High-frequency (1min) | TD3 | Transformer | Continuous | Latency critical |
| Budget constraints | PPO | MLP (64) | Discrete | Minimize compute |
| Optimal performance | TD3 | Transformer | Continuous | CPU/GPU needed |

---

**Navigation Hint:** Print this document. Reference during implementation. When stuck, follow the decision tree for your situation.

**Last Updated:** January 12, 2026  
**Status:** Implementation-ready
