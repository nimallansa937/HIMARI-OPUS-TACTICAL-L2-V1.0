# OPUS 2 Failure Mode Catalog

## Purpose

Document every anticipated failure mode and its defense mechanism.
Each component must justify its existence by preventing a specific failure.

---

## FM-1: Cascade Averaging Problem

**Description:** During market crashes, averaging signals leads to "moderate caution" when you need "STOP EVERYTHING"

**Historical Example:**

- Terra/Luna collapse (May 2022)
- All signals showed stress, but weighted average = 0.6
- System stayed 40% invested → lost 35% in 48 hours

**Defense Mechanism:**

- Subsumption architecture (Level 3 Emergency Stop overrides all lower levels)
- Implementation: `tactical_layer.py`

**Test Strategy:**

```python
def test_cascade_averaging_failure():
    signals = {'funding': -0.8, 'oi': -0.7, 'whale': -0.9}
    # Emergency stop should trigger, not weighted average
    assert subsumption_controller.process(signals) == 0.0
```

**Status:** ✅ Implemented

---

## FM-2: False Precision Under Uncertainty

**Description:** System outputs confident predictions when data is unreliable

**Historical Example:**

- FTX collapse (November 2022)
- Models trained on "normal" volatility gave precise predictions
- Precise but completely wrong

**Defense Mechanism:**

- Conformal prediction kill-switch
- If uncertainty > 50%, abstain from trading entirely
- Implementation: `safety/conformal_uncertainty.py`

**Status:** ✅ Implemented

---

## FM-3: Overfitting to Crypto Bull Market

**Description:** Strategy performs well in backtests (2020-2021 bull) but fails in live bear market

**Defense Mechanism:**

- HIFA with mandatory bear market holdout
- Must test on 2022 bear market data
- Must test on 2018 bear market data

**Status:** ✅ Implemented (HIFA)

---

## FM-4: Slippage Underestimation

**Description:** Backtest assumes mid-price fills; live trading gets worse prices

**Historical Example:**

- Backtest Sharpe: 2.1
- Live Sharpe: 0.4
- Difference entirely due to 15bps average slippage

**Defense Mechanism:**

- Real-time bid-ask spread monitoring
- Order book depth checks before trade
- If spread > 10bps OR depth < 10x position size → skip trade

**Status:** 🔴 Not Implemented

---

## FM-5: Data Provenance Failure

**Description:** Upstream API returns garbage data, system trades on it

**Historical Example:**

- Binance API congestion during volatility
- Funding rate stuck at 0.05% for 20 minutes
- System didn't detect staleness

**Defense Mechanism:**

- Data provenance chain with freshness checks
- Each data point tagged with: source, timestamp, staleness_threshold
- If data older than threshold, flag and skip
- Implementation: `srm/data_provenance.py`

**Status:** ✅ Implemented

---

## FM-6: Signal Edge Decay (Silent Failure)

**Description:** Signal edge gradually decays over months, system continues trading on it

**Historical Example:**

- Funding rate arbitrage edge in 2020-2021
- Win rate degraded from 72% → 58% → 52% over 12 months
- System kept trading until Sharpe went negative

**Defense Mechanism:**

- Signal half-life tracking
- Auto-sunset signals when win rate < 52%
- Implementation: `signal_health_monitor.py`

**Status:** ✅ Implemented

---

## FM-7: Correlation Breakdown During Stress

**Description:** Signals that are independent during normal times become correlated during stress

**Historical Example:**

- March 2020 COVID crash
- All signals showed same direction (sell)
- No diversification benefit when needed most

**Defense Mechanism:**

- Correlation breakdown monitoring
- Position size multiplier based on effective signal count
- Implementation: `monitoring/correlation_monitor.py`

**Status:** ✅ Implemented

---

## FM-8: Capacity Ceiling Breach

**Description:** Strategy works at $100K but fails at $10M due to market impact

**Defense Mechanism:**

- Capacity ceiling calculator
- Hard limits on AUM based on market microstructure
- Implementation: `src/core/capacity_ceiling.py`

**Status:** ✅ Implemented

---

## Template for New Failure Modes

```markdown
## FM-X: [Failure Mode Name]
**Description:** [What goes wrong]

**Historical Example:** [Real-world case]

**Defense Mechanism:** [How OPUS 2 prevents this]

**Test Strategy:** [Code to verify defense works]

**Status:** [✅ Implemented | 🟡 In Progress | 🔴 Not Implemented]
```

---

## Review Cadence

- **Weekly:** Check for new failure modes in production logs
- **Monthly:** Run all failure mode tests
- **Quarterly:** Adversarial audit (try to break each defense)

---

## Maintenance

- Each new component must document what failure mode it prevents
- If a component doesn't prevent a failure mode, remove it

---

**Last Updated:** 2026-01-09
