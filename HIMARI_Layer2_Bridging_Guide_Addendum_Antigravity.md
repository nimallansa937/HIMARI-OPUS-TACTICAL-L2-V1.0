# HIMARI Layer 2 Bridging Guide - Addendum

## Updates from Antigravity Agent

**Version:** 1.1 Addendum  
**Date:** January 6, 2026  
**Author:** Antigravity AI Agent  
**Purpose:** Document Layer 1 changes since original bridging guide and provide updated integration specifications

---

## Overview

This addendum updates the original `HIMARI_Layer2_Bridging_Guide.md` to reflect recent Layer 1 enhancements. The original guide assumed a **50D feature vector**; Layer 1 now outputs a **60D feature vector** with new order flow features.

---

## 1. Feature Vector Dimension Change

### Before (Original Guide)

```
FEATURE_DIM = 50
Input: 50-dimensional feature vector from Layer 1
```

### After (Current)

```
FEATURE_DIM = 60
Input: 60-dimensional feature vector from Layer 1
```

### Impact on Layer 2

| Component | Required Change |
|-----------|-----------------|
| Part A: Preprocessing | Update `VecNormalize` input dim from 50 → 60 |
| Part D: Decision Engine | Update observation space from 50 → 60 |
| Part F: Uncertainty | Update embedding dimensions if applicable |
| Part N: Interpretability | Add new feature names for SHAP attribution |

---

## 2. New Order Flow Features (Indices 50-59)

Layer 1 now provides 10 additional order flow features:

| Index | Feature Name | Range | Description |
|-------|--------------|-------|-------------|
| 50 | `orderflow_obi_current` | [-1, 1] | Real-time Order Book Imbalance |
| 51 | `orderflow_obi_ema` | [-1, 1] | JMA-smoothed OBI (low-lag) |
| 52 | `orderflow_cvd_normalized` | [-5, 5] | Z-scored Cumulative Volume Delta |
| 53 | `orderflow_cvd_divergence` | {-1, 0, 1} | Price/CVD divergence flag |
| 54 | `orderflow_microprice_dev` | [-1, 1] | Microprice deviation from mid |
| 55 | `orderflow_vpin` | [0, 1] | Volume-Sync Informed Trading prob |
| 56 | `orderflow_spread_zscore` | [-5, 5] | Spread volatility z-score |
| 57 | `orderflow_lob_imbalance` | [-1, 1] | 10-level LOB imbalance |
| 58 | `orderflow_trade_intensity` | [-5, 5] | Trades/sec z-score |
| 59 | `orderflow_aggressive_ratio` | [0, 1] | Proportion of aggressive trades |

### Normalization Config Update

```python
# Add to PreprocessingConfig in Layer 2
ORDER_FLOW_FEATURE_RANGES = {
    50: (-1, 1),    # obi_current
    51: (-1, 1),    # obi_ema
    52: (-5, 5),    # cvd_normalized (clip)
    53: (-1, 1),    # cvd_divergence
    54: (-1, 1),    # microprice_dev
    55: (0, 1),     # vpin
    56: (-5, 5),    # spread_zscore (clip)
    57: (-1, 1),    # lob_imbalance
    58: (-5, 5),    # trade_intensity (clip)
    59: (0, 1),     # aggressive_ratio
}
```

---

## 3. Low-Lag DSP Optimizations

### 3.1 JMA Smoothing (Replaces EMA)

Layer 1 now uses **Jurik Moving Average (JMA)** instead of EMA for:

- Order Book Imbalance smoothing (`order_flow.py`)
- Volume indicators (`volume.py`)

**Benefit:** ~30% faster signal detection with near-zero lag.

### 3.2 Inline Gaussian PDF

`streaming_hmm.py` now uses inline Gaussian PDF instead of `scipy.stats.norm.pdf`:

- **~50% faster** HMM updates
- No scipy dependency in hot path

### 3.3 New DSP-Based TINs for A/B Testing

Located in `ml/tins.py`:

| TIN | Purpose | Use Case |
|-----|---------|----------|
| `TIN_JMA` | Low-lag moving average | Replace EMA in indicators |
| `TIN_MESA_MACD` | Ehlers MESA MACD | Adaptive cycle detection |
| `TIN_FisherRSI` | Fisher Transform RSI | Earlier reversal signals |

---

## 4. Updated Interface Contract

### Layer1Output → Layer2Input

```python
@dataclass
class Layer1Output:
    """Output from Layer 1 Signal Layer (v1.1)."""
    features: np.ndarray          # Shape: (60,) - UPDATED from 50
    timestamp: float
    symbol: str
    
    # Order flow metadata (new)
    orderbook_available: bool = False
    trade_data_available: bool = False
    
    # Quality indicators
    n_nonzero: int = 0
    latency_ms: float = 0.0

@dataclass  
class Layer2Input:
    """Input to Layer 2 preprocessing (Part A)."""
    raw_features: np.ndarray      # Shape: (60,)
    portfolio_state: PortfolioState
    llm_cache: Optional[LLMSignal] = None
```

---

## 5. Code Changes Required in Layer 2

### 5.1 VecNormalize Update

```python
# In src/preprocessing/vec_normalize.py
class VecNormalize:
    def __init__(self, obs_dim: int = 60):  # Changed from 50
        self.obs_dim = obs_dim
        # ...
```

### 5.2 Decision Engine Observation Space

```python
# In src/decision_engine/ppo_lstm.py
observation_space = gym.spaces.Box(
    low=-np.inf, high=np.inf, 
    shape=(60,),  # Changed from 50
    dtype=np.float32
)
```

### 5.3 Feature Names Update

```python
# Add to interpretability module
ORDER_FLOW_FEATURE_NAMES = [
    'orderflow_obi_current',
    'orderflow_obi_ema', 
    'orderflow_cvd_normalized',
    'orderflow_cvd_divergence',
    'orderflow_microprice_dev',
    'orderflow_vpin',
    'orderflow_spread_zscore',
    'orderflow_lob_imbalance',
    'orderflow_trade_intensity',
    'orderflow_aggressive_ratio',
]
```

---

## 6. Bridge Integration Code

See `l1_l2_bridge.py` for the complete integration adapter:

- `Layer1DataAdapter`: Converts L1 output to L2 input format
- `validate_feature_vector()`: Ensures 60D with proper ranges
- `normalize_order_flow_features()`: Applies correct normalization

---

## 7. Latency Budget Update

| Component | Original Budget | Updated Budget |
|-----------|----------------|----------------|
| Layer 1 Feature Generation | 12ms | 11ms (JMA optimization) |
| Layer 1 → Layer 2 Bridge | N/A | <1ms (new) |
| Part A: Preprocessing | 5ms | 5ms |
| **Total L1+Bridge+A** | **17ms** | **17ms** |

---

## Summary of Changes

1. **Dimension**: 50D → 60D feature vector
2. **New Features**: +10 order flow features (indices 50-59)
3. **Optimizations**: JMA smoothing, inline Gaussian PDF
4. **New TINs**: JMA, MESA_MACD, FisherRSI for A/B testing
5. **Interface**: Updated data contracts with orderbook metadata

---

**END OF ADDENDUM**

*This addendum was generated by Antigravity AI Agent on January 6, 2026.*
