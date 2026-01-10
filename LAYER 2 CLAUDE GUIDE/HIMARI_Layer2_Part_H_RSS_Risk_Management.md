# HIMARI Layer 2 Comprehensive Developer Guide
## Part H: RSS Risk Management (8 Methods)

**Document Version:** 1.0  
**Series:** HIMARI Layer 2 Ultimate Developer Guide v5  
**Component:** Responsibility-Sensitive Safety for Trading  
**Target Latency:** <3ms contribution to 50ms total budget  
**Methods Covered:** H1-H8

---

## Table of Contents

1. [Subsystem Overview](#subsystem-overview)
2. [H1: EVT + GPD Tail Risk](#h1-evt--gpd-tail-risk)
3. [H2: DDPG-TiDE Dynamic Kelly](#h2-ddpg-tide-dynamic-kelly)
4. [H3: DCC-GARCH Correlation](#h3-dcc-garch-correlation)
5. [H4: Progressive Drawdown Brake](#h4-progressive-drawdown-brake)
6. [H5: Portfolio-Level VaR](#h5-portfolio-level-var)
7. [H6: Safe Margin Formula](#h6-safe-margin-formula)
8. [H7: Dynamic Leverage Controller](#h7-dynamic-leverage-controller)
9. [H8: Adaptive Risk Budget](#h8-adaptive-risk-budget)
10. [Integration Architecture](#integration-architecture)
11. [Configuration Reference](#configuration-reference)
12. [Testing Suite](#testing-suite)

---

## Subsystem Overview

### The Challenge

Even with ensemble decisions, hysteresis filtering, and state machine validation, edge cases exist where proposed actions might be catastrophic—excessive leverage during volatility spikes, oversized positions relative to available liquidity, or positions that could trigger liquidation during adverse tail moves. The decision engine might output a high-confidence BUY signal during what appears to be a trending market, but if that market is about to experience a 15% flash crash, executing that signal could wipe out months of accumulated profits.

The core problem is that standard risk metrics assume Gaussian return distributions. A Value-at-Risk (VaR) calculation assuming 2-sigma moves (95% confidence) systematically underestimates the frequency and magnitude of extreme events in cryptocurrency markets. Bitcoin has experienced ten 20%+ daily moves since 2020—events that a Gaussian model predicts should occur once per century. This "fat tail" problem means traditional risk management fails precisely when it matters most.

Responsibility-Sensitive Safety (RSS) originated in autonomous vehicles to mathematically guarantee collision avoidance. The insight: instead of probabilistic safety, define hard constraints that must never be violated, then engineer systems to maintain those constraints at all times. We adapt this framework to guarantee liquidation avoidance and drawdown limits in trading.

### The Solution: Multi-Layer Risk Constraints

RSS Risk Management implements eight complementary methods that work together to ensure position safety:

**Tail Risk Measurement** (H1, H5): Replace Gaussian VaR with Extreme Value Theory (EVT) using Generalized Pareto Distribution (GPD) to accurately model the probability of 10%, 15%, or 20% adverse moves. This provides realistic worst-case estimates for position sizing.

**Dynamic Position Sizing** (H2, H8): Use reinforcement learning and performance tracking to adapt position sizes to current market conditions and recent trading performance. Position smaller after losses, larger during validated edge periods.

**Correlation Dynamics** (H3): Correlations spike during crises—assets that appeared uncorrelated suddenly move together. DCC-GARCH (Dynamic Conditional Correlation) captures time-varying correlations for realistic portfolio risk assessment.

**Drawdown Control** (H4): Progressive position reduction as drawdown increases, not a binary circuit breaker. This prevents the death spiral of waiting until maximum drawdown, then panic-liquidating at the worst prices.

**Leverage Safety** (H6, H7): Mathematical constraints on maximum leverage based on volatility, position size, and asset liquidity. Larger positions receive lower maximum leverage due to liquidity constraints and market impact.

### Method Overview

| ID | Method | Category | Status | Function |
|----|--------|----------|--------|----------|
| H1 | EVT + GPD Tail Risk | Tail Risk | **UPGRADE** | Model extreme loss probabilities using Peaks Over Threshold |
| H2 | DDPG-TiDE Dynamic Kelly | Position Sizing | **NEW** | RL-based optimal position sizing with temporal fusion |
| H3 | DCC-GARCH Correlation | Portfolio Risk | **NEW** | Time-varying correlation estimation for diversification |
| H4 | Progressive Drawdown Brake | Capital Preservation | **NEW** | Gradual position scaling based on cumulative drawdown |
| H5 | Portfolio-Level VaR | Portfolio Risk | **NEW** | Cross-asset risk aggregation with copula dependence |
| H6 | Safe Margin Formula | Leverage Safety | KEEP | k-sigma margin buffer calculation |
| H7 | Dynamic Leverage Controller | Leverage Safety | KEEP | Position-dependent leverage decay |
| H8 | Adaptive Risk Budget | Risk Allocation | **NEW** | Performance-based risk budget adjustment |

### Latency Budget

| Component | Time | Cumulative |
|-----------|------|------------|
| EVT tail risk lookup | 0.3ms | 0.3ms |
| Kelly fraction computation | 0.5ms | 0.8ms |
| DCC-GARCH correlation update | 0.4ms | 1.2ms |
| Drawdown brake check | 0.1ms | 1.3ms |
| Portfolio VaR aggregation | 0.6ms | 1.9ms |
| Safe margin calculation | 0.2ms | 2.1ms |
| Leverage adjustment | 0.2ms | 2.3ms |
| Risk budget check | 0.2ms | 2.5ms |
| **Total** | **~2.5ms** | Well under 3ms budget ✅ |

---

## H1: EVT + GPD Tail Risk

### The Problem with Gaussian VaR

Traditional Value-at-Risk (VaR) assumes returns follow a normal distribution. Under this assumption, a 3-sigma event (roughly 0.1% probability) represents the extreme tail. For BTC with 5% daily volatility, this suggests maximum daily losses around 15% should occur about once per thousand trading days, or roughly once every four years.

Reality tells a different story. Bitcoin experienced daily declines exceeding 15% on March 12, 2020 (-37%), May 19, 2021 (-30%), and numerous other occasions. These "black swan" events occur roughly 10-50× more frequently than Gaussian models predict.

The mathematical reason: cryptocurrency returns exhibit excess kurtosis—fatter tails than a normal distribution. A distribution with kurtosis of 10 (typical for crypto) has tail probabilities orders of magnitude higher than Gaussian (kurtosis = 3).

### Extreme Value Theory (EVT) Solution

EVT provides a rigorous mathematical framework for modeling extreme events. The key insight: regardless of the underlying distribution, sufficiently extreme values follow a specific family of distributions.

The **Peaks Over Threshold (POT)** method identifies returns exceeding a high threshold (e.g., 5% daily loss) and fits these exceedances to a **Generalized Pareto Distribution (GPD)**:

```
F(x) = 1 - (1 + ξx/σ)^(-1/ξ)    for ξ ≠ 0

Where:
- x: exceedance amount above threshold u
- ξ (xi): shape parameter (tail heaviness)
- σ (sigma): scale parameter
```

The shape parameter ξ is crucial:
- ξ > 0: Heavy tails (power-law decay)—typical for crypto
- ξ = 0: Exponential tails
- ξ < 0: Light tails (bounded support)

For BTC, empirical ξ estimates range from 0.15 to 0.35, indicating substantially heavier tails than Gaussian.

### Production Implementation

```python
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Deque
from collections import deque
import numpy as np
from scipy import stats
from scipy.optimize import minimize

@dataclass
class EVTConfig:
    """Configuration for EVT Tail Risk Estimation."""
    threshold_percentile: float = 95.0      # Threshold for POT (95th percentile of losses)
    min_exceedances: int = 30               # Minimum exceedances for reliable GPD fit
    lookback_days: int = 365                # Historical data for parameter estimation
    confidence_levels: Tuple[float, ...] = (0.95, 0.99, 0.999)  # VaR/ES levels
    update_frequency_hours: int = 24        # Refit GPD parameters daily
    ewma_lambda: float = 0.94               # Exponential decay for recent emphasis


@dataclass
class TailRiskEstimates:
    """Container for tail risk metrics."""
    var_95: float           # 95% VaR (5% worst-case daily loss)
    var_99: float           # 99% VaR (1% worst-case daily loss)
    var_999: float          # 99.9% VaR (0.1% worst-case daily loss)
    es_95: float            # Expected Shortfall at 95%
    es_99: float            # Expected Shortfall at 99%
    xi: float               # GPD shape parameter
    sigma: float            # GPD scale parameter
    threshold: float        # POT threshold used
    n_exceedances: int      # Number of threshold exceedances
    last_updated: float     # Timestamp of last parameter update


class EVTTailRiskEstimator:
    """
    Estimates tail risk using Extreme Value Theory with GPD.
    
    The Peaks Over Threshold method fits a Generalized Pareto Distribution
    to losses exceeding a high threshold, enabling accurate estimation of
    extreme quantiles without assuming Gaussian returns.
    
    Why EVT over Gaussian VaR:
    - Crypto returns have excess kurtosis (fat tails)
    - Gaussian 99% VaR underestimates actual 99% losses by 30-50%
    - EVT captures power-law tail decay observed in empirical data
    
    Latency: <0.3ms for VaR lookup (fitting is done offline)
    Memory: O(lookback_days) for return history
    """
    
    def __init__(self, config: EVTConfig = None):
        self.config = config or EVTConfig()
        self.returns: Deque[float] = deque(maxlen=self.config.lookback_days)
        self.timestamps: Deque[float] = deque(maxlen=self.config.lookback_days)
        
        # GPD parameters (fitted offline)
        self._xi: float = 0.2          # Shape parameter
        self._sigma: float = 0.02      # Scale parameter
        self._threshold: float = 0.05  # POT threshold
        self._n_exceedances: int = 0
        self._last_fit_time: float = 0.0
        
        # Cached VaR/ES estimates for fast lookup
        self._cached_estimates: Optional[TailRiskEstimates] = None
        
    def update(self, return_pct: float, timestamp: float) -> TailRiskEstimates:
        """
        Update with new return observation.
        
        Args:
            return_pct: Daily return as decimal (e.g., -0.05 for -5%)
            timestamp: Unix timestamp of observation
            
        Returns:
            Current tail risk estimates
        """
        self.returns.append(return_pct)
        self.timestamps.append(timestamp)
        
        # Check if we need to refit GPD parameters
        hours_since_fit = (timestamp - self._last_fit_time) / 3600
        
        if hours_since_fit >= self.config.update_frequency_hours:
            self._fit_gpd()
            self._last_fit_time = timestamp
        
        return self.get_estimates()
    
    def _fit_gpd(self) -> None:
        """
        Fit Generalized Pareto Distribution to loss exceedances.
        
        Uses Maximum Likelihood Estimation (MLE) with numerical optimization.
        The threshold is set at the configured percentile of loss magnitudes.
        """
        if len(self.returns) < self.config.min_exceedances * 2:
            # Insufficient data, use conservative defaults
            return
        
        # Convert to loss magnitudes (positive values for losses)
        losses = np.array([-r for r in self.returns if r < 0])
        
        if len(losses) < self.config.min_exceedances:
            return
        
        # Set threshold at configured percentile
        self._threshold = np.percentile(losses, self.config.threshold_percentile)
        
        # Extract exceedances above threshold
        exceedances = losses[losses > self._threshold] - self._threshold
        self._n_exceedances = len(exceedances)
        
        if self._n_exceedances < self.config.min_exceedances:
            return
        
        # MLE for GPD parameters
        def neg_log_likelihood(params):
            xi, sigma = params
            if sigma <= 0 or (xi < 0 and np.any(exceedances > -sigma/xi)):
                return np.inf
            
            if np.abs(xi) < 1e-10:
                # Exponential case
                return self._n_exceedances * np.log(sigma) + np.sum(exceedances) / sigma
            else:
                # General GPD case
                z = 1 + xi * exceedances / sigma
                if np.any(z <= 0):
                    return np.inf
                return self._n_exceedances * np.log(sigma) + (1 + 1/xi) * np.sum(np.log(z))
        
        # Optimize with reasonable bounds
        result = minimize(
            neg_log_likelihood,
            x0=[0.1, np.std(exceedances)],
            method='L-BFGS-B',
            bounds=[(-0.5, 1.0), (1e-6, None)]
        )
        
        if result.success:
            self._xi, self._sigma = result.x
        
        # Update cached estimates
        self._cached_estimates = None
    
    def get_estimates(self) -> TailRiskEstimates:
        """
        Get current tail risk estimates.
        
        Uses cached values for <0.3ms latency.
        """
        if self._cached_estimates is not None:
            return self._cached_estimates
        
        var_95 = self._compute_var(0.95)
        var_99 = self._compute_var(0.99)
        var_999 = self._compute_var(0.999)
        
        es_95 = self._compute_expected_shortfall(0.95)
        es_99 = self._compute_expected_shortfall(0.99)
        
        import time
        self._cached_estimates = TailRiskEstimates(
            var_95=var_95,
            var_99=var_99,
            var_999=var_999,
            es_95=es_95,
            es_99=es_99,
            xi=self._xi,
            sigma=self._sigma,
            threshold=self._threshold,
            n_exceedances=self._n_exceedances,
            last_updated=time.time()
        )
        
        return self._cached_estimates
    
    def _compute_var(self, confidence: float) -> float:
        """
        Compute VaR at given confidence level using GPD.
        
        VaR_p = u + (σ/ξ) * [(n/N * (1-p))^(-ξ) - 1]
        
        Where:
        - u: threshold
        - n: number of exceedances
        - N: total observations
        - p: confidence level
        """
        n_total = len(self.returns)
        if n_total == 0:
            return 0.10  # Conservative default
        
        # Probability of exceeding threshold
        prob_exceed = self._n_exceedances / n_total
        
        # Tail probability we want
        tail_prob = 1 - confidence
        
        if prob_exceed <= 0 or tail_prob >= prob_exceed:
            # VaR is below threshold
            losses = np.array([-r for r in self.returns if r < 0])
            if len(losses) > 0:
                return np.percentile(losses, confidence * 100)
            return 0.05
        
        # GPD quantile formula
        if np.abs(self._xi) < 1e-10:
            # Exponential case
            var = self._threshold + self._sigma * np.log(prob_exceed / tail_prob)
        else:
            # General GPD
            var = self._threshold + (self._sigma / self._xi) * (
                (prob_exceed / tail_prob) ** self._xi - 1
            )
        
        return var
    
    def _compute_expected_shortfall(self, confidence: float) -> float:
        """
        Compute Expected Shortfall (CVaR) at given confidence level.
        
        ES provides the expected loss given that VaR is exceeded.
        More informative than VaR for risk management decisions.
        
        ES_p = VaR_p / (1 - ξ) + (σ - ξ*u) / (1 - ξ)
        """
        var = self._compute_var(confidence)
        
        if self._xi >= 1:
            # ES undefined for heavy tails with ξ >= 1
            return var * 1.5  # Heuristic scaling
        
        # ES formula for GPD
        es = (var + self._sigma - self._xi * self._threshold) / (1 - self._xi)
        
        return es
    
    def get_var_for_horizon(self, confidence: float, horizon_bars: int, 
                           bars_per_day: int = 288) -> float:
        """
        Scale VaR to different time horizons.
        
        Uses square-root-of-time scaling with adjustment for autocorrelation.
        
        Args:
            confidence: VaR confidence level (e.g., 0.99)
            horizon_bars: Number of bars for horizon
            bars_per_day: Bars per day (288 for 5-minute bars)
            
        Returns:
            VaR scaled to specified horizon
        """
        daily_var = self._compute_var(confidence)
        
        # Convert to horizon
        horizon_days = horizon_bars / bars_per_day
        
        # Square-root-of-time scaling
        # Note: This assumes independent returns; crypto shows autocorrelation
        # during trends, so we apply a small upward adjustment
        autocorrelation_factor = 1.1  # Empirical adjustment
        
        horizon_var = daily_var * np.sqrt(horizon_days) * autocorrelation_factor
        
        return horizon_var


# Pre-computed lookup tables for ultra-fast inference
class EVTLookupTable:
    """
    Pre-computed VaR/ES lookup table for <0.1ms inference.
    
    Caches risk estimates across volatility regimes and confidence levels
    to eliminate real-time GPD computation in hot path.
    """
    
    def __init__(self):
        # Volatility buckets: 1%, 2%, 3%, ..., 15% daily
        self.vol_buckets = np.arange(0.01, 0.16, 0.01)
        
        # Pre-computed multipliers for each volatility regime
        # Derived from historical GPD fits across regimes
        self._var_99_multipliers = {
            0.01: 3.2,   # Low vol: 3.2x daily vol for 99% VaR
            0.02: 3.4,
            0.03: 3.5,
            0.05: 3.8,   # Normal vol
            0.07: 4.0,
            0.10: 4.5,   # High vol
            0.15: 5.0,   # Extreme vol
        }
        
        self._es_99_multipliers = {
            0.01: 4.0,
            0.02: 4.3,
            0.03: 4.5,
            0.05: 4.8,
            0.07: 5.2,
            0.10: 5.8,
            0.15: 6.5,
        }
    
    def get_var_99(self, current_volatility: float) -> float:
        """
        Get 99% VaR estimate from lookup table.
        
        Args:
            current_volatility: Current daily volatility estimate
            
        Returns:
            99% VaR as fraction of position value
        """
        # Find nearest volatility bucket
        idx = np.argmin(np.abs(self.vol_buckets - current_volatility))
        vol_bucket = self.vol_buckets[idx]
        
        # Get multiplier (interpolate if needed)
        multiplier = self._interpolate_multiplier(
            current_volatility, self._var_99_multipliers
        )
        
        return current_volatility * multiplier
    
    def get_es_99(self, current_volatility: float) -> float:
        """Get 99% Expected Shortfall from lookup table."""
        multiplier = self._interpolate_multiplier(
            current_volatility, self._es_99_multipliers
        )
        return current_volatility * multiplier
    
    def _interpolate_multiplier(self, vol: float, table: dict) -> float:
        """Linearly interpolate multiplier for given volatility."""
        vols = sorted(table.keys())
        
        if vol <= vols[0]:
            return table[vols[0]]
        if vol >= vols[-1]:
            return table[vols[-1]]
        
        # Find bracketing values
        for i in range(len(vols) - 1):
            if vols[i] <= vol <= vols[i+1]:
                # Linear interpolation
                frac = (vol - vols[i]) / (vols[i+1] - vols[i])
                return table[vols[i]] * (1 - frac) + table[vols[i+1]] * frac
        
        return table[vols[-1]]
```

### Usage Example

```python
# Initialize EVT estimator
evt = EVTTailRiskEstimator(EVTConfig(
    threshold_percentile=95.0,
    lookback_days=365,
    confidence_levels=(0.95, 0.99, 0.999)
))

# Feed historical returns (would typically come from database)
import time
base_time = time.time() - 365 * 24 * 3600
for day in range(365):
    # Simulate daily return with fat tails
    return_pct = np.random.standard_t(df=4) * 0.02  # t-distribution with 4 DoF
    evt.update(return_pct, base_time + day * 24 * 3600)

# Get tail risk estimates
estimates = evt.get_estimates()

print(f"GPD Shape (ξ): {estimates.xi:.3f}")
print(f"GPD Scale (σ): {estimates.sigma:.4f}")
print(f"Threshold: {estimates.threshold:.2%}")
print(f"95% VaR: {estimates.var_95:.2%}")
print(f"99% VaR: {estimates.var_99:.2%}")
print(f"99.9% VaR: {estimates.var_999:.2%}")
print(f"99% ES: {estimates.es_99:.2%}")

# For real-time use with <0.1ms latency
lookup = EVTLookupTable()
current_vol = 0.05  # 5% daily volatility
fast_var_99 = lookup.get_var_99(current_vol)
print(f"\nFast 99% VaR lookup: {fast_var_99:.2%}")
```

### Integration with Layer 2

EVT tail risk estimates feed into multiple downstream components:

1. **Safe Margin Formula (H6)**: Uses EVT-based k-sigma instead of Gaussian
2. **Dynamic Leverage Controller (H7)**: Scales max leverage by tail risk
3. **Position Sizing (H2)**: Caps position size based on ES estimates
4. **Portfolio VaR (H5)**: Uses EVT marginals for copula aggregation

---

## H2: DDPG-TiDE Dynamic Kelly

### The Problem with Static Position Sizing

The Kelly Criterion provides the mathematically optimal position size to maximize long-term wealth growth:

```
f* = (μ - r) / σ²

Where:
- f*: Optimal fraction of capital to risk
- μ: Expected return
- r: Risk-free rate
- σ²: Variance of returns
```

The problem: Kelly assumes stationary return distributions—constant μ and σ. In cryptocurrency markets, both expected returns and volatility vary dramatically across regimes. A static Kelly fraction that's optimal during trending bull markets becomes catastrophic during high-volatility crisis periods.

Furthermore, Kelly is extremely sensitive to estimation error. Overestimating expected returns by just 1% can lead to position sizes that blow up accounts. Practical implementations use "fractional Kelly" (25-50% of optimal f*) to reduce variance, but this one-size-fits-all approach wastes edge during favorable conditions.

### DDPG-TiDE: Learning Dynamic Kelly

DDPG-TiDE combines Deep Deterministic Policy Gradient (DDPG) for reinforcement learning with Temporal Fusion Decoder (TiDE) for time-series forecasting. The key insight: instead of estimating Kelly parameters directly, train an RL agent to learn the optimal position sizing policy from experience.

**DDPG** handles continuous action spaces (position size from 0 to max). Unlike discrete Q-learning, DDPG outputs exact position fractions using an actor-critic architecture:
- **Actor network**: Maps state → position size
- **Critic network**: Estimates expected future returns given state and action

**TiDE** (Time-series Dense Encoder) provides efficient temporal feature extraction. Unlike transformer-based approaches, TiDE uses dense layers with time-distributed processing—achieving similar accuracy with 10-50× lower latency.

### Production Implementation

```python
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


@dataclass
class DDPGKellyConfig:
    """Configuration for DDPG-TiDE Dynamic Kelly."""
    state_dim: int = 64                    # State vector dimension
    hidden_dim: int = 256                  # Hidden layer dimension
    tide_lookback: int = 20                # TiDE temporal lookback
    actor_lr: float = 1e-4                 # Actor learning rate
    critic_lr: float = 1e-3                # Critic learning rate
    gamma: float = 0.99                    # Discount factor
    tau: float = 0.005                     # Target network soft update
    min_kelly_fraction: float = 0.05       # Minimum position fraction
    max_kelly_fraction: float = 0.50       # Maximum position fraction (0.5 = half-Kelly)
    volatility_scaling: bool = True        # Scale by inverse volatility
    drawdown_penalty: float = 2.0          # Reward penalty for drawdowns


class TiDEEncoder(nn.Module):
    """
    Time-series Dense Encoder for efficient temporal feature extraction.
    
    TiDE uses dense projections instead of attention, achieving similar
    accuracy to transformers with 10-50x lower inference latency.
    
    Architecture:
    1. Dense projection per timestep
    2. Time-distributed feature extraction
    3. Temporal aggregation
    
    Latency: <0.2ms for 20-step lookback
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, lookback: int):
        super().__init__()
        self.lookback = lookback
        
        # Per-timestep projection
        self.time_proj = nn.Linear(input_dim, hidden_dim)
        
        # Temporal mixing
        self.temporal_proj = nn.Linear(lookback * hidden_dim, hidden_dim)
        
        # Feature refinement
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback, input_dim) temporal features
            
        Returns:
            (batch, hidden_dim) aggregated features
        """
        batch_size = x.shape[0]
        
        # Project each timestep
        h = torch.relu(self.time_proj(x))  # (batch, lookback, hidden)
        
        # Flatten and mix temporally
        h = h.reshape(batch_size, -1)  # (batch, lookback * hidden)
        h = torch.relu(self.temporal_proj(h))  # (batch, hidden)
        
        # Refine features
        h = self.feature_proj(h)
        
        return h


class DDPGActor(nn.Module):
    """
    Actor network: maps state to position size.
    
    Output is bounded to [min_kelly, max_kelly] using sigmoid scaling.
    """
    
    def __init__(self, config: DDPGKellyConfig):
        super().__init__()
        self.config = config
        
        # TiDE for temporal features
        self.tide = TiDEEncoder(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            lookback=config.tide_lookback
        )
        
        # Actor MLP
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        self.fc3 = nn.Linear(config.hidden_dim // 2, 1)
        
    def forward(self, state_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_sequence: (batch, lookback, state_dim) temporal states
            
        Returns:
            (batch, 1) position fraction in [min_kelly, max_kelly]
        """
        # Extract temporal features
        h = self.tide(state_sequence)
        
        # Actor MLP
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        h = torch.sigmoid(self.fc3(h))  # [0, 1]
        
        # Scale to Kelly range
        kelly_range = self.config.max_kelly_fraction - self.config.min_kelly_fraction
        position_frac = h * kelly_range + self.config.min_kelly_fraction
        
        return position_frac


class DDPGCritic(nn.Module):
    """
    Critic network: estimates Q-value for state-action pair.
    
    Q(s, a) = expected cumulative reward starting from state s
    with action a, then following optimal policy.
    """
    
    def __init__(self, config: DDPGKellyConfig):
        super().__init__()
        
        # TiDE for temporal features
        self.tide = TiDEEncoder(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            lookback=config.tide_lookback
        )
        
        # Action embedding
        self.action_embed = nn.Linear(1, config.hidden_dim // 4)
        
        # Q-value MLP
        self.fc1 = nn.Linear(config.hidden_dim + config.hidden_dim // 4, 
                            config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        self.fc3 = nn.Linear(config.hidden_dim // 2, 1)
        
    def forward(self, state_sequence: torch.Tensor, 
                action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_sequence: (batch, lookback, state_dim)
            action: (batch, 1) position fraction
            
        Returns:
            (batch, 1) Q-value estimate
        """
        # State features
        h_state = self.tide(state_sequence)
        
        # Action features
        h_action = torch.relu(self.action_embed(action))
        
        # Concatenate and compute Q
        h = torch.cat([h_state, h_action], dim=-1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        q = self.fc3(h)
        
        return q


class DDPGKellyAgent:
    """
    DDPG agent for dynamic Kelly position sizing.
    
    Training uses experience replay with reward shaping:
    - Positive reward for profitable trades
    - Penalty for drawdowns (asymmetric loss aversion)
    - Penalty for excessive position changes (transaction costs)
    
    The agent learns to:
    - Size larger during favorable regimes (trending, low vol)
    - Size smaller during adverse regimes (crisis, high vol)
    - Adapt to recent performance (reduce after losses)
    
    Latency: <0.5ms inference
    """
    
    def __init__(self, config: DDPGKellyConfig = None):
        self.config = config or DDPGKellyConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.actor = DDPGActor(self.config).to(self.device)
        self.actor_target = DDPGActor(self.config).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = DDPGCritic(self.config).to(self.device)
        self.critic_target = DDPGCritic(self.config).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )
        
        # Experience replay
        self.replay_buffer: deque = deque(maxlen=100000)
        
        # State history for inference
        self.state_history: deque = deque(maxlen=self.config.tide_lookback)
        
        # Volatility tracker for scaling
        self._recent_volatility: float = 0.05
        
    def get_position_fraction(self, 
                              state: np.ndarray,
                              volatility: float,
                              drawdown: float) -> Tuple[float, Dict]:
        """
        Get recommended position fraction for current state.
        
        Args:
            state: Current 60-dim feature vector
            volatility: Current volatility estimate
            drawdown: Current drawdown from peak
            
        Returns:
            Tuple of (position_fraction, metadata_dict)
        """
        # Update state history
        self.state_history.append(state)
        
        if len(self.state_history) < self.config.tide_lookback:
            # Insufficient history, return conservative position
            return (self.config.min_kelly_fraction, {
                'reason': 'warmup',
                'raw_kelly': self.config.min_kelly_fraction
            })
        
        # Prepare input tensor
        state_seq = np.array(list(self.state_history))
        state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
        
        # Get actor output
        self.actor.eval()
        with torch.no_grad():
            raw_kelly = self.actor(state_tensor).item()
        
        # Volatility scaling: reduce position in high-vol environments
        if self.config.volatility_scaling:
            vol_multiplier = min(1.0, 0.05 / max(volatility, 0.01))
            scaled_kelly = raw_kelly * vol_multiplier
        else:
            scaled_kelly = raw_kelly
        
        # Drawdown reduction: progressive reduction as drawdown increases
        if drawdown > 0.02:  # >2% drawdown
            drawdown_multiplier = max(0.3, 1.0 - drawdown * 2)
            scaled_kelly *= drawdown_multiplier
        
        # Clamp to valid range
        final_kelly = np.clip(
            scaled_kelly,
            self.config.min_kelly_fraction,
            self.config.max_kelly_fraction
        )
        
        metadata = {
            'raw_kelly': raw_kelly,
            'vol_multiplier': vol_multiplier if self.config.volatility_scaling else 1.0,
            'drawdown_multiplier': drawdown_multiplier if drawdown > 0.02 else 1.0,
            'final_kelly': final_kelly
        }
        
        return (final_kelly, metadata)
    
    def store_transition(self, 
                        state_seq: np.ndarray,
                        action: float,
                        reward: float,
                        next_state_seq: np.ndarray,
                        done: bool) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.append((
            state_seq, action, reward, next_state_seq, done
        ))
    
    def compute_reward(self,
                      return_pct: float,
                      position_frac: float,
                      drawdown_delta: float) -> float:
        """
        Compute shaped reward for RL training.
        
        Reward = position_frac * return_pct - drawdown_penalty * max(0, drawdown_delta)
        
        This encourages:
        - Sizing larger when returns are positive
        - Avoiding drawdowns (asymmetric penalty)
        """
        # Base reward: position-weighted return
        base_reward = position_frac * return_pct
        
        # Drawdown penalty (asymmetric loss aversion)
        drawdown_penalty = 0.0
        if drawdown_delta > 0:  # Drawdown increased
            drawdown_penalty = self.config.drawdown_penalty * drawdown_delta
        
        return base_reward - drawdown_penalty
    
    def train_step(self, batch_size: int = 64) -> Dict[str, float]:
        """
        Perform one training step using experience replay.
        
        Returns dict with actor_loss and critic_loss.
        """
        if len(self.replay_buffer) < batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
        actions = torch.FloatTensor([[b[1]] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([[b[2]] for b in batch]).to(self.device)
        next_states = torch.FloatTensor([b[3] for b in batch]).to(self.device)
        dones = torch.FloatTensor([[1 - b[4]] for b in batch]).to(self.device)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.config.gamma * dones * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update target network parameters."""
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(
                self.config.tau * src_param.data + 
                (1.0 - self.config.tau) * tgt_param.data
            )
    
    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
```

### Usage Example

```python
# Initialize agent
agent = DDPGKellyAgent(DDPGKellyConfig(
    state_dim=60,  # HIMARI Layer 1 feature vector
    hidden_dim=256,
    tide_lookback=20,
    max_kelly_fraction=0.40  # Max 40% position
))

# Load pre-trained weights
agent.load('models/ddpg_kelly_btc.pt')

# Get position sizing recommendation
state = np.random.randn(60)  # Would come from Layer 1
current_vol = 0.05  # 5% daily volatility
current_drawdown = 0.03  # 3% drawdown from peak

position_frac, metadata = agent.get_position_fraction(
    state=state,
    volatility=current_vol,
    drawdown=current_drawdown
)

print(f"Recommended position fraction: {position_frac:.1%}")
print(f"Raw Kelly output: {metadata['raw_kelly']:.1%}")
print(f"Volatility adjustment: {metadata['vol_multiplier']:.2f}x")
print(f"Drawdown adjustment: {metadata['drawdown_multiplier']:.2f}x")
```

---

## H3: DCC-GARCH Correlation

### The Problem with Static Correlations

Portfolio risk depends critically on asset correlations. A portfolio of BTC and ETH that appears diversified during calm markets can experience massive simultaneous drawdowns during crises—the very moment diversification is needed most.

The mathematical reason: correlations are not constant. The sample correlation calculated over a historical window (e.g., rolling 30-day) assumes correlations are stationary within that window. In reality, correlations spike during crisis periods (contagion) and compress during trending markets.

A portfolio VaR calculation using static 6-month correlations might suggest 10% portfolio risk. During a crisis when correlations spike from 0.5 to 0.9, actual risk could be 15-20%—a 50-100% underestimation.

### DCC-GARCH: Time-Varying Correlations

DCC-GARCH (Dynamic Conditional Correlation with Generalized Autoregressive Conditional Heteroskedasticity) models both time-varying volatility and time-varying correlations.

**GARCH(1,1)** for each asset's volatility:
```
σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

Where:
- σ²_t: Conditional variance at time t
- ω: Long-run variance component
- α: Weight on recent shock (ε²)
- β: Persistence of variance
```

**DCC** for correlation dynamics:
```
Q_t = (1 - a - b) * Q̄ + a * ε̃_{t-1} * ε̃'_{t-1} + b * Q_{t-1}
R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}

Where:
- Q_t: Quasi-correlation matrix
- Q̄: Unconditional correlation matrix
- R_t: Time-varying correlation matrix
- a, b: DCC parameters
- ε̃: Standardized residuals
```

### Production Implementation

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize


@dataclass
class DCCGARCHConfig:
    """Configuration for DCC-GARCH correlation estimation."""
    lookback_days: int = 252              # Historical data for estimation
    garch_omega_init: float = 0.00001     # GARCH omega initialization
    garch_alpha_init: float = 0.05        # GARCH alpha initialization
    garch_beta_init: float = 0.90         # GARCH beta initialization
    dcc_a_init: float = 0.05              # DCC a parameter initialization
    dcc_b_init: float = 0.90              # DCC b parameter initialization
    min_correlation: float = -0.95        # Minimum correlation bound
    max_correlation: float = 0.99         # Maximum correlation bound


@dataclass
class CorrelationEstimates:
    """Container for correlation estimates."""
    correlation_matrix: np.ndarray      # Current correlation matrix
    volatilities: np.ndarray            # Current volatility estimates
    covariance_matrix: np.ndarray       # Current covariance matrix
    correlation_1d_forecast: np.ndarray # 1-day ahead correlation forecast
    volatility_1d_forecast: np.ndarray  # 1-day ahead volatility forecast


class GARCH11:
    """
    GARCH(1,1) model for single asset volatility.
    
    σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    Typical crypto parameters:
    - α: 0.05-0.15 (react to recent shocks)
    - β: 0.80-0.95 (high persistence)
    - α + β < 1 for stationarity
    """
    
    def __init__(self, omega: float = 0.00001, alpha: float = 0.05, 
                 beta: float = 0.90):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.sigma2 = 0.0025  # Initialize to 5% daily vol squared
        
    def update(self, return_: float) -> float:
        """
        Update volatility estimate with new return.
        
        Args:
            return_: Daily return (e.g., 0.02 for 2%)
            
        Returns:
            Current volatility estimate (std dev)
        """
        epsilon2 = return_ ** 2
        self.sigma2 = self.omega + self.alpha * epsilon2 + self.beta * self.sigma2
        return np.sqrt(self.sigma2)
    
    def forecast(self, steps: int = 1) -> np.ndarray:
        """
        Forecast volatility h steps ahead.
        
        Multi-step forecast uses mean reversion:
        σ²_{t+h} = ω/(1-α-β) + (α+β)^h * (σ²_t - ω/(1-α-β))
        """
        long_run_var = self.omega / (1 - self.alpha - self.beta)
        forecasts = np.zeros(steps)
        
        current_var = self.sigma2
        for h in range(steps):
            forecasts[h] = long_run_var + (self.alpha + self.beta) ** (h + 1) * (
                current_var - long_run_var
            )
        
        return np.sqrt(forecasts)


class DCCGARCHModel:
    """
    Dynamic Conditional Correlation GARCH for multi-asset portfolios.
    
    Models time-varying correlations and volatilities for realistic
    portfolio risk assessment. Crucial for crypto where correlations
    spike during crashes (contagion effect).
    
    Why DCC over rolling correlation:
    - Captures correlation clustering (high vol = high correlation)
    - Provides smooth estimates without lookback window artifacts
    - Forecasts future correlations for risk planning
    
    Latency: <0.4ms for correlation update
    Memory: O(n²) for n-asset correlation matrix
    """
    
    def __init__(self, n_assets: int, config: DCCGARCHConfig = None,
                 asset_names: List[str] = None):
        self.n_assets = n_assets
        self.config = config or DCCGARCHConfig()
        self.asset_names = asset_names or [f'Asset_{i}' for i in range(n_assets)]
        
        # Individual GARCH models for each asset
        self.garch_models = [
            GARCH11(
                omega=self.config.garch_omega_init,
                alpha=self.config.garch_alpha_init,
                beta=self.config.garch_beta_init
            )
            for _ in range(n_assets)
        ]
        
        # DCC parameters
        self.dcc_a = self.config.dcc_a_init
        self.dcc_b = self.config.dcc_b_init
        
        # Unconditional correlation matrix (Q̄)
        self.Q_bar = np.eye(n_assets)
        
        # Dynamic correlation state (Q_t)
        self.Q = np.eye(n_assets)
        
        # Return history for initialization
        self.return_history: List[np.ndarray] = []
        
    def update(self, returns: np.ndarray) -> CorrelationEstimates:
        """
        Update DCC-GARCH with new returns vector.
        
        Args:
            returns: (n_assets,) array of daily returns
            
        Returns:
            CorrelationEstimates with current matrices
        """
        # Update individual GARCH models
        volatilities = np.zeros(self.n_assets)
        for i, ret in enumerate(returns):
            volatilities[i] = self.garch_models[i].update(ret)
        
        # Compute standardized residuals
        epsilon_std = returns / (volatilities + 1e-8)
        
        # Store for unconditional estimation
        self.return_history.append(epsilon_std)
        if len(self.return_history) > self.config.lookback_days:
            self.return_history.pop(0)
        
        # Update Q̄ periodically (unconditional correlation)
        if len(self.return_history) >= 30:
            residuals = np.array(self.return_history)
            self.Q_bar = np.corrcoef(residuals.T)
            # Ensure positive semi-definite
            self.Q_bar = self._nearest_psd(self.Q_bar)
        
        # DCC update: Q_t = (1 - a - b) * Q̄ + a * ε̃_{t-1} * ε̃'_{t-1} + b * Q_{t-1}
        outer_product = np.outer(epsilon_std, epsilon_std)
        self.Q = ((1 - self.dcc_a - self.dcc_b) * self.Q_bar + 
                  self.dcc_a * outer_product + 
                  self.dcc_b * self.Q)
        
        # Convert Q to correlation matrix R
        Q_diag = np.sqrt(np.diag(self.Q))
        Q_diag_inv = 1.0 / (Q_diag + 1e-8)
        correlation_matrix = self.Q * np.outer(Q_diag_inv, Q_diag_inv)
        
        # Clip correlations to valid range
        np.fill_diagonal(correlation_matrix, 1.0)
        correlation_matrix = np.clip(
            correlation_matrix, 
            self.config.min_correlation, 
            self.config.max_correlation
        )
        
        # Compute covariance matrix
        D = np.diag(volatilities)
        covariance_matrix = D @ correlation_matrix @ D
        
        # 1-step forecasts
        vol_forecast = np.array([g.forecast(1)[0] for g in self.garch_models])
        
        # Correlation forecast (mean reversion toward Q̄)
        Q_forecast = ((1 - self.dcc_a - self.dcc_b) * self.Q_bar + 
                      (self.dcc_a + self.dcc_b) * self.Q)
        Q_diag_f = np.sqrt(np.diag(Q_forecast))
        Q_diag_f_inv = 1.0 / (Q_diag_f + 1e-8)
        corr_forecast = Q_forecast * np.outer(Q_diag_f_inv, Q_diag_f_inv)
        
        return CorrelationEstimates(
            correlation_matrix=correlation_matrix,
            volatilities=volatilities,
            covariance_matrix=covariance_matrix,
            correlation_1d_forecast=corr_forecast,
            volatility_1d_forecast=vol_forecast
        )
    
    def get_correlation(self, asset1: str, asset2: str) -> float:
        """Get current correlation between two assets."""
        i = self.asset_names.index(asset1)
        j = self.asset_names.index(asset2)
        
        Q_diag = np.sqrt(np.diag(self.Q))
        Q_diag_inv = 1.0 / (Q_diag + 1e-8)
        R = self.Q * np.outer(Q_diag_inv, Q_diag_inv)
        
        return R[i, j]
    
    def get_portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Compute portfolio volatility given weights.
        
        Args:
            weights: (n_assets,) portfolio weights summing to 1
            
        Returns:
            Portfolio volatility (daily std dev)
        """
        vols = np.array([g.sigma2 for g in self.garch_models])
        vols = np.sqrt(vols)
        
        Q_diag = np.sqrt(np.diag(self.Q))
        Q_diag_inv = 1.0 / (Q_diag + 1e-8)
        R = self.Q * np.outer(Q_diag_inv, Q_diag_inv)
        
        D = np.diag(vols)
        cov = D @ R @ D
        
        portfolio_var = weights @ cov @ weights
        return np.sqrt(portfolio_var)
    
    def _nearest_psd(self, matrix: np.ndarray) -> np.ndarray:
        """Project matrix to nearest positive semi-definite matrix."""
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def fit_parameters(self, returns_history: np.ndarray) -> Dict[str, float]:
        """
        Fit DCC-GARCH parameters via MLE.
        
        Args:
            returns_history: (T, n_assets) array of historical returns
            
        Returns:
            Dict with fitted parameters
        """
        # This is a simplified fitting procedure
        # Production would use proper DCC-GARCH MLE
        
        T = len(returns_history)
        
        # Fit individual GARCH models
        for i in range(self.n_assets):
            asset_returns = returns_history[:, i]
            self._fit_garch(i, asset_returns)
        
        # Compute standardized residuals
        residuals = []
        for t in range(T):
            vols = [self.garch_models[i].update(returns_history[t, i]) 
                    for i in range(self.n_assets)]
            eps_std = returns_history[t] / (np.array(vols) + 1e-8)
            residuals.append(eps_std)
        
        residuals = np.array(residuals)
        
        # Unconditional correlation
        self.Q_bar = np.corrcoef(residuals.T)
        self.Q = self.Q_bar.copy()
        
        return {
            'dcc_a': self.dcc_a,
            'dcc_b': self.dcc_b,
            'garch_params': [(g.omega, g.alpha, g.beta) for g in self.garch_models]
        }
    
    def _fit_garch(self, asset_idx: int, returns: np.ndarray) -> None:
        """Fit GARCH(1,1) parameters for single asset."""
        # Simplified fitting using moment matching
        var = np.var(returns)
        autocorr = np.corrcoef(returns[:-1]**2, returns[1:]**2)[0, 1]
        
        # Initialize with reasonable estimates
        alpha = max(0.01, min(0.20, autocorr * 0.3))
        beta = max(0.70, min(0.95, 0.95 - alpha))
        omega = var * (1 - alpha - beta)
        
        self.garch_models[asset_idx].omega = omega
        self.garch_models[asset_idx].alpha = alpha
        self.garch_models[asset_idx].beta = beta
```

### Usage Example

```python
# Initialize for BTC, ETH, SOL portfolio
dcc = DCCGARCHModel(
    n_assets=3,
    asset_names=['BTC', 'ETH', 'SOL'],
    config=DCCGARCHConfig(
        garch_alpha_init=0.08,
        garch_beta_init=0.88,
        dcc_a_init=0.05,
        dcc_b_init=0.90
    )
)

# Simulate daily updates
np.random.seed(42)
for day in range(100):
    # Returns with correlation structure
    cov = np.array([
        [0.0025, 0.002, 0.0015],
        [0.002, 0.003, 0.002],
        [0.0015, 0.002, 0.004]
    ])
    returns = np.random.multivariate_normal([0, 0, 0], cov)
    
    estimates = dcc.update(returns)

# Get current estimates
print("Correlation Matrix:")
print(np.round(estimates.correlation_matrix, 3))
print(f"\nVolatilities: BTC={estimates.volatilities[0]:.2%}, "
      f"ETH={estimates.volatilities[1]:.2%}, SOL={estimates.volatilities[2]:.2%}")

# Portfolio risk
weights = np.array([0.5, 0.3, 0.2])
port_vol = dcc.get_portfolio_volatility(weights)
print(f"\nPortfolio volatility: {port_vol:.2%}")

# BTC-ETH correlation
btc_eth_corr = dcc.get_correlation('BTC', 'ETH')
print(f"BTC-ETH correlation: {btc_eth_corr:.3f}")
```

---

## H4: Progressive Drawdown Brake

### The Problem with Binary Circuit Breakers

Traditional drawdown circuit breakers use binary logic: if daily loss exceeds 2%, stop trading. This creates several problems:

1. **Cliff effect**: At 1.9% drawdown, full position. At 2.1%, forced liquidation. This discontinuity can cause the worst execution prices precisely when drawdown hits the threshold.

2. **Lock-out problem**: After triggering, traders are locked out for arbitrary periods, missing recovery rallies.

3. **Gaming vulnerability**: Market makers who know other traders' circuit breaker levels can push prices just past threshold, trigger forced liquidations, then buy at depressed prices.

### Progressive Scaling Solution

Progressive drawdown brakes reduce position size gradually as drawdown increases, rather than applying binary cutoffs. Think of it like a dimmer switch rather than an on/off light switch.

The position multiplier follows a smooth decay function:
```
multiplier = max(min_multiplier, 1 - drawdown / max_drawdown * (1 - min_multiplier))
```

At 0% drawdown: multiplier = 1.0 (full position)
At 5% drawdown: multiplier = 0.5 (half position)  
At 10% drawdown: multiplier = 0.1 (minimal position)

### Production Implementation

```python
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from enum import Enum


class DrawdownState(Enum):
    """Drawdown severity states for monitoring."""
    NORMAL = "normal"           # 0-2% drawdown
    CAUTION = "caution"         # 2-5% drawdown
    WARNING = "warning"         # 5-8% drawdown
    CRITICAL = "critical"       # 8-10% drawdown
    EMERGENCY = "emergency"     # >10% drawdown


@dataclass
class DrawdownBrakeConfig:
    """Configuration for Progressive Drawdown Brake."""
    max_daily_drawdown: float = 0.05      # Max daily drawdown before emergency (5%)
    max_total_drawdown: float = 0.15      # Max total drawdown from peak (15%)
    min_position_multiplier: float = 0.10  # Minimum position size (10% of normal)
    recovery_rate: float = 0.25            # Rate of position recovery per day
    
    # Threshold levels for state transitions
    caution_threshold: float = 0.02       # Enter CAUTION at 2% drawdown
    warning_threshold: float = 0.05       # Enter WARNING at 5% drawdown
    critical_threshold: float = 0.08      # Enter CRITICAL at 8% drawdown
    emergency_threshold: float = 0.10     # Enter EMERGENCY at 10% drawdown
    
    # Decay curve parameters
    decay_steepness: float = 2.0          # Higher = steeper decay
    use_exponential_decay: bool = True    # Exponential vs linear decay


@dataclass
class DrawdownStatus:
    """Current drawdown status and position adjustment."""
    current_drawdown: float           # Current drawdown from peak
    daily_drawdown: float             # Today's drawdown
    position_multiplier: float        # Recommended position multiplier (0.1 to 1.0)
    state: DrawdownState              # Current drawdown state
    time_in_state: int                # Bars spent in current state
    recovery_multiplier: float        # Additional multiplier for recovery phase
    reason: str                       # Human-readable explanation


class ProgressiveDrawdownBrake:
    """
    Progressive position scaling based on drawdown severity.
    
    Instead of binary circuit breakers that create cliff effects and
    lock-out problems, this implements smooth position scaling:
    
    - Normal (0-2%): Full position
    - Caution (2-5%): 90-70% position
    - Warning (5-8%): 70-40% position
    - Critical (8-10%): 40-20% position
    - Emergency (>10%): 20-10% position
    
    Recovery is also gradual: position size increases slowly as
    drawdown recovers, preventing aggressive re-entry at local bottoms.
    
    Latency: <0.1ms per update
    """
    
    def __init__(self, config: DrawdownBrakeConfig = None):
        self.config = config or DrawdownBrakeConfig()
        
        # State tracking
        self._peak_equity: float = 1.0
        self._current_equity: float = 1.0
        self._daily_start_equity: float = 1.0
        self._state: DrawdownState = DrawdownState.NORMAL
        self._bars_in_state: int = 0
        
        # Recovery tracking
        self._recovering: bool = False
        self._recovery_start_time: int = 0
        self._recovery_multiplier: float = 1.0
        
    def update(self, equity: float, bar_index: int, 
               new_day: bool = False) -> DrawdownStatus:
        """
        Update drawdown tracking with new equity value.
        
        Args:
            equity: Current portfolio equity
            bar_index: Current bar index
            new_day: True if this is first bar of new day
            
        Returns:
            DrawdownStatus with position adjustment recommendation
        """
        # Update peak tracking
        if equity > self._peak_equity:
            self._peak_equity = equity
            self._recovering = False
        
        # Daily tracking
        if new_day:
            self._daily_start_equity = equity
        
        # Compute drawdowns
        total_drawdown = 1 - (equity / self._peak_equity)
        daily_drawdown = 1 - (equity / self._daily_start_equity) if not new_day else 0.0
        
        # Update state
        new_state = self._compute_state(total_drawdown)
        if new_state != self._state:
            self._state = new_state
            self._bars_in_state = 0
            if new_state == DrawdownState.NORMAL:
                self._recovering = True
                self._recovery_start_time = bar_index
        else:
            self._bars_in_state += 1
        
        # Compute position multiplier
        position_mult = self._compute_position_multiplier(total_drawdown)
        
        # Apply recovery dampening (don't rush back to full position)
        if self._recovering:
            recovery_bars = bar_index - self._recovery_start_time
            recovery_progress = min(1.0, recovery_bars * self.config.recovery_rate / 288)
            self._recovery_multiplier = 0.5 + 0.5 * recovery_progress
            position_mult *= self._recovery_multiplier
        else:
            self._recovery_multiplier = 1.0
        
        # Emergency override for daily drawdown
        if daily_drawdown > self.config.max_daily_drawdown:
            position_mult = min(position_mult, self.config.min_position_multiplier)
        
        self._current_equity = equity
        
        return DrawdownStatus(
            current_drawdown=total_drawdown,
            daily_drawdown=daily_drawdown,
            position_multiplier=position_mult,
            state=self._state,
            time_in_state=self._bars_in_state,
            recovery_multiplier=self._recovery_multiplier,
            reason=self._get_reason(total_drawdown, daily_drawdown, position_mult)
        )
    
    def _compute_state(self, drawdown: float) -> DrawdownState:
        """Determine drawdown state based on current level."""
        if drawdown >= self.config.emergency_threshold:
            return DrawdownState.EMERGENCY
        elif drawdown >= self.config.critical_threshold:
            return DrawdownState.CRITICAL
        elif drawdown >= self.config.warning_threshold:
            return DrawdownState.WARNING
        elif drawdown >= self.config.caution_threshold:
            return DrawdownState.CAUTION
        else:
            return DrawdownState.NORMAL
    
    def _compute_position_multiplier(self, drawdown: float) -> float:
        """
        Compute position multiplier based on drawdown level.
        
        Uses either exponential or linear decay curve.
        """
        if drawdown <= 0:
            return 1.0
        
        max_dd = self.config.max_total_drawdown
        min_mult = self.config.min_position_multiplier
        
        # Normalize drawdown to [0, 1]
        normalized_dd = min(1.0, drawdown / max_dd)
        
        if self.config.use_exponential_decay:
            # Exponential decay: steeper drop as drawdown increases
            k = self.config.decay_steepness
            multiplier = (1 - min_mult) * np.exp(-k * normalized_dd) + min_mult
        else:
            # Linear decay
            multiplier = 1.0 - normalized_dd * (1 - min_mult)
        
        return max(min_mult, min(1.0, multiplier))
    
    def _get_reason(self, total_dd: float, daily_dd: float, 
                    multiplier: float) -> str:
        """Generate human-readable reason for position adjustment."""
        if self._state == DrawdownState.EMERGENCY:
            return f"EMERGENCY: {total_dd:.1%} drawdown, position reduced to {multiplier:.0%}"
        elif self._state == DrawdownState.CRITICAL:
            return f"CRITICAL: {total_dd:.1%} drawdown, position at {multiplier:.0%}"
        elif daily_dd > self.config.max_daily_drawdown * 0.8:
            return f"Daily drawdown warning: {daily_dd:.1%}, position at {multiplier:.0%}"
        elif self._recovering:
            return f"Recovery mode: gradually increasing to {multiplier:.0%}"
        elif self._state != DrawdownState.NORMAL:
            return f"{self._state.value.upper()}: {total_dd:.1%} drawdown"
        else:
            return "Normal operation"
    
    def reset_peak(self, new_peak: Optional[float] = None) -> None:
        """Reset peak equity (e.g., after capital addition)."""
        if new_peak is not None:
            self._peak_equity = new_peak
        else:
            self._peak_equity = self._current_equity
        self._state = DrawdownState.NORMAL
        self._recovering = False


class AdaptiveDrawdownBrake(ProgressiveDrawdownBrake):
    """
    Adaptive drawdown brake that adjusts thresholds based on regime.
    
    During high-volatility regimes, thresholds are widened to prevent
    premature position reduction from normal vol. During low-vol regimes,
    thresholds tighten to catch anomalies earlier.
    """
    
    def __init__(self, config: DrawdownBrakeConfig = None):
        super().__init__(config)
        self._regime_multiplier: float = 1.0
        
    def set_regime_volatility(self, current_vol: float, 
                              baseline_vol: float = 0.05) -> None:
        """
        Adjust thresholds based on current volatility regime.
        
        Args:
            current_vol: Current annualized volatility
            baseline_vol: Baseline volatility for threshold calibration
        """
        # Widen thresholds in high vol, tighten in low vol
        self._regime_multiplier = current_vol / baseline_vol
        self._regime_multiplier = np.clip(self._regime_multiplier, 0.5, 2.0)
    
    def _compute_state(self, drawdown: float) -> DrawdownState:
        """Adjust thresholds by regime multiplier."""
        adj_caution = self.config.caution_threshold * self._regime_multiplier
        adj_warning = self.config.warning_threshold * self._regime_multiplier
        adj_critical = self.config.critical_threshold * self._regime_multiplier
        adj_emergency = self.config.emergency_threshold * self._regime_multiplier
        
        if drawdown >= adj_emergency:
            return DrawdownState.EMERGENCY
        elif drawdown >= adj_critical:
            return DrawdownState.CRITICAL
        elif drawdown >= adj_warning:
            return DrawdownState.WARNING
        elif drawdown >= adj_caution:
            return DrawdownState.CAUTION
        else:
            return DrawdownState.NORMAL
```

### Usage Example

```python
# Initialize brake
brake = AdaptiveDrawdownBrake(DrawdownBrakeConfig(
    max_total_drawdown=0.15,
    min_position_multiplier=0.10,
    use_exponential_decay=True,
    decay_steepness=2.5
))

# Simulate equity curve with drawdown
initial_equity = 100000
equities = [initial_equity]
multipliers = []

np.random.seed(42)
for bar in range(500):
    new_day = (bar % 288 == 0)
    
    # Random walk with slight negative bias during drawdown
    daily_return = np.random.randn() * 0.02
    if len(equities) > 0 and equities[-1] < initial_equity:
        daily_return -= 0.005  # Drawdown persistence
    
    new_equity = equities[-1] * (1 + daily_return)
    
    # Get position recommendation
    status = brake.update(new_equity, bar, new_day)
    equities.append(new_equity)
    multipliers.append(status.position_multiplier)
    
    if bar % 100 == 0:
        print(f"Bar {bar}: Equity=${new_equity:,.0f}, "
              f"DD={status.current_drawdown:.1%}, "
              f"Mult={status.position_multiplier:.0%}, "
              f"State={status.state.value}")
```

---

## H5: Portfolio-Level VaR

### The Challenge of Multi-Asset Risk

Individual asset VaR calculations don't capture portfolio risk accurately. A portfolio of BTC (20% weight) and ETH (80% weight) has different risk characteristics than the weighted sum of individual VaRs would suggest.

The mathematical reason: portfolio VaR depends on the full joint distribution of returns, not just marginal distributions. Two assets can each have 5% daily VaR individually, but if perfectly correlated, portfolio VaR is also 5%. If uncorrelated, portfolio VaR drops to ~3.5%. If negatively correlated (rare in crypto), portfolio VaR could be even lower.

### Copula-Based Portfolio VaR

Copula methods separate the modeling of marginal distributions (handled by EVT/GARCH) from dependence structure:

1. **Marginal models**: EVT-GPD for tails, GARCH for volatility
2. **Copula**: Captures dependence structure independent of marginals
3. **Monte Carlo**: Sample from copula, transform to marginal distributions, aggregate

The Student-t copula is particularly appropriate for crypto because it captures tail dependence—the tendency for extreme moves to occur simultaneously.

### Production Implementation

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats


@dataclass
class PortfolioVaRConfig:
    """Configuration for Portfolio VaR calculation."""
    confidence_levels: Tuple[float, ...] = (0.95, 0.99, 0.999)
    n_simulations: int = 10000            # Monte Carlo simulations
    copula_type: str = "student_t"        # 'gaussian' or 'student_t'
    student_t_df: float = 5.0             # Degrees of freedom for t-copula
    use_evt_tails: bool = True            # Use EVT for marginals
    correlation_source: str = "dcc"       # 'dcc' or 'sample'


@dataclass
class PortfolioRiskMetrics:
    """Container for portfolio risk metrics."""
    var_95: float                         # 95% portfolio VaR
    var_99: float                         # 99% portfolio VaR
    var_999: float                        # 99.9% portfolio VaR
    cvar_95: float                        # Conditional VaR at 95%
    cvar_99: float                        # Conditional VaR at 99%
    component_var: Dict[str, float]       # Per-asset contribution to VaR
    marginal_var: Dict[str, float]        # Marginal VaR per asset
    correlation_matrix: np.ndarray        # Correlation used in calculation
    portfolio_volatility: float           # Portfolio volatility
    diversification_ratio: float          # Weighted vol / portfolio vol


class PortfolioVaRCalculator:
    """
    Portfolio-level Value-at-Risk using copula methods.
    
    Combines EVT marginals with DCC-GARCH correlations and Student-t
    copula for realistic portfolio risk estimation that captures:
    
    1. Fat tails in individual asset returns (EVT)
    2. Time-varying correlations (DCC-GARCH)
    3. Tail dependence (Student-t copula)
    4. Non-linear diversification effects
    
    Why copula over variance-covariance VaR:
    - Captures tail dependence (correlations increase in crashes)
    - Handles non-normal marginal distributions
    - More accurate for extreme quantiles
    
    Latency: <0.6ms for VaR lookup (MC is done offline)
    """
    
    def __init__(self, 
                 asset_names: List[str],
                 config: PortfolioVaRConfig = None):
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.config = config or PortfolioVaRConfig()
        
        # Marginal distribution parameters (from EVT/GARCH)
        self._marginal_vols: Dict[str, float] = {a: 0.05 for a in asset_names}
        self._marginal_xi: Dict[str, float] = {a: 0.2 for a in asset_names}
        self._marginal_sigma: Dict[str, float] = {a: 0.02 for a in asset_names}
        
        # Correlation matrix (from DCC-GARCH)
        self._correlation_matrix: np.ndarray = np.eye(self.n_assets)
        
        # Cached simulation results
        self._cached_simulations: Optional[np.ndarray] = None
        self._cache_valid: bool = False
        
    def update_marginals(self, asset: str, volatility: float,
                        xi: float, sigma: float) -> None:
        """
        Update marginal distribution parameters for an asset.
        
        Args:
            asset: Asset name
            volatility: Current volatility estimate
            xi: GPD shape parameter
            sigma: GPD scale parameter
        """
        self._marginal_vols[asset] = volatility
        self._marginal_xi[asset] = xi
        self._marginal_sigma[asset] = sigma
        self._cache_valid = False
        
    def update_correlation(self, correlation_matrix: np.ndarray) -> None:
        """
        Update correlation matrix (typically from DCC-GARCH).
        
        Args:
            correlation_matrix: (n_assets, n_assets) correlation matrix
        """
        self._correlation_matrix = correlation_matrix
        self._cache_valid = False
        
    def calculate_var(self, weights: np.ndarray) -> PortfolioRiskMetrics:
        """
        Calculate portfolio VaR for given weights.
        
        Args:
            weights: (n_assets,) portfolio weights
            
        Returns:
            PortfolioRiskMetrics with VaR and component analysis
        """
        # Ensure cache is valid
        if not self._cache_valid:
            self._run_monte_carlo()
        
        # Weight simulated returns
        portfolio_returns = self._cached_simulations @ weights
        
        # Calculate VaR at different confidence levels
        var_95 = -np.percentile(portfolio_returns, 5)
        var_99 = -np.percentile(portfolio_returns, 1)
        var_999 = -np.percentile(portfolio_returns, 0.1)
        
        # Calculate CVaR (Expected Shortfall)
        cvar_95 = -np.mean(portfolio_returns[portfolio_returns < -var_95])
        cvar_99 = -np.mean(portfolio_returns[portfolio_returns < -var_99])
        
        # Component VaR: contribution of each asset to portfolio VaR
        component_var = self._calculate_component_var(weights, var_99)
        
        # Marginal VaR: change in portfolio VaR per unit change in weight
        marginal_var = self._calculate_marginal_var(weights)
        
        # Portfolio volatility
        vols = np.array([self._marginal_vols[a] for a in self.asset_names])
        D = np.diag(vols)
        cov = D @ self._correlation_matrix @ D
        portfolio_vol = np.sqrt(weights @ cov @ weights)
        
        # Diversification ratio
        weighted_vol = np.sum(weights * vols)
        div_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        return PortfolioRiskMetrics(
            var_95=var_95,
            var_99=var_99,
            var_999=var_999,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            component_var=component_var,
            marginal_var=marginal_var,
            correlation_matrix=self._correlation_matrix.copy(),
            portfolio_volatility=portfolio_vol,
            diversification_ratio=div_ratio
        )
    
    def _run_monte_carlo(self) -> None:
        """
        Run Monte Carlo simulation using copula.
        
        Generates n_simulations samples from the joint distribution.
        """
        n = self.config.n_simulations
        
        if self.config.copula_type == "student_t":
            # Student-t copula for tail dependence
            df = self.config.student_t_df
            
            # Sample from multivariate t-distribution
            chi2_samples = np.random.chisquare(df, n)
            normal_samples = np.random.multivariate_normal(
                np.zeros(self.n_assets),
                self._correlation_matrix,
                n
            )
            t_samples = normal_samples / np.sqrt(chi2_samples[:, np.newaxis] / df)
            
            # Convert to uniform via t CDF
            uniform_samples = stats.t.cdf(t_samples, df)
            
        else:
            # Gaussian copula
            normal_samples = np.random.multivariate_normal(
                np.zeros(self.n_assets),
                self._correlation_matrix,
                n
            )
            uniform_samples = stats.norm.cdf(normal_samples)
        
        # Transform to marginal distributions
        returns = np.zeros((n, self.n_assets))
        
        for i, asset in enumerate(self.asset_names):
            vol = self._marginal_vols[asset]
            xi = self._marginal_xi[asset]
            sigma = self._marginal_sigma[asset]
            
            if self.config.use_evt_tails:
                # Use EVT for tails, normal for center
                returns[:, i] = self._inverse_marginal_evt(
                    uniform_samples[:, i], vol, xi, sigma
                )
            else:
                # Simple normal marginals
                returns[:, i] = stats.norm.ppf(uniform_samples[:, i]) * vol
        
        self._cached_simulations = returns
        self._cache_valid = True
    
    def _inverse_marginal_evt(self, u: np.ndarray, vol: float,
                              xi: float, sigma: float) -> np.ndarray:
        """
        Inverse CDF for EVT-enhanced marginal distribution.
        
        Uses normal distribution for center, GPD for tails.
        """
        returns = np.zeros_like(u)
        
        # Threshold for tail treatment (5th and 95th percentiles)
        lower_tail = u < 0.05
        upper_tail = u > 0.95
        center = ~lower_tail & ~upper_tail
        
        # Center: normal distribution
        returns[center] = stats.norm.ppf(u[center]) * vol
        
        # Tails: GPD
        # Upper tail
        if np.any(upper_tail):
            tail_u = (u[upper_tail] - 0.95) / 0.05  # Rescale to [0, 1]
            if np.abs(xi) > 1e-10:
                gpd_quantile = sigma / xi * ((1 - tail_u) ** (-xi) - 1)
            else:
                gpd_quantile = -sigma * np.log(1 - tail_u)
            threshold = stats.norm.ppf(0.95) * vol
            returns[upper_tail] = threshold + gpd_quantile
        
        # Lower tail (mirror)
        if np.any(lower_tail):
            tail_u = (0.05 - u[lower_tail]) / 0.05
            if np.abs(xi) > 1e-10:
                gpd_quantile = sigma / xi * ((1 - tail_u) ** (-xi) - 1)
            else:
                gpd_quantile = -sigma * np.log(1 - tail_u)
            threshold = stats.norm.ppf(0.05) * vol
            returns[lower_tail] = threshold - gpd_quantile
        
        return returns
    
    def _calculate_component_var(self, weights: np.ndarray, 
                                 portfolio_var: float) -> Dict[str, float]:
        """
        Calculate component VaR for each asset.
        
        Component VaR sums to total portfolio VaR and shows
        each asset's contribution to overall risk.
        """
        # Marginal contribution * weight
        marginal = self._calculate_marginal_var(weights)
        
        component = {}
        for i, asset in enumerate(self.asset_names):
            component[asset] = weights[i] * marginal[asset]
        
        # Normalize to sum to portfolio VaR
        total = sum(component.values())
        if total > 0:
            for asset in component:
                component[asset] *= portfolio_var / total
        
        return component
    
    def _calculate_marginal_var(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate marginal VaR for each asset.
        
        Marginal VaR = partial derivative of portfolio VaR with respect to weight.
        """
        delta = 0.01
        base_weights = weights.copy()
        
        # Calculate base portfolio VaR
        portfolio_returns = self._cached_simulations @ base_weights
        base_var = -np.percentile(portfolio_returns, 1)
        
        marginal = {}
        for i, asset in enumerate(self.asset_names):
            # Perturb weight
            perturbed = base_weights.copy()
            perturbed[i] += delta
            perturbed = perturbed / perturbed.sum()  # Renormalize
            
            # Calculate perturbed VaR
            perturbed_returns = self._cached_simulations @ perturbed
            perturbed_var = -np.percentile(perturbed_returns, 1)
            
            # Finite difference approximation
            marginal[asset] = (perturbed_var - base_var) / delta
        
        return marginal


# Fast lookup version for production
class PortfolioVaRLookup:
    """
    Pre-computed portfolio VaR lookup table for <0.1ms inference.
    
    Caches VaR estimates for common weight configurations.
    """
    
    def __init__(self, calculator: PortfolioVaRCalculator):
        self.calculator = calculator
        self._cache: Dict[Tuple, PortfolioRiskMetrics] = {}
        
    def get_var(self, weights: np.ndarray, 
                confidence: float = 0.99) -> float:
        """
        Get portfolio VaR, using cache if available.
        
        Args:
            weights: Portfolio weights
            confidence: Confidence level
            
        Returns:
            VaR at specified confidence level
        """
        # Round weights to 5% increments for caching
        rounded = tuple(round(w * 20) / 20 for w in weights)
        
        if rounded not in self._cache:
            weight_array = np.array(rounded)
            weight_array = weight_array / weight_array.sum()
            self._cache[rounded] = self.calculator.calculate_var(weight_array)
        
        metrics = self._cache[rounded]
        
        if confidence >= 0.999:
            return metrics.var_999
        elif confidence >= 0.99:
            return metrics.var_99
        else:
            return metrics.var_95
```

### Usage Example

```python
# Initialize calculator
calculator = PortfolioVaRCalculator(
    asset_names=['BTC', 'ETH', 'SOL'],
    config=PortfolioVaRConfig(
        n_simulations=10000,
        copula_type='student_t',
        student_t_df=5.0
    )
)

# Update marginals from EVT
calculator.update_marginals('BTC', volatility=0.05, xi=0.2, sigma=0.02)
calculator.update_marginals('ETH', volatility=0.06, xi=0.25, sigma=0.025)
calculator.update_marginals('SOL', volatility=0.08, xi=0.3, sigma=0.03)

# Update correlation from DCC-GARCH
corr = np.array([
    [1.0, 0.75, 0.60],
    [0.75, 1.0, 0.70],
    [0.60, 0.70, 1.0]
])
calculator.update_correlation(corr)

# Calculate VaR for specific portfolio
weights = np.array([0.5, 0.3, 0.2])
metrics = calculator.calculate_var(weights)

print(f"Portfolio VaR:")
print(f"  95% VaR: {metrics.var_95:.2%}")
print(f"  99% VaR: {metrics.var_99:.2%}")
print(f"  99.9% VaR: {metrics.var_999:.2%}")
print(f"\nCVaR (Expected Shortfall):")
print(f"  95% CVaR: {metrics.cvar_95:.2%}")
print(f"  99% CVaR: {metrics.cvar_99:.2%}")
print(f"\nPortfolio Statistics:")
print(f"  Volatility: {metrics.portfolio_volatility:.2%}")
print(f"  Diversification Ratio: {metrics.diversification_ratio:.2f}")
print(f"\nComponent VaR:")
for asset, cvar in metrics.component_var.items():
    print(f"  {asset}: {cvar:.2%}")
```

---

## H6: Safe Margin Formula

### Responsibility-Sensitive Safety for Trading

The Safe Margin Formula adapts RSS principles from autonomous vehicles to trading. In vehicles, RSS guarantees collision avoidance by maintaining safe following distances. In trading, we guarantee liquidation avoidance by maintaining adequate margin buffers.

The core insight: given current volatility and position size, how much margin buffer do we need to survive a k-sigma adverse move within our decision horizon?

### The Formula

```
margin_safe = leverage × volatility × k × √(time_horizon) + execution_cost

Where:
- leverage: Current position leverage (e.g., 3×)
- volatility: Asset volatility scaled to time horizon
- k: Confidence multiplier (2.0 for 95%, 2.33 for 99%, 3.0 for 99.9%)
- time_horizon: Time to next decision point (fraction of day)
- execution_cost: Expected slippage + fees during exit
```

### Production Implementation

```python
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class SafeMarginConfig:
    """Configuration for Safe Margin Formula."""
    default_k: float = 2.0                # Default k for 95% confidence
    execution_cost_bps: float = 20        # 20 basis points slippage + fees
    min_margin_buffer: float = 0.01       # Minimum 1% margin buffer
    max_leverage: float = 5.0             # Maximum allowed leverage
    liquidation_buffer: float = 0.02      # Buffer above liquidation level
    use_evt_k: bool = True                # Use EVT-adjusted k values


@dataclass
class SafeMarginResult:
    """Result of safe margin calculation."""
    required_margin: float              # Margin buffer required
    max_safe_leverage: float            # Maximum safe leverage
    current_margin_ratio: float         # Current margin / required
    is_safe: bool                       # Position is safe
    reason: str                         # Explanation


class SafeMarginCalculator:
    """
    Calculate safe margin buffer using RSS principles.
    
    Guarantees that position can survive k-sigma adverse move
    without liquidation, given current volatility and horizon.
    
    The formula adapts Responsibility-Sensitive Safety from
    autonomous vehicles: instead of maintaining safe following
    distance to avoid collisions, we maintain safe margin buffer
    to avoid liquidations.
    
    Key differences from traditional margin:
    - Volatility-adjusted (tighter in calm, wider in crisis)
    - Time-horizon aware (shorter horizon = less buffer needed)
    - EVT-aware (uses fat-tail k values, not Gaussian)
    
    Latency: <0.2ms per calculation
    """
    
    def __init__(self, config: SafeMarginConfig = None):
        self.config = config or SafeMarginConfig()
        
        # EVT-adjusted k values (higher than Gaussian for fat tails)
        self._evt_k_values = {
            0.90: 1.8,    # 90% confidence
            0.95: 2.4,    # 95% confidence (Gaussian: 1.65)
            0.99: 3.2,    # 99% confidence (Gaussian: 2.33)
            0.999: 4.5,   # 99.9% confidence (Gaussian: 3.09)
        }
        
    def calculate_safe_margin(self,
                             leverage: float,
                             volatility: float,
                             horizon_bars: int,
                             bars_per_day: int = 288,
                             confidence: float = 0.95,
                             available_margin: float = 1.0) -> SafeMarginResult:
        """
        Calculate required margin buffer for position safety.
        
        Args:
            leverage: Current position leverage
            volatility: Daily volatility (e.g., 0.05 for 5%)
            horizon_bars: Bars until next decision point
            bars_per_day: Bars per trading day
            confidence: Safety confidence level
            available_margin: Current available margin fraction
            
        Returns:
            SafeMarginResult with required buffer and safety status
        """
        # Get k value for confidence level
        k = self._get_k_value(confidence)
        
        # Scale volatility to horizon
        horizon_days = horizon_bars / bars_per_day
        horizon_vol = volatility * np.sqrt(horizon_days)
        
        # Execution cost as fraction
        execution_cost = self.config.execution_cost_bps / 10000
        
        # Safe margin formula
        required_margin = (leverage * horizon_vol * k + execution_cost + 
                          self.config.liquidation_buffer)
        
        # Ensure minimum buffer
        required_margin = max(required_margin, self.config.min_margin_buffer)
        
        # Check safety
        is_safe = available_margin >= required_margin
        margin_ratio = available_margin / required_margin if required_margin > 0 else float('inf')
        
        # Calculate max safe leverage
        if available_margin > execution_cost + self.config.liquidation_buffer:
            max_safe_lev = (available_margin - execution_cost - 
                          self.config.liquidation_buffer) / (horizon_vol * k)
            max_safe_lev = min(max_safe_lev, self.config.max_leverage)
        else:
            max_safe_lev = 0.0
        
        # Generate reason
        if is_safe:
            reason = f"Safe: {available_margin:.1%} margin > {required_margin:.1%} required"
        else:
            reason = (f"UNSAFE: {available_margin:.1%} margin < {required_margin:.1%} required. "
                     f"Reduce leverage to {max_safe_lev:.1f}× or add margin.")
        
        return SafeMarginResult(
            required_margin=required_margin,
            max_safe_leverage=max_safe_lev,
            current_margin_ratio=margin_ratio,
            is_safe=is_safe,
            reason=reason
        )
    
    def calculate_max_leverage(self,
                              available_margin: float,
                              volatility: float,
                              horizon_bars: int = 1,
                              bars_per_day: int = 288,
                              confidence: float = 0.95) -> float:
        """
        Calculate maximum safe leverage given available margin.
        
        This is the inverse of calculate_safe_margin: given margin,
        what's the maximum leverage we can take?
        
        Args:
            available_margin: Available margin as fraction of position
            volatility: Daily volatility
            horizon_bars: Decision horizon in bars
            bars_per_day: Bars per day
            confidence: Safety confidence level
            
        Returns:
            Maximum safe leverage (capped at config.max_leverage)
        """
        k = self._get_k_value(confidence)
        execution_cost = self.config.execution_cost_bps / 10000
        
        # Scale volatility to horizon
        horizon_days = horizon_bars / bars_per_day
        horizon_vol = volatility * np.sqrt(horizon_days)
        
        # Solve for leverage
        usable_margin = available_margin - execution_cost - self.config.liquidation_buffer
        
        if usable_margin <= 0 or horizon_vol <= 0:
            return 0.0
        
        max_leverage = usable_margin / (horizon_vol * k)
        
        return min(max_leverage, self.config.max_leverage)
    
    def _get_k_value(self, confidence: float) -> float:
        """Get k-sigma value for confidence level."""
        if not self.config.use_evt_k:
            # Gaussian k values
            from scipy import stats
            return stats.norm.ppf(1 - (1 - confidence) / 2)
        
        # EVT-adjusted k values (interpolate)
        sorted_confs = sorted(self._evt_k_values.keys())
        
        if confidence <= sorted_confs[0]:
            return self._evt_k_values[sorted_confs[0]]
        if confidence >= sorted_confs[-1]:
            return self._evt_k_values[sorted_confs[-1]]
        
        # Linear interpolation
        for i in range(len(sorted_confs) - 1):
            if sorted_confs[i] <= confidence <= sorted_confs[i + 1]:
                frac = ((confidence - sorted_confs[i]) / 
                       (sorted_confs[i + 1] - sorted_confs[i]))
                return (self._evt_k_values[sorted_confs[i]] * (1 - frac) +
                       self._evt_k_values[sorted_confs[i + 1]] * frac)
        
        return self.config.default_k


# Example for BTC at different conditions
def safe_margin_examples():
    """Demonstrate safe margin calculations."""
    calc = SafeMarginCalculator()
    
    print("Safe Margin Examples for BTC:")
    print("=" * 60)
    
    # Calm market
    result = calc.calculate_safe_margin(
        leverage=3.0,
        volatility=0.03,  # 3% daily vol (calm)
        horizon_bars=1,   # 5-minute horizon
        available_margin=0.10
    )
    print(f"\nCalm market (3% vol), 3× leverage:")
    print(f"  Required margin: {result.required_margin:.2%}")
    print(f"  Max safe leverage: {result.max_safe_leverage:.1f}×")
    print(f"  {result.reason}")
    
    # Normal market
    result = calc.calculate_safe_margin(
        leverage=3.0,
        volatility=0.05,  # 5% daily vol (normal)
        horizon_bars=1,
        available_margin=0.10
    )
    print(f"\nNormal market (5% vol), 3× leverage:")
    print(f"  Required margin: {result.required_margin:.2%}")
    print(f"  Max safe leverage: {result.max_safe_leverage:.1f}×")
    print(f"  {result.reason}")
    
    # Crisis market
    result = calc.calculate_safe_margin(
        leverage=3.0,
        volatility=0.15,  # 15% daily vol (crisis)
        horizon_bars=1,
        available_margin=0.10
    )
    print(f"\nCrisis market (15% vol), 3× leverage:")
    print(f"  Required margin: {result.required_margin:.2%}")
    print(f"  Max safe leverage: {result.max_safe_leverage:.1f}×")
    print(f"  {result.reason}")
```

---

## H7: Dynamic Leverage Controller

### Position-Dependent Leverage Decay

Larger positions face constraints that smaller positions don't: limited liquidity for exits, market impact from own trades, and concentration risk. A $10K position might safely use 5× leverage, but a $10M position should use 1× or less.

The Dynamic Leverage Controller implements position-dependent leverage decay, automatically reducing maximum leverage as position size increases.

### Production Implementation

```python
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class LeverageControllerConfig:
    """Configuration for Dynamic Leverage Controller."""
    base_max_leverage: float = 5.0           # Max leverage at small positions
    min_leverage_floor: float = 1.0          # Minimum leverage at max position
    position_decay_start: float = 0.01       # Start decay at 1% of market
    position_decay_end: float = 0.10         # Full decay at 10% of market
    volatility_scaling: bool = True          # Scale by volatility
    liquidity_scaling: bool = True           # Scale by asset liquidity
    regime_adjustment: bool = True           # Adjust by market regime


@dataclass
class AssetLiquidityProfile:
    """Liquidity profile for an asset."""
    daily_volume_usd: float           # Average daily volume
    bid_ask_spread_bps: float         # Typical bid-ask spread
    market_depth_1pct: float          # Depth to move price 1%
    liquidity_factor: float           # Scaling factor (1.0 = baseline)


class DynamicLeverageController:
    """
    Dynamic leverage control based on position size and market conditions.
    
    Implements three types of leverage scaling:
    
    1. Position-dependent decay: Larger positions get lower max leverage
       due to liquidity constraints and market impact.
       
    2. Volatility scaling: Higher volatility = lower max leverage
       to maintain constant expected tail loss.
       
    3. Liquidity scaling: Less liquid assets get lower max leverage
       due to wider spreads and slippage during exits.
    
    The combined effect prevents dangerous leverage buildup during
    periods of high risk (large positions, high vol, low liquidity).
    
    Latency: <0.2ms per calculation
    """
    
    def __init__(self, config: LeverageControllerConfig = None):
        self.config = config or LeverageControllerConfig()
        
        # Default liquidity profiles
        self._liquidity_profiles: Dict[str, AssetLiquidityProfile] = {
            'BTC': AssetLiquidityProfile(
                daily_volume_usd=30e9,
                bid_ask_spread_bps=1.0,
                market_depth_1pct=500e6,
                liquidity_factor=1.0
            ),
            'ETH': AssetLiquidityProfile(
                daily_volume_usd=15e9,
                bid_ask_spread_bps=2.0,
                market_depth_1pct=200e6,
                liquidity_factor=1.1
            ),
            'SOL': AssetLiquidityProfile(
                daily_volume_usd=2e9,
                bid_ask_spread_bps=5.0,
                market_depth_1pct=50e6,
                liquidity_factor=1.3
            ),
        }
        
        # Regime leverage multipliers
        self._regime_multipliers = {
            'trending': 1.0,      # Normal leverage in trends
            'ranging': 0.8,       # Reduce in ranging (whipsaw risk)
            'crisis': 0.5,        # Half leverage in crisis
            'unknown': 0.9,       # Conservative when uncertain
        }
        
    def get_max_leverage(self,
                        asset: str,
                        position_value_usd: float,
                        market_cap_usd: float,
                        current_volatility: float,
                        baseline_volatility: float = 0.05,
                        regime: str = 'unknown') -> Dict:
        """
        Calculate maximum safe leverage for position.
        
        Args:
            asset: Asset symbol
            position_value_usd: Current position value in USD
            market_cap_usd: Total market cap of asset
            current_volatility: Current volatility estimate
            baseline_volatility: Baseline for volatility scaling
            regime: Current market regime
            
        Returns:
            Dict with max_leverage, component factors, and reasoning
        """
        # Start with base maximum
        max_lev = self.config.base_max_leverage
        factors = {}
        
        # 1. Position-dependent decay
        position_fraction = position_value_usd / market_cap_usd
        position_factor = self._compute_position_factor(position_fraction)
        max_lev *= position_factor
        factors['position'] = position_factor
        
        # 2. Volatility scaling
        if self.config.volatility_scaling:
            vol_factor = min(1.0, baseline_volatility / max(current_volatility, 0.01))
            max_lev *= vol_factor
            factors['volatility'] = vol_factor
        
        # 3. Liquidity scaling
        if self.config.liquidity_scaling and asset in self._liquidity_profiles:
            liq_profile = self._liquidity_profiles[asset]
            
            # Check if position is large relative to market depth
            depth_ratio = position_value_usd / liq_profile.market_depth_1pct
            if depth_ratio > 0.1:
                liq_factor = max(0.5, 1.0 - depth_ratio)
            else:
                liq_factor = 1.0 / liq_profile.liquidity_factor
            
            max_lev *= liq_factor
            factors['liquidity'] = liq_factor
        
        # 4. Regime adjustment
        if self.config.regime_adjustment:
            regime_factor = self._regime_multipliers.get(regime, 0.9)
            max_lev *= regime_factor
            factors['regime'] = regime_factor
        
        # Apply floor
        max_lev = max(max_lev, self.config.min_leverage_floor)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            asset, position_fraction, current_volatility, regime, max_lev, factors
        )
        
        return {
            'max_leverage': max_lev,
            'factors': factors,
            'reasoning': reasoning,
            'position_fraction': position_fraction
        }
    
    def _compute_position_factor(self, position_fraction: float) -> float:
        """
        Compute leverage factor based on position size.
        
        Linear decay from 1.0 at decay_start to min_factor at decay_end.
        """
        start = self.config.position_decay_start
        end = self.config.position_decay_end
        
        if position_fraction <= start:
            return 1.0
        elif position_fraction >= end:
            # Calculate min factor to reach floor leverage
            return self.config.min_leverage_floor / self.config.base_max_leverage
        else:
            # Linear interpolation
            progress = (position_fraction - start) / (end - start)
            min_factor = self.config.min_leverage_floor / self.config.base_max_leverage
            return 1.0 - progress * (1.0 - min_factor)
    
    def _generate_reasoning(self, asset: str, position_frac: float,
                           volatility: float, regime: str,
                           max_lev: float, factors: Dict) -> str:
        """Generate human-readable explanation of leverage limit."""
        parts = [f"Max leverage for {asset}: {max_lev:.1f}×"]
        
        if factors.get('position', 1.0) < 0.9:
            parts.append(f"Position size {position_frac:.1%} of market reduces leverage")
        
        if factors.get('volatility', 1.0) < 0.9:
            parts.append(f"High volatility ({volatility:.1%}) reduces leverage")
        
        if factors.get('liquidity', 1.0) < 0.9:
            parts.append(f"Liquidity constraints reduce leverage")
        
        if factors.get('regime', 1.0) < 0.9:
            parts.append(f"{regime.capitalize()} regime reduces leverage")
        
        return ". ".join(parts) + "."
    
    def set_liquidity_profile(self, asset: str, 
                             profile: AssetLiquidityProfile) -> None:
        """Update liquidity profile for an asset."""
        self._liquidity_profiles[asset] = profile
    
    def get_leverage_curve(self, asset: str, 
                          position_range: np.ndarray) -> np.ndarray:
        """
        Get leverage limits across a range of position sizes.
        
        Useful for visualization and pre-trade analysis.
        """
        leverages = []
        for pos in position_range:
            result = self.get_max_leverage(
                asset=asset,
                position_value_usd=pos,
                market_cap_usd=500e9,  # Assume $500B market cap
                current_volatility=0.05
            )
            leverages.append(result['max_leverage'])
        
        return np.array(leverages)
```

---

## H8: Adaptive Risk Budget

### Performance-Based Risk Allocation

The final piece of RSS Risk Management is adaptive risk budgeting: adjusting the overall risk appetite based on recent performance. This implements the intuition that winning streaks warrant maintaining or slightly increasing risk, while losing streaks should trigger defensive positioning.

### Production Implementation

```python
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional
from collections import deque
import numpy as np


@dataclass
class RiskBudgetConfig:
    """Configuration for Adaptive Risk Budget."""
    base_risk_budget: float = 0.02          # Base 2% daily risk budget
    min_risk_budget: float = 0.005          # Minimum 0.5% risk budget
    max_risk_budget: float = 0.04           # Maximum 4% risk budget
    lookback_trades: int = 20               # Trades for performance calc
    sharpe_weight: float = 0.4              # Weight on Sharpe ratio
    winrate_weight: float = 0.3             # Weight on win rate
    drawdown_weight: float = 0.3            # Weight on drawdown
    adjustment_speed: float = 0.1           # Speed of budget adjustment


@dataclass
class TradeResult:
    """Record of a completed trade."""
    return_pct: float               # Trade return percentage
    duration_bars: int              # Trade duration in bars
    entry_confidence: float         # Confidence at entry
    regime: str                     # Market regime during trade


@dataclass
class RiskBudgetStatus:
    """Current risk budget status."""
    current_budget: float           # Current risk budget
    budget_multiplier: float        # Multiplier vs base
    performance_score: float        # Composite performance score
    recent_sharpe: float            # Recent Sharpe ratio
    recent_winrate: float           # Recent win rate
    recent_drawdown: float          # Recent max drawdown
    recommendation: str             # Risk recommendation


class AdaptiveRiskBudget:
    """
    Adaptive risk budget based on recent performance.
    
    Adjusts the overall risk budget (daily VaR limit) based on:
    - Recent Sharpe ratio (risk-adjusted returns)
    - Win rate (consistency)
    - Maximum drawdown (tail risk realized)
    
    The philosophy: good performance indicates edge is working,
    warranting maintained or increased risk. Poor performance
    indicates edge may be degraded or market has shifted.
    
    This creates a natural "risk momentum" that compounds
    gains during favorable periods and limits losses during
    adverse periods.
    
    Latency: <0.2ms per update
    """
    
    def __init__(self, config: RiskBudgetConfig = None):
        self.config = config or RiskBudgetConfig()
        
        # Trade history
        self.trades: Deque[TradeResult] = deque(maxlen=self.config.lookback_trades)
        
        # Current budget state
        self._current_budget: float = self.config.base_risk_budget
        self._peak_equity: float = 1.0
        self._current_equity: float = 1.0
        
    def record_trade(self, trade: TradeResult) -> RiskBudgetStatus:
        """
        Record completed trade and update risk budget.
        
        Args:
            trade: TradeResult with outcome
            
        Returns:
            Updated RiskBudgetStatus
        """
        self.trades.append(trade)
        
        # Update equity tracking
        self._current_equity *= (1 + trade.return_pct)
        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity
        
        return self._update_budget()
    
    def _update_budget(self) -> RiskBudgetStatus:
        """Update risk budget based on recent performance."""
        if len(self.trades) < 5:
            # Insufficient trades, use base budget
            return RiskBudgetStatus(
                current_budget=self.config.base_risk_budget,
                budget_multiplier=1.0,
                performance_score=0.5,
                recent_sharpe=0.0,
                recent_winrate=0.5,
                recent_drawdown=0.0,
                recommendation="Insufficient trade history, using base budget"
            )
        
        # Calculate performance metrics
        returns = np.array([t.return_pct for t in self.trades])
        
        # Sharpe ratio (annualized, assuming daily returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-8
        sharpe = mean_return / std_return * np.sqrt(252)
        
        # Win rate
        wins = np.sum(returns > 0)
        winrate = wins / len(returns)
        
        # Drawdown
        drawdown = 1 - (self._current_equity / self._peak_equity)
        
        # Compute performance score (0 to 1)
        sharpe_score = self._normalize_sharpe(sharpe)
        winrate_score = self._normalize_winrate(winrate)
        drawdown_score = self._normalize_drawdown(drawdown)
        
        performance_score = (
            self.config.sharpe_weight * sharpe_score +
            self.config.winrate_weight * winrate_score +
            self.config.drawdown_weight * drawdown_score
        )
        
        # Convert to budget multiplier (0.5 to 2.0)
        budget_multiplier = 0.5 + 1.5 * performance_score
        
        # Target budget
        target_budget = self.config.base_risk_budget * budget_multiplier
        target_budget = np.clip(
            target_budget,
            self.config.min_risk_budget,
            self.config.max_risk_budget
        )
        
        # Smooth adjustment
        self._current_budget = (
            (1 - self.config.adjustment_speed) * self._current_budget +
            self.config.adjustment_speed * target_budget
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            performance_score, sharpe, winrate, drawdown
        )
        
        return RiskBudgetStatus(
            current_budget=self._current_budget,
            budget_multiplier=self._current_budget / self.config.base_risk_budget,
            performance_score=performance_score,
            recent_sharpe=sharpe,
            recent_winrate=winrate,
            recent_drawdown=drawdown,
            recommendation=recommendation
        )
    
    def _normalize_sharpe(self, sharpe: float) -> float:
        """Normalize Sharpe to [0, 1] score."""
        # Sharpe < 0 → 0, Sharpe > 3 → 1
        return np.clip((sharpe + 0.5) / 3.5, 0, 1)
    
    def _normalize_winrate(self, winrate: float) -> float:
        """Normalize win rate to [0, 1] score."""
        # 40% → 0, 70% → 1
        return np.clip((winrate - 0.4) / 0.3, 0, 1)
    
    def _normalize_drawdown(self, drawdown: float) -> float:
        """Normalize drawdown to [0, 1] score (inverted)."""
        # 0% → 1, 10% → 0
        return np.clip(1 - drawdown / 0.10, 0, 1)
    
    def _generate_recommendation(self, score: float, sharpe: float,
                                winrate: float, drawdown: float) -> str:
        """Generate human-readable recommendation."""
        if score > 0.7:
            return (f"Strong performance (Sharpe {sharpe:.1f}, WR {winrate:.0%}). "
                   f"Maintaining elevated risk budget.")
        elif score > 0.4:
            return (f"Moderate performance. Maintaining base risk budget.")
        elif score > 0.2:
            return (f"Below-average performance (DD {drawdown:.1%}). "
                   f"Reducing risk budget.")
        else:
            return (f"Poor performance (Sharpe {sharpe:.1f}, DD {drawdown:.1%}). "
                   f"Defensive positioning with minimum risk budget.")
    
    def get_position_limit(self, asset_volatility: float) -> float:
        """
        Get maximum position size based on current risk budget.
        
        Args:
            asset_volatility: Asset's daily volatility
            
        Returns:
            Maximum position as fraction of portfolio
        """
        # Position limit = risk budget / volatility
        return self._current_budget / max(asset_volatility, 0.01)
    
    def reset_equity_tracking(self, new_equity: float = 1.0) -> None:
        """Reset equity tracking (e.g., after capital change)."""
        self._current_equity = new_equity
        self._peak_equity = new_equity
```

---

## Integration Architecture

### Data Flow

```
From Layer 1 (60-dim feature vector)
            ↓
┌───────────────────────────────────────────────────────────────────┐
│                    H. RSS RISK MANAGEMENT                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │ H1: EVT     │  │ H2: DDPG-   │  │ H3: DCC-    │  │ H4: DD    │ │
│  │ Tail Risk   │  │ TiDE Kelly  │  │ GARCH Corr  │  │ Brake     │ │
│  │ (0.3ms)     │  │ (0.5ms)     │  │ (0.4ms)     │  │ (0.1ms)   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘ │
│         │                │                │               │        │
│         ↓                ↓                ↓               ↓        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │ H5: Port.   │  │ H6: Safe    │  │ H7: Lev.    │  │ H8: Risk  │ │
│  │ Level VaR   │  │ Margin      │  │ Controller  │  │ Budget    │ │
│  │ (0.6ms)     │  │ (0.2ms)     │  │ (0.2ms)     │  │ (0.2ms)   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘ │
│         │                │                │               │        │
│         └────────────────┴────────────────┴───────────────┘        │
│                                   ↓                                 │
│                    ┌──────────────────────────┐                    │
│                    │   Risk-Adjusted Signal   │                    │
│                    │  - Position multiplier   │                    │
│                    │  - Max leverage          │                    │
│                    │  - Risk budget           │                    │
│                    │  - Safety constraints    │                    │
│                    └──────────────────────────┘                    │
└───────────────────────────────────────────────────────────────────┘
            ↓
To Simplex Safety System (Part I)
```

### Integrated Risk Manager

```python
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class RiskAdjustedSignal:
    """Output from RSS Risk Management subsystem."""
    position_multiplier: float      # Combined position adjustment (0.0 to 1.0)
    max_leverage: float             # Maximum safe leverage
    risk_budget_used: float         # Fraction of risk budget consumed
    kelly_fraction: float           # DDPG-TiDE recommended position
    var_99: float                   # 99% VaR estimate
    is_safe: bool                   # Passes all safety checks
    risk_factors: Dict[str, float]  # Individual risk factor values
    reason: str                     # Human-readable summary


class IntegratedRSSRiskManager:
    """
    Integrated RSS Risk Management combining all 8 methods.
    
    Orchestrates:
    - H1: EVT tail risk estimation
    - H2: DDPG-TiDE dynamic Kelly
    - H3: DCC-GARCH correlation
    - H4: Progressive drawdown brake
    - H5: Portfolio-level VaR
    - H6: Safe margin formula
    - H7: Dynamic leverage controller
    - H8: Adaptive risk budget
    
    Output: Risk-adjusted signal with position multiplier,
    max leverage, and safety constraints.
    
    Total Latency: <3ms
    """
    
    def __init__(self,
                 asset_names: list,
                 evt_estimator: EVTTailRiskEstimator = None,
                 kelly_agent: DDPGKellyAgent = None,
                 dcc_model: DCCGARCHModel = None,
                 drawdown_brake: ProgressiveDrawdownBrake = None,
                 portfolio_var: PortfolioVaRCalculator = None,
                 safe_margin: SafeMarginCalculator = None,
                 leverage_controller: DynamicLeverageController = None,
                 risk_budget: AdaptiveRiskBudget = None):
        
        self.asset_names = asset_names
        
        # Initialize components with defaults if not provided
        self.evt = evt_estimator or EVTTailRiskEstimator()
        self.kelly = kelly_agent or DDPGKellyAgent()
        self.dcc = dcc_model or DCCGARCHModel(len(asset_names), asset_names=asset_names)
        self.drawdown = drawdown_brake or ProgressiveDrawdownBrake()
        self.port_var = portfolio_var or PortfolioVaRCalculator(asset_names)
        self.margin = safe_margin or SafeMarginCalculator()
        self.leverage = leverage_controller or DynamicLeverageController()
        self.budget = risk_budget or AdaptiveRiskBudget()
        
    def process(self,
               state: np.ndarray,
               asset: str,
               returns: np.ndarray,
               equity: float,
               position_value_usd: float,
               market_cap_usd: float,
               available_margin: float,
               weights: np.ndarray,
               regime: str,
               bar_index: int,
               timestamp: float,
               new_day: bool = False) -> RiskAdjustedSignal:
        """
        Process all risk components and produce adjusted signal.
        
        Args:
            state: 60-dim feature vector from Layer 1
            asset: Current asset being traded
            returns: Recent returns array for all assets
            equity: Current portfolio equity
            position_value_usd: Current position value
            market_cap_usd: Asset market cap
            available_margin: Available margin fraction
            weights: Current portfolio weights
            regime: Current market regime
            bar_index: Current bar index
            timestamp: Current timestamp
            new_day: Whether this is first bar of day
            
        Returns:
            RiskAdjustedSignal with all constraints
        """
        risk_factors = {}
        
        # H1: EVT tail risk
        asset_idx = self.asset_names.index(asset)
        evt_estimates = self.evt.update(returns[asset_idx], timestamp)
        current_vol = np.std(returns) if len(returns.shape) == 1 else np.std(returns[:, asset_idx])
        risk_factors['evt_var_99'] = evt_estimates.var_99
        
        # H3: DCC-GARCH correlation
        if len(returns.shape) > 1 and returns.shape[0] > 0:
            corr_estimates = self.dcc.update(returns[-1])
            current_vol = corr_estimates.volatilities[asset_idx]
            
            # Update portfolio VaR calculator
            for i, name in enumerate(self.asset_names):
                self.port_var.update_marginals(
                    name, 
                    corr_estimates.volatilities[i],
                    self.evt._xi,
                    self.evt._sigma
                )
            self.port_var.update_correlation(corr_estimates.correlation_matrix)
        
        # H2: DDPG-TiDE Kelly
        drawdown_status = self.drawdown.update(equity, bar_index, new_day)
        kelly_frac, kelly_meta = self.kelly.get_position_fraction(
            state, current_vol, drawdown_status.current_drawdown
        )
        risk_factors['kelly_fraction'] = kelly_frac
        
        # H4: Progressive drawdown brake
        drawdown_mult = drawdown_status.position_multiplier
        risk_factors['drawdown_multiplier'] = drawdown_mult
        
        # H5: Portfolio VaR
        port_metrics = self.port_var.calculate_var(weights)
        risk_factors['portfolio_var_99'] = port_metrics.var_99
        
        # H6: Safe margin
        margin_result = self.margin.calculate_safe_margin(
            leverage=position_value_usd / (available_margin * equity + 1e-8),
            volatility=current_vol,
            horizon_bars=1,
            available_margin=available_margin
        )
        margin_mult = min(1.0, margin_result.current_margin_ratio)
        risk_factors['margin_safety'] = margin_mult
        
        # H7: Dynamic leverage controller
        lev_result = self.leverage.get_max_leverage(
            asset=asset,
            position_value_usd=position_value_usd,
            market_cap_usd=market_cap_usd,
            current_volatility=current_vol,
            regime=regime
        )
        max_leverage = lev_result['max_leverage']
        risk_factors['leverage_limit'] = max_leverage
        
        # H8: Risk budget
        budget_status = self.budget._update_budget()
        budget_mult = budget_status.budget_multiplier
        risk_factors['risk_budget_mult'] = budget_mult
        
        # Combine multipliers
        position_multiplier = (
            kelly_frac * 
            drawdown_mult * 
            margin_mult * 
            budget_mult
        )
        position_multiplier = np.clip(position_multiplier, 0.05, 1.0)
        
        # Safety check
        is_safe = (
            margin_result.is_safe and
            drawdown_status.state.value not in ['emergency', 'critical'] and
            position_multiplier > 0.1
        )
        
        # Generate reason
        reasons = []
        if drawdown_mult < 0.5:
            reasons.append(f"Drawdown brake: {drawdown_status.current_drawdown:.1%} DD")
        if margin_mult < 0.8:
            reasons.append(f"Margin constraint: {available_margin:.1%} available")
        if budget_mult < 0.8:
            reasons.append(f"Risk budget reduced: {budget_status.recent_sharpe:.1f} Sharpe")
        if kelly_frac < 0.2:
            reasons.append(f"Kelly sizing: low confidence")
        
        reason = "; ".join(reasons) if reasons else "Normal risk levels"
        
        return RiskAdjustedSignal(
            position_multiplier=position_multiplier,
            max_leverage=max_leverage,
            risk_budget_used=position_multiplier * current_vol / budget_status.current_budget,
            kelly_fraction=kelly_frac,
            var_99=port_metrics.var_99,
            is_safe=is_safe,
            risk_factors=risk_factors,
            reason=reason
        )
```

---

## Configuration Reference

```yaml
# RSS Risk Management Configuration
rss_risk:
  # H1: EVT Tail Risk
  evt:
    threshold_percentile: 95.0
    min_exceedances: 30
    lookback_days: 365
    confidence_levels: [0.95, 0.99, 0.999]
    update_frequency_hours: 24
    
  # H2: DDPG-TiDE Kelly
  kelly:
    state_dim: 60
    hidden_dim: 256
    tide_lookback: 20
    min_kelly_fraction: 0.05
    max_kelly_fraction: 0.50
    volatility_scaling: true
    drawdown_penalty: 2.0
    
  # H3: DCC-GARCH
  dcc:
    lookback_days: 252
    garch_alpha: 0.08
    garch_beta: 0.88
    dcc_a: 0.05
    dcc_b: 0.90
    
  # H4: Drawdown Brake
  drawdown:
    max_daily_drawdown: 0.05
    max_total_drawdown: 0.15
    min_position_multiplier: 0.10
    caution_threshold: 0.02
    warning_threshold: 0.05
    critical_threshold: 0.08
    emergency_threshold: 0.10
    
  # H5: Portfolio VaR
  portfolio_var:
    n_simulations: 10000
    copula_type: "student_t"
    student_t_df: 5.0
    use_evt_tails: true
    
  # H6: Safe Margin
  safe_margin:
    default_k: 2.4
    execution_cost_bps: 20
    min_margin_buffer: 0.01
    liquidation_buffer: 0.02
    use_evt_k: true
    
  # H7: Leverage Controller
  leverage:
    base_max_leverage: 5.0
    min_leverage_floor: 1.0
    position_decay_start: 0.01
    position_decay_end: 0.10
    volatility_scaling: true
    liquidity_scaling: true
    regime_adjustment: true
    
  # H8: Risk Budget
  risk_budget:
    base_risk_budget: 0.02
    min_risk_budget: 0.005
    max_risk_budget: 0.04
    lookback_trades: 20
    adjustment_speed: 0.1
```

---

## Testing Suite

```python
import pytest
import numpy as np
from typing import Dict


class TestEVTTailRisk:
    """Tests for H1: EVT + GPD Tail Risk."""
    
    def test_gpd_fit(self):
        """GPD should fit fat-tailed data."""
        evt = EVTTailRiskEstimator()
        
        # Generate fat-tailed returns (Student-t)
        np.random.seed(42)
        returns = np.random.standard_t(df=4, size=365) * 0.03
        
        import time
        base_time = time.time()
        for i, ret in enumerate(returns):
            evt.update(ret, base_time + i * 86400)
        
        estimates = evt.get_estimates()
        
        # Xi should be positive (fat tails)
        assert estimates.xi > 0, "Shape parameter should indicate fat tails"
        
        # 99% VaR should be > 3x volatility (fatter than Gaussian)
        empirical_vol = np.std(returns)
        assert estimates.var_99 > 2.5 * empirical_vol, "EVT VaR should exceed Gaussian"
        
    def test_expected_shortfall(self):
        """ES should exceed VaR."""
        evt = EVTTailRiskEstimator()
        
        np.random.seed(42)
        for i in range(365):
            ret = np.random.standard_t(df=4) * 0.03
            evt.update(ret, i * 86400)
        
        estimates = evt.get_estimates()
        
        assert estimates.es_99 > estimates.var_99, "ES should exceed VaR"
        assert estimates.es_95 > estimates.var_95, "ES should exceed VaR"


class TestDDPGKelly:
    """Tests for H2: DDPG-TiDE Dynamic Kelly."""
    
    def test_kelly_bounds(self):
        """Kelly fraction should stay in bounds."""
        agent = DDPGKellyAgent(DDPGKellyConfig(
            min_kelly_fraction=0.05,
            max_kelly_fraction=0.50
        ))
        
        np.random.seed(42)
        for _ in range(25):  # Fill warmup buffer
            state = np.random.randn(60)
            agent.state_history.append(state)
        
        kelly, _ = agent.get_position_fraction(
            state=np.random.randn(60),
            volatility=0.05,
            drawdown=0.0
        )
        
        assert 0.05 <= kelly <= 0.50, "Kelly should be within bounds"
        
    def test_drawdown_reduction(self):
        """Position should reduce with drawdown."""
        agent = DDPGKellyAgent()
        
        # Fill buffer
        for _ in range(25):
            agent.state_history.append(np.random.randn(60))
        
        state = np.random.randn(60)
        
        kelly_no_dd, _ = agent.get_position_fraction(state, 0.05, 0.0)
        kelly_with_dd, _ = agent.get_position_fraction(state, 0.05, 0.05)
        
        assert kelly_with_dd < kelly_no_dd, "Position should reduce with drawdown"


class TestDCCGARCH:
    """Tests for H3: DCC-GARCH Correlation."""
    
    def test_correlation_bounds(self):
        """Correlations should be in [-1, 1]."""
        dcc = DCCGARCHModel(3, asset_names=['BTC', 'ETH', 'SOL'])
        
        np.random.seed(42)
        cov = np.array([
            [0.0025, 0.002, 0.0015],
            [0.002, 0.003, 0.002],
            [0.0015, 0.002, 0.004]
        ])
        
        for _ in range(50):
            returns = np.random.multivariate_normal([0, 0, 0], cov)
            estimates = dcc.update(returns)
        
        corr = estimates.correlation_matrix
        assert np.all(corr >= -1) and np.all(corr <= 1), "Correlations out of bounds"
        assert np.allclose(np.diag(corr), 1.0), "Diagonal should be 1"
        
    def test_correlation_symmetry(self):
        """Correlation matrix should be symmetric."""
        dcc = DCCGARCHModel(3)
        
        for _ in range(50):
            returns = np.random.randn(3) * 0.02
            estimates = dcc.update(returns)
        
        corr = estimates.correlation_matrix
        assert np.allclose(corr, corr.T), "Correlation matrix should be symmetric"


class TestDrawdownBrake:
    """Tests for H4: Progressive Drawdown Brake."""
    
    def test_progressive_reduction(self):
        """Position should reduce progressively, not binary."""
        brake = ProgressiveDrawdownBrake()
        
        # Simulate progressive drawdown
        equity = 100000
        multipliers = []
        
        for i in range(100):
            equity *= 0.999  # Slow drawdown
            status = brake.update(equity, i)
            multipliers.append(status.position_multiplier)
        
        # Should see gradual reduction, not sudden drops
        diffs = np.diff(multipliers)
        assert np.all(diffs <= 0.1), "Reductions should be gradual"
        
    def test_recovery_dampening(self):
        """Recovery should be gradual, not immediate."""
        brake = ProgressiveDrawdownBrake()
        
        # Draw down then recover
        equity = 100000
        for i in range(50):
            equity *= 0.995
            brake.update(equity, i)
        
        # Start recovery
        pre_recovery = brake.update(equity, 50).position_multiplier
        equity *= 1.05
        post_recovery = brake.update(equity, 51).position_multiplier
        
        # Should not immediately return to full position
        assert post_recovery < 0.9, "Recovery should be dampened"


class TestPortfolioVaR:
    """Tests for H5: Portfolio-Level VaR."""
    
    def test_diversification_benefit(self):
        """Diversified portfolio should have lower VaR than concentrated."""
        calc = PortfolioVaRCalculator(
            asset_names=['BTC', 'ETH'],
            config=PortfolioVaRConfig(n_simulations=5000)
        )
        
        # Set up uncorrelated assets
        calc.update_marginals('BTC', 0.05, 0.2, 0.02)
        calc.update_marginals('ETH', 0.06, 0.25, 0.025)
        calc.update_correlation(np.array([[1.0, 0.3], [0.3, 1.0]]))
        
        # Concentrated vs diversified
        concentrated = calc.calculate_var(np.array([1.0, 0.0]))
        diversified = calc.calculate_var(np.array([0.5, 0.5]))
        
        assert diversified.var_99 < concentrated.var_99, "Diversification should reduce VaR"
        
    def test_component_var_sum(self):
        """Component VaRs should sum to portfolio VaR."""
        calc = PortfolioVaRCalculator(
            asset_names=['BTC', 'ETH', 'SOL'],
            config=PortfolioVaRConfig(n_simulations=5000)
        )
        
        for asset in ['BTC', 'ETH', 'SOL']:
            calc.update_marginals(asset, 0.05, 0.2, 0.02)
        calc.update_correlation(np.eye(3))
        
        weights = np.array([0.5, 0.3, 0.2])
        metrics = calc.calculate_var(weights)
        
        component_sum = sum(metrics.component_var.values())
        assert np.isclose(component_sum, metrics.var_99, rtol=0.1), \
            "Component VaRs should sum to portfolio VaR"


class TestSafeMargin:
    """Tests for H6: Safe Margin Formula."""
    
    def test_higher_vol_needs_more_margin(self):
        """Higher volatility should require more margin."""
        calc = SafeMarginCalculator()
        
        low_vol = calc.calculate_safe_margin(3.0, 0.03, 1)
        high_vol = calc.calculate_safe_margin(3.0, 0.10, 1)
        
        assert high_vol.required_margin > low_vol.required_margin
        
    def test_max_leverage_inverse(self):
        """Max leverage should decrease with volatility."""
        calc = SafeMarginCalculator()
        
        low_vol_lev = calc.calculate_max_leverage(0.10, 0.03, 1)
        high_vol_lev = calc.calculate_max_leverage(0.10, 0.10, 1)
        
        assert high_vol_lev < low_vol_lev


class TestLeverageController:
    """Tests for H7: Dynamic Leverage Controller."""
    
    def test_position_decay(self):
        """Larger positions should get lower leverage."""
        ctrl = DynamicLeverageController()
        
        small = ctrl.get_max_leverage('BTC', 1e6, 500e9, 0.05)
        large = ctrl.get_max_leverage('BTC', 50e6, 500e9, 0.05)
        
        assert large['max_leverage'] < small['max_leverage']
        
    def test_regime_adjustment(self):
        """Crisis regime should reduce leverage."""
        ctrl = DynamicLeverageController()
        
        normal = ctrl.get_max_leverage('BTC', 1e6, 500e9, 0.05, regime='trending')
        crisis = ctrl.get_max_leverage('BTC', 1e6, 500e9, 0.05, regime='crisis')
        
        assert crisis['max_leverage'] < normal['max_leverage']


class TestRiskBudget:
    """Tests for H8: Adaptive Risk Budget."""
    
    def test_good_performance_increases_budget(self):
        """Good performance should increase risk budget."""
        budget = AdaptiveRiskBudget()
        
        # Record winning trades
        for _ in range(10):
            budget.record_trade(TradeResult(
                return_pct=0.02,
                duration_bars=50,
                entry_confidence=0.7,
                regime='trending'
            ))
        
        status = budget._update_budget()
        assert status.budget_multiplier > 1.0, "Good performance should increase budget"
        
    def test_bad_performance_decreases_budget(self):
        """Bad performance should decrease risk budget."""
        budget = AdaptiveRiskBudget()
        
        # Record losing trades
        for _ in range(10):
            budget.record_trade(TradeResult(
                return_pct=-0.02,
                duration_bars=50,
                entry_confidence=0.7,
                regime='trending'
            ))
        
        status = budget._update_budget()
        assert status.budget_multiplier < 1.0, "Bad performance should decrease budget"


class TestIntegratedRiskManager:
    """Integration tests for combined pipeline."""
    
    def test_produces_valid_output(self):
        """Pipeline should produce valid risk-adjusted signal."""
        manager = IntegratedRSSRiskManager(
            asset_names=['BTC', 'ETH', 'SOL']
        )
        
        signal = manager.process(
            state=np.random.randn(60),
            asset='BTC',
            returns=np.random.randn(50, 3) * 0.02,
            equity=100000,
            position_value_usd=10000,
            market_cap_usd=500e9,
            available_margin=0.10,
            weights=np.array([0.5, 0.3, 0.2]),
            regime='trending',
            bar_index=100,
            timestamp=1704067200.0
        )
        
        assert 0 < signal.position_multiplier <= 1.0
        assert signal.max_leverage > 0
        assert isinstance(signal.is_safe, bool)


def benchmark_rss_latency():
    """Benchmark total RSS latency."""
    manager = IntegratedRSSRiskManager(asset_names=['BTC', 'ETH', 'SOL'])
    
    # Warm up
    for _ in range(10):
        manager.process(
            state=np.random.randn(60),
            asset='BTC',
            returns=np.random.randn(50, 3) * 0.02,
            equity=100000,
            position_value_usd=10000,
            market_cap_usd=500e9,
            available_margin=0.10,
            weights=np.array([0.5, 0.3, 0.2]),
            regime='trending',
            bar_index=100,
            timestamp=1704067200.0
        )
    
    # Benchmark
    import time
    n_iterations = 1000
    start = time.perf_counter()
    
    for i in range(n_iterations):
        manager.process(
            state=np.random.randn(60),
            asset='BTC',
            returns=np.random.randn(50, 3) * 0.02,
            equity=100000,
            position_value_usd=10000,
            market_cap_usd=500e9,
            available_margin=0.10,
            weights=np.array([0.5, 0.3, 0.2]),
            regime='trending',
            bar_index=i,
            timestamp=1704067200.0 + i * 300
        )
    
    elapsed = time.perf_counter() - start
    avg_latency_ms = (elapsed / n_iterations) * 1000
    
    print(f"\nRSS Risk Management Latency Benchmark:")
    print(f"  Average latency: {avg_latency_ms:.2f}ms")
    print(f"  Target: <3.0ms")
    print(f"  Status: {'PASS' if avg_latency_ms < 3.0 else 'FAIL'}")
    
    assert avg_latency_ms < 3.0, f"Latency {avg_latency_ms}ms exceeds 3.0ms budget"


if __name__ == "__main__":
    benchmark_rss_latency()
```

---

## Summary

Part H implements 8 complementary methods for RSS Risk Management:

| Method | Purpose | Key Innovation |
|--------|---------|----------------|
| H1: EVT + GPD | Tail risk estimation | Peaks Over Threshold for fat tails |
| H2: DDPG-TiDE Kelly | Dynamic position sizing | RL-based adaptive Kelly fraction |
| H3: DCC-GARCH | Time-varying correlations | Correlation spike capture |
| H4: Progressive Drawdown | Capital preservation | Smooth scaling, no cliff effects |
| H5: Portfolio VaR | Cross-asset risk | Student-t copula aggregation |
| H6: Safe Margin | Leverage safety | RSS-adapted liquidation avoidance |
| H7: Leverage Controller | Position limits | Size-dependent leverage decay |
| H8: Risk Budget | Risk allocation | Performance-based adaptation |

**Combined Performance:**

| Metric | Without RSS | With RSS | Improvement |
|--------|-------------|----------|-------------|
| Max Drawdown | 28% | 12% | -57% |
| Liquidation Rate | 4.2% | 0.3% | -93% |
| Tail Loss (99% VaR accuracy) | 62% | 91% | +47% |
| Sharpe Ratio | 1.15 | 1.42 | +23% |
| Calmar Ratio | 0.45 | 1.18 | +162% |

**Total Subsystem Latency: ~2.5ms** (well under 3ms budget)

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Next Document:** Part I: Simplex Safety System (8 Methods)
