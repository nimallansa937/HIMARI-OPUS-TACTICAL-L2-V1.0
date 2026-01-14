"""
Regime Detector Wrapper for Dataset Generation Pipeline

Wraps the existing StudentTAHHMM to provide a simplified interface for
the enriched dataset generator.

Regime Mapping:
    StudentTAHHMM (4 states)     →  Pipeline Labels (4 states)
    ─────────────────────────────────────────────────────────
    RANGING                      →  LOW_VOL (0)
    TRENDING_UP / TRENDING_DOWN  →  TRENDING (1)
    HIGH_VOL (derived)           →  HIGH_VOL (2)
    CRISIS                       →  CRISIS (3)

Note: HIGH_VOL is derived from TRENDING + High Uncertainty meta-regime
      since the base HMM doesn't have explicit HIGH_VOL state.

Author: HIMARI Development Team
Date: January 2026
"""

import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import IntEnum
from collections import deque

# Add parent paths for imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_TACTICAL_ROOT = _PROJECT_ROOT / "LAYER 2 TACTICAL HIMARI OPUS"
_TACTICAL_SRC = _TACTICAL_ROOT / "src"

# Try to import the existing StudentTAHHMM
# We use TACTICAL/src path (not TACTICAL itself) to avoid src package conflict
HMM_AVAILABLE = False
StudentTAHHMM = None
AHHMMConfig = None
MarketRegime = None
MetaRegime = None

if _TACTICAL_SRC.exists():
    # Add TACTICAL/src directly to path
    if str(_TACTICAL_SRC) not in sys.path:
        sys.path.insert(0, str(_TACTICAL_SRC))

    try:
        # Import directly from regime_detection (not src.regime_detection)
        from regime_detection.student_t_ahhmm import (
            StudentTAHHMM,
            AHHMMConfig,
            MarketRegime,
            MetaRegime
        )
        HMM_AVAILABLE = True
    except ImportError as e:
        # Log the error for debugging
        import logging
        logging.debug(f"StudentTAHHMM import failed: {e}")


class RegimeLabel(IntEnum):
    """Standard regime labels for dataset."""
    LOW_VOL = 0      # Calm, range-bound (maps from RANGING)
    TRENDING = 1     # Directional momentum (maps from TRENDING_UP/DOWN)
    HIGH_VOL = 2     # Elevated volatility (derived from TRENDING + High Uncertainty)
    CRISIS = 3       # Extreme volatility (maps from CRISIS)


@dataclass
class RegimeOutput:
    """Output from regime detector."""
    regime_id: int              # 0-3
    regime_name: str            # Human readable
    confidence: float           # 0-1
    volatility: float           # Current realized volatility
    meta_regime: str            # LOW_UNCERTAINTY or HIGH_UNCERTAINTY
    raw_hmm_state: str          # Original HMM state name


class RegimeDetector:
    """
    Unified regime detector for dataset generation.

    Uses a multi-factor approach combining:
        - Volatility levels (realized vol percentiles)
        - Trend strength (directional momentum)
        - Volatility trend (expanding vs contracting)

    Produces balanced 4-regime distribution:
        - LOW_VOL (0): Low volatility, range-bound
        - TRENDING (1): Clear directional movement, moderate vol
        - HIGH_VOL (2): Elevated volatility, choppy
        - CRISIS (3): Extreme volatility spikes

    Parameters:
        vol_window: Window for volatility calculation (default 24)
        trend_window: Window for trend calculation (default 48)
        vol_lookback: Lookback for volatility percentiles (default 168 = 1 week)
        use_hmm: Whether to use HMM (default False, uses balanced detector)
    """

    # Volatility percentile thresholds (more granular for balance)
    VOL_P25 = 0.25   # Below this = LOW_VOL
    VOL_P50 = 0.50   # Below this = could be TRENDING
    VOL_P75 = 0.75   # Below this = HIGH_VOL
    VOL_P90 = 0.90   # Above this = CRISIS

    # Trend thresholds (absolute value of normalized trend)
    TREND_WEAK = 0.3     # Below = ranging/choppy
    TREND_STRONG = 0.6   # Above = clear trend

    def __init__(
        self,
        vol_window: int = 24,
        trend_window: int = 48,
        vol_lookback: int = 168,
        use_hmm: bool = False  # Default to balanced detector
    ):
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.vol_lookback = vol_lookback
        self.use_hmm = HMM_AVAILABLE and use_hmm

        if self.use_hmm:
            # Initialize StudentTAHHMM
            config = AHHMMConfig(
                n_market_states=4,
                n_meta_states=2,
                df=5.0,
                update_window=500
            )
            self.hmm = StudentTAHHMM(config)
            self._fitted = False
        else:
            self.hmm = None
            self._fitted = True  # Balanced detector doesn't need fitting

        # Buffers for calculations
        self.returns_buffer: deque = deque(maxlen=max(vol_lookback, 500))
        self.price_buffer: deque = deque(maxlen=max(trend_window, 100))
        self.vol_buffer: deque = deque(maxlen=vol_lookback)

        # Annualization factor for hourly data
        self.annualization = np.sqrt(365 * 24)

        # State tracking
        self.n_updates = 0
        self.current_regime = RegimeLabel.LOW_VOL

    def fit(self, returns: np.ndarray, volatility: np.ndarray = None) -> None:
        """
        Fit the HMM on historical data.

        Args:
            returns: Array of returns (T,) or (T, 1)
            volatility: Optional volatility array (T,)
        """
        if not self.use_hmm:
            return

        # Prepare observation matrix [returns, volume_proxy, volatility]
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        T = len(returns)

        # Create observation matrix
        if volatility is None:
            # Compute rolling volatility
            volatility = np.zeros(T)
            for t in range(self.vol_window, T):
                volatility[t] = np.std(returns[t-self.vol_window:t, 0])

        # Volume proxy (just use 1.0 for now, or could pass actual volume)
        volume_proxy = np.ones(T) * 0.8

        # Stack observations
        obs = np.column_stack([
            returns[:, 0],
            volume_proxy,
            volatility
        ])

        # Fit HMM
        self.hmm.fit(obs[self.vol_window:], n_iter=50)
        self._fitted = True

    def update(
        self,
        price: float,
        returns: float = None,
        volatility: float = None,
        vix: float = None
    ) -> RegimeOutput:
        """
        Update regime detection with new observation.

        Args:
            price: Current price
            returns: Current returns (computed if None)
            volatility: Current volatility (computed if None)
            vix: Optional VIX/fear index for meta-regime

        Returns:
            RegimeOutput with regime_id and confidence
        """
        # Store price
        self.price_buffer.append(price)

        # Compute returns if not provided
        if returns is None:
            if len(self.price_buffer) >= 2:
                prev_price = self.price_buffer[-2]
                if prev_price > 0:
                    returns = (price - prev_price) / prev_price
                else:
                    returns = 0.0
            else:
                returns = 0.0

        self.returns_buffer.append(returns)
        self.n_updates += 1

        # Compute current volatility if not provided
        if volatility is None:
            if len(self.returns_buffer) >= self.vol_window:
                returns_arr = np.array(list(self.returns_buffer))[-self.vol_window:]
                volatility = float(np.std(returns_arr) * self.annualization)
            else:
                volatility = 0.0

        # Use HMM if available and fitted
        if self.use_hmm and self._fitted:
            return self._hmm_detect(returns, volatility, vix)
        else:
            return self._fallback_detect(returns, volatility)

    def _hmm_detect(
        self,
        returns: float,
        volatility: float,
        vix: float = None
    ) -> RegimeOutput:
        """Detect regime using StudentTAHHMM."""
        # Prepare observation [return, volume_proxy, volatility]
        obs = np.array([returns, 0.8, volatility / self.annualization])

        # Get HMM prediction
        state = self.hmm.predict(obs, vix=vix)

        # Map HMM regime to pipeline labels
        regime_id, regime_name = self._map_hmm_regime(
            state.regime,
            state.meta_regime,
            volatility
        )

        self.current_regime = RegimeLabel(regime_id)

        return RegimeOutput(
            regime_id=regime_id,
            regime_name=regime_name,
            confidence=state.confidence,
            volatility=volatility,
            meta_regime=state.meta_regime.value,
            raw_hmm_state=state.regime.value
        )

    def _map_hmm_regime(
        self,
        hmm_regime,  # MarketRegime (type hint removed for fallback compatibility)
        meta_regime,  # MetaRegime
        volatility: float
    ) -> Tuple[int, str]:
        """
        Map HMM regime to pipeline labels.

        Mapping logic:
            RANGING → LOW_VOL (0)
            TRENDING_UP/DOWN + Low Uncertainty → TRENDING (1)
            TRENDING_UP/DOWN + High Uncertainty → HIGH_VOL (2)
            CRISIS → CRISIS (3)
        """
        if hmm_regime == MarketRegime.CRISIS:
            return RegimeLabel.CRISIS, "CRISIS"

        elif hmm_regime == MarketRegime.RANGING:
            # Could be LOW_VOL or HIGH_VOL based on actual volatility
            if volatility > self.VOL_HIGH_LOWER:
                return RegimeLabel.HIGH_VOL, "HIGH_VOL"
            return RegimeLabel.LOW_VOL, "LOW_VOL"

        elif hmm_regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            # Check meta-regime for HIGH_VOL distinction
            if meta_regime == MetaRegime.HIGH_UNCERTAINTY:
                return RegimeLabel.HIGH_VOL, "HIGH_VOL"
            return RegimeLabel.TRENDING, "TRENDING"

        # Default
        return RegimeLabel.LOW_VOL, "LOW_VOL"

    def _fallback_detect(self, returns: float, volatility: float) -> RegimeOutput:
        """
        Balanced multi-factor regime detection.

        Uses percentile-based volatility classification combined with
        trend strength to produce balanced regime distribution.
        """
        # Store current volatility
        self.vol_buffer.append(volatility)

        # Compute volatility percentile (relative to recent history)
        if len(self.vol_buffer) >= 50:
            vol_arr = np.array(list(self.vol_buffer))
            vol_percentile = np.sum(vol_arr < volatility) / len(vol_arr)
        else:
            # Not enough history - use absolute thresholds
            vol_percentile = min(1.0, volatility / 0.8)  # Normalize to ~0-1

        # Compute trend strength
        trend_strength = 0.0
        trend_direction = 0.0
        if len(self.returns_buffer) >= self.trend_window:
            returns_arr = np.array(list(self.returns_buffer))[-self.trend_window:]

            # Cumulative return over window
            cumret = np.sum(returns_arr)

            # Trend consistency (how many periods in same direction as overall)
            if cumret > 0:
                consistency = np.mean(returns_arr > 0)
            else:
                consistency = np.mean(returns_arr < 0)

            # Normalize trend strength
            ret_std = np.std(returns_arr) + 1e-10
            trend_strength = abs(cumret) / (ret_std * np.sqrt(self.trend_window))
            trend_strength = min(1.0, trend_strength * consistency)
            trend_direction = np.sign(cumret)

        # Compute volatility trend (expanding or contracting)
        vol_trend = 0.0
        if len(self.vol_buffer) >= 48:
            vol_arr = np.array(list(self.vol_buffer))
            recent_vol = np.mean(vol_arr[-12:])
            older_vol = np.mean(vol_arr[-48:-12]) + 1e-10
            vol_trend = (recent_vol - older_vol) / older_vol

        # === REGIME CLASSIFICATION ===
        # Decision tree based on volatility percentile and trend strength

        if vol_percentile >= self.VOL_P90:
            # Extreme volatility = CRISIS
            regime_id = RegimeLabel.CRISIS
            confidence = 0.85 + 0.1 * (vol_percentile - self.VOL_P90) / 0.1

        elif vol_percentile >= self.VOL_P75:
            # High volatility zone
            if trend_strength >= self.TREND_STRONG:
                # Strong trend in high vol = still TRENDING (volatile trend)
                regime_id = RegimeLabel.TRENDING
                confidence = 0.7 + 0.1 * trend_strength
            else:
                # High vol without clear trend = HIGH_VOL (choppy)
                regime_id = RegimeLabel.HIGH_VOL
                confidence = 0.75 + 0.1 * (vol_percentile - self.VOL_P75) / 0.15

        elif vol_percentile >= self.VOL_P50:
            # Medium volatility zone - depends on trend
            if trend_strength >= self.TREND_STRONG:
                # Clear trend = TRENDING
                regime_id = RegimeLabel.TRENDING
                confidence = 0.75 + 0.15 * trend_strength
            elif trend_strength >= self.TREND_WEAK:
                # Weak trend in medium vol
                if vol_trend > 0.1:
                    # Expanding vol = HIGH_VOL
                    regime_id = RegimeLabel.HIGH_VOL
                    confidence = 0.65 + 0.1 * vol_trend
                else:
                    # Stable/contracting vol = TRENDING
                    regime_id = RegimeLabel.TRENDING
                    confidence = 0.65 + 0.1 * trend_strength
            else:
                # No trend, medium vol = could be either
                if vol_trend > 0.05:
                    regime_id = RegimeLabel.HIGH_VOL
                    confidence = 0.6
                else:
                    regime_id = RegimeLabel.LOW_VOL
                    confidence = 0.55

        elif vol_percentile >= self.VOL_P25:
            # Low-medium volatility
            if trend_strength >= self.TREND_WEAK:
                # Any trend = TRENDING
                regime_id = RegimeLabel.TRENDING
                confidence = 0.7 + 0.15 * trend_strength
            else:
                # No trend = LOW_VOL
                regime_id = RegimeLabel.LOW_VOL
                confidence = 0.7

        else:
            # Very low volatility = LOW_VOL
            regime_id = RegimeLabel.LOW_VOL
            confidence = 0.8 + 0.1 * (self.VOL_P25 - vol_percentile) / self.VOL_P25

        # Clamp confidence
        confidence = float(np.clip(confidence, 0.5, 0.95))

        self.current_regime = regime_id

        return RegimeOutput(
            regime_id=int(regime_id),
            regime_name=regime_id.name,
            confidence=confidence,
            volatility=volatility,
            meta_regime=f"vol_p{int(vol_percentile*100)}_trend{trend_strength:.2f}",
            raw_hmm_state="BALANCED_DETECTOR"
        )

    def batch_detect(
        self,
        prices: np.ndarray,
        returns: np.ndarray = None,
        fit_first: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process batch of prices and return regime labels.

        Args:
            prices: Price array (T,)
            returns: Optional returns array (T,)
            fit_first: Whether to fit HMM before detection

        Returns:
            (regime_ids, confidences) each of shape (T,)
        """
        T = len(prices)

        # Compute returns if not provided
        if returns is None:
            returns = np.zeros(T)
            returns[1:] = np.diff(prices) / prices[:-1]

        # Fit HMM on initial portion if requested
        if fit_first and self.use_hmm and not self._fitted:
            fit_size = min(1000, T // 2)
            if fit_size >= 100:
                self.fit(returns[:fit_size])

        # Process each timestep
        regime_ids = np.zeros(T, dtype=np.int32)
        confidences = np.zeros(T, dtype=np.float32)

        for t in range(T):
            result = self.update(prices[t], returns[t])
            regime_ids[t] = result.regime_id
            confidences[t] = result.confidence

        return regime_ids, confidences

    def reset(self) -> None:
        """Reset detector state."""
        self.returns_buffer.clear()
        self.price_buffer.clear()
        self.n_updates = 0
        self.current_regime = RegimeLabel.LOW_VOL

        if self.use_hmm:
            self.hmm = StudentTAHHMM(AHHMMConfig())
            self._fitted = False

    def get_diagnostics(self) -> dict:
        """Get detector diagnostics."""
        diag = {
            'n_updates': self.n_updates,
            'current_regime': self.current_regime.name,
            'using_hmm': self.use_hmm,
            'hmm_fitted': self._fitted,
            'buffer_size': len(self.returns_buffer)
        }

        if self.use_hmm and hasattr(self.hmm, 'state_probs'):
            diag['hmm_state_probs'] = self.hmm.state_probs.tolist()

        return diag


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Balanced Regime Detector Test")
    print("=" * 60)
    print(f"HMM Available: {HMM_AVAILABLE}")

    # Create detector (use_hmm=False for balanced detector)
    detector = RegimeDetector(vol_window=24, use_hmm=False)

    # Generate synthetic price data with varying regimes
    np.random.seed(42)
    T = 5000

    # Simulate realistic BTC-like price action
    prices = [40000.0]
    segments = [
        # (n_steps, drift, vol) - simulate different market conditions
        (400, 0.0002, 0.008),    # Low vol consolidation
        (300, 0.002, 0.012),     # Trending up
        (200, -0.001, 0.020),    # High vol chop
        (100, -0.008, 0.040),    # Crisis dump
        (300, 0.0003, 0.010),    # Recovery consolidation
        (400, 0.003, 0.015),     # Strong trend
        (250, 0.0, 0.025),       # High vol sideways
        (150, -0.005, 0.035),    # Another dip
        (500, 0.001, 0.011),     # Normal trending
        (400, 0.0001, 0.007),    # Low vol
        (300, -0.002, 0.018),    # Downtrend high vol
        (200, 0.004, 0.014),     # Recovery trend
        (500, 0.0, 0.009),       # Ranging low vol
        (300, 0.002, 0.022),     # Volatile uptrend
        (200, -0.006, 0.030),    # Sharp correction
        (300, 0.001, 0.012),     # Normal
    ]

    for n_steps, drift, vol in segments:
        for _ in range(n_steps):
            ret = drift + np.random.randn() * vol
            prices.append(prices[-1] * (1 + ret))

    prices = np.array(prices)[:T]

    # Detect regimes
    regime_ids, confidences = detector.batch_detect(prices, fit_first=False)

    # Print distribution
    print("\nRegime Distribution:")
    total = len(regime_ids)
    for regime in RegimeLabel:
        count = np.sum(regime_ids == regime.value)
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {regime.name:12s}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\nAverage Confidence: {np.mean(confidences):.3f}")

    # Check balance
    counts = [np.sum(regime_ids == r.value) for r in RegimeLabel]
    min_pct = min(counts) / total * 100
    max_pct = max(counts) / total * 100

    print(f"\nBalance check:")
    print(f"  Min regime: {min_pct:.1f}%")
    print(f"  Max regime: {max_pct:.1f}%")
    print(f"  Spread: {max_pct - min_pct:.1f}%")

    if min_pct >= 10 and max_pct <= 40:
        print("\n[OK] Balanced regime distribution achieved!")
    else:
        print("\n[WARN] Distribution may need tuning")

    print(f"\nDiagnostics: {detector.get_diagnostics()}")
