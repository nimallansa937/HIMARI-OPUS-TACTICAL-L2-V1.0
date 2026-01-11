"""
HIMARI Layer 2 - Meta-Regime Configuration
Subsystem B: Regime Detection (Method B2)

Configuration for Meta-Regime Layer that tracks structural market conditions.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np


@dataclass
class MetaRegimeConfig:
    """
    Configuration for Meta-Regime Layer.
    
    The meta-regime layer monitors macroeconomic indicators to determine
    whether the market is in a structurally calm or stressed environment.
    This governs transition dynamics in the fast-moving market regime layer.
    """
    
    # Indicator weights (must sum to 1.0)
    indicator_weights: Dict[str, float] = field(default_factory=lambda: {
        "vix": 0.30,           # Equity volatility (proxy for risk appetite)
        "dvol": 0.25,          # Crypto volatility (direct measure)
        "epu": 0.20,           # Economic Policy Uncertainty
        "funding_rate": 0.15,  # Perpetual funding rate (leverage indicator)
        "oi_change": 0.10      # Open interest change (positioning indicator)
    })
    
    # Thresholds for each indicator (low, high)
    indicator_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "vix": (18.0, 28.0),
        "dvol": (50.0, 80.0),
        "epu": (100.0, 200.0),
        "funding_rate": (-0.01, 0.03),  # Extreme negative or positive
        "oi_change": (-0.05, 0.10)      # -5% to +10% daily change
    })
    
    # Hysteresis parameters
    transition_up_threshold: float = 0.65    # Score to switch LOW → HIGH
    transition_down_threshold: float = 0.35  # Score to switch HIGH → LOW
    min_bars_between_transitions: int = 48   # 4 hours at 5min bars
    
    # EMA smoothing
    score_ema_span: int = 24  # 2 hours of smoothing
    
    # Impact on market regime
    crisis_probability_boost: float = 2.0    # HIGH unc: crisis prob × 2
    transition_probability_boost: float = 1.5 # HIGH unc: all transitions × 1.5


DEFAULT_META_REGIME_CONFIG = MetaRegimeConfig()
