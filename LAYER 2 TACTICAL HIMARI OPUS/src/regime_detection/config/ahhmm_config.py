"""
HIMARI Layer 2 - AH-HMM Configuration
Subsystem B: Regime Detection (Method B1)

Configuration for Adaptive Hierarchical Hidden Markov Model with Student-t emissions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class MetaRegime(Enum):
    """
    Meta-regime layer: slow-moving structural market conditions.
    
    LOW_UNCERTAINTY: Central bank accommodation, stable growth, low VIX
    HIGH_UNCERTAINTY: Policy tightening, geopolitical risk, elevated VIX
    
    Meta-regime governs transition probabilities in the market regime layer.
    """
    LOW_UNCERTAINTY = "low_uncertainty"
    HIGH_UNCERTAINTY = "high_uncertainty"


class MarketRegime(Enum):
    """
    Market regime layer: fast-moving tactical market states.
    
    BULL: Trending up, momentum strategies favored
    BEAR: Trending down, defensive positioning
    SIDEWAYS: Range-bound, mean-reversion strategies favored
    CRISIS: Extreme volatility, capital preservation mode
    """
    BULL = 0
    BEAR = 1
    SIDEWAYS = 2
    CRISIS = 3


@dataclass
class EmissionParams:
    """
    Student-t emission parameters for a single regime.
    
    Each regime has characteristic statistical fingerprints:
    - mean: Expected return, volume ratio, volatility level
    - scale: Dispersion around the mean
    - df: Degrees of freedom (lower = fatter tails)
    """
    mean: np.ndarray
    scale: np.ndarray
    df: float


@dataclass
class AHHMMConfig:
    """
    Configuration for Adaptive Hierarchical Hidden Markov Model.
    
    The AH-HMM operates on two hierarchical levels:
    1. Meta-regime (slow): LOW_UNCERTAINTY or HIGH_UNCERTAINTY
    2. Market regime (fast): BULL, BEAR, SIDEWAYS, CRISIS
    
    The meta-regime modifies transition probabilities—crisis transitions
    are much more likely during high uncertainty periods.
    """
    
    # Structural parameters
    n_market_states: int = 4
    n_meta_states: int = 2
    n_features: int = 3  # [return, volume_norm, volatility]
    
    # Student-t parameters
    df_normal: float = 5.0      # Degrees of freedom for normal regimes
    df_crisis: float = 3.0      # Even fatter tails during crisis
    
    # Online learning parameters
    update_window: int = 500    # Observations for online Baum-Welch
    transition_prior: float = 0.1  # Dirichlet prior strength
    learning_rate: float = 0.01    # EMA decay for parameter updates
    
    # Meta-regime thresholds
    vix_high_threshold: float = 30.0   # VIX > 30 → high uncertainty
    vix_low_threshold: float = 20.0    # VIX < 20 → low uncertainty
    epu_high_threshold: float = 200.0  # EPU > 200 → high uncertainty
    epu_low_threshold: float = 100.0   # EPU < 100 → low uncertainty
    
    # Transition matrices (meta-regime specific)
    # These are learned during training but initialized with domain knowledge
    trans_low_uncertainty: np.ndarray = field(default_factory=lambda: np.array([
        [0.92, 0.04, 0.03, 0.01],  # Bull: sticky, rare crisis
        [0.08, 0.88, 0.03, 0.01],  # Bear: sticky, rare crisis
        [0.12, 0.12, 0.75, 0.01],  # Sideways: can go either way
        [0.25, 0.15, 0.10, 0.50]   # Crisis: short-lived
    ]))
    
    trans_high_uncertainty: np.ndarray = field(default_factory=lambda: np.array([
        [0.70, 0.12, 0.08, 0.10],  # Bull: fragile, crisis-prone
        [0.05, 0.72, 0.08, 0.15],  # Bear: sticky, crisis-prone
        [0.08, 0.12, 0.60, 0.20],  # Sideways: unstable
        [0.08, 0.08, 0.04, 0.80]   # Crisis: persistent
    ]))
    
    # Emission parameters per regime
    emission_params: Dict[MarketRegime, EmissionParams] = field(
        default_factory=lambda: {
            MarketRegime.BULL: EmissionParams(
                mean=np.array([0.002, 0.8, 0.015]),   # +0.2%/bar, normal volume, low vol
                scale=np.array([0.010, 0.3, 0.005]),
                df=5.0
            ),
            MarketRegime.BEAR: EmissionParams(
                mean=np.array([-0.002, 1.0, 0.025]),  # -0.2%/bar, high volume, elevated vol
                scale=np.array([0.015, 0.4, 0.008]),
                df=5.0
            ),
            MarketRegime.SIDEWAYS: EmissionParams(
                mean=np.array([0.0, 0.6, 0.012]),     # 0% drift, low volume, low vol
                scale=np.array([0.008, 0.2, 0.004]),
                df=5.0
            ),
            MarketRegime.CRISIS: EmissionParams(
                mean=np.array([-0.015, 2.5, 0.080]),  # -1.5%/bar, extreme volume, extreme vol
                scale=np.array([0.040, 1.0, 0.030]),
                df=3.0  # Fatter tails during crisis
            ),
        }
    )


# Global configuration instance
DEFAULT_AHHMM_CONFIG = AHHMMConfig()
