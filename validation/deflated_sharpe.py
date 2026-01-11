"""HIMARI Layer 2 - Part L: Deflated Sharpe Ratio"""

import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DeflatedSharpe:
    """Deflated Sharpe Ratio for multiple testing correction."""
    def __init__(self):
        pass
        
    def compute(self, sharpe: float, n_trials: int, n_obs: int, skew: float = 0, kurt: float = 3) -> float:
        """Compute probability that Sharpe is genuine."""
        # Bailey-Lopez de Prado deflated Sharpe
        e_max_sr = (1 - 0.5772) * stats.norm.ppf(1 - 1/n_trials) + 0.5772 * stats.norm.ppf(1 - 1/(n_trials * np.e))
        var_sr = (1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 3)/4 * sharpe**2) / (n_obs - 1)
        psr = stats.norm.cdf((sharpe - e_max_sr) / np.sqrt(var_sr))
        return psr
