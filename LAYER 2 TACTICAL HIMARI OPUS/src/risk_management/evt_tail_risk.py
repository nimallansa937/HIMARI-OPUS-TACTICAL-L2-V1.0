"""
HIMARI Layer 2 - Part H: EVT Tail Risk
H1: Extreme Value Theory for tail risk estimation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class EVTConfig:
    """EVT configuration."""
    threshold_quantile: float = 0.95
    min_exceedances: int = 30


class EVTTailRisk:
    """
    EVT-GPD tail risk estimation.
    Uses Generalized Pareto Distribution for extreme losses.
    """
    
    def __init__(self, config: Optional[EVTConfig] = None):
        self.config = config or EVTConfig()
        self.losses: list = []
        self.xi = 0.1  # Shape parameter
        self.beta = 0.01  # Scale parameter
        self.threshold = 0.0
        
    def fit(self, returns: np.ndarray) -> None:
        """Fit GPD to loss tail."""
        losses = -returns[returns < 0]
        self.losses = list(losses)
        
        self.threshold = np.quantile(losses, self.config.threshold_quantile)
        exceedances = losses[losses > self.threshold] - self.threshold
        
        if len(exceedances) >= self.config.min_exceedances:
            self.beta = np.mean(exceedances)
            self.xi = 0.5 * (np.var(exceedances) / self.beta**2 - 1)
            
    def var(self, alpha: float = 0.99) -> float:
        """Calculate VaR at confidence level."""
        n = len(self.losses)
        nu = sum(1 for l in self.losses if l > self.threshold)
        
        if nu == 0:
            return self.threshold
            
        return self.threshold + (self.beta / self.xi) * (
            ((n / nu) * (1 - alpha)) ** (-self.xi) - 1
        )
        
    def cvar(self, alpha: float = 0.99) -> float:
        """Calculate CVaR (Expected Shortfall)."""
        var = self.var(alpha)
        return (var + self.beta - self.xi * self.threshold) / (1 - self.xi)
