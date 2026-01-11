"""HIMARI Layer 2 - Part L: Sequential Testing, Performance Persistence"""

import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class SequentialTesting:
    """Sequential hypothesis testing for strategy validation."""
    def __init__(self, alpha: float = 0.05, beta: float = 0.20):
        self.alpha = alpha
        self.beta = beta
        self.log_likelihood_ratio = 0.0
        
    def update(self, observation: float, null_mean: float = 0, alt_mean: float = 0.01) -> str:
        # SPRT test
        lr = (observation - null_mean) / (alt_mean - null_mean) if alt_mean != null_mean else 0
        self.log_likelihood_ratio += lr
        
        upper = np.log((1 - self.beta) / self.alpha)
        lower = np.log(self.beta / (1 - self.alpha))
        
        if self.log_likelihood_ratio >= upper:
            return "ACCEPT_ALT"
        elif self.log_likelihood_ratio <= lower:
            return "ACCEPT_NULL"
        return "CONTINUE"


class PerformancePersistence:
    """Analyze performance persistence across time."""
    def __init__(self, window: int = 20):
        self.window = window
        self.returns: List[float] = []
        
    def add_return(self, ret: float) -> None:
        self.returns.append(ret)
        
    def compute_persistence(self) -> float:
        if len(self.returns) < self.window * 2:
            return 0.5
        first_half = self.returns[-self.window*2:-self.window]
        second_half = self.returns[-self.window:]
        corr = np.corrcoef(first_half, second_half)[0, 1]
        return corr if not np.isnan(corr) else 0.0
