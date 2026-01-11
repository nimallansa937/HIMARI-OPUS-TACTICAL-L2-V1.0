"""
HIMARI Layer 2 - Part G: KAMA Adaptive Thresholds
G1: Kaufman Adaptive Moving Average for dynamic thresholds.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class KAMAConfig:
    """KAMA configuration."""
    period: int = 10
    fast_sc: int = 2
    slow_sc: int = 30


class KAMAAdaptive:
    """
    Kaufman Adaptive Moving Average for dynamic thresholds.
    Adjusts smoothing based on market efficiency ratio.
    """
    
    def __init__(self, config: Optional[KAMAConfig] = None):
        self.config = config or KAMAConfig()
        self.kama = None
        self.values = []
        
    def _efficiency_ratio(self, prices: np.ndarray) -> float:
        """Calculate efficiency ratio."""
        change = abs(prices[-1] - prices[0])
        volatility = np.sum(np.abs(np.diff(prices)))
        return change / volatility if volatility > 0 else 0
        
    def update(self, price: float) -> float:
        """Update KAMA with new price."""
        self.values.append(price)
        
        if len(self.values) < self.config.period:
            self.kama = price
            return self.kama
            
        prices = np.array(self.values[-self.config.period:])
        er = self._efficiency_ratio(prices)
        
        fast = 2 / (self.config.fast_sc + 1)
        slow = 2 / (self.config.slow_sc + 1)
        sc = (er * (fast - slow) + slow) ** 2
        
        if self.kama is None:
            self.kama = price
        else:
            self.kama = self.kama + sc * (price - self.kama)
            
        return self.kama
        
    def get_threshold(self, atr: float, multiplier: float = 2.0) -> float:
        """Get dynamic threshold based on KAMA."""
        return atr * multiplier * (1 + self._efficiency_ratio(np.array(self.values[-self.config.period:])))
