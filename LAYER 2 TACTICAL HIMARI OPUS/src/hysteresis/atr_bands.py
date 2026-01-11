"""
HIMARI Layer 2 - Part G: ATR-Scaled Bands
G3: Average True Range scaled bands for signal filtering.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ATRBandsConfig:
    """ATR bands configuration."""
    period: int = 14
    multiplier: float = 2.0


class ATRBands:
    """
    ATR-scaled bands for signal filtering.
    Signals outside bands are more likely valid.
    """
    
    def __init__(self, config: Optional[ATRBandsConfig] = None):
        self.config = config or ATRBandsConfig()
        self.highs: list = []
        self.lows: list = []
        self.closes: list = []
        
    def update(self, high: float, low: float, close: float) -> None:
        """Update with new bar."""
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
    def _calculate_atr(self) -> float:
        """Calculate ATR."""
        if len(self.closes) < 2:
            return 0.0
            
        n = min(len(self.closes), self.config.period)
        trs = []
        
        for i in range(-n, 0):
            high = self.highs[i]
            low = self.lows[i]
            prev_close = self.closes[i - 1] if i > -len(self.closes) else self.closes[i]
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
            
        return np.mean(trs)
        
    def get_bands(self) -> Tuple[float, float, float]:
        """Get upper band, middle, lower band."""
        if len(self.closes) == 0:
            return 0, 0, 0
            
        atr = self._calculate_atr()
        middle = self.closes[-1]
        upper = middle + self.config.multiplier * atr
        lower = middle - self.config.multiplier * atr
        
        return upper, middle, lower
        
    def is_breakout(self, price: float) -> Tuple[bool, str]:
        """Check if price breaks out of bands."""
        upper, middle, lower = self.get_bands()
        
        if price > upper:
            return True, 'UPPER'
        elif price < lower:
            return True, 'LOWER'
        return False, 'NONE'
