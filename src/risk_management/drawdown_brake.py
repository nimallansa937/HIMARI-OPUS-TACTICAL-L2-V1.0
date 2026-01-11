"""
HIMARI Layer 2 - Part H: Drawdown Brake
H4: Progressive position reduction based on drawdown.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DrawdownBrakeConfig:
    """Drawdown brake configuration."""
    thresholds: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = [
                (0.05, 1.0),   # 5% DD -> 100%
                (0.10, 0.5),   # 10% DD -> 50%
                (0.15, 0.25),  # 15% DD -> 25%
                (0.20, 0.0),   # 20% DD -> 0%
            ]


class DrawdownBrake:
    """
    Progressive drawdown-based position scaling.
    Reduces exposure as drawdown increases.
    """
    
    def __init__(self, config: Optional[DrawdownBrakeConfig] = None):
        self.config = config or DrawdownBrakeConfig()
        self.peak = 0.0
        self.current = 0.0
        
    def update(self, equity: float) -> None:
        """Update equity tracking."""
        self.current = equity
        self.peak = max(self.peak, equity)
        
    def get_drawdown(self) -> float:
        """Get current drawdown."""
        if self.peak == 0:
            return 0.0
        return (self.peak - self.current) / self.peak
        
    def get_scale(self) -> float:
        """Get position scale based on drawdown."""
        dd = self.get_drawdown()
        
        for threshold, scale in sorted(self.config.thresholds, reverse=True):
            if dd >= threshold:
                return scale
                
        return 1.0
        
    def is_trading_halted(self) -> bool:
        """Check if trading should be halted."""
        return self.get_scale() == 0.0
