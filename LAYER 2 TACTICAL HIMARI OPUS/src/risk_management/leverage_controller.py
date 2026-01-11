"""HIMARI Layer 2 - Part H: Leverage Controller"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class LeverageControllerConfig:
    max_leverage: float = 2.0
    target_leverage: float = 1.0
    regime_adjustments: dict = None
    
    def __post_init__(self):
        if self.regime_adjustments is None:
            self.regime_adjustments = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5}


class LeverageController:
    """Dynamic leverage control based on regime."""
    def __init__(self, config: Optional[LeverageControllerConfig] = None):
        self.config = config or LeverageControllerConfig()
        
    def get_leverage(self, regime: int, volatility_scale: float = 1.0) -> float:
        base = self.config.regime_adjustments.get(regime, 1.0)
        adjusted = base * self.config.target_leverage / volatility_scale
        return min(adjusted, self.config.max_leverage)
