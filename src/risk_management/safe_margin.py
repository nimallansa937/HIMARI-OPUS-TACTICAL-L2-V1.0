"""HIMARI Layer 2 - Part H: Safe Margin, Leverage Controller, Risk Budget"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SafeMarginConfig:
    min_margin_ratio: float = 0.3
    warning_margin_ratio: float = 0.5


class SafeMargin:
    """Safe margin monitoring."""
    def __init__(self, config: Optional[SafeMarginConfig] = None):
        self.config = config or SafeMarginConfig()
        
    def check_margin(self, equity: float, margin_used: float) -> dict:
        ratio = equity / margin_used if margin_used > 0 else float('inf')
        return {
            'ratio': ratio,
            'safe': ratio >= self.config.min_margin_ratio,
            'warning': ratio < self.config.warning_margin_ratio
        }
