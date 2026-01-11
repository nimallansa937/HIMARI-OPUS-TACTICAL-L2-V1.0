"""HIMARI Layer 2 - Part H: Risk Budget"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskBudgetConfig:
    daily_budget: float = 0.02
    weekly_budget: float = 0.05
    monthly_budget: float = 0.10


class RiskBudget:
    """Track and enforce risk budget."""
    def __init__(self, config: Optional[RiskBudgetConfig] = None):
        self.config = config or RiskBudgetConfig()
        self.daily_losses: list = []
        self.weekly_losses: list = []
        
    def add_loss(self, loss: float) -> None:
        self.daily_losses.append(loss)
        self.weekly_losses.append(loss)
        
    def get_remaining_budget(self) -> Dict[str, float]:
        daily_used = sum(self.daily_losses[-1:]) if self.daily_losses else 0
        return {
            'daily_remaining': max(0, self.config.daily_budget - daily_used),
            'daily_utilization': daily_used / self.config.daily_budget
        }
        
    def can_trade(self) -> bool:
        budget = self.get_remaining_budget()
        return budget['daily_remaining'] > 0
