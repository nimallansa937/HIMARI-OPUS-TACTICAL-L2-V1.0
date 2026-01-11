"""
HIMARI Layer 2 - Part H: Portfolio VaR
H5: Portfolio-level Value at Risk calculation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioVaRConfig:
    """Portfolio VaR configuration."""
    confidence: float = 0.99
    lookback: int = 252
    method: str = "historical"  # or "parametric"


class PortfolioVaR:
    """
    Portfolio VaR calculation.
    Supports historical and parametric methods.
    """
    
    def __init__(self, config: Optional[PortfolioVaRConfig] = None):
        self.config = config or PortfolioVaRConfig()
        self.returns: list = []
        
    def update(self, return_val: float) -> None:
        """Update with portfolio return."""
        self.returns.append(return_val)
        if len(self.returns) > self.config.lookback:
            self.returns.pop(0)
            
    def calculate_var(self) -> float:
        """Calculate VaR at confidence level."""
        if len(self.returns) < 10:
            return 0.0
            
        if self.config.method == "historical":
            return -np.percentile(self.returns, (1 - self.config.confidence) * 100)
        else:
            mu = np.mean(self.returns)
            sigma = np.std(self.returns)
            z = 2.326 if self.config.confidence == 0.99 else 1.645
            return -(mu - z * sigma)
            
    def calculate_cvar(self) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self.calculate_var()
        tail_returns = [r for r in self.returns if r < -var]
        return -np.mean(tail_returns) if tail_returns else var
        
    def check_limit(self, limit: float) -> Dict:
        """Check if VaR exceeds limit."""
        var = self.calculate_var()
        return {
            'var': var,
            'limit': limit,
            'exceeded': var > limit,
            'utilization': var / limit if limit > 0 else 0
        }
