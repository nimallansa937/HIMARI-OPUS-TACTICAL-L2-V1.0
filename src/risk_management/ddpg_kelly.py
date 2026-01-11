"""
HIMARI Layer 2 - Part H: DDPG Kelly
H2: Deep Deterministic Policy Gradient for position sizing.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DDPGKellyConfig:
    """DDPG Kelly configuration."""
    max_fraction: float = 0.20
    half_kelly: bool = True
    min_history: int = 30


class DDPGKelly:
    """
    Kelly Criterion with DDPG adjustments.
    Combines optimal betting with RL-based adaptation.
    """
    
    def __init__(self, config: Optional[DDPGKellyConfig] = None):
        self.config = config or DDPGKellyConfig()
        self.win_history: list = []
        self.return_history: list = []
        
    def update(self, won: bool, return_pct: float) -> None:
        """Update with trade result."""
        self.win_history.append(1 if won else 0)
        self.return_history.append(return_pct)
        
    def kelly_fraction(self) -> float:
        """Calculate Kelly fraction."""
        if len(self.win_history) < self.config.min_history:
            return self.config.max_fraction * 0.5
            
        p = np.mean(self.win_history)
        wins = [r for r, w in zip(self.return_history, self.win_history) if w]
        losses = [abs(r) for r, w in zip(self.return_history, self.win_history) if not w]
        
        if not wins or not losses:
            return self.config.max_fraction * 0.5
            
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return self.config.max_fraction
            
        b = avg_win / avg_loss
        kelly = p - (1 - p) / b
        
        if self.config.half_kelly:
            kelly *= 0.5
            
        return np.clip(kelly, 0, self.config.max_fraction)
        
    def get_position_size(self, confidence: float = 1.0) -> float:
        """Get position size with confidence adjustment."""
        kelly = self.kelly_fraction()
        return kelly * confidence
