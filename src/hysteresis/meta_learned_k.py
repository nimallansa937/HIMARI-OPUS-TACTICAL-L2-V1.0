"""
HIMARI Layer 2 - Part G: Meta-Learned K
G4: Meta-learning for adaptive hysteresis parameter K.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetaLearnedKConfig:
    """Meta-learned K configuration."""
    initial_k: float = 0.5
    learning_rate: float = 0.01
    decay: float = 0.99
    min_k: float = 0.1
    max_k: float = 2.0


class MetaLearnedK:
    """
    Meta-learning for adaptive hysteresis parameter K.
    Learns optimal K based on regime and recent performance.
    """
    
    def __init__(self, config: Optional[MetaLearnedKConfig] = None):
        self.config = config or MetaLearnedKConfig()
        self.k = self.config.initial_k
        self.regime_ks: Dict[int, float] = {0: 1.5, 1: 1.0, 2: 0.5, 3: 0.7}
        self.performance_history: List[float] = []
        
    def get_k(self, regime: int = 2) -> float:
        """Get adaptive K value for regime."""
        base_k = self.regime_ks.get(regime, self.k)
        return np.clip(base_k, self.config.min_k, self.config.max_k)
        
    def update(self, result: float, regime: int = 2) -> None:
        """Update K based on result (positive = good, negative = bad)."""
        self.performance_history.append(result)
        
        if len(self.performance_history) >= 10:
            recent_perf = np.mean(self.performance_history[-10:])
            
            if recent_perf < 0:
                # Increase K (more filtering)
                delta = self.config.learning_rate * abs(recent_perf)
            else:
                # Decrease K (less filtering)
                delta = -self.config.learning_rate * recent_perf * 0.5
                
            self.regime_ks[regime] = np.clip(
                self.regime_ks.get(regime, self.k) + delta,
                self.config.min_k,
                self.config.max_k
            )
            
    def get_all_ks(self) -> Dict[int, float]:
        """Get all regime K values."""
        return dict(self.regime_ks)
