"""
HIMARI Layer 2 - Part F: Predictive Uncertainty
F8: Forecast future uncertainty for proactive position sizing.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PredictiveUncertaintyConfig:
    """Predictive uncertainty configuration."""
    lookback: int = 50
    horizon: int = 5
    ar_order: int = 3


class PredictiveUncertainty:
    """
    Forecast future uncertainty levels.
    Uses AR model on historical uncertainty to enable proactive risk scaling.
    """
    
    def __init__(self, config: Optional[PredictiveUncertaintyConfig] = None):
        self.config = config or PredictiveUncertaintyConfig()
        self.history = deque(maxlen=self.config.lookback)
        self.ar_coeffs: Optional[np.ndarray] = None
        
    def update(self, uncertainty: float) -> None:
        """Add new uncertainty observation."""
        self.history.append(uncertainty)
        if len(self.history) >= self.config.ar_order + 10:
            self._fit_ar()
            
    def _fit_ar(self) -> None:
        """Fit AR model on history."""
        y = np.array(self.history)
        n = len(y)
        p = self.config.ar_order
        
        if n < p + 5:
            return
            
        X = np.zeros((n - p, p))
        for i in range(p):
            X[:, i] = y[p - i - 1:n - i - 1]
        Y = y[p:]
        
        try:
            self.ar_coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        except:
            self.ar_coeffs = np.ones(p) / p
            
    def forecast(self) -> Tuple[List[float], Dict]:
        """Forecast uncertainty for next horizon bars."""
        if len(self.history) < self.config.ar_order:
            current = self.history[-1] if self.history else 0.5
            return [current] * self.config.horizon, {'fitted': False}
            
        if self.ar_coeffs is None:
            self._fit_ar()
            
        recent = list(self.history)[-self.config.ar_order:]
        forecasts = []
        
        for _ in range(self.config.horizon):
            if self.ar_coeffs is not None:
                next_val = np.dot(self.ar_coeffs, recent[::-1])
            else:
                next_val = np.mean(recent)
                
            next_val = np.clip(next_val, 0, 1)
            forecasts.append(float(next_val))
            recent = recent[1:] + [next_val]
            
        return forecasts, {
            'fitted': self.ar_coeffs is not None,
            'current': self.history[-1] if self.history else 0.5,
            'max_forecast': max(forecasts)
        }
        
    def get_proactive_scale(self) -> float:
        """Get position scale factor based on predicted uncertainty."""
        forecasts, _ = self.forecast()
        max_unc = max(forecasts)
        return max(1.0 - 0.75 * max_unc, 0.1)
