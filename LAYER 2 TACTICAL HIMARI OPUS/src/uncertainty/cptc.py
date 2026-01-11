"""
HIMARI Layer 2 - Part F: CPTC Regime Change Points
F2: Conformal Prediction with Temporal Covariate for regime-aware intervals.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class CPTCConfig:
    """CPTC configuration."""
    base_alpha: float = 0.10
    regime_expansion: float = 2.0
    decay_rate: float = 0.95
    lookback_window: int = 100
    change_threshold: float = 2.0
    min_samples_per_regime: int = 20


class RegimeChangeDetector:
    """Detect regime changes using CUSUM-like statistics."""
    
    def __init__(self, config: CPTCConfig):
        self.config = config
        self.scores = deque(maxlen=config.lookback_window)
        self.current_regime = 0
        self.regime_start_idx = 0
        
    def update(self, score: float) -> Tuple[bool, int]:
        """Update with new score, return (is_change, regime_id)."""
        self.scores.append(score)
        
        if len(self.scores) < self.config.min_samples_per_regime * 2:
            return False, self.current_regime
            
        recent = list(self.scores)[-self.config.min_samples_per_regime:]
        historical = list(self.scores)[:-self.config.min_samples_per_regime]
        
        recent_mean = np.mean(recent)
        hist_mean = np.mean(historical)
        hist_std = np.std(historical) + 1e-8
        
        z_score = abs(recent_mean - hist_mean) / hist_std
        
        if z_score > self.config.change_threshold:
            self.current_regime += 1
            self.regime_start_idx = len(self.scores)
            return True, self.current_regime
            
        return False, self.current_regime


class CPTC:
    """
    Conformal Prediction with Temporal Covariate.
    Handles non-stationarity by detecting regime changes and adjusting intervals.
    """
    
    def __init__(self, config: Optional[CPTCConfig] = None):
        self.config = config or CPTCConfig()
        self.detector = RegimeChangeDetector(self.config)
        self.global_scores: List[float] = []
        self.expansion_factor = 1.0
        
    def calibrate(self, residuals: np.ndarray) -> Dict[str, float]:
        """Calibrate on absolute residuals."""
        self.global_scores = list(np.abs(residuals))
        quantile = np.quantile(self.global_scores, 1 - self.config.base_alpha)
        return {
            'n_samples': len(residuals),
            'base_quantile': float(quantile)
        }
        
    def predict_interval(self, point_pred: float) -> Tuple[float, float, Dict]:
        """Predict regime-aware conformal interval."""
        if len(self.global_scores) == 0:
            q = 0.1
        else:
            q = np.quantile(self.global_scores, 1 - self.config.base_alpha)
            
        adjusted_q = q * self.expansion_factor
        lower = point_pred - adjusted_q
        upper = point_pred + adjusted_q
        
        return lower, upper, {
            'quantile': adjusted_q,
            'expansion_factor': self.expansion_factor,
            'regime': self.detector.current_regime
        }
        
    def update(self, residual: float) -> Dict:
        """Update with observed residual."""
        abs_res = abs(residual)
        is_change, regime = self.detector.update(abs_res)
        
        self.global_scores.append(abs_res)
        if len(self.global_scores) > self.config.lookback_window * 2:
            self.global_scores = self.global_scores[-self.config.lookback_window * 2:]
            
        if is_change:
            self.expansion_factor = self.config.regime_expansion
        else:
            self.expansion_factor = 1.0 + (self.expansion_factor - 1.0) * self.config.decay_rate
            
        return {'is_change': is_change, 'regime': regime, 'expansion': self.expansion_factor}
