"""
HIMARI Layer 2 - Meta-Regime Layer
Subsystem B: Regime Detection (Method B2)

Purpose:
    Track structural market conditions via macroeconomic indicators.
    Governs transition dynamics in the fast-moving market regime layer.

Performance:
    +0.05 Sharpe from better crisis anticipation
    Latency: ~0.5ms (simple indicator aggregation)
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging

from .config.meta_regime_config import MetaRegimeConfig, DEFAULT_META_REGIME_CONFIG
from .config.ahhmm_config import MetaRegime


logger = logging.getLogger(__name__)


@dataclass
class MetaRegimeOutput:
    """Complete output from meta-regime layer."""
    regime: MetaRegime
    score: float                    # Current uncertainty score [0, 1]
    score_ema: float               # Smoothed score
    transition_probability: float  # P(transition in next period)
    indicator_contributions: Dict[str, float]  # Per-indicator scores


class MetaRegimeLayer:
    """
    Meta-Regime Layer: Structural market condition tracker.
    
    The meta-regime layer operates on a slower timescale than the market
    regime HMM, tracking structural shifts in market conditions. It monitors
    a weighted combination of macro indicators:
    
    - VIX: Equity implied volatility (risk appetite proxy)
    - DVOL: Crypto implied volatility (direct vol measure)
    - EPU: Economic Policy Uncertainty (policy risk)
    - Funding Rate: Perpetual swap funding (leverage indicator)
    - OI Change: Open interest dynamics (positioning indicator)
    
    When the meta-regime shifts to HIGH_UNCERTAINTY:
    - Crisis transition probabilities increase 2×
    - All regime transition probabilities increase 1.5×
    - Confidence requirements for regime changes decrease
    
    Performance: +0.05 Sharpe from better crisis anticipation
    Latency: ~0.5ms (simple indicator aggregation)
    """
    
    def __init__(self, config: Optional[MetaRegimeConfig] = None):
        self.config = config or DEFAULT_META_REGIME_CONFIG
        
        # Current state
        self._regime = MetaRegime.LOW_UNCERTAINTY
        self._score = 0.5
        self._score_ema = 0.5
        
        # EMA parameters
        self._ema_alpha = 2.0 / (self.config.score_ema_span + 1)
        
        # Transition tracking
        self._bars_since_transition = 0
        self._transition_count = 0
        
        # Indicator history for trend detection
        self._score_history: List[float] = []
        
    def _normalize_indicator(
        self, 
        value: float, 
        indicator_name: str
    ) -> float:
        """
        Normalize indicator to [0, 1] uncertainty score.
        
        0 = maximally calm (below low threshold)
        1 = maximally stressed (above high threshold)
        Linear interpolation between thresholds
        """
        thresholds = self.config.indicator_thresholds.get(indicator_name)
        if thresholds is None:
            return 0.5  # Unknown indicator
        
        low, high = thresholds
        
        if value <= low:
            return 0.0
        elif value >= high:
            return 1.0
        else:
            return (value - low) / (high - low)
    
    def _compute_composite_score(
        self, 
        indicators: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted composite uncertainty score.
        
        Returns:
            score: Composite score [0, 1]
            contributions: Per-indicator normalized values
        """
        contributions = {}
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for name, weight in self.config.indicator_weights.items():
            if name in indicators:
                normalized = self._normalize_indicator(indicators[name], name)
                contributions[name] = normalized
                weighted_sum += weight * normalized
                weight_sum += weight
        
        if weight_sum == 0:
            return 0.5, contributions
        
        score = weighted_sum / weight_sum
        return score, contributions
    
    def update(self, indicators: Dict[str, float]) -> MetaRegimeOutput:
        """
        Update meta-regime based on current indicators.
        
        Args:
            indicators: Dictionary with keys from indicator_weights
                       e.g., {"vix": 25.0, "dvol": 65.0, "epu": 150.0, ...}
        
        Returns:
            MetaRegimeOutput with current state and diagnostics
        """
        # Compute raw score
        raw_score, contributions = self._compute_composite_score(indicators)
        self._score = raw_score
        
        # Apply EMA smoothing
        self._score_ema = (
            self._ema_alpha * raw_score + 
            (1 - self._ema_alpha) * self._score_ema
        )
        
        # Store history
        self._score_history.append(self._score_ema)
        if len(self._score_history) > 100:
            self._score_history.pop(0)
        
        # Check for transition
        self._bars_since_transition += 1
        old_regime = self._regime
        
        if self._bars_since_transition >= self.config.min_bars_between_transitions:
            if (self._score_ema > self.config.transition_up_threshold and 
                self._regime == MetaRegime.LOW_UNCERTAINTY):
                self._regime = MetaRegime.HIGH_UNCERTAINTY
                self._bars_since_transition = 0
                self._transition_count += 1
                logger.info(
                    f"Meta-regime → HIGH_UNCERTAINTY "
                    f"(score={self._score_ema:.3f})"
                )
            elif (self._score_ema < self.config.transition_down_threshold and 
                  self._regime == MetaRegime.HIGH_UNCERTAINTY):
                self._regime = MetaRegime.LOW_UNCERTAINTY
                self._bars_since_transition = 0
                self._transition_count += 1
                logger.info(
                    f"Meta-regime → LOW_UNCERTAINTY "
                    f"(score={self._score_ema:.3f})"
                )
        
        # Compute transition probability from score trend
        if len(self._score_history) >= 10:
            recent = np.array(self._score_history[-10:])
            trend = np.polyfit(range(10), recent, 1)[0]
            
            if self._regime == MetaRegime.LOW_UNCERTAINTY:
                # Probability of transitioning to HIGH
                trans_prob = np.clip(trend * 10 + 0.1, 0, 1)
            else:
                # Probability of transitioning to LOW
                trans_prob = np.clip(-trend * 10 + 0.1, 0, 1)
        else:
            trans_prob = 0.1
        
        return MetaRegimeOutput(
            regime=self._regime,
            score=self._score,
            score_ema=self._score_ema,
            transition_probability=trans_prob,
            indicator_contributions=contributions
        )
    
    def get_transition_modifier(self) -> float:
        """
        Get modifier for market regime transition probabilities.
        
        HIGH_UNCERTAINTY increases all transition probabilities by 1.5×.
        This makes regime switches more frequent during stress.
        """
        if self._regime == MetaRegime.HIGH_UNCERTAINTY:
            return self.config.transition_probability_boost
        return 1.0
    
    def get_crisis_modifier(self) -> float:
        """
        Get modifier for crisis probability specifically.
        
        HIGH_UNCERTAINTY doubles crisis probability in the HMM.
        """
        if self._regime == MetaRegime.HIGH_UNCERTAINTY:
            return self.config.crisis_probability_boost
        return 1.0
    
    def get_diagnostics(self) -> Dict:
        """Return diagnostic information for monitoring."""
        return {
            "regime": self._regime.name,
            "score": self._score,
            "score_ema": self._score_ema,
            "bars_since_transition": self._bars_since_transition,
            "transition_count": self._transition_count,
            "history_length": len(self._score_history)
        }


class IntegratedRegimeDetector:
    """
    Complete regime detection integrating HMM and Meta-Regime layers.
    
    This class coordinates the two hierarchical levels:
    1. Meta-regime layer updates first (slow)
    2. Meta-regime modifiers applied to HMM
    3. HMM processes observation (fast)
    """
    
    def __init__(self):
        from .student_t_ahhmm import create_regime_detector_v5
        
        self.meta_layer = MetaRegimeLayer()
        self.hmm = create_regime_detector_v5()
        
    def predict(
        self, 
        obs: np.ndarray,
        macro_indicators: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Full prediction pipeline.
        
        Args:
            obs: Feature vector [return, volume_norm, volatility]
            macro_indicators: Dict with vix, dvol, epu, funding_rate, oi_change
        """
        # Update meta-regime
        if macro_indicators:
            meta_result = self.meta_layer.update(macro_indicators)
            # Update HMM meta-regime
            self.hmm._meta_regime = meta_result.regime
        
        # Get HMM prediction
        vix = macro_indicators.get("vix", 25.0) if macro_indicators else 25.0
        epu = macro_indicators.get("epu", 150.0) if macro_indicators else 150.0
        hmm_result = self.hmm.predict(obs, vix=vix, epu=epu)
        
        return {
            "market_regime": hmm_result.regime.name,
            "meta_regime": self.meta_layer._regime.name,
            "probabilities": hmm_result.state_probabilities,
            "confidence": hmm_result.confidence,
            "is_crisis": hmm_result.is_crisis,
            "meta_score": self.meta_layer._score_ema if macro_indicators else None
        }
