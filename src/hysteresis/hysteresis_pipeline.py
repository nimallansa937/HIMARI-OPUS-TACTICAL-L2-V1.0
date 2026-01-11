"""
HIMARI Layer 2 - Part G: Hysteresis Filter
6 methods for anti-whipsaw signal filtering.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Deque
from collections import deque
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# G1: KAMA Adaptive Thresholds
# ============================================================================

@dataclass
class KAMAConfig:
    er_period: int = 10
    fast_threshold_entry: float = 0.25
    slow_threshold_entry: float = 0.45
    fast_threshold_exit: float = 0.25
    slow_threshold_exit: float = 0.12
    smoothing_factor: float = 0.3

class KAMAThresholdAdapter:
    """Adapts thresholds based on Kaufman's Efficiency Ratio."""
    def __init__(self, config=None):
        self.config = config or KAMAConfig()
        self.prices = deque(maxlen=self.config.er_period + 1)
        self.smoothed_er = 0.5
        
    def update(self, price: float) -> Tuple[float, float]:
        self.prices.append(price)
        if len(self.prices) < self.config.er_period + 1:
            return 0.35, 0.16
            
        prices = list(self.prices)
        direction = abs(prices[-1] - prices[0])
        volatility = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices)))
        er = min(1.0, direction / (volatility + 1e-10))
        
        self.smoothed_er = self.config.smoothing_factor * er + (1 - self.config.smoothing_factor) * self.smoothed_er
        
        entry = self.config.slow_threshold_entry + self.smoothed_er * (self.config.fast_threshold_entry - self.config.slow_threshold_entry)
        exit_t = self.config.slow_threshold_exit + self.smoothed_er * (self.config.fast_threshold_exit - self.config.slow_threshold_exit)
        return entry, exit_t


# ============================================================================
# G2: KNN Pattern Matching (simplified without FAISS)
# ============================================================================

class KNNPatternMatcher:
    """Identifies similar historical patterns for threshold adjustment."""
    def __init__(self, k: int = 20, threshold_adjustment: float = 0.08):
        self.k = k
        self.threshold_adjustment = threshold_adjustment
        self.features_db = None
        self.outcomes_db = None
        
    def build_index(self, features: np.ndarray, outcomes: np.ndarray):
        self.features_db = features
        self.outcomes_db = outcomes
        self.means = features.mean(axis=0)
        self.stds = features.std(axis=0) + 1e-8
        
    def query(self, current_features: np.ndarray) -> Tuple[float, float]:
        if self.features_db is None:
            return 0.0, 0.0
        normalized = (current_features - self.means) / self.stds
        distances = np.linalg.norm(self.features_db - normalized, axis=1)
        k_idx = np.argsort(distances)[:self.k]
        false_rate = 1.0 - self.outcomes_db[k_idx].mean()
        adjustment = self.threshold_adjustment * max(0, false_rate - 0.4)
        return false_rate, adjustment


# ============================================================================
# G3: ATR-Scaled Bands
# ============================================================================

@dataclass
class ATRBandConfig:
    atr_period: int = 14
    baseline_period: int = 100
    entry_sensitivity: float = 0.7
    base_entry: float = 0.40
    base_exit: float = 0.18

class ATRBandCalculator:
    """Computes volatility-adjusted threshold bands."""
    def __init__(self, config=None):
        self.config = config or ATRBandConfig()
        self.tr_buffer = deque(maxlen=self.config.baseline_period)
        self._prev_close = None
        
    def update(self, high: float, low: float, close: float) -> Tuple[float, float]:
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        self._prev_close = close
        self.tr_buffer.append(tr)
        
        if len(self.tr_buffer) < self.config.atr_period:
            return self.config.base_entry, self.config.base_exit
            
        atr_short = np.mean(list(self.tr_buffer)[-self.config.atr_period:])
        atr_baseline = np.mean(list(self.tr_buffer))
        ratio = atr_short / (atr_baseline + 1e-10)
        
        entry = self.config.base_entry * (1 + self.config.entry_sensitivity * (ratio - 1))
        exit_t = self.config.base_exit * (1 + 0.5 * (ratio - 1))
        return np.clip(entry, 0.25, 0.65), np.clip(exit_t, 0.08, 0.30)


# ============================================================================
# G4: Meta-Learned k Values
# ============================================================================

class MetaLearnedKSelector:
    """Per-regime k-value optimization."""
    def __init__(self):
        self.regime_k = {0: 15, 1: 20, 2: 25, 3: 10}  # Default per-regime
        
    def get_k(self, regime: int) -> int:
        return self.regime_k.get(regime, 20)
    
    def update_k(self, regime: int, performance: float):
        # Simple adaptation based on performance
        if performance < 0:
            self.regime_k[regime] = min(50, self.regime_k.get(regime, 20) + 5)
        else:
            self.regime_k[regime] = max(5, self.regime_k.get(regime, 20) - 2)


# ============================================================================
# G5: 2.2× Loss Aversion Ratio
# ============================================================================

class LossAversionThresholds:
    """Prospect theory asymmetric thresholds."""
    def __init__(self, loss_aversion: float = 2.2):
        self.loss_aversion = loss_aversion
        
    def adjust(self, entry_threshold: float, exit_threshold: float, 
               is_winning: bool) -> Tuple[float, float]:
        if is_winning:
            # Easier to exit winners (bird in hand)
            return entry_threshold, exit_threshold * 0.9
        else:
            # Harder to exit losers (loss aversion)
            return entry_threshold, exit_threshold * self.loss_aversion


# ============================================================================
# G6: Whipsaw Learning
# ============================================================================

class WhipsawLearner:
    """Online threshold adjustment after false signals."""
    def __init__(self, learning_rate: float = 0.02, max_adjustment: float = 0.15):
        self.learning_rate = learning_rate
        self.max_adjustment = max_adjustment
        self.adjustment = 0.0
        self.false_signals = deque(maxlen=50)
        
    def record_outcome(self, was_false_signal: bool):
        self.false_signals.append(1.0 if was_false_signal else 0.0)
        false_rate = np.mean(self.false_signals) if self.false_signals else 0.0
        
        if false_rate > 0.3:
            self.adjustment = min(self.max_adjustment, self.adjustment + self.learning_rate)
        elif false_rate < 0.15:
            self.adjustment = max(-0.05, self.adjustment - self.learning_rate * 0.5)
            
    def get_adjustment(self) -> float:
        return self.adjustment


# ============================================================================
# Complete Hysteresis Pipeline
# ============================================================================

@dataclass
class HysteresisConfig:
    use_kama: bool = True
    use_atr: bool = True
    use_knn: bool = False  # Requires pattern database
    use_whipsaw: bool = True

class HysteresisFilter:
    """Complete hysteresis filter pipeline with all 6 methods."""
    
    def __init__(self, config=None):
        self.config = config or HysteresisConfig()
        self.kama = KAMAThresholdAdapter()
        self.atr = ATRBandCalculator()
        self.knn = KNNPatternMatcher()
        self.meta_k = MetaLearnedKSelector()
        self.loss_aversion = LossAversionThresholds()
        self.whipsaw = WhipsawLearner()
        self.current_position = 0
        
    def process(self, price: float, high: float, low: float,
               confidence: float, signal: int, regime: int = 2,
               is_winning: bool = True) -> int:
        """Process signal through hysteresis filter."""
        
        # G1: KAMA adaptive thresholds
        entry_t, exit_t = self.kama.update(price)
        
        # G3: ATR scaling
        if self.config.use_atr:
            atr_entry, atr_exit = self.atr.update(high, low, price)
            entry_t = (entry_t + atr_entry) / 2
            exit_t = (exit_t + atr_exit) / 2
        
        # G5: Loss aversion
        entry_t, exit_t = self.loss_aversion.adjust(entry_t, exit_t, is_winning)
        
        # G6: Whipsaw adjustment
        if self.config.use_whipsaw:
            entry_t += self.whipsaw.get_adjustment()
        
        # Apply hysteresis logic
        if self.current_position == 0:
            if signal != 0 and confidence >= entry_t:
                self.current_position = signal
                return signal
            return 0
        elif self.current_position == 1:
            if confidence < exit_t or (signal == -1 and confidence >= entry_t):
                self.current_position = 0 if confidence < exit_t else -1
                return -1
            return 0
        else:  # -1
            if confidence < exit_t or (signal == 1 and confidence >= entry_t):
                self.current_position = 0 if confidence < exit_t else 1
                return 1
            return 0
    
    def record_outcome(self, was_false: bool):
        self.whipsaw.record_outcome(was_false)
