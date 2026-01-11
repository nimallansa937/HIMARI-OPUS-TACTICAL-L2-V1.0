"""
HIMARI Layer 2 - Part G6: Whipsaw Learning
Online threshold adjustment after false signals.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class WhipsawConfig:
    """Configuration for whipsaw learning."""
    learning_rate: float = 0.02           # Adjustment learning rate
    max_adjustment: float = 0.15          # Maximum threshold increase
    min_adjustment: float = -0.05         # Minimum threshold decrease
    lookback_window: int = 50             # Window for false signal tracking
    false_rate_threshold: float = 0.30    # Trigger adjustment if > this
    recovery_rate: float = 0.5            # How fast to reduce adjustment
    decay_factor: float = 0.995           # Per-update decay of adjustment
    regime_specific: bool = True          # Track per-regime thresholds


@dataclass
class WhipsawEvent:
    """Record of a whipsaw event."""
    timestamp: float
    signal: int                           # 1=buy, -1=sell
    entry_price: float
    exit_price: float
    holding_bars: int
    was_false_signal: bool                # True if quickly reversed
    pnl: float
    regime: int


class WhipsawLearner:
    """
    Online threshold adjustment based on recent whipsaw detection.
    
    Learns from false signals and adjusts thresholds to reduce them:
    - Tracks recent signal outcomes
    - Increases thresholds when false signals are frequent
    - Gradually decreases thresholds when false rate drops
    - Supports regime-specific adjustments
    """
    
    def __init__(self, config: WhipsawConfig = None):
        self.config = config or WhipsawConfig()
        self.events = deque(maxlen=self.config.lookback_window)
        self.current_adjustment = 0.0
        
        # Regime-specific adjustments
        self.regime_adjustments = {i: 0.0 for i in range(5)}
        
        # Statistics
        self._total_signals = 0
        self._false_signals = 0
        self._adjustment_history = []
    
    def record_outcome(
        self, 
        was_false_signal: bool,
        regime: int = 2,
        pnl: float = 0.0
    ):
        """
        Record signal outcome for learning.
        
        Args:
            was_false_signal: True if signal was quickly reversed (whipsaw)
            regime: Current regime for regime-specific learning
            pnl: P&L of the trade
        """
        self._total_signals += 1
        
        if was_false_signal:
            self._false_signals += 1
            
        self.events.append({
            'was_false': was_false_signal,
            'regime': regime,
            'pnl': pnl
        })
        
        # Update adjustments
        self._update_adjustment(regime)
    
    def _update_adjustment(self, current_regime: int):
        """Update threshold adjustments based on recent outcomes."""
        if len(self.events) < 10:
            return
        
        # Calculate false signal rate
        recent = list(self.events)
        false_rate = np.mean([e['was_false'] for e in recent])
        
        # Update global adjustment
        if false_rate > self.config.false_rate_threshold:
            # Increase thresholds
            self.current_adjustment = min(
                self.config.max_adjustment,
                self.current_adjustment + self.config.learning_rate
            )
            logger.debug(f"Whipsaw: Increasing threshold adjustment to {self.current_adjustment:.3f}")
        elif false_rate < self.config.false_rate_threshold * 0.5:
            # Decrease thresholds (can trade more)
            self.current_adjustment = max(
                self.config.min_adjustment,
                self.current_adjustment - self.config.learning_rate * self.config.recovery_rate
            )
        
        # Apply decay
        self.current_adjustment *= self.config.decay_factor
        
        # Update regime-specific adjustment
        if self.config.regime_specific:
            regime_events = [e for e in recent if e['regime'] == current_regime]
            if len(regime_events) >= 5:
                regime_false_rate = np.mean([e['was_false'] for e in regime_events])
                if regime_false_rate > self.config.false_rate_threshold:
                    self.regime_adjustments[current_regime] = min(
                        self.config.max_adjustment,
                        self.regime_adjustments[current_regime] + self.config.learning_rate * 0.5
                    )
                else:
                    self.regime_adjustments[current_regime] *= self.config.decay_factor
        
        self._adjustment_history.append(self.current_adjustment)
    
    def get_adjustment(self, regime: int = None) -> float:
        """
        Get current threshold adjustment.
        
        Args:
            regime: Optional regime for regime-specific adjustment
            
        Returns:
            Threshold adjustment to add to base thresholds
        """
        if regime is not None and self.config.regime_specific:
            return self.current_adjustment + self.regime_adjustments.get(regime, 0.0)
        return self.current_adjustment
    
    def adjust_thresholds(
        self, 
        entry_threshold: float, 
        exit_threshold: float,
        regime: int = None
    ) -> Tuple[float, float]:
        """
        Apply whipsaw adjustment to thresholds.
        
        Args:
            entry_threshold: Base entry threshold
            exit_threshold: Base exit threshold
            regime: Current regime
            
        Returns:
            Adjusted (entry_threshold, exit_threshold)
        """
        adjustment = self.get_adjustment(regime)
        
        # Increase entry threshold to reduce false entries
        adjusted_entry = entry_threshold + adjustment
        
        # Keep exit threshold relatively stable (don't trap in positions)
        adjusted_exit = exit_threshold + adjustment * 0.3
        
        return adjusted_entry, adjusted_exit
    
    def get_false_signal_rate(self) -> float:
        """Get current false signal rate."""
        if not self.events:
            return 0.0
        return np.mean([e['was_false'] for e in self.events])
    
    def get_regime_stats(self) -> Dict[int, Dict]:
        """Get per-regime statistics."""
        stats = {}
        for regime in range(5):
            regime_events = [e for e in self.events if e['regime'] == regime]
            if regime_events:
                stats[regime] = {
                    'count': len(regime_events),
                    'false_rate': np.mean([e['was_false'] for e in regime_events]),
                    'avg_pnl': np.mean([e['pnl'] for e in regime_events]),
                    'adjustment': self.regime_adjustments[regime]
                }
        return stats
    
    def get_diagnostics(self) -> Dict:
        return {
            'total_signals': self._total_signals,
            'false_signals': self._false_signals,
            'false_rate': self.get_false_signal_rate(),
            'current_adjustment': self.current_adjustment,
            'regime_adjustments': dict(self.regime_adjustments),
            'events_tracked': len(self.events)
        }
    
    def reset(self):
        """Reset learner state."""
        self.events.clear()
        self.current_adjustment = 0.0
        self.regime_adjustments = {i: 0.0 for i in range(5)}
        self._adjustment_history.clear()


# Factory function
def create_whipsaw_learner(
    learning_rate: float = 0.02,
    lookback: int = 50,
    **kwargs
) -> WhipsawLearner:
    """Create WhipsawLearner with specified parameters."""
    config = WhipsawConfig(
        learning_rate=learning_rate,
        lookback_window=lookback,
        **kwargs
    )
    return WhipsawLearner(config)
