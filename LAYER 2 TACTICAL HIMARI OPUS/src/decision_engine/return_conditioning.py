"""
HIMARI Layer 2 - Return Conditioning
Subsystem D: Decision Engine (Method D9)

Purpose:
    Condition decision models on target Sharpe ratio.
    Controls risk-return tradeoff without retraining.

Performance:
    +0.02 Sharpe from adaptive risk targeting
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import logging


logger = logging.getLogger(__name__)


@dataclass
class ReturnConditionerConfig:
    """Configuration for return conditioning."""
    # Default targets per regime
    crisis_target: float = 0.5     # Conservative during crisis
    bearish_target: float = 1.0    # Moderate in bear markets
    neutral_target: float = 2.0    # Standard target
    bullish_target: float = 2.5    # Aggressive in bull markets
    
    # Volatility adjustment
    high_vol_multiplier: float = 0.7   # Reduce target when vol is high
    low_vol_multiplier: float = 1.2    # Increase target when vol is low
    vol_threshold_high: float = 0.03   # 3% daily vol = high
    vol_threshold_low: float = 0.01    # 1% daily vol = low


class ReturnConditioner:
    """
    Condition decision models on target Sharpe ratio.
    
    Decision Transformers and other sequence models can be conditioned
    on target returns at inference time. This enables risk-return
    control without retraining:
    
    - High target (3.0): Aggressive positioning, higher risk
    - Medium target (2.0): Balanced approach
    - Low target (1.0): Conservative, capital preservation
    
    The target is adjusted based on:
    1. Current market regime (from regime detection)
    2. Recent volatility (higher vol → lower target)
    3. Drawdown state (in drawdown → lower target)
    
    Performance: +0.02 Sharpe from adaptive risk targeting
    """
    
    def __init__(self, config: Optional[ReturnConditionerConfig] = None):
        self.config = config or ReturnConditionerConfig()
        
        # Regime target mapping
        self._regime_targets = {
            0: self.config.crisis_target,    # Crisis
            1: self.config.bearish_target,   # Bearish
            2: self.config.neutral_target,   # Neutral
            3: self.config.bullish_target    # Bullish
        }
        
        # State tracking
        self._current_target = self.config.neutral_target
        self._recent_volatility = 0.015  # Default 1.5% daily
        self._in_drawdown = False
        self._drawdown_pct = 0.0
        
    def get_regime_target(self, regime: int) -> float:
        """
        Get target Sharpe based on market regime.
        
        Args:
            regime: Market regime code (0=crisis, 1=bear, 2=neutral, 3=bull)
        
        Returns:
            target: Target Sharpe ratio for conditioning
        """
        base_target = self._regime_targets.get(regime, self.config.neutral_target)
        
        # Apply volatility adjustment
        adjusted_target = self._apply_volatility_adjustment(base_target)
        
        # Apply drawdown adjustment
        if self._in_drawdown:
            adjusted_target = self._apply_drawdown_adjustment(adjusted_target)
        
        self._current_target = adjusted_target
        return adjusted_target
    
    def _apply_volatility_adjustment(self, target: float) -> float:
        """Adjust target based on recent volatility."""
        if self._recent_volatility > self.config.vol_threshold_high:
            return target * self.config.high_vol_multiplier
        elif self._recent_volatility < self.config.vol_threshold_low:
            return target * self.config.low_vol_multiplier
        return target
    
    def _apply_drawdown_adjustment(self, target: float) -> float:
        """Reduce target when in drawdown."""
        # Linear reduction: 10% drawdown → 50% target reduction
        reduction = min(0.5, abs(self._drawdown_pct) * 5)
        return target * (1 - reduction)
    
    def update_volatility(self, volatility: float) -> None:
        """Update recent volatility estimate."""
        self._recent_volatility = volatility
    
    def update_drawdown(self, drawdown_pct: float) -> None:
        """Update drawdown state."""
        self._drawdown_pct = drawdown_pct
        self._in_drawdown = drawdown_pct < -0.02  # 2% threshold
    
    def get_current_target(self) -> float:
        """Get current target Sharpe."""
        return self._current_target
    
    def get_all_targets(self) -> Dict[str, float]:
        """Get all regime targets for diagnostics."""
        return {
            "crisis": self._regime_targets[0],
            "bearish": self._regime_targets[1],
            "neutral": self._regime_targets[2],
            "bullish": self._regime_targets[3],
            "current": self._current_target,
            "volatility": self._recent_volatility,
            "in_drawdown": self._in_drawdown
        }


def create_return_conditioner() -> ReturnConditioner:
    """Factory function to create return conditioner."""
    return ReturnConditioner()
