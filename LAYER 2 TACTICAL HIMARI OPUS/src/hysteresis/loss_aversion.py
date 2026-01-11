"""
HIMARI Layer 2 - Part G5: 2.2× Loss Aversion Ratio
Prospect theory asymmetric thresholds for entry/exit.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class LossAversionConfig:
    """Configuration for prospect theory loss aversion."""
    loss_aversion_ratio: float = 2.2        # Kahneman-Tversky loss aversion
    gain_sensitivity: float = 0.88          # Diminishing sensitivity to gains
    loss_sensitivity: float = 0.88          # Diminishing sensitivity to losses
    reference_point_ema: float = 0.1        # EMA for reference point adaptation
    entry_base: float = 0.35                # Base entry threshold
    exit_win_base: float = 0.18             # Base exit threshold (winning)
    exit_loss_base: float = 0.40            # Base exit threshold (losing)


class LossAversionThresholds:
    """
    Computes asymmetric entry/exit thresholds based on prospect theory.
    
    Implemented based on:
    - Kahneman & Tversky (1979): Loss aversion ratio of ~2.2
    - Thaler (1980): Mental accounting for trading positions
    
    Key behaviors:
    - Harder to exit losing positions (reluctance to realize losses)
    - Easier to exit winning positions (desire to lock in gains)  
    - Reference point adapts to position entry price
    """
    
    def __init__(self, config: LossAversionConfig = None):
        self.config = config or LossAversionConfig()
        self.reference_point = None
        self.entry_price = None
        self.current_pnl = 0.0
        
    def set_reference_point(self, price: float):
        """Set reference point (usually entry price)."""
        self.entry_price = price
        if self.reference_point is None:
            self.reference_point = price
        else:
            alpha = self.config.reference_point_ema
            self.reference_point = alpha * price + (1 - alpha) * self.reference_point
    
    def update_pnl(self, current_price: float):
        """Update current P&L relative to entry."""
        if self.entry_price is not None:
            self.current_pnl = (current_price - self.entry_price) / self.entry_price
    
    def compute_value_function(self, outcome: float) -> float:
        """
        Compute prospect theory value function.
        
        V(x) = x^α           if x >= 0
        V(x) = -λ(-x)^β      if x < 0
        
        Where:
        - α = β = 0.88 (diminishing sensitivity)
        - λ = 2.2 (loss aversion)
        """
        if outcome >= 0:
            return np.power(outcome, self.config.gain_sensitivity)
        else:
            return -self.config.loss_aversion_ratio * np.power(
                -outcome, self.config.loss_sensitivity
            )
    
    def get_thresholds(self, is_long: bool = True) -> Tuple[float, float]:
        """
        Compute entry and exit thresholds based on current position status.
        
        Returns:
            Tuple of (entry_threshold, exit_threshold)
        """
        if self.current_pnl >= 0:
            # Winning position: easier to exit
            exit_threshold = self.config.exit_win_base
            # But harder to double down (higher entry for adding)
            entry_threshold = self.config.entry_base * 1.1
        else:
            # Losing position: harder to exit (loss aversion)
            exit_threshold = self.config.exit_loss_base * self.config.loss_aversion_ratio
            # Easier to add (mental accounting - "averaging down")
            entry_threshold = self.config.entry_base * 0.9
        
        return entry_threshold, exit_threshold
    
    def adjust_thresholds(
        self, 
        base_entry: float, 
        base_exit: float, 
        is_winning: bool
    ) -> Tuple[float, float]:
        """
        Adjust base thresholds using loss aversion.
        
        Args:
            base_entry: Base entry threshold from other methods
            base_exit: Base exit threshold from other methods
            is_winning: Whether current position is profitable
            
        Returns:
            Adjusted (entry_threshold, exit_threshold)
        """
        if is_winning:
            # Bird in hand effect: easier to exit winners
            exit_multiplier = 0.85
            entry_multiplier = 1.05
        else:
            # Loss aversion: harder to exit losers
            exit_multiplier = self.config.loss_aversion_ratio * 0.5  # 1.1x
            entry_multiplier = 0.95
        
        return base_entry * entry_multiplier, base_exit * exit_multiplier
    
    def get_disposition_effect_score(self) -> float:
        """
        Compute disposition effect tendency score.
        
        Score > 1: More likely to hold losers / sell winners (typical bias)
        Score < 1: Rational behavior
        """
        return self.config.loss_aversion_ratio
    
    def get_diagnostics(self) -> Dict:
        return {
            'reference_point': self.reference_point,
            'entry_price': self.entry_price,
            'current_pnl': self.current_pnl,
            'loss_aversion_ratio': self.config.loss_aversion_ratio,
            'is_winning': self.current_pnl >= 0
        }
    
    def reset(self):
        """Reset state for new position."""
        self.entry_price = None
        self.current_pnl = 0.0


# Factory function
def create_loss_aversion_thresholds(
    loss_ratio: float = 2.2,
    **kwargs
) -> LossAversionThresholds:
    """Create LossAversionThresholds with specified parameters."""
    config = LossAversionConfig(loss_aversion_ratio=loss_ratio, **kwargs)
    return LossAversionThresholds(config)
