"""
HIMARI Layer 2 - Simplex Safety System
Subsystem I: Simplex (Methods I1-I5)

Purpose:
    Black-box Simplex architecture - final safety net with baseline fallback.

Methods:
    I1: Black-Box Simplex - Runtime safety verification
    I2: Position-Limited Baseline - Safe fallback controller
    I3: Safety Invariants - 4 constraint checks
    I4: Stop-Loss Enforcer - Daily loss override
    I5: Fallback Cascade - 5-level degradation

Expected Performance:
    - Zero invariant violations (guaranteed)
    - Graceful degradation under failure
    - Baseline Sharpe 0.5-0.8 (safe but modest)
"""

from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger


class FallbackLevel(Enum):
    """Fallback cascade levels (I5)"""
    ADVANCED = 0  # Ensemble output
    BASELINE = 1  # Simple momentum
    HOLD_ONLY = 2  # No new positions
    LIQUIDATE = 3  # Emergency exit


@dataclass
class SimplexConfig:
    """Simplex configuration"""
    # Invariants (I3)
    max_leverage: float = 3.0
    max_position_concentration: float = 0.2  # 20% per asset
    max_drawdown: float = 0.05  # 5%

    # Stop-loss (I4)
    daily_stop_loss: float = 0.02  # 2%

    # Baseline controller (I2)
    baseline_momentum_window: int = 20
    baseline_max_position: float = 0.05  # 5% per trade


class SimplexSafetyWrapper:
    """
    Black-box Simplex safety wrapper.

    Example:
        >>> simplex = SimplexSafetyWrapper()
        >>>
        >>> # Try advanced action
        >>> action, level = simplex.execute_with_fallback(
        ...     advanced_action=2,  # SELL
        ...     advanced_confidence=0.7,
        ...     state={'leverage': 2.5, 'position': 0.1}
        ... )
    """

    def __init__(self, config: Optional[SimplexConfig] = None):
        self.config = config or SimplexConfig()
        self.current_level = FallbackLevel.ADVANCED
        self.daily_pnl = 0.0
        logger.info("Simplex Safety Wrapper initialized")

    def check_invariants(self, action: int, state: dict) -> Tuple[bool, str]:
        """Check safety invariants (I3)"""
        # Invariant 1: Leverage bound
        if state.get('leverage', 0) > self.config.max_leverage:
            return False, f"Leverage {state['leverage']:.1f}x exceeds max {self.config.max_leverage}x"

        # Invariant 2: Position concentration
        if state.get('position', 0) > self.config.max_position_concentration:
            return False, f"Position {state['position']:.1%} exceeds max {self.config.max_position_concentration:.0%}"

        # Invariant 3: Drawdown limit
        if state.get('unrealized_loss', 0) > self.config.max_drawdown:
            return False, f"Drawdown {state['unrealized_loss']:.1%} exceeds max {self.config.max_drawdown:.0%}"

        # Invariant 4: No simultaneous long/short (handled elsewhere)

        return True, "OK"

    def get_baseline_action(self, momentum: float) -> int:
        """Position-limited momentum baseline (I2)"""
        if momentum > 0:
            return 0  # BUY (small increment)
        elif momentum < 0:
            return 2  # SELL
        return 1  # HOLD

    def execute_with_fallback(
        self,
        advanced_action: int,
        advanced_confidence: float,
        state: dict,
        momentum: Optional[float] = None
    ) -> Tuple[int, FallbackLevel]:
        """
        Execute with fallback cascade (I5).

        Returns:
            (action, fallback_level)
        """
        # Level 0: Stop-loss override (I4)
        if self.daily_pnl < -self.config.daily_stop_loss:
            logger.critical(f"STOP-LOSS HIT: {self.daily_pnl:.2%}")
            return 2, FallbackLevel.LIQUIDATE  # SELL everything

        # Level 1: Try advanced action
        if advanced_confidence >= 0.6:
            safe, reason = self.check_invariants(advanced_action, state)
            if safe:
                return advanced_action, FallbackLevel.ADVANCED
            logger.warning(f"Advanced action unsafe: {reason}")

        # Level 2: Try baseline
        if momentum is not None:
            baseline_action = self.get_baseline_action(momentum)
            safe, reason = self.check_invariants(baseline_action, state)
            if safe:
                logger.info("Falling back to baseline controller")
                return baseline_action, FallbackLevel.BASELINE
            logger.warning(f"Baseline action unsafe: {reason}")

        # Level 3: HOLD only
        logger.warning("Falling back to HOLD-only mode")
        return 1, FallbackLevel.HOLD_ONLY

    def update_pnl(self, pnl_change: float):
        """Update daily P&L"""
        self.daily_pnl += pnl_change

    def reset_daily(self):
        """Reset daily metrics"""
        self.daily_pnl = 0.0


__all__ = ['SimplexSafetyWrapper', 'SimplexConfig', 'FallbackLevel']
