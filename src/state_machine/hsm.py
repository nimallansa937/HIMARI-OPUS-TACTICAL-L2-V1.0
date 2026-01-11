"""
HIMARI Layer 2 - Hierarchical State Machine (HSM)
Subsystem E: State Machine (Methods E1-E4)

Purpose:
    Track trading state and enforce state-dependent rules using hierarchical
    state machine with orthogonal regions for position and regime tracking.

Methods:
    E1: Orthogonal Regions - Separate position and regime state spaces
    E2: Hierarchical Nesting - Super-states containing sub-states
    E3: History States - Resume previous state after interruption
    E4: Synchronized Events - Cross-region coordination

Key Features:
    - Position region: FLAT, LONG_ENTRY, LONG_HOLD, LONG_EXIT, SHORT_ENTRY, SHORT_HOLD, SHORT_EXIT
    - Regime region: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, CRISIS
    - Transition validation prevents invalid state changes
    - Crisis events trigger synchronized position reduction

Expected Benefits:
    - Prevents invalid trades (e.g., exit when no position)
    - Enforces minimum hold times
    - Coordinates regime changes with position management
    - Provides clear audit trail

Reference:
    - Harel "Statecharts: A Visual Formalism for Complex Systems" (1987)
    - UML State Machine specification
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
from loguru import logger


# Position States (Orthogonal Region 1)
class PositionState(Enum):
    """Position state machine states"""
    FLAT = "flat"  # No position
    LONG_ENTRY = "long_entry"  # Entering long
    LONG_HOLD = "long_hold"  # Holding long
    LONG_EXIT = "long_exit"  # Exiting long
    SHORT_ENTRY = "short_entry"  # Entering short
    SHORT_HOLD = "short_hold"  # Holding short
    SHORT_EXIT = "short_exit"  # Exiting short


# Regime States (Orthogonal Region 2)
class RegimeState(Enum):
    """Regime state machine states"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"


# Super-states (Hierarchical Nesting - Method E2)
class SuperState(Enum):
    """Hierarchical super-states"""
    LONG_MODE = "long_mode"  # Contains LONG_ENTRY, LONG_HOLD, LONG_EXIT
    SHORT_MODE = "short_mode"  # Contains SHORT_ENTRY, SHORT_HOLD, SHORT_EXIT
    NORMAL_REGIME = "normal_regime"  # Contains TRENDING_*, RANGING
    ABNORMAL_REGIME = "abnormal_regime"  # Contains HIGH_VOLATILITY, CRISIS


# Events for state transitions
class StateEvent(Enum):
    """Events triggering state transitions"""
    # Position events
    BUY_SIGNAL = "buy_signal"
    SELL_SIGNAL = "sell_signal"
    POSITION_ENTERED = "position_entered"
    POSITION_EXITED = "position_exited"
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"

    # Regime events (synchronized - Method E4)
    REGIME_CHANGE = "regime_change"
    CRISIS_DETECTED = "crisis_detected"
    CRISIS_RESOLVED = "crisis_resolved"
    VOLATILITY_SPIKE = "volatility_spike"

    # Time-based
    MIN_HOLD_EXPIRED = "min_hold_expired"
    MAX_HOLD_EXPIRED = "max_hold_expired"


@dataclass
class Transition:
    """State transition definition"""
    from_state: PositionState
    to_state: PositionState
    event: StateEvent
    guard: Optional[str] = None  # Condition that must be true
    action: Optional[str] = None  # Action to execute on transition


@dataclass
class HSMConfig:
    """Configuration for HSM"""
    # Position constraints
    min_hold_bars: int = 3  # Minimum bars to hold position
    max_hold_bars: int = 100  # Maximum bars to hold position

    # Transition costs (discourage frequent transitions)
    transition_costs: Dict[Tuple[PositionState, PositionState], float] = field(default_factory=lambda: {
        (PositionState.FLAT, PositionState.LONG_ENTRY): 0.001,
        (PositionState.FLAT, PositionState.SHORT_ENTRY): 0.001,
        (PositionState.LONG_EXIT, PositionState.FLAT): 0.0005,
        (PositionState.SHORT_EXIT, PositionState.FLAT): 0.0005,
    })

    # Crisis behavior
    crisis_position_reduction: float = 0.5  # Reduce position by 50% in crisis


class HierarchicalStateMachine:
    """
    Hierarchical State Machine for trading.

    Manages two orthogonal regions (Method E1):
    1. Position region: Track position lifecycle
    2. Regime region: Track market regime

    Features hierarchical nesting (Method E2), history states (Method E3),
    and synchronized events (Method E4).

    Example:
        >>> hsm = HierarchicalStateMachine()
        >>>
        >>> # Check if transition is allowed
        >>> can_buy = hsm.can_transition(PositionState.LONG_ENTRY)
        >>>
        >>> # Execute transition
        >>> if can_buy:
        ...     hsm.transition(PositionState.LONG_ENTRY, StateEvent.BUY_SIGNAL)
        >>>
        >>> # Handle synchronized crisis event
        >>> hsm.handle_synchronized_event(StateEvent.CRISIS_DETECTED)
    """

    def __init__(self, config: Optional[HSMConfig] = None):
        """
        Initialize HSM.

        Args:
            config: HSM configuration
        """
        self.config = config or HSMConfig()

        # Orthogonal Region 1: Position State (Method E1)
        self.position_state = PositionState.FLAT
        self.position_history = []  # For Method E3

        # Orthogonal Region 2: Regime State (Method E1)
        self.regime_state = RegimeState.RANGING
        self.regime_history = []

        # State entry times
        self.position_entry_time = 0
        self.regime_entry_time = 0

        # Current bar counter
        self.current_bar = 0

        # History state stack (Method E3)
        self.saved_position_state: Optional[PositionState] = None

        # Define valid transitions
        self._define_transitions()

        logger.info("Hierarchical State Machine initialized")

    def _define_transitions(self):
        """Define valid state transitions"""
        self.valid_transitions = {
            PositionState.FLAT: [
                PositionState.LONG_ENTRY,
                PositionState.SHORT_ENTRY
            ],
            PositionState.LONG_ENTRY: [
                PositionState.LONG_HOLD,
                PositionState.FLAT  # Entry failed
            ],
            PositionState.LONG_HOLD: [
                PositionState.LONG_EXIT,
                PositionState.FLAT  # Emergency exit
            ],
            PositionState.LONG_EXIT: [
                PositionState.FLAT,
                PositionState.LONG_HOLD  # Exit failed
            ],
            PositionState.SHORT_ENTRY: [
                PositionState.SHORT_HOLD,
                PositionState.FLAT
            ],
            PositionState.SHORT_HOLD: [
                PositionState.SHORT_EXIT,
                PositionState.FLAT
            ],
            PositionState.SHORT_EXIT: [
                PositionState.FLAT,
                PositionState.SHORT_HOLD
            ]
        }

    def get_super_state(self, state: PositionState) -> Optional[SuperState]:
        """Get super-state for given state (Method E2)"""
        if state in [PositionState.LONG_ENTRY, PositionState.LONG_HOLD, PositionState.LONG_EXIT]:
            return SuperState.LONG_MODE
        elif state in [PositionState.SHORT_ENTRY, PositionState.SHORT_HOLD, PositionState.SHORT_EXIT]:
            return SuperState.SHORT_MODE
        return None

    def can_transition(self, to_state: PositionState) -> Tuple[bool, str]:
        """
        Check if transition is allowed.

        Args:
            to_state: Target state

        Returns:
            (is_allowed, reason)
        """
        # Check if transition is valid
        if to_state not in self.valid_transitions.get(self.position_state, []):
            return False, f"Invalid transition: {self.position_state} -> {to_state}"

        # Check minimum hold time
        bars_in_state = self.current_bar - self.position_entry_time
        if bars_in_state < self.config.min_hold_bars:
            if self.position_state in [PositionState.LONG_HOLD, PositionState.SHORT_HOLD]:
                return False, f"Minimum hold time not met ({bars_in_state}/{self.config.min_hold_bars})"

        # Check maximum hold time
        if bars_in_state >= self.config.max_hold_bars:
            if self.position_state in [PositionState.LONG_HOLD, PositionState.SHORT_HOLD]:
                if to_state not in [PositionState.LONG_EXIT, PositionState.SHORT_EXIT, PositionState.FLAT]:
                    return False, "Maximum hold time exceeded, must exit"

        return True, "OK"

    def transition(
        self,
        to_state: PositionState,
        event: StateEvent,
        save_history: bool = False
    ) -> bool:
        """
        Execute state transition.

        Args:
            to_state: Target state
            event: Event triggering transition
            save_history: Save current state for later resume (Method E3)

        Returns:
            success: True if transition executed
        """
        # Check if allowed
        allowed, reason = self.can_transition(to_state)
        if not allowed:
            logger.warning(f"Transition blocked: {reason}")
            return False

        # Save history if requested (Method E3)
        if save_history:
            self.saved_position_state = self.position_state
            logger.debug(f"Saved history state: {self.saved_position_state}")

        # Record transition
        old_state = self.position_state
        self.position_state = to_state
        self.position_entry_time = self.current_bar

        # Add to history
        self.position_history.append({
            'from': old_state,
            'to': to_state,
            'event': event,
            'bar': self.current_bar
        })

        logger.info(f"State transition: {old_state} -> {to_state} (event: {event})")

        return True

    def resume_history_state(self) -> bool:
        """
        Resume saved history state (Method E3).

        Returns:
            success: True if resumed
        """
        if self.saved_position_state is None:
            logger.warning("No saved history state to resume")
            return False

        logger.info(f"Resuming history state: {self.saved_position_state}")

        # Transition to saved state
        success = self.transition(
            self.saved_position_state,
            StateEvent.REGIME_CHANGE
        )

        if success:
            self.saved_position_state = None

        return success

    def update_regime(self, new_regime: RegimeState):
        """
        Update regime state (Orthogonal Region 2).

        Args:
            new_regime: New regime state
        """
        if new_regime != self.regime_state:
            old_regime = self.regime_state
            self.regime_state = new_regime
            self.regime_entry_time = self.current_bar

            self.regime_history.append({
                'from': old_regime,
                'to': new_regime,
                'bar': self.current_bar
            })

            logger.info(f"Regime transition: {old_regime} -> {new_regime}")

            # Trigger synchronized event if crisis (Method E4)
            if new_regime == RegimeState.CRISIS:
                self.handle_synchronized_event(StateEvent.CRISIS_DETECTED)
            elif old_regime == RegimeState.CRISIS:
                self.handle_synchronized_event(StateEvent.CRISIS_RESOLVED)

    def handle_synchronized_event(self, event: StateEvent):
        """
        Handle synchronized event across regions (Method E4).

        Synchronized events coordinate between orthogonal regions.

        Args:
            event: Synchronized event
        """
        logger.info(f"Synchronized event: {event}")

        if event == StateEvent.CRISIS_DETECTED:
            # Crisis: Force position reduction or exit
            if self.position_state in [PositionState.LONG_HOLD, PositionState.SHORT_HOLD]:
                # Save current state
                logger.warning("Crisis detected - reducing position")

                # Force transition to exit state
                if self.position_state == PositionState.LONG_HOLD:
                    self.transition(PositionState.LONG_EXIT, event, save_history=True)
                else:
                    self.transition(PositionState.SHORT_EXIT, event, save_history=True)

        elif event == StateEvent.CRISIS_RESOLVED:
            # Crisis resolved: Can resume saved state
            logger.info("Crisis resolved - can resume trading")
            # Don't auto-resume, let strategy decide

    def get_allowed_actions(self) -> List[str]:
        """
        Get actions allowed in current state.

        Returns:
            actions: List of allowed action names
        """
        if self.position_state == PositionState.FLAT:
            return ['BUY', 'SELL']
        elif self.position_state in [PositionState.LONG_ENTRY, PositionState.LONG_HOLD]:
            return ['HOLD', 'SELL']
        elif self.position_state in [PositionState.SHORT_ENTRY, PositionState.SHORT_HOLD]:
            return ['HOLD', 'BUY']
        elif self.position_state in [PositionState.LONG_EXIT, PositionState.SHORT_EXIT]:
            return ['HOLD']  # Waiting for exit to complete

        return []

    def step(self):
        """Advance state machine by one bar"""
        self.current_bar += 1

    def get_status(self) -> Dict:
        """
        Get current HSM status.

        Returns:
            status: Dict with state info
        """
        return {
            'position_state': self.position_state.value,
            'regime_state': self.regime_state.value,
            'bars_in_position': self.current_bar - self.position_entry_time,
            'bars_in_regime': self.current_bar - self.regime_entry_time,
            'super_state': self.get_super_state(self.position_state).value if self.get_super_state(self.position_state) else None,
            'allowed_actions': self.get_allowed_actions(),
            'has_saved_state': self.saved_position_state is not None
        }

    def reset(self):
        """Reset state machine"""
        self.position_state = PositionState.FLAT
        self.regime_state = RegimeState.RANGING
        self.current_bar = 0
        self.position_entry_time = 0
        self.regime_entry_time = 0
        self.position_history = []
        self.regime_history = []
        self.saved_position_state = None
        logger.info("HSM reset")
