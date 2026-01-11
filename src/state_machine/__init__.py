"""
HIMARI Layer 2 - State Machine Module
Subsystem E: Hierarchical State Machine (6 Methods)

Components:
    - E1: Orthogonal Regions - Independent parallel state dimensions
    - E2: Hierarchical Nesting - Super-state transitions
    - E3: History States - State memory for resumption
    - E4: Synchronized Events - Cross-region coordination
    - E5: Learned Transitions - ML-based transition timing
    - E6: Oscillation Detection - Anti-churn filter
"""

from .hsm import (
    HierarchicalStateMachine,
    HSMConfig,
    PositionState,
    RegimeState,
    SuperState,
    StateEvent,
    Transition
)

# E1: Orthogonal Regions
from .orthogonal_regions import (
    OrthogonalHSM,
    StateRegion,
    PositionState as OrthogonalPositionState,
    RegimeState as OrthogonalRegimeState,
    create_trading_hsm
)

# E2: Hierarchical Nesting
from .hierarchical_nesting import (
    HierarchicalHSM,
    HierarchicalState,
    create_hierarchical_trading_fsm
)

# E3: History States
from .history_states import (
    HistoryStateManager,
    HistoryConfig
)

# E4: Synchronized Events
from .synchronized_events import (
    SynchronizedEventBus,
    SyncEventConfig,
    EventSubscription,
    TradingEvents,
    create_trading_event_bus
)

# E5: Learned Transitions
from .learned_transitions import (
    LearnedTransitionManager,
    LearnedTransitionConfig,
    TransitionPredictor
)

# E6: Oscillation Detection
from .oscillation_detection import (
    OscillationDetector,
    OscillationConfig,
    TransitionRecord
)

# Section 7: Complete HSM Integration
from .trading_hsm import (
    TradingHSM,
    TradingHSMConfig
)

__all__ = [
    # Original HSM
    'HierarchicalStateMachine',
    'HSMConfig',
    'PositionState',
    'RegimeState',
    'SuperState',
    'StateEvent',
    'Transition',
    # E1: Orthogonal Regions
    'OrthogonalHSM',
    'StateRegion',
    'OrthogonalPositionState',
    'OrthogonalRegimeState',
    'create_trading_hsm',
    # E2: Hierarchical Nesting
    'HierarchicalHSM',
    'HierarchicalState',
    'create_hierarchical_trading_fsm',
    # E3: History States
    'HistoryStateManager',
    'HistoryConfig',
    # E4: Synchronized Events
    'SynchronizedEventBus',
    'SyncEventConfig',
    'EventSubscription',
    'TradingEvents',
    'create_trading_event_bus',
    # E5: Learned Transitions
    'LearnedTransitionManager',
    'LearnedTransitionConfig',
    'TransitionPredictor',
    # E6: Oscillation Detection
    'OscillationDetector',
    'OscillationConfig',
    'TransitionRecord',
    # Section 7: Complete HSM Integration
    'TradingHSM',
    'TradingHSMConfig',
]
