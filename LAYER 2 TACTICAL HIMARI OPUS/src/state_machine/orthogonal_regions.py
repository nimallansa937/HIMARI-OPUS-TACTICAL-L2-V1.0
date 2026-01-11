"""
HIMARI Layer 2 - Orthogonal Regions (E1)
HSM with independent parallel state dimensions.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Callable, List
import logging

logger = logging.getLogger(__name__)


class PositionState(Enum):
    """Position region states."""
    FLAT = auto()
    LONG_ENTRY = auto()
    LONG_HOLD = auto()
    LONG_EXIT = auto()
    SHORT_ENTRY = auto()
    SHORT_HOLD = auto()
    SHORT_EXIT = auto()


class RegimeState(Enum):
    """Regime region states."""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING = auto()
    HIGH_VOLATILITY = auto()
    CRISIS = auto()


@dataclass
class StateRegion:
    """A single orthogonal region in the HSM."""
    name: str
    states: type
    current: Enum = None
    history: Optional[Enum] = None
    entry_time: float = 0.0
    transitions: Dict[tuple, Enum] = field(default_factory=dict)
    on_enter: Dict[Enum, List[Callable]] = field(default_factory=dict)
    on_exit: Dict[Enum, List[Callable]] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.current is None:
            self.current = list(self.states)[0]
    
    def can_transition(self, event: str) -> bool:
        return (self.current, event) in self.transitions
    
    def transition(self, event: str, timestamp: float = 0.0) -> bool:
        key = (self.current, event)
        if key not in self.transitions:
            return False
            
        old_state = self.current
        new_state = self.transitions[key]
        
        for callback in self.on_exit.get(old_state, []):
            callback(old_state, new_state, event)
            
        self.history = old_state
        self.current = new_state
        self.entry_time = timestamp
        
        for callback in self.on_enter.get(new_state, []):
            callback(old_state, new_state, event)
            
        return True
    
    def time_in_state(self, current_time: float) -> float:
        return current_time - self.entry_time


class OrthogonalHSM:
    """HSM with orthogonal regions for O(sum) instead of O(product) states."""
    
    def __init__(self):
        self.regions: Dict[str, StateRegion] = {}
        self.synchronized_events: Dict[str, List[str]] = {}
        self._event_log: List[tuple] = []
        
    def add_region(self, name: str, states: type, initial: Enum,
                   transitions: Optional[Dict[tuple, Enum]] = None) -> None:
        region = StateRegion(name=name, states=states, current=initial,
                            transitions=transitions or {})
        self.regions[name] = region
        
    def add_transition(self, region: str, from_state: Enum, 
                      event: str, to_state: Enum) -> None:
        if region not in self.regions:
            raise ValueError(f"Unknown region: {region}")
        self.regions[region].transitions[(from_state, event)] = to_state
        
    def get_state(self, region: str) -> Enum:
        return self.regions[region].current
    
    def get_all_states(self) -> Dict[str, Enum]:
        return {name: r.current for name, r in self.regions.items()}
    
    def process_event(self, event: str, timestamp: float = 0.0,
                     target_regions: Optional[List[str]] = None) -> Dict[str, bool]:
        results = {}
        regions_to_check = target_regions or list(self.regions.keys())
        
        for region_name in regions_to_check:
            if region_name in self.regions:
                results[region_name] = self.regions[region_name].transition(event, timestamp)
                
        if event in self.synchronized_events:
            for sync_event in self.synchronized_events[event]:
                self.process_event(sync_event, timestamp)
                
        self._event_log.append((timestamp, event, dict(results)))
        return results
    
    def add_synchronized_event(self, trigger: str, sync_events: List[str]) -> None:
        self.synchronized_events[trigger] = sync_events
        
    def validate_action(self, action: str, region: str = 'position') -> bool:
        return self.regions[region].can_transition(action)
    
    def get_valid_actions(self, region: str = 'position') -> List[str]:
        current = self.regions[region].current
        return [event for (state, event), _ in self.regions[region].transitions.items()
                if state == current]


def create_trading_hsm() -> OrthogonalHSM:
    """Factory for configured trading HSM."""
    hsm = OrthogonalHSM()
    
    hsm.add_region('position', PositionState, PositionState.FLAT)
    
    position_transitions = [
        (PositionState.FLAT, 'buy_signal', PositionState.LONG_ENTRY),
        (PositionState.FLAT, 'sell_signal', PositionState.SHORT_ENTRY),
        (PositionState.LONG_ENTRY, 'entry_confirmed', PositionState.LONG_HOLD),
        (PositionState.LONG_ENTRY, 'entry_failed', PositionState.FLAT),
        (PositionState.LONG_HOLD, 'exit_signal', PositionState.LONG_EXIT),
        (PositionState.LONG_HOLD, 'stop_loss', PositionState.LONG_EXIT),
        (PositionState.LONG_EXIT, 'exit_confirmed', PositionState.FLAT),
        (PositionState.SHORT_ENTRY, 'entry_confirmed', PositionState.SHORT_HOLD),
        (PositionState.SHORT_ENTRY, 'entry_failed', PositionState.FLAT),
        (PositionState.SHORT_HOLD, 'exit_signal', PositionState.SHORT_EXIT),
        (PositionState.SHORT_HOLD, 'stop_loss', PositionState.SHORT_EXIT),
        (PositionState.SHORT_EXIT, 'exit_confirmed', PositionState.FLAT),
        (PositionState.LONG_ENTRY, 'crisis_exit', PositionState.FLAT),
        (PositionState.LONG_HOLD, 'crisis_exit', PositionState.FLAT),
        (PositionState.LONG_EXIT, 'crisis_exit', PositionState.FLAT),
        (PositionState.SHORT_ENTRY, 'crisis_exit', PositionState.FLAT),
        (PositionState.SHORT_HOLD, 'crisis_exit', PositionState.FLAT),
        (PositionState.SHORT_EXIT, 'crisis_exit', PositionState.FLAT),
    ]
    
    for from_state, event, to_state in position_transitions:
        hsm.add_transition('position', from_state, event, to_state)
        
    hsm.add_region('regime', RegimeState, RegimeState.RANGING)
    
    regime_transitions = [
        (RegimeState.RANGING, 'trend_detected', RegimeState.TRENDING_UP),
        (RegimeState.RANGING, 'downtrend_detected', RegimeState.TRENDING_DOWN),
        (RegimeState.RANGING, 'volatility_spike', RegimeState.HIGH_VOLATILITY),
        (RegimeState.TRENDING_UP, 'trend_reversal', RegimeState.TRENDING_DOWN),
        (RegimeState.TRENDING_UP, 'trend_end', RegimeState.RANGING),
        (RegimeState.TRENDING_DOWN, 'trend_reversal', RegimeState.TRENDING_UP),
        (RegimeState.TRENDING_DOWN, 'trend_end', RegimeState.RANGING),
        (RegimeState.HIGH_VOLATILITY, 'volatility_normalize', RegimeState.RANGING),
        (RegimeState.RANGING, 'crisis_detected', RegimeState.CRISIS),
        (RegimeState.TRENDING_UP, 'crisis_detected', RegimeState.CRISIS),
        (RegimeState.TRENDING_DOWN, 'crisis_detected', RegimeState.CRISIS),
        (RegimeState.HIGH_VOLATILITY, 'crisis_detected', RegimeState.CRISIS),
        (RegimeState.CRISIS, 'crisis_resolved', RegimeState.HIGH_VOLATILITY),
    ]
    
    for from_state, event, to_state in regime_transitions:
        hsm.add_transition('regime', from_state, event, to_state)
        
    hsm.add_synchronized_event('crisis_detected', ['crisis_exit'])
    
    return hsm
