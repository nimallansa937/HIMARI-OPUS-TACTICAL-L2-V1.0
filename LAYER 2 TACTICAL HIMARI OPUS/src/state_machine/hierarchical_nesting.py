"""
HIMARI Layer 2 - Hierarchical Nesting (E2)
Super-state transitions for 70% rule reduction.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalState:
    """A state that can contain sub-states."""
    name: str
    parent: Optional['HierarchicalState'] = None
    children: List['HierarchicalState'] = field(default_factory=list)
    is_initial: bool = False
    
    def __post_init__(self):
        if self.parent:
            self.parent.children.append(self)
            
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    @property
    def is_super_state(self) -> bool:
        return len(self.children) > 0
    
    def get_ancestors(self) -> List['HierarchicalState']:
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current)
            current = current.parent
        return ancestors
    
    def get_initial_leaf(self) -> 'HierarchicalState':
        if self.is_leaf:
            return self
        for child in self.children:
            if child.is_initial:
                return child.get_initial_leaf()
        return self.children[0].get_initial_leaf() if self.children else self
    
    def contains(self, other: 'HierarchicalState') -> bool:
        if other == self:
            return True
        return any(child.contains(other) for child in self.children)


class HierarchicalHSM:
    """HSM with hierarchical state nesting for 70% fewer rules."""
    
    def __init__(self):
        self.states: Dict[str, HierarchicalState] = {}
        self.transitions: Dict[Tuple[str, str], str] = {}
        self.current: Optional[HierarchicalState] = None
        self.history: Dict[str, HierarchicalState] = {}
        
    def add_state(self, name: str, parent: Optional[HierarchicalState] = None,
                  is_initial: bool = False) -> HierarchicalState:
        state = HierarchicalState(name=name, parent=parent, is_initial=is_initial)
        self.states[name] = state
        return state
    
    def add_transition(self, from_state: HierarchicalState, event: str,
                      to_state: HierarchicalState) -> None:
        self.transitions[(from_state.name, event)] = to_state.name
        
    def set_initial(self, state: HierarchicalState) -> None:
        self.current = state.get_initial_leaf()
        
    def _find_transition(self, event: str) -> Optional[HierarchicalState]:
        key = (self.current.name, event)
        if key in self.transitions:
            return self.states[self.transitions[key]]
            
        for ancestor in self.current.get_ancestors():
            key = (ancestor.name, event)
            if key in self.transitions:
                return self.states[self.transitions[key]]
                
        return None
    
    def process_event(self, event: str) -> bool:
        target = self._find_transition(event)
        if target is None:
            return False
            
        for ancestor in self.current.get_ancestors():
            self.history[ancestor.name] = self.current
            
        old_state = self.current
        self.current = target.get_initial_leaf()
        
        logger.debug(f"Transition: {old_state.name} -> {self.current.name} via {event}")
        return True
    
    def restore_history(self, super_state: HierarchicalState) -> bool:
        if super_state.name in self.history:
            self.current = self.history[super_state.name]
            return True
        return False


def create_hierarchical_trading_fsm() -> HierarchicalHSM:
    """Create trading FSM with hierarchical structure."""
    hsm = HierarchicalHSM()
    
    flat = hsm.add_state('FLAT')
    long_mode = hsm.add_state('LONG_MODE')
    short_mode = hsm.add_state('SHORT_MODE')
    
    long_entry = hsm.add_state('LONG_ENTRY', parent=long_mode, is_initial=True)
    long_hold = hsm.add_state('LONG_HOLD', parent=long_mode)
    long_exit = hsm.add_state('LONG_EXIT', parent=long_mode)
    
    short_entry = hsm.add_state('SHORT_ENTRY', parent=short_mode, is_initial=True)
    short_hold = hsm.add_state('SHORT_HOLD', parent=short_mode)
    short_exit = hsm.add_state('SHORT_EXIT', parent=short_mode)
    
    hsm.add_transition(flat, 'buy_signal', long_entry)
    hsm.add_transition(flat, 'sell_signal', short_entry)
    
    hsm.add_transition(long_entry, 'confirmed', long_hold)
    hsm.add_transition(long_hold, 'exit_signal', long_exit)
    hsm.add_transition(long_exit, 'executed', flat)
    
    hsm.add_transition(short_entry, 'confirmed', short_hold)
    hsm.add_transition(short_hold, 'exit_signal', short_exit)
    hsm.add_transition(short_exit, 'executed', flat)
    
    # Super-state transitions (key benefit - replaces 3 rules each)
    hsm.add_transition(long_mode, 'crisis', flat)
    hsm.add_transition(short_mode, 'crisis', flat)
    hsm.add_transition(long_mode, 'stop_loss', flat)
    hsm.add_transition(short_mode, 'stop_loss', flat)
    
    hsm.set_initial(flat)
    return hsm
