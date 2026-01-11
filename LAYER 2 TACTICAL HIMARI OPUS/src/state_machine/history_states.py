"""
HIMARI Layer 2 - History States (E3)
State memory for resumption after interruption.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass 
class HistoryConfig:
    """Configuration for history state behavior."""
    use_deep_history: bool = False
    default_to_initial: bool = True
    clear_on_restore: bool = True
    conservative_reentry: bool = True  # Always start from ENTRY, not HOLD


class HistoryStateManager:
    """
    Manages historical state for super-states.
    
    HIMARI uses conservative re-entry: after any interruption, always
    restart from ENTRY state rather than resuming HOLD.
    """
    
    def __init__(self, config: Optional[HistoryConfig] = None):
        self.config = config or HistoryConfig()
        self.history: Dict[str, str] = {}
        self.deep_history: Dict[str, list] = {}
        
    def record_exit(self, super_state: str, current_sub_state: str,
                   ancestors: Optional[list] = None) -> None:
        self.history[super_state] = current_sub_state
        
        if self.config.use_deep_history and ancestors:
            self.deep_history[super_state] = ancestors.copy()
            
        logger.debug(f"Recorded history for {super_state}: {current_sub_state}")
        
    def get_restoration_state(self, super_state: str, initial_state: str) -> str:
        if self.config.conservative_reentry:
            logger.debug(f"Conservative re-entry for {super_state}: using {initial_state}")
            return initial_state
            
        if super_state not in self.history:
            return initial_state
            
        restored = self.history[super_state]
        
        if self.config.clear_on_restore:
            del self.history[super_state]
            
        logger.debug(f"Restoring {super_state} to {restored}")
        return restored
    
    def has_history(self, super_state: str) -> bool:
        return super_state in self.history
    
    def get_history(self, super_state: str) -> Optional[str]:
        return self.history.get(super_state)
    
    def clear_history(self, super_state: Optional[str] = None) -> None:
        if super_state:
            self.history.pop(super_state, None)
            self.deep_history.pop(super_state, None)
        else:
            self.history.clear()
            self.deep_history.clear()
    
    def get_all_history(self) -> Dict[str, str]:
        return dict(self.history)
