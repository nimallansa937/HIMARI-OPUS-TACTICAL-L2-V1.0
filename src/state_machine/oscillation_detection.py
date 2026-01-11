"""
HIMARI Layer 2 - Oscillation Detection (E6)
Anti-churn filter to prevent state flip-flopping (-30% trades).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class OscillationConfig:
    window_seconds: float = 300.0  # 5 minutes
    max_transitions_in_window: int = 4
    cooldown_seconds: float = 60.0
    detect_reversal_patterns: bool = True
    min_transition_interval: float = 5.0
    exempt_transitions: List[Tuple[str, str]] = field(default_factory=lambda: [('ANY', 'FLAT')])


@dataclass
class TransitionRecord:
    timestamp: float
    from_state: str
    to_state: str
    was_blocked: bool = False
    block_reason: Optional[str] = None


class OscillationDetector:
    """
    Detects and prevents rapid state oscillation.
    
    Patterns detected:
    1. Too many transitions in time window
    2. Reversal pattern: A → B → A
    3. Transitions too close together
    """
    
    def __init__(self, config: Optional[OscillationConfig] = None):
        self.config = config or OscillationConfig()
        self.history: deque = deque(maxlen=100)
        self.cooldown_until: float = 0.0
        self.oscillation_count: int = 0
        
    def _current_time(self) -> float:
        return time.time()
    
    def _is_exempt(self, from_state: str, to_state: str) -> bool:
        for exempt_from, exempt_to in self.config.exempt_transitions:
            if (exempt_from == 'ANY' or exempt_from == from_state) and \
               (exempt_to == 'ANY' or exempt_to == to_state):
                return True
        return False
    
    def _get_recent_transitions(self) -> List[TransitionRecord]:
        cutoff = self._current_time() - self.config.window_seconds
        return [t for t in self.history if t.timestamp > cutoff and not t.was_blocked]
    
    def _check_reversal_pattern(self, from_state: str, to_state: str) -> bool:
        if not self.config.detect_reversal_patterns:
            return False
            
        recent = self._get_recent_transitions()
        if len(recent) < 1:
            return False
            
        last = recent[-1]
        if last.to_state == from_state and last.from_state == to_state:
            time_since_last = self._current_time() - last.timestamp
            if time_since_last < self.config.window_seconds / 2:
                return True
                
        return False
    
    def _check_transition_flood(self) -> bool:
        recent = self._get_recent_transitions()
        return len(recent) >= self.config.max_transitions_in_window
    
    def _check_too_fast(self) -> bool:
        recent = self._get_recent_transitions()
        if not recent:
            return False
        time_since_last = self._current_time() - recent[-1].timestamp
        return time_since_last < self.config.min_transition_interval
    
    def _in_cooldown(self) -> bool:
        return self._current_time() < self.cooldown_until
    
    def should_block(self, from_state: str, to_state: str) -> Tuple[bool, Optional[str]]:
        if self._is_exempt(from_state, to_state):
            return False, None
            
        if self._in_cooldown():
            return True, f"In cooldown until {self.cooldown_until:.1f}"
            
        if self._check_reversal_pattern(from_state, to_state):
            self._enter_cooldown("reversal pattern detected")
            return True, "Reversal pattern (A→B→A) detected"
            
        if self._check_transition_flood():
            self._enter_cooldown("transition flood")
            return True, f"Too many transitions in {self.config.window_seconds}s window"
            
        if self._check_too_fast():
            return True, f"Transition too fast (min interval: {self.config.min_transition_interval}s)"
            
        return False, None
    
    def _enter_cooldown(self, reason: str) -> None:
        self.cooldown_until = self._current_time() + self.config.cooldown_seconds
        self.oscillation_count += 1
        logger.warning(f"Oscillation detected ({reason}), cooldown until {self.cooldown_until:.1f}")
    
    def record_transition(self, from_state: str, to_state: str,
                         was_blocked: bool = False, block_reason: Optional[str] = None) -> None:
        record = TransitionRecord(
            timestamp=self._current_time(),
            from_state=from_state,
            to_state=to_state,
            was_blocked=was_blocked,
            block_reason=block_reason
        )
        self.history.append(record)
    
    def get_statistics(self) -> Dict:
        recent = self._get_recent_transitions()
        blocked = [t for t in self.history if t.was_blocked]
        
        return {
            'transitions_in_window': len(recent),
            'total_blocked': len(blocked),
            'oscillation_count': self.oscillation_count,
            'in_cooldown': self._in_cooldown(),
            'cooldown_remaining': max(0, self.cooldown_until - self._current_time()),
        }
    
    def reset(self) -> None:
        self.history.clear()
        self.cooldown_until = 0.0
        self.oscillation_count = 0
