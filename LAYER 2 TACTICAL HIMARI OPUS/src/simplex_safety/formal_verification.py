"""HIMARI Layer 2 - Part I: Formal Verification & Reachability"""

from dataclasses import dataclass
from typing import Optional, Set
import logging

logger = logging.getLogger(__name__)


class FormalVerification:
    """Formal verification of safety properties."""
    def __init__(self):
        self.invariants: list = []
        
    def add_invariant(self, name: str, check_fn) -> None:
        self.invariants.append((name, check_fn))
        
    def verify_all(self, state: dict) -> tuple:
        violated = []
        for name, check in self.invariants:
            if not check(state):
                violated.append(name)
        return len(violated) == 0, violated


class Reachability:
    """Reachability analysis for safety states."""
    def __init__(self):
        self.unsafe_states: Set[str] = set()
        
    def add_unsafe(self, state: str) -> None:
        self.unsafe_states.add(state)
        
    def is_reachable(self, current: str, target: str, transitions: dict) -> bool:
        visited = set()
        queue = [current]
        while queue:
            state = queue.pop(0)
            if state == target:
                return True
            if state in visited:
                continue
            visited.add(state)
            queue.extend(transitions.get(state, []))
        return False
