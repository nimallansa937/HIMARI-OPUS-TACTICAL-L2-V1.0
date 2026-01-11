"""HIMARI Layer 2 - Part I: Reachability Analysis"""

from typing import Set, Dict, List
import logging

logger = logging.getLogger(__name__)


class ReachabilityAnalyzer:
    """Analyze reachability of safety-critical states."""
    def __init__(self):
        self.safe_states: Set[str] = {"FLAT", "HEDGED"}
        self.unsafe_states: Set[str] = {"OVERLEVERAGED", "MARGIN_CALL"}
        
    def compute_reachable(self, start: str, transitions: Dict[str, List[str]]) -> Set[str]:
        reachable = {start}
        queue = [start]
        while queue:
            state = queue.pop(0)
            for next_state in transitions.get(state, []):
                if next_state not in reachable:
                    reachable.add(next_state)
                    queue.append(next_state)
        return reachable
        
    def can_reach_unsafe(self, start: str, transitions: dict) -> bool:
        reachable = self.compute_reachable(start, transitions)
        return bool(reachable & self.unsafe_states)
