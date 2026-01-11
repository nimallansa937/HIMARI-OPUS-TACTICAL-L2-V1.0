"""HIMARI Layer 2 - Part I: Safety Invariants, Monitor, Stop Loss, Recovery"""

from dataclasses import dataclass
from typing import Optional, Callable, List
import logging

logger = logging.getLogger(__name__)


class SafetyInvariants:
    """Maintain safety invariants."""
    def __init__(self):
        self.invariants: List[tuple] = []
        
    def add(self, name: str, check: Callable) -> None:
        self.invariants.append((name, check))
        
    def check_all(self, state: dict) -> List[str]:
        return [name for name, check in self.invariants if not check(state)]


class SafetyMonitor:
    """Real-time safety monitoring."""
    def __init__(self):
        self.alerts: list = []
        
    def monitor(self, metrics: dict) -> list:
        alerts = []
        if metrics.get('leverage', 0) > 3:
            alerts.append("HIGH_LEVERAGE")
        if metrics.get('drawdown', 0) > 0.2:
            alerts.append("HIGH_DRAWDOWN")
        self.alerts.extend(alerts)
        return alerts


class StopLossEnforcer:
    """Enforce stop-loss rules."""
    def __init__(self, stop_loss_pct: float = 0.02):
        self.stop_loss_pct = stop_loss_pct
        
    def should_stop(self, entry: float, current: float, is_long: bool) -> bool:
        if is_long:
            return (entry - current) / entry > self.stop_loss_pct
        return (current - entry) / entry > self.stop_loss_pct


class RecoveryProtocol:
    """Recovery protocol after safety violation."""
    def __init__(self):
        self.recovery_steps = ["HALT", "ASSESS", "REDUCE", "MONITOR", "RESUME"]
        self.current_step = 0
        
    def start_recovery(self) -> str:
        self.current_step = 0
        return self.recovery_steps[0]
        
    def next_step(self) -> str:
        self.current_step = min(self.current_step + 1, len(self.recovery_steps) - 1)
        return self.recovery_steps[self.current_step]
