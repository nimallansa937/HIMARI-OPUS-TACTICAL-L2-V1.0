"""
HIMARI Layer 2 - Safety Module (v5.0)
Subsystem I: Simplex Safety System

Components (v5.0 - 78 methods architecture):
    - 4-Level Simplex (I1): Runtime assurance with fallback cascade - UPGRADED
    - Predictive Safety (I2): N-step lookahead - NEW
    - Formal Verification (I3): Invariant checking - NEW
    - Safety Monitor (I6): Real-time monitoring - KEPT
    - Stop-Loss Enforcer (I7): Hard limits - KEPT
"""

from .simplex_safety import (
    SimplexSafetySystem,
    SafetyConfig,
    SafetyState,
    SafetyLevel,
    SafetyViolation,
    MarketState,
    rule_based_baseline,
    create_simplex_safety
)

__all__ = [
    # v5.0 Components
    'SimplexSafetySystem',
    'SafetyConfig',
    'SafetyState',
    'SafetyLevel',
    'SafetyViolation',
    'MarketState',
    'rule_based_baseline',
    'create_simplex_safety'
]
