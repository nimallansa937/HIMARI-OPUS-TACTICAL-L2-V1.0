"""
HIMARI Layer 2 - 4-Level Simplex Safety System
Subsystem I: Simplex Safety (Method I1)

Purpose:
    Runtime safety monitor with 4-level fallback cascade to prevent 
    catastrophic actions from RL agents during market stress.

Why 4-Level over 2-Level?
    - More granular degradation under uncertainty
    - Level 3 (CQL) provides conservative fallback before rule-based
    - Smoother transition maintains partial alpha capture
    - Formal verification at each level

Architecture:
    Level 0: Primary agent (FLAG-TRADER) - full alpha
    Level 1: Secondary agent (CGDT) - reduced risk  
    Level 2: Conservative agent (CQL) - minimal risk
    Level 3: Rule-based baseline - capital preservation only

Invariants enforced at all levels:
    - Max position size
    - Max leverage  
    - Max drawdown
    - Liquidity check
    - Volatility gate

Reference:
    - Simplex architecture for runtime assurance
    - RSS (Responsibility-Sensitive Safety) principles
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple
from enum import Enum
from loguru import logger


class SafetyLevel(Enum):
    """Simplex safety levels"""
    LEVEL_0 = 0  # Primary (FLAG-TRADER)
    LEVEL_1 = 1  # Secondary (CGDT)
    LEVEL_2 = 2  # Conservative (CQL)
    LEVEL_3 = 3  # Rule-based baseline


class SafetyViolation(Enum):
    """Types of safety violations"""
    NONE = "none"
    POSITION_SIZE = "position_size"
    LEVERAGE = "leverage"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"
    UNCERTAINTY = "uncertainty"


@dataclass
class SafetyConfig:
    """Simplex safety configuration"""
    max_position_size: float = 0.20      # Max 20% of portfolio per asset
    max_leverage: float = 3.0            # Max 3x leverage
    max_drawdown: float = 0.05           # Max 5% drawdown before intervention
    volatility_threshold: float = 0.05   # High volatility gate
    min_liquidity_ratio: float = 0.10    # Min liquidity requirement
    correlation_threshold: float = 0.80  # Max correlation for diversification
    uncertainty_threshold: float = 0.70  # Uncertainty gate for fallback
    
    # Level transition thresholds
    l0_to_l1_threshold: float = 0.30     # Confidence threshold
    l1_to_l2_threshold: float = 0.50     # Risk threshold
    l2_to_l3_threshold: float = 0.70     # Crisis threshold


@dataclass
class SafetyState:
    """Current safety system state"""
    current_level: SafetyLevel = SafetyLevel.LEVEL_0
    violations: List[SafetyViolation] = field(default_factory=list)
    risk_score: float = 0.0
    uncertainty_score: float = 0.0
    is_safe: bool = True
    recommended_action: Optional[str] = None
    position_multiplier: float = 1.0


@dataclass 
class MarketState:
    """Market state for safety evaluation"""
    current_position: float = 0.0
    current_leverage: float = 1.0
    current_drawdown: float = 0.0
    current_volatility: float = 0.02
    available_liquidity: float = 1.0
    portfolio_correlation: float = 0.0
    model_uncertainty: float = 0.0
    proposed_action: Optional[str] = None
    proposed_size: float = 0.0


class SimplexSafetySystem:
    """
    4-Level Simplex Safety System.
    
    Provides runtime assurance through:
    1. Invariant monitoring (position, leverage, drawdown, etc.)
    2. Graceful degradation via 4-level fallback cascade
    3. Action filtering and size scaling
    
    Example:
        >>> config = SafetyConfig(max_leverage=3.0, max_drawdown=0.05)
        >>> safety = SimplexSafetySystem(config)
        >>> state = safety.evaluate(market_state, agent_action)
        >>> safe_action = safety.filter_action(agent_action, state)
    """
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.current_level = SafetyLevel.LEVEL_0
        self.violation_history: List[SafetyViolation] = []
        self.level_history: List[SafetyLevel] = []
        
        # Fallback agents (to be registered)
        self.fallback_agents: Dict[SafetyLevel, Callable] = {}
        
        logger.debug(
            f"SimplexSafetySystem initialized: max_leverage={self.config.max_leverage}, "
            f"max_dd={self.config.max_drawdown}"
        )
    
    def register_fallback(self, level: SafetyLevel, agent: Callable):
        """
        Register fallback agent for a level.
        
        Args:
            level: SafetyLevel to register for
            agent: Callable that takes MarketState and returns action
        """
        self.fallback_agents[level] = agent
        logger.debug(f"Registered fallback agent for {level.name}")
    
    def check_invariants(self, state: MarketState) -> List[SafetyViolation]:
        """
        Check all safety invariants.
        
        Args:
            state: Current market state
            
        Returns:
            List of violations (empty if all invariants satisfied)
        """
        violations = []
        
        # Position size check
        if abs(state.current_position) > self.config.max_position_size:
            violations.append(SafetyViolation.POSITION_SIZE)
        
        # Leverage check
        if state.current_leverage > self.config.max_leverage:
            violations.append(SafetyViolation.LEVERAGE)
        
        # Drawdown check
        if state.current_drawdown > self.config.max_drawdown:
            violations.append(SafetyViolation.DRAWDOWN)
        
        # Volatility gate
        if state.current_volatility > self.config.volatility_threshold:
            violations.append(SafetyViolation.VOLATILITY)
        
        # Liquidity check
        if state.available_liquidity < self.config.min_liquidity_ratio:
            violations.append(SafetyViolation.LIQUIDITY)
        
        # Correlation check
        if state.portfolio_correlation > self.config.correlation_threshold:
            violations.append(SafetyViolation.CORRELATION)
        
        # Uncertainty gate
        if state.model_uncertainty > self.config.uncertainty_threshold:
            violations.append(SafetyViolation.UNCERTAINTY)
        
        return violations
    
    def compute_risk_score(self, state: MarketState) -> float:
        """
        Compute aggregate risk score [0, 1].
        
        Higher score = higher risk = more likely to trigger fallback.
        """
        scores = []
        
        # Position risk
        pos_risk = abs(state.current_position) / self.config.max_position_size
        scores.append(min(pos_risk, 1.0))
        
        # Leverage risk
        lev_risk = state.current_leverage / self.config.max_leverage
        scores.append(min(lev_risk, 1.0))
        
        # Drawdown risk (exponential weighting)
        dd_risk = (state.current_drawdown / self.config.max_drawdown) ** 2
        scores.append(min(dd_risk, 1.0))
        
        # Volatility risk
        vol_risk = state.current_volatility / self.config.volatility_threshold
        scores.append(min(vol_risk, 1.0))
        
        # Uncertainty risk
        scores.append(state.model_uncertainty)
        
        # Weighted average (drawdown and uncertainty weighted higher)
        weights = [0.15, 0.15, 0.30, 0.20, 0.20]
        return sum(w * s for w, s in zip(weights, scores))
    
    def determine_level(self, risk_score: float, violations: List[SafetyViolation]) -> SafetyLevel:
        """
        Determine appropriate safety level based on risk.
        
        Args:
            risk_score: Aggregate risk score [0, 1]
            violations: Current invariant violations
            
        Returns:
            Appropriate SafetyLevel
        """
        # Immediate escalation for critical violations
        critical = {SafetyViolation.DRAWDOWN, SafetyViolation.LEVERAGE}
        if any(v in critical for v in violations):
            return SafetyLevel.LEVEL_3
        
        # Risk-based escalation
        if risk_score > self.config.l2_to_l3_threshold:
            return SafetyLevel.LEVEL_3
        elif risk_score > self.config.l1_to_l2_threshold:
            return SafetyLevel.LEVEL_2
        elif risk_score > self.config.l0_to_l1_threshold:
            return SafetyLevel.LEVEL_1
        else:
            return SafetyLevel.LEVEL_0
    
    def evaluate(self, state: MarketState) -> SafetyState:
        """
        Evaluate safety state and determine appropriate level.
        
        Args:
            state: Current market state
            
        Returns:
            SafetyState with recommendations
        """
        # Check invariants
        violations = self.check_invariants(state)
        
        # Compute risk score
        risk_score = self.compute_risk_score(state)
        
        # Determine level
        new_level = self.determine_level(risk_score, violations)
        
        # Log level changes
        if new_level != self.current_level:
            logger.info(
                f"Safety level change: {self.current_level.name} → {new_level.name} "
                f"(risk={risk_score:.2f}, violations={[v.value for v in violations]})"
            )
            self.current_level = new_level
        
        # Track history
        self.level_history.append(new_level)
        if len(self.level_history) > 1000:
            self.level_history.pop(0)
        
        # Compute position multiplier based on level
        position_multipliers = {
            SafetyLevel.LEVEL_0: 1.0,
            SafetyLevel.LEVEL_1: 0.7,
            SafetyLevel.LEVEL_2: 0.4,
            SafetyLevel.LEVEL_3: 0.1
        }
        
        return SafetyState(
            current_level=new_level,
            violations=violations,
            risk_score=risk_score,
            uncertainty_score=state.model_uncertainty,
            is_safe=len(violations) == 0,
            position_multiplier=position_multipliers[new_level],
            recommended_action=self._get_recommendation(new_level, violations)
        )
    
    def _get_recommendation(self, level: SafetyLevel, 
                           violations: List[SafetyViolation]) -> str:
        """Get human-readable recommendation"""
        if level == SafetyLevel.LEVEL_0:
            return "NORMAL: Full trading allowed"
        elif level == SafetyLevel.LEVEL_1:
            return "CAUTION: Reduced position sizing"
        elif level == SafetyLevel.LEVEL_2:
            return "WARNING: Conservative mode, minimal new positions"
        else:
            viol_str = ", ".join(v.value for v in violations)
            return f"CRITICAL: Rule-based only. Violations: {viol_str}"
    
    def filter_action(
        self,
        proposed_action: str,
        proposed_size: float,
        safety_state: SafetyState
    ) -> Tuple[str, float]:
        """
        Filter proposed action through safety system.
        
        Args:
            proposed_action: Action from primary agent ("BUY", "SELL", "HOLD")
            proposed_size: Proposed position size
            safety_state: Current safety state
            
        Returns:
            (filtered_action, filtered_size)
        """
        # Apply position multiplier
        filtered_size = proposed_size * safety_state.position_multiplier
        
        # At LEVEL_3, only allow position reduction
        if safety_state.current_level == SafetyLevel.LEVEL_3:
            if proposed_action in ["BUY", "STRONG_BUY"]:
                return ("HOLD", 0.0)
            elif proposed_action in ["SELL", "STRONG_SELL"]:
                return (proposed_action, filtered_size)
            else:
                return ("HOLD", 0.0)
        
        # At LEVEL_2, no strong positions
        if safety_state.current_level == SafetyLevel.LEVEL_2:
            if proposed_action in ["STRONG_BUY", "STRONG_SELL"]:
                action = "BUY" if "BUY" in proposed_action else "SELL"
                return (action, filtered_size * 0.5)
        
        return (proposed_action, filtered_size)
    
    def get_fallback_action(
        self,
        state: MarketState,
        safety_state: SafetyState
    ) -> Optional[str]:
        """
        Get action from appropriate fallback agent.
        
        Args:
            state: Current market state
            safety_state: Current safety state
            
        Returns:
            Action from fallback agent, or None if not registered
        """
        level = safety_state.current_level
        
        if level in self.fallback_agents:
            try:
                return self.fallback_agents[level](state)
            except Exception as e:
                logger.error(f"Fallback agent {level.name} failed: {e}")
                # Escalate to next level
                next_level = SafetyLevel(min(level.value + 1, 3))
                if next_level in self.fallback_agents:
                    return self.fallback_agents[next_level](state)
        
        # Ultimate fallback: HOLD
        return "HOLD"
    
    def get_statistics(self) -> Dict[str, float]:
        """Get safety system statistics"""
        if not self.level_history:
            return {}
        
        level_counts = {level: 0 for level in SafetyLevel}
        for level in self.level_history:
            level_counts[level] += 1
        
        total = len(self.level_history)
        return {
            'pct_level_0': level_counts[SafetyLevel.LEVEL_0] / total,
            'pct_level_1': level_counts[SafetyLevel.LEVEL_1] / total,
            'pct_level_2': level_counts[SafetyLevel.LEVEL_2] / total,
            'pct_level_3': level_counts[SafetyLevel.LEVEL_3] / total,
            'total_evaluations': total
        }


# Rule-based baseline agent for LEVEL_3
def rule_based_baseline(state: MarketState) -> str:
    """
    Simple rule-based agent for capital preservation.
    
    Rules:
    - If in drawdown: HOLD (no new positions)
    - If position exists and in profit: Consider taking profit
    - If high volatility: HOLD
    - Otherwise: HOLD
    """
    if state.current_drawdown > 0.03:
        return "HOLD"
    
    if state.current_volatility > 0.04:
        return "HOLD"
    
    if state.current_position > 0.1 and state.current_drawdown < 0:
        return "SELL"  # Take profit
    
    return "HOLD"


# Factory function
def create_simplex_safety(max_drawdown: float = 0.05, 
                          max_leverage: float = 3.0) -> SimplexSafetySystem:
    """Create Simplex safety system"""
    config = SafetyConfig(max_drawdown=max_drawdown, max_leverage=max_leverage)
    safety = SimplexSafetySystem(config)
    safety.register_fallback(SafetyLevel.LEVEL_3, rule_based_baseline)
    return safety
