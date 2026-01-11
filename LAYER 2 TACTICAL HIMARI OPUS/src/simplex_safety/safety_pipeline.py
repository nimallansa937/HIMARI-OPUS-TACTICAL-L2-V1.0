"""
HIMARI Layer 2 - Part I: Simplex Safety System
Multi-layer safety system for trading protection.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# I1: Circuit Breakers
# ============================================================================

@dataclass
class CircuitBreakerConfig:
    max_loss_per_trade: float = 0.02
    max_daily_loss: float = 0.05
    max_consecutive_losses: int = 5
    cooldown_seconds: float = 300

class CircuitBreaker:
    """Halts trading when risk limits exceeded."""
    def __init__(self, config=None):
        self.config = config or CircuitBreakerConfig()
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.halt_until = 0.0
        self.trade_count = 0
        
    def check_trade(self, potential_loss: float) -> Tuple[bool, str]:
        """Check if trade allowed. Returns (allowed, reason)."""
        if time.time() < self.halt_until:
            return False, f"Circuit breaker active until {self.halt_until}"
        if potential_loss > self.config.max_loss_per_trade:
            return False, "Exceeds max loss per trade"
        if self.daily_pnl + potential_loss < -self.config.max_daily_loss:
            return False, "Would exceed max daily loss"
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False, "Too many consecutive losses"
        return True, ""
    
    def record_trade(self, pnl: float):
        self.daily_pnl += pnl
        self.trade_count += 1
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.config.max_consecutive_losses:
                self.halt_until = time.time() + self.config.cooldown_seconds
        else:
            self.consecutive_losses = 0
            
    def reset_daily(self):
        self.daily_pnl = 0.0
        self.trade_count = 0


# ============================================================================
# I2: Anomaly Detection
# ============================================================================

class AnomalyDetector:
    """Detects anomalous market conditions or system behavior."""
    def __init__(self, z_threshold: float = 3.0, window: int = 100):
        self.z_threshold = z_threshold
        self.returns = deque(maxlen=window)
        self.latencies = deque(maxlen=window)
        
    def check_price(self, price_change: float) -> Tuple[bool, float]:
        self.returns.append(price_change)
        if len(self.returns) < 20:
            return False, 0.0
        mean, std = np.mean(self.returns), np.std(self.returns)
        z_score = abs(price_change - mean) / (std + 1e-8)
        return z_score > self.z_threshold, z_score
    
    def check_latency(self, latency_ms: float) -> bool:
        self.latencies.append(latency_ms)
        if len(self.latencies) < 20:
            return False
        threshold = np.percentile(self.latencies, 95) * 2
        return latency_ms > threshold


# ============================================================================
# I3: Position Limits
# ============================================================================

@dataclass
class PositionLimitsConfig:
    max_position_value: float = 100000
    max_position_pct: float = 0.20
    max_notional: float = 500000
    max_positions: int = 10

class PositionLimiter:
    """Enforces position limits."""
    def __init__(self, config=None):
        self.config = config or PositionLimitsConfig()
        self.positions = {}
        
    def check_order(self, symbol: str, value: float, capital: float) -> Tuple[bool, str]:
        if value > self.config.max_position_value:
            return False, "Exceeds max position value"
        if value / capital > self.config.max_position_pct:
            return False, "Exceeds max position percentage"
        total_notional = sum(self.positions.values()) + value
        if total_notional > self.config.max_notional:
            return False, "Exceeds max notional"
        if symbol not in self.positions and len(self.positions) >= self.config.max_positions:
            return False, "Max positions reached"
        return True, ""
    
    def update_position(self, symbol: str, value: float):
        if value == 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = value


# ============================================================================
# I4: Order Validation
# ============================================================================

class OrderValidator:
    """Validates order parameters before execution."""
    def __init__(self):
        self.last_order_time = {}
        self.min_order_interval = 1.0  # seconds
        
    def validate(self, symbol: str, side: str, quantity: float, 
                price: float) -> Tuple[bool, str]:
        # Basic validation
        if quantity <= 0:
            return False, "Invalid quantity"
        if price <= 0:
            return False, "Invalid price"
        if side not in ['buy', 'sell']:
            return False, "Invalid side"
            
        # Rate limiting
        last_time = self.last_order_time.get(symbol, 0)
        if time.time() - last_time < self.min_order_interval:
            return False, "Order rate limit exceeded"
            
        return True, ""
    
    def record_order(self, symbol: str):
        self.last_order_time[symbol] = time.time()


# ============================================================================
# I5: Emergency Shutdown
# ============================================================================

class EmergencyShutdown:
    """Emergency shutdown mechanism."""
    def __init__(self):
        self.is_shutdown = False
        self.shutdown_reason = None
        self.shutdown_time = None
        
    def trigger(self, reason: str):
        self.is_shutdown = True
        self.shutdown_reason = reason
        self.shutdown_time = time.time()
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        
    def check_active(self) -> bool:
        return self.is_shutdown
    
    def reset(self, authorization_code: str = None):
        if authorization_code == "RESET_CONFIRMED":
            self.is_shutdown = False
            self.shutdown_reason = None
            logger.info("Emergency shutdown reset")
            return True
        return False


# ============================================================================
# I6: Fail-Safe Defaults
# ============================================================================

class FailSafeDefaults:
    """Default safe actions when components fail."""
    def __init__(self):
        self.component_status = {}
        self.safe_actions = {
            'decision_engine': ('HOLD', 0.0),
            'risk_manager': ('REDUCE', 0.5),
            'data_feed': ('HALT', 0.0),
            'execution': ('CANCEL_ALL', 0.0),
        }
        
    def report_failure(self, component: str, error: str):
        self.component_status[component] = {'status': 'failed', 'error': error}
        logger.error(f"Component failure: {component} - {error}")
        
    def get_safe_action(self, component: str) -> Tuple[str, float]:
        return self.safe_actions.get(component, ('HALT', 0.0))
    
    def is_healthy(self) -> bool:
        return all(s.get('status') != 'failed' for s in self.component_status.values())


# ============================================================================
# Complete Simplex Safety Pipeline
# ============================================================================

@dataclass
class SafetyConfig:
    max_daily_loss: float = 0.05
    max_position_pct: float = 0.20
    z_threshold: float = 3.0

class SimplexSafetySystem:
    """Complete safety system integrating all 6 methods."""
    
    def __init__(self, config=None):
        self.config = config or SafetyConfig()
        self.circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(max_daily_loss=self.config.max_daily_loss)
        )
        self.anomaly = AnomalyDetector(z_threshold=self.config.z_threshold)
        self.position_limits = PositionLimiter(
            PositionLimitsConfig(max_position_pct=self.config.max_position_pct)
        )
        self.order_validator = OrderValidator()
        self.emergency = EmergencyShutdown()
        self.fail_safe = FailSafeDefaults()
        
    def check_trade(self, symbol: str, side: str, quantity: float,
                   price: float, capital: float, 
                   potential_loss: float) -> Tuple[bool, str]:
        """Full safety check before trade execution."""
        
        # Emergency check
        if self.emergency.check_active():
            return False, "Emergency shutdown active"
        
        # Circuit breaker
        allowed, reason = self.circuit_breaker.check_trade(potential_loss)
        if not allowed:
            return False, reason
        
        # Position limits
        value = quantity * price
        allowed, reason = self.position_limits.check_order(symbol, value, capital)
        if not allowed:
            return False, reason
        
        # Order validation
        allowed, reason = self.order_validator.validate(symbol, side, quantity, price)
        if not allowed:
            return False, reason
        
        return True, ""
    
    def check_market(self, price_change: float, latency_ms: float) -> Tuple[bool, str]:
        """Check market conditions."""
        is_anomaly, z_score = self.anomaly.check_price(price_change)
        if is_anomaly:
            return False, f"Price anomaly detected (z={z_score:.2f})"
        
        if self.anomaly.check_latency(latency_ms):
            return False, "Latency anomaly detected"
        
        return True, ""
    
    def record_trade(self, symbol: str, pnl: float, value: float):
        self.circuit_breaker.record_trade(pnl)
        self.position_limits.update_position(symbol, value)
        self.order_validator.record_order(symbol)
    
    def trigger_emergency(self, reason: str):
        self.emergency.trigger(reason)
        
    def get_status(self) -> Dict:
        return {
            'emergency_active': self.emergency.check_active(),
            'daily_pnl': self.circuit_breaker.daily_pnl,
            'consecutive_losses': self.circuit_breaker.consecutive_losses,
            'healthy': self.fail_safe.is_healthy()
        }
