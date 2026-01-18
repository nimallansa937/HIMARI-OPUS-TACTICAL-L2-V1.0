"""
Layer 2 Redis Integration

Consumes Layer 1 outputs and publishes Layer 2 tactical decisions via Redis.

Usage:
    from redis_integration import Layer2RedisIntegration

    # Initialize
    integration = Layer2RedisIntegration(symbols=["BTCUSDT"])

    # Option 1: Poll for latest L1 output
    l1_output = integration.get_l1_output("BTCUSDT")

    # Option 2: Subscribe to real-time L1 outputs
    def on_l1_output(data):
        # Process L1 output
        pass
    integration.subscribe_to_l1(on_l1_output)

    # Publish L2 decision
    integration.publish_decision("BTCUSDT", {
        "direction": 1,
        "direction_label": "LONG",
        "confidence": 0.75,
        "prob_flat": 0.1,
        "prob_long": 0.75,
        "prob_short": 0.15,
    })
"""

import sys
import os
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
import numpy as np

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

try:
    from redis_client import get_redis_client, HimariRedisClient
    from message_types import Layer1Output, Layer2Decision, RegimeState
    from constants import (
        LAYER_2,
        CHANNEL_L1_DECISIONS,
        CHANNEL_L2_DECISIONS,
        CHANNEL_CIRCUIT_BREAKER,
        STATE_L1_OUTPUT,
        STATE_L2_OUTPUT,
        STATE_REGIME,
        STATE_DATA_QUALITY,
        STREAM_DECISIONS,
        TTL_STATE,
        TTL_L2_OUTPUT,
        STREAM_MAXLEN_DECISIONS,
        SYMBOLS,
        REGIME_MULTIPLIERS,
    )
except ImportError as e:
    print(f"Failed to import shared modules: {e}")
    print("Make sure the 'shared' directory exists in the HIMARI OPUS 2 folder")
    raise

logger = logging.getLogger(__name__)


class Layer2RedisIntegration:
    """
    Layer 2 Redis Integration.

    Responsibilities:
    - Consume Layer 1 outputs (60D features + metadata)
    - Consume data quality scores for confidence adjustment
    - Publish Layer 2 tactical decisions
    - Handle circuit breaker messages
    - Apply regime-based adjustments
    """

    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize Layer 2 Redis integration.

        Args:
            symbols: List of symbols to track
        """
        self.redis = get_redis_client(LAYER_2)
        self.symbols = symbols or SYMBOLS

        # Local cache
        self._l1_cache: Dict[str, Dict] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 1.0

        # Callbacks
        self._l1_callbacks: List[Callable] = []

        # Circuit breaker state
        self._circuit_breaker_level = "normal"
        self._circuit_breaker_max_position = 1.0
        self._trading_allowed = True

        # Metrics
        self._l1_received = 0
        self._decisions_published = 0
        self._errors = 0

        # Running state
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None

        logger.info(f"Layer2RedisIntegration initialized for symbols: {self.symbols}")

    # =========================================================================
    # CONSUME LAYER 1 OUTPUT
    # =========================================================================

    def get_l1_output(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest Layer 1 output for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Layer 1 output dictionary or None
        """
        now = time.time()

        # Check cache
        if symbol in self._l1_cache:
            if now - self._cache_time.get(symbol, 0) < self._cache_ttl:
                return self._l1_cache[symbol]

        # Fetch from Redis
        state_key = STATE_L1_OUTPUT.format(symbol=symbol)
        data = self.redis.get_state(state_key)

        if data:
            self._l1_cache[symbol] = data
            self._cache_time[symbol] = now
            return data

        logger.warning(f"No L1 output found for {symbol}")
        return None

    def get_feature_vector(self, symbol: str) -> Optional[np.ndarray]:
        """
        Get 60D feature vector from L1 output.

        Args:
            symbol: Trading symbol

        Returns:
            60D numpy array or None
        """
        l1_output = self.get_l1_output(symbol)
        if l1_output and "feature_vector" in l1_output:
            features = l1_output["feature_vector"]
            if len(features) == 60:
                return np.array(features)
        return None

    def get_regime_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current regime state."""
        state_key = STATE_REGIME.format(symbol=symbol)
        return self.redis.get_state(state_key)

    def get_data_quality(self, symbol: str) -> float:
        """
        Get upstream data quality score.

        Returns quality score from L1 output (which inherits from L0).
        """
        l1_output = self.get_l1_output(symbol)
        if l1_output:
            return l1_output.get("upstream_data_quality", 1.0)
        return 1.0

    def is_in_transition_window(self, symbol: str) -> bool:
        """Check if in regime transition window."""
        l1_output = self.get_l1_output(symbol)
        if l1_output:
            return l1_output.get("is_regime_transition", False)

        regime = self.get_regime_state(symbol)
        if regime:
            return regime.get("in_transition_window", False)

        return False

    def get_transition_type(self, symbol: str) -> Optional[str]:
        """Get current transition type."""
        regime = self.get_regime_state(symbol)
        if regime and regime.get("in_transition_window"):
            return regime.get("transition_type")
        return None

    # =========================================================================
    # SUBSCRIBE TO LAYER 1
    # =========================================================================

    def subscribe_to_l1(self, callback: Callable[[Dict], None]) -> None:
        """
        Subscribe to Layer 1 decision outputs.

        Args:
            callback: Function called when L1 output arrives
        """
        self._l1_callbacks.append(callback)
        self.redis.subscribe(CHANNEL_L1_DECISIONS, self._handle_l1_output)

    def _handle_l1_output(self, data: Dict[str, Any]) -> None:
        """Internal handler for L1 output messages."""
        self._l1_received += 1
        symbol = data.get("symbol", "UNKNOWN")

        # Update cache
        self._l1_cache[symbol] = data
        self._cache_time[symbol] = time.time()

        # Call registered callbacks
        for callback in self._l1_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"L1 callback error: {e}")
                self._errors += 1

    def start_listening(self) -> None:
        """Start listening for subscribed channels (blocking)."""
        self._running = True
        self.redis.start_listening()

    def start_listening_thread(self) -> threading.Thread:
        """Start listening in background thread."""
        self._running = True
        return self.redis.start_listening_thread()

    # =========================================================================
    # PUBLISH LAYER 2 DECISION
    # =========================================================================

    def publish_decision(self, symbol: str, decision_data: Dict[str, Any]) -> bool:
        """
        Publish Layer 2 decision to Redis.

        Args:
            symbol: Trading symbol
            decision_data: Decision data dictionary containing:
                - direction: int (0=FLAT, 1=LONG, 2=SHORT)
                - direction_label: str
                - confidence: float (0-1)
                - prob_flat: float
                - prob_long: float
                - prob_short: float
                - value_estimate: float
                - epistemic_uncertainty: float
                - And other L2 outputs

        Returns:
            True if successful
        """
        try:
            timestamp = int(time.time() * 1000)

            # Get upstream quality
            l1_output = self.get_l1_output(symbol)
            inherited_quality = 1.0
            if l1_output:
                inherited_quality = l1_output.get("upstream_data_quality", 1.0)

            # Get regime
            regime_state = self.get_regime_state(symbol)
            current_regime = "RANGING"
            is_transition = False
            transition_type = None
            transition_hours = 0.0

            if regime_state:
                current_regime = regime_state.get("current_regime", "RANGING")
                is_transition = regime_state.get("in_transition_window", False)
                transition_type = regime_state.get("transition_type")
                transition_hours = regime_state.get("window_hours_elapsed", 0.0)
            elif l1_output:
                current_regime = l1_output.get("current_regime", "RANGING")
                is_transition = l1_output.get("is_regime_transition", False)

            # Apply regime adjustment
            raw_confidence = decision_data.get("confidence", 0.5)
            regime_adjusted_confidence = self._apply_regime_adjustment(
                raw_confidence,
                decision_data.get("direction", 0),
                current_regime
            )

            # Apply quality adjustment
            quality_adjusted_confidence = regime_adjusted_confidence * inherited_quality

            # Build message
            direction = decision_data.get("direction", 0)
            direction_labels = {0: "FLAT", 1: "LONG", 2: "SHORT"}

            message = Layer2Decision(
                timestamp=timestamp,
                source=LAYER_2,
                symbol=symbol,
                direction=direction,
                direction_label=decision_data.get("direction_label", direction_labels.get(direction, "FLAT")),
                confidence=raw_confidence,
                prob_flat=decision_data.get("prob_flat", 0.34),
                prob_long=decision_data.get("prob_long", 0.33),
                prob_short=decision_data.get("prob_short", 0.33),
                value_estimate=decision_data.get("value_estimate", 0.0),
                advantage_estimate=decision_data.get("advantage_estimate", 0.0),
                epistemic_uncertainty=decision_data.get("epistemic_uncertainty", 0.0),
                aleatoric_uncertainty=decision_data.get("aleatoric_uncertainty", 0.0),
                total_uncertainty=decision_data.get("total_uncertainty", 0.0),
                model_confidence_score=decision_data.get("model_confidence_score", 1.0),
                ood_detected=decision_data.get("ood_detected", False),
                ensemble_disagreement=decision_data.get("ensemble_disagreement", 0.0),
                inherited_quality_score=inherited_quality,
                decision_confidence_adjusted=quality_adjusted_confidence,
                current_regime=current_regime,
                regime_confidence_threshold=decision_data.get("regime_confidence_threshold", 0.5),
                regime_adjusted_confidence=regime_adjusted_confidence,
                is_regime_transition=is_transition,
                transition_type=transition_type,
                transition_hours=transition_hours,
                executing_strategy_id=decision_data.get("executing_strategy_id"),
                strategy_regime_sharpe=decision_data.get("strategy_regime_sharpe", 0.0),
                strategy_hifa_stage=decision_data.get("strategy_hifa_stage", 0),
                requires_governance_approval=decision_data.get("requires_governance_approval", False),
                governance_approval_reason=decision_data.get("governance_approval_reason"),
                governance_request_id=decision_data.get("governance_request_id"),
                input_dim_validated=decision_data.get("input_dim_validated", True),
                nan_inf_detected=decision_data.get("nan_inf_detected", False),
                feature_staleness_count=decision_data.get("feature_staleness_count", 0),
                urgency=decision_data.get("urgency", "normal"),
                max_slippage_bps=decision_data.get("max_slippage_bps", 10),
                time_in_force=decision_data.get("time_in_force", "GTC"),
                inference_latency_ms=decision_data.get("inference_latency_ms", 0.0),
                latency_budget_ms=100.0,
                latency_budget_exceeded=decision_data.get("inference_latency_ms", 0.0) > 100.0,
            )

            message_dict = message.to_dict()

            # 1. Publish to channel
            self.redis.publish(CHANNEL_L2_DECISIONS, message_dict)

            # 2. Update state key
            state_key = STATE_L2_OUTPUT.format(symbol=symbol)
            self.redis.set_state(state_key, message_dict, ttl=TTL_L2_OUTPUT)

            # 3. Add to stream
            stream_key = STREAM_DECISIONS.format(symbol=symbol)
            self.redis.stream_add(stream_key, message_dict, maxlen=STREAM_MAXLEN_DECISIONS)

            # Update metrics
            self._decisions_published += 1

            logger.debug(
                f"Published L2 decision for {symbol}: "
                f"{message.direction_label} conf={message.confidence:.3f} "
                f"regime={current_regime}"
            )
            return True

        except Exception as e:
            self._errors += 1
            logger.error(f"Failed to publish L2 decision for {symbol}: {e}")
            return False

    def _apply_regime_adjustment(
        self,
        confidence: float,
        direction: int,
        regime: str
    ) -> float:
        """
        Apply regime-based confidence adjustment.

        Args:
            confidence: Raw model confidence
            direction: 0=FLAT, 1=LONG, 2=SHORT
            regime: Current regime

        Returns:
            Adjusted confidence
        """
        regime_upper = regime.upper()

        # Crisis regime: Force FLAT
        if regime_upper in ("CRISIS", "HIGH_VOLATILITY"):
            if direction != 0:  # Not FLAT
                return confidence * 0.1  # Heavily penalize non-FLAT decisions
            return confidence

        # Bull regime
        if regime_upper in ("TRENDING_UP", "BULL"):
            if direction == 1:  # LONG
                return min(1.0, confidence * 1.2)
            elif direction == 2:  # SHORT
                return confidence * 0.5

        # Bear regime
        if regime_upper in ("TRENDING_DOWN", "BEAR"):
            if direction == 2:  # SHORT
                return min(1.0, confidence * 1.2)
            elif direction == 1:  # LONG
                if confidence < 0.8:
                    return confidence * 0.3  # Force toward FLAT
                return confidence * 0.5

        # Range regime
        if regime_upper in ("RANGING", "RANGE"):
            return confidence * 0.9

        return confidence

    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================

    def setup_circuit_breaker(self) -> None:
        """Subscribe to circuit breaker channel."""
        self.redis.subscribe(CHANNEL_CIRCUIT_BREAKER, self._handle_circuit_breaker)

    def _handle_circuit_breaker(self, message: Dict[str, Any]) -> None:
        """Handle circuit breaker message."""
        level = message.get("level", "warning")
        code = message.get("trigger_code", "UNKNOWN")

        logger.warning(f"[Layer2] Circuit breaker {level}: {code}")

        if level == "emergency":
            self._circuit_breaker_level = "emergency"
            self._circuit_breaker_max_position = 0.0
            self._trading_allowed = False
        elif level == "halt":
            self._circuit_breaker_level = "halt"
            self._circuit_breaker_max_position = 0.0
            self._trading_allowed = False
        elif level == "warning":
            self._circuit_breaker_level = "warning"
            self._circuit_breaker_max_position = message.get("max_position_allowed", 0.25)
            self._trading_allowed = True
        elif level == "normal" or message.get("type") == "clear":
            self._circuit_breaker_level = "normal"
            self._circuit_breaker_max_position = 1.0
            self._trading_allowed = True

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed under current circuit breaker state."""
        return self._trading_allowed

    def get_circuit_breaker_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "level": self._circuit_breaker_level,
            "max_position": self._circuit_breaker_max_position,
            "trading_allowed": self._trading_allowed,
        }

    # =========================================================================
    # HEARTBEAT
    # =========================================================================

    def start_heartbeat(self, interval: int = 5) -> None:
        """Start background heartbeat publishing."""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return

        self._running = True

        def heartbeat_loop():
            while self._running:
                try:
                    metrics = {
                        "l1_received": self._l1_received,
                        "decisions_published": self._decisions_published,
                        "errors": self._errors,
                        "circuit_breaker": self._circuit_breaker_level,
                        "trading_allowed": self._trading_allowed,
                    }
                    self.redis.publish_heartbeat(status="healthy", metrics=metrics)
                except Exception as e:
                    logger.error(f"Heartbeat failed: {e}")
                time.sleep(interval)

        self._heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            daemon=True,
            name="layer2-heartbeat"
        )
        self._heartbeat_thread.start()
        logger.info("Started Layer 2 heartbeat")

    def stop_heartbeat(self) -> None:
        """Stop heartbeat."""
        self._running = False

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get integration metrics."""
        return {
            "l1_received": self._l1_received,
            "decisions_published": self._decisions_published,
            "errors": self._errors,
            "circuit_breaker": self._circuit_breaker_level,
            "trading_allowed": self._trading_allowed,
            "redis_connected": self.redis.is_connected(),
        }

    def close(self) -> None:
        """Close integration and cleanup."""
        self._running = False
        self.redis.stop_listening()
        logger.info("Layer2RedisIntegration closed")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_integration(symbols: Optional[List[str]] = None) -> Layer2RedisIntegration:
    """Create and initialize Layer 2 Redis integration."""
    integration = Layer2RedisIntegration(symbols=symbols)
    integration.setup_circuit_breaker()
    integration.start_heartbeat()
    integration.start_listening_thread()
    return integration


if __name__ == "__main__":
    # Test the integration
    logging.basicConfig(level=logging.INFO)

    print("Testing Layer2RedisIntegration...")

    integration = Layer2RedisIntegration(symbols=["BTCUSDT"])
    integration.setup_circuit_breaker()
    integration.start_heartbeat()

    # Test getting L1 output
    l1_output = integration.get_l1_output("BTCUSDT")
    print(f"L1 output: {l1_output}")

    # Test publishing decision
    test_decision = {
        "direction": 1,
        "direction_label": "LONG",
        "confidence": 0.75,
        "prob_flat": 0.1,
        "prob_long": 0.75,
        "prob_short": 0.15,
        "value_estimate": 0.05,
        "epistemic_uncertainty": 0.1,
    }

    success = integration.publish_decision("BTCUSDT", test_decision)
    print(f"Decision published: {success}")

    print(f"Metrics: {integration.get_metrics()}")

    # Cleanup
    integration.close()
    print("Test complete!")
