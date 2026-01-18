"""
HIMARI OPUS 2 - Layer 2 Main Runner

Integrates Redis communication with the Tactical Layer.

Consumes L1 outputs from Redis, processes through TacticalLayerV2_1_1,
and publishes L2 decisions back to Redis for Layer 3 consumption.

Usage:
    python main.py
    python main.py --poll  # Use polling mode instead of subscription
"""

import sys
import os
import time
import logging
import argparse
import threading
from typing import Dict, Any, Optional

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LAYER 2 TACTICAL HIMARI OPUS'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

from redis_integration import Layer2RedisIntegration

# Import tactical layer
from himari_layer2.tactical_layer import TacticalLayerV2_1_1, evaluate_tactical
from himari_layer2.core.contracts import SignalInput, RiskContext, MultimodalInput
from himari_layer2.core.types import RegimeLabel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('himari.layer2.main')


class Layer2Processor:
    """
    Layer 2 Processor - Integrates Redis with Tactical Layer.
    """

    def __init__(self, symbols: list = None):
        """Initialize Layer 2 processor."""
        self.symbols = symbols or ["BTCUSDT"]

        # Initialize Redis integration
        self.redis = Layer2RedisIntegration(symbols=self.symbols)
        self.redis.setup_circuit_breaker()
        self.redis.start_heartbeat()

        # Initialize tactical layer
        self.tactical = TacticalLayerV2_1_1()

        # State tracking
        self._running = False
        self._process_count = 0
        self._errors = 0

        logger.info(f"Layer2Processor initialized for symbols: {self.symbols}")

    def process_l1_output(self, l1_data: Dict[str, Any]) -> None:
        """
        Process L1 output and publish L2 decision.

        Called either by subscription callback or polling loop.

        Args:
            l1_data: Layer 1 output data from Redis
        """
        try:
            symbol = l1_data.get("symbol", "BTCUSDT")

            # Check circuit breaker
            if not self.redis.is_trading_allowed():
                logger.warning(f"Trading not allowed - circuit breaker active")
                return

            # Build signal input from L1 features
            signals = self._build_signal_input(l1_data)
            risk_context = self._build_risk_context(l1_data)
            multimodal = self._build_multimodal_input(l1_data)

            # Evaluate through tactical layer
            start_time = time.perf_counter()
            decision = self.tactical.evaluate(signals, risk_context, multimodal)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Build L2 decision data
            l2_decision = {
                "direction": self._action_to_direction(decision.action.name),
                "direction_label": self._action_to_label(decision.action.name),
                "confidence": decision.confidence,
                "prob_flat": 1.0 - decision.confidence if decision.action.name == "HOLD" else 0.1,
                "prob_long": decision.confidence if decision.action.name in ("BUY", "STRONG_BUY") else 0.1,
                "prob_short": decision.confidence if decision.action.name in ("SELL", "STRONG_SELL") else 0.1,
                "value_estimate": decision.composite_score if decision.composite_score else 0.0,
                "epistemic_uncertainty": 1.0 - decision.confidence,
                "inference_latency_ms": latency_ms,
                "governance_tier": decision.tier.name,
                "tactical_reason": decision.reason,
            }

            # Publish to Redis
            success = self.redis.publish_decision(symbol, l2_decision)

            if success:
                self._process_count += 1
                logger.debug(
                    f"L2 decision for {symbol}: {decision.action.name} "
                    f"conf={decision.confidence:.3f} tier={decision.tier.name} "
                    f"latency={latency_ms:.2f}ms"
                )
            else:
                self._errors += 1

        except Exception as e:
            self._errors += 1
            logger.error(f"Error processing L1 output: {e}", exc_info=True)

    def _build_signal_input(self, l1_data: Dict[str, Any]) -> SignalInput:
        """Build SignalInput from L1 data."""
        # Extract features from L1 output
        # Map from 60D vector or individual fields

        momentum = l1_data.get("directional_bias", 0.0)
        volatility = l1_data.get("volatility_percentile", 50.0) / 100.0

        # Estimate reversion from regime
        regime = l1_data.get("current_regime", "RANGING")
        if regime in ("RANGING", "RANGE"):
            reversion = 0.3
        else:
            reversion = 0.0

        return SignalInput(
            momentum_ema=momentum,
            reversion_bb=reversion,
            volatility=volatility,
            flow_volume=0.0,  # Not available in L1 output
        )

    def _build_risk_context(self, l1_data: Dict[str, Any]) -> RiskContext:
        """Build RiskContext from L1 data and Redis state."""
        # Get regime from L1 output
        regime_str = l1_data.get("current_regime", "RANGING")
        regime = RegimeLabel.from_string(regime_str)

        # Get cascade risk from circuit breaker state
        cb_state = self.redis.get_circuit_breaker_state()
        cascade_risk = 0.0
        if cb_state["level"] == "warning":
            cascade_risk = 0.3
        elif cb_state["level"] == "halt":
            cascade_risk = 0.8
        elif cb_state["level"] == "emergency":
            cascade_risk = 1.0

        return RiskContext(
            regime_label=regime,
            regime_confidence=l1_data.get("regime_confidence", 0.5),
            cascade_risk=cascade_risk,
            daily_pnl=0.0,  # Would come from Layer 3/4
            daily_dd=0.0,
            exchange_health=True,
            onchain_whale_pressure=0.0,
            cascade_flag=l1_data.get("causal_event_active", False),
            kill_switch=False,
        )

    def _build_multimodal_input(self, l1_data: Dict[str, Any]) -> MultimodalInput:
        """Build MultimodalInput from L1 data."""
        return MultimodalInput(
            sentiment_event_active=l1_data.get("causal_event_active", False),
            sentiment_shock_magnitude=0.0,
            sentiment_trend=0.0,
            event_magnitude=0.0,
        )

    def _action_to_direction(self, action: str) -> int:
        """Convert action name to direction code."""
        if action in ("BUY", "STRONG_BUY"):
            return 1  # LONG
        elif action in ("SELL", "STRONG_SELL"):
            return 2  # SHORT
        else:
            return 0  # FLAT

    def _action_to_label(self, action: str) -> str:
        """Convert action name to direction label."""
        if action in ("BUY", "STRONG_BUY"):
            return "LONG"
        elif action in ("SELL", "STRONG_SELL"):
            return "SHORT"
        else:
            return "FLAT"

    def run_polling(self, interval_seconds: float = 1.0) -> None:
        """
        Run in polling mode.

        Polls Redis for latest L1 output at specified interval.

        Args:
            interval_seconds: Polling interval
        """
        self._running = True
        logger.info(f"Starting Layer 2 polling loop (interval={interval_seconds}s)")

        while self._running:
            try:
                for symbol in self.symbols:
                    l1_output = self.redis.get_l1_output(symbol)
                    if l1_output:
                        self.process_l1_output(l1_output)

                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Polling error: {e}")
                time.sleep(1)

    def run_subscription(self) -> None:
        """
        Run in subscription mode.

        Subscribes to L1 channel and processes in real-time.
        """
        self._running = True
        logger.info("Starting Layer 2 subscription mode")

        # Subscribe to L1 outputs
        self.redis.subscribe_to_l1(self.process_l1_output)

        # Start listening thread
        listener_thread = self.redis.start_listening_thread()

        logger.info("Layer 2 listening for L1 outputs...")

        try:
            # Keep main thread alive
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the processor."""
        self._running = False
        self.redis.close()
        logger.info(f"Layer 2 stopped. Processed: {self._process_count}, Errors: {self._errors}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics."""
        return {
            "process_count": self._process_count,
            "errors": self._errors,
            "tactical_metrics": self.tactical.get_metrics(),
            "redis_metrics": self.redis.get_metrics(),
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='HIMARI OPUS 2 - Layer 2 Tactical Processor'
    )

    parser.add_argument(
        '--poll',
        action='store_true',
        help='Use polling mode instead of subscription'
    )

    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Polling interval in seconds (default: 1.0)'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        default='BTCUSDT',
        help='Comma-separated list of symbols (default: BTCUSDT)'
    )

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]

    processor = Layer2Processor(symbols=symbols)

    try:
        if args.poll:
            processor.run_polling(interval_seconds=args.interval)
        else:
            processor.run_subscription()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        processor.stop()
        metrics = processor.get_metrics()
        logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
