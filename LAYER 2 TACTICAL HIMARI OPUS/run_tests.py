#!/usr/bin/env python
"""
Quick Test Script for HIMARI OPUS 2 Layer 2
Run from the himari_layer2 directory.
"""

import sys
import os

# Add parent directory to path for local testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.types import TradeAction, Tier, RegimeLabel
from core.contracts import SignalInput, RiskContext, MultimodalInput
from tactical_layer import TacticalLayerV2_1_1

def run_tests():
    print("=" * 60)
    print("HIMARI OPUS 2 Layer 2 v2.1.1 - Quick Test")
    print("=" * 60)
    
    layer = TacticalLayerV2_1_1()
    
    # Test 1: Strong bullish signals
    print("\n[Test 1] Strong Bullish Signals")
    signals = SignalInput(
        momentum_ema=0.85,
        reversion_bb=0.70,
        volatility=0.50,
        flow_volume=0.80,
    )
    risk = RiskContext(
        regime_label=RegimeLabel.TRENDING_UP,
        regime_confidence=0.85,
        cascade_risk=0.10,
        daily_pnl=100.0,
        daily_dd=-0.03,
        exchange_health=True,
    )
    multi = MultimodalInput(
        sentiment_event_active=False,
        sentiment_shock_magnitude=0.0,
        sentiment_trend=0.1,
    )
    
    decision = layer.evaluate(signals, risk, multi)
    print(f"  Composite Score: {decision.composite_score:.3f}")
    print(f"  Action: {decision.action.name}")
    print(f"  Confidence: {decision.confidence:.3f}")
    print(f"  Tier: {decision.tier.name}")
    print(f"  Latency: {decision.latency_ms:.3f} ms")
    
    assert decision.action in (TradeAction.BUY, TradeAction.STRONG_BUY), f"Expected BUY/STRONG_BUY, got {decision.action}"
    print("  ✓ PASSED")
    
    # Test 2: Crisis regime dampening
    print("\n[Test 2] Crisis Regime Dampening")
    risk_crisis = RiskContext(
        regime_label=RegimeLabel.CRISIS,
        regime_confidence=0.90,
        cascade_risk=0.65,
        daily_pnl=-500.0,
        daily_dd=-0.10,
        exchange_health=True,
    )
    
    decision = layer.evaluate(signals, risk_crisis, multi)
    print(f"  Action: {decision.action.name}")
    print(f"  Confidence: {decision.confidence:.3f} (should be dampened)")
    print(f"  Tier: {decision.tier.name} (should be escalated)")
    print(f"  Reason: {decision.reason}")
    
    # During crisis, confidence should be significantly dampened
    assert decision.confidence < 0.5, "Crisis should dampen confidence"
    print("  ✓ PASSED")
    
    # Test 3: Emergency stop
    print("\n[Test 3] Emergency Stop (DD Exceeded)")
    risk_dd = RiskContext(
        regime_label=RegimeLabel.TRENDING_UP,
        regime_confidence=0.85,
        cascade_risk=0.10,
        daily_pnl=-2000.0,
        daily_dd=-0.20,  # Exceeds -15% threshold
        exchange_health=True,
    )
    
    decision = layer.evaluate(signals, risk_dd, multi)
    print(f"  Action: {decision.action.name}")
    print(f"  Confidence: {decision.confidence:.3f}")
    print(f"  Tier: {decision.tier.name}")
    print(f"  Reason: {decision.reason}")
    
    assert decision.action == TradeAction.HOLD, "Emergency stop should force HOLD"
    assert decision.tier == Tier.T4, "Emergency stop should be T4"
    print("  ✓ PASSED")
    
    # Test 4: v2.1.1 sentiment boost check
    print("\n[Test 4] v2.1.1 Sentiment Boost (0.9 dampen)")
    from layers.regime_sentiment import RegimeSentimentGate
    gate = RegimeSentimentGate()
    
    bearish_multi = MultimodalInput(
        sentiment_event_active=True,
        sentiment_shock_magnitude=0.8,
        sentiment_trend=-0.7,
    )
    
    boost = gate.compute_sentiment_boost(bearish_multi, TradeAction.BUY)
    print(f"  Conflicting trade boost: {boost}")
    assert boost == 0.9, f"v2.1.1 requires 0.9 dampen, got {boost}"
    print("  ✓ PASSED")
    
    # Test 5: Performance metrics
    print("\n[Test 5] Performance Metrics")
    metrics = layer.get_metrics()
    print(f"  Total evaluations: {metrics['eval_count']}")
    print(f"  Avg latency: {metrics['avg_latency_ms']:.3f} ms")
    print(f"  Tier distribution: {metrics['tier_distribution']}")
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! HIMARI Layer 2 v2.1.1 is working correctly.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
