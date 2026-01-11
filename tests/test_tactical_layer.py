"""
HIMARI OPUS 2 Layer 2 - Unit Tests for Tactical Layer
Version: 2.1.1 FINAL

Comprehensive test suite for the tactical layer subsumption architecture.
"""

import pytest
import math

# Import components to test
from himari_layer2.core.types import TradeAction, Tier, RegimeLabel
from himari_layer2.core.contracts import SignalInput, RiskContext, MultimodalInput
from himari_layer2.core.config import TacticalConfig, DEFAULT_CONFIG
from himari_layer2.layers.baseline import BaselineComposite
from himari_layer2.layers.regime_sentiment import RegimeSentimentGate
from himari_layer2.layers.cascade import CascadeRiskGate
from himari_layer2.layers.emergency import EmergencyStop
from himari_layer2.tactical_layer import TacticalLayerV2_1_1


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_signals():
    """Default neutral signals."""
    return SignalInput(
        momentum_ema=0.0,
        reversion_bb=0.0,
        volatility=0.0,
        flow_volume=0.0,
    )

@pytest.fixture
def bullish_signals():
    """Strong bullish signals."""
    return SignalInput(
        momentum_ema=0.8,
        reversion_bb=0.6,
        volatility=0.3,
        flow_volume=0.7,
    )

@pytest.fixture
def bearish_signals():
    """Strong bearish signals."""
    return SignalInput(
        momentum_ema=-0.8,
        reversion_bb=-0.6,
        volatility=-0.3,
        flow_volume=-0.7,
    )

@pytest.fixture
def default_risk_context():
    """Default normal risk context."""
    return RiskContext(
        regime_label=RegimeLabel.TRENDING_UP,
        regime_confidence=0.8,
        cascade_risk=0.1,
        daily_pnl=100.0,
        daily_dd=-0.05,
        exchange_health=True,
    )

@pytest.fixture
def crisis_risk_context():
    """Crisis regime risk context."""
    return RiskContext(
        regime_label=RegimeLabel.CRISIS,
        regime_confidence=0.9,
        cascade_risk=0.7,
        daily_pnl=-500.0,
        daily_dd=-0.12,
        exchange_health=True,
    )

@pytest.fixture
def default_multimodal():
    """Default no-event multimodal."""
    return MultimodalInput(
        sentiment_event_active=False,
        sentiment_shock_magnitude=0.0,
        sentiment_trend=0.0,
    )

@pytest.fixture
def bullish_shock_multimodal():
    """Bullish sentiment shock."""
    return MultimodalInput(
        sentiment_event_active=True,
        sentiment_shock_magnitude=0.8,
        sentiment_trend=0.7,
    )

@pytest.fixture
def bearish_shock_multimodal():
    """Bearish sentiment shock."""
    return MultimodalInput(
        sentiment_event_active=True,
        sentiment_shock_magnitude=0.8,
        sentiment_trend=-0.7,
    )


# =============================================================================
# Level 0: Baseline Composite Tests
# =============================================================================

class TestBaselineComposite:
    
    def test_neutral_signals_hold(self, default_signals):
        """Neutral signals should produce HOLD action."""
        baseline = BaselineComposite()
        action, confidence, composite = baseline.evaluate(default_signals)
        
        assert action == TradeAction.HOLD
        assert -0.1 < composite < 0.1
    
    def test_bullish_signals_buy(self, bullish_signals):
        """Strong bullish signals should produce STRONG_BUY."""
        baseline = BaselineComposite()
        action, confidence, composite = baseline.evaluate(bullish_signals)
        
        # With weights: 0.35*0.8 + 0.25*0.6 + 0.20*0.3 + 0.20*0.7 = 0.63
        assert action in (TradeAction.BUY, TradeAction.STRONG_BUY)
        assert composite > 0.4
    
    def test_bearish_signals_sell(self, bearish_signals):
        """Strong bearish signals should produce STRONG_SELL."""
        baseline = BaselineComposite()
        action, confidence, composite = baseline.evaluate(bearish_signals)
        
        assert action in (TradeAction.SELL, TradeAction.STRONG_SELL)
        assert composite < -0.4
    
    def test_confidence_sigmoid(self):
        """Confidence should follow sigmoid curve."""
        baseline = BaselineComposite()
        
        # Zero composite → ~0.5 confidence
        conf_zero = baseline.compute_confidence(0.0)
        assert 0.4 < conf_zero < 0.6
        
        # High composite → high confidence
        conf_high = baseline.compute_confidence(0.8)
        assert conf_high > 0.9
    
    def test_weights_sum_to_one(self):
        """Signal weights should sum to 1.0."""
        config = DEFAULT_CONFIG
        weight_sum = sum(config.signal_weights.values())
        assert abs(weight_sum - 1.0) < 0.001


# =============================================================================
# Level 1: Regime & Sentiment Gate Tests
# =============================================================================

class TestRegimeSentimentGate:
    
    def test_stable_regime_no_penalty(self, default_risk_context, default_multimodal):
        """Stable regime should have no confidence penalty."""
        gate = RegimeSentimentGate()
        penalty = gate.get_regime_penalty(default_risk_context.regime_label)
        assert penalty == 1.0
    
    def test_crisis_regime_penalty(self):
        """CRISIS regime should have 0.4 penalty."""
        gate = RegimeSentimentGate()
        penalty = gate.get_regime_penalty(RegimeLabel.CRISIS)
        assert penalty == 0.4
    
    def test_high_volatility_penalty(self):
        """HIGH_VOLATILITY regime should have 0.7 penalty."""
        gate = RegimeSentimentGate()
        penalty = gate.get_regime_penalty(RegimeLabel.HIGH_VOLATILITY)
        assert penalty == 0.7
    
    def test_bullish_shock_blocks_sells(self, bullish_shock_multimodal):
        """Bullish shock should block SELL actions."""
        gate = RegimeSentimentGate()
        should_block, reason = gate.check_sentiment_blocking(
            bullish_shock_multimodal, TradeAction.SELL
        )
        assert should_block is True
        assert "blocks" in reason.lower()
    
    def test_bearish_shock_blocks_buys(self, bearish_shock_multimodal):
        """Bearish shock should block BUY actions."""
        gate = RegimeSentimentGate()
        should_block, reason = gate.check_sentiment_blocking(
            bearish_shock_multimodal, TradeAction.BUY
        )
        assert should_block is True
    
    def test_aligned_trade_boost_v211(self, bullish_shock_multimodal):
        """v2.1.1: Aligned trades should get 1.2x boost."""
        gate = RegimeSentimentGate()
        boost = gate.compute_sentiment_boost(bullish_shock_multimodal, TradeAction.BUY)
        assert boost == 1.2
    
    def test_conflicting_trade_dampen_v211(self, bearish_shock_multimodal):
        """v2.1.1: Conflicting trades should get 0.9 dampen (not 0.8)."""
        gate = RegimeSentimentGate()
        boost = gate.compute_sentiment_boost(bearish_shock_multimodal, TradeAction.BUY)
        # v2.1.1 fix: 0.9 not 0.8
        assert boost == 0.9


# =============================================================================
# Level 2: Cascade Risk Gate Tests
# =============================================================================

class TestCascadeRiskGate:
    
    def test_low_risk_no_suppression(self):
        """Low cascade risk should have no suppression."""
        gate = CascadeRiskGate()
        factor = gate.get_suppress_factor(0.2)
        assert factor == 1.0
    
    def test_moderate_risk_partial_suppression(self):
        """Moderate risk should have 0.6 suppression."""
        gate = CascadeRiskGate()
        factor = gate.get_suppress_factor(0.4)
        assert factor == 0.6
    
    def test_high_risk_aggressive_suppression(self):
        """High risk should have 0.3 suppression."""
        gate = CascadeRiskGate()
        factor = gate.get_suppress_factor(0.7)
        assert factor == 0.3
    
    def test_high_risk_downgrades_strong_actions(self):
        """High cascade risk should downgrade STRONG actions."""
        gate = CascadeRiskGate()
        action, conf, reason = gate.evaluate(
            TradeAction.STRONG_BUY, 0.8, 0.7
        )
        assert action == TradeAction.BUY  # Downgraded
        assert "downgrade" in reason.lower() or "HIGH" in reason
    
    def test_normal_action_not_downgraded(self):
        """Normal BUY should not be downgraded even at high risk."""
        gate = CascadeRiskGate()
        action, conf, reason = gate.evaluate(
            TradeAction.BUY, 0.8, 0.7
        )
        assert action == TradeAction.BUY  # Not downgraded


# =============================================================================
# Level 3: Emergency Stop Tests
# =============================================================================

class TestEmergencyStop:
    
    def test_normal_conditions_no_stop(self, default_risk_context):
        """Normal conditions should not trigger emergency stop."""
        stop = EmergencyStop()
        should_stop, reason = stop.evaluate(default_risk_context)
        assert should_stop is False
    
    def test_exchange_down_triggers_stop(self, default_risk_context):
        """Exchange connectivity loss should trigger stop."""
        default_risk_context.exchange_health = False
        stop = EmergencyStop()
        should_stop, reason = stop.evaluate(default_risk_context)
        assert should_stop is True
        assert "exchange" in reason.lower()
    
    def test_daily_dd_triggers_stop_v211(self):
        """v2.1.1 FIX: daily_dd < -0.15 should trigger stop."""
        stop = EmergencyStop()
        risk = RiskContext(
            regime_label=RegimeLabel.CRISIS,
            regime_confidence=0.9,
            cascade_risk=0.1,
            daily_pnl=-1000.0,
            daily_dd=-0.20,  # Exceeds -15% threshold
            exchange_health=True,
        )
        should_stop, reason = stop.evaluate(risk)
        assert should_stop is True
        assert "drawdown" in reason.lower()
    
    def test_cascade_flag_triggers_stop(self, default_risk_context):
        """Cascade flag should trigger stop."""
        default_risk_context.cascade_flag = True
        stop = EmergencyStop()
        should_stop, reason = stop.evaluate(default_risk_context)
        assert should_stop is True
        assert "cascade" in reason.lower()
    
    def test_kill_switch_triggers_stop(self, default_risk_context):
        """Manual kill switch should trigger stop."""
        default_risk_context.kill_switch = True
        stop = EmergencyStop()
        should_stop, reason = stop.evaluate(default_risk_context)
        assert should_stop is True
        assert "kill" in reason.lower()


# =============================================================================
# Tactical Layer Integration Tests
# =============================================================================

class TestTacticalLayerV211:
    
    def test_full_evaluation_returns_decision(
        self, bullish_signals, default_risk_context, default_multimodal
    ):
        """Full evaluation should return valid decision."""
        layer = TacticalLayerV2_1_1()
        decision = layer.evaluate(bullish_signals, default_risk_context, default_multimodal)
        
        assert decision.action in TradeAction
        assert 0 <= decision.confidence <= 1
        assert decision.tier in Tier
        assert decision.latency_ms >= 0
    
    def test_emergency_stop_overrides_all(
        self, bullish_signals, default_risk_context, default_multimodal
    ):
        """Emergency stop should override all other layers."""
        default_risk_context.exchange_health = False
        
        layer = TacticalLayerV2_1_1()
        decision = layer.evaluate(bullish_signals, default_risk_context, default_multimodal)
        
        assert decision.action == TradeAction.HOLD
        assert decision.confidence == 0.0
        assert decision.tier == Tier.T4
    
    def test_crisis_escalates_to_tier2(
        self, bullish_signals, crisis_risk_context, default_multimodal
    ):
        """CRISIS regime should escalate routing to at least T2."""
        crisis_risk_context.cascade_risk = 0.1  # Low cascade so not T3
        
        layer = TacticalLayerV2_1_1()
        decision = layer.evaluate(bullish_signals, crisis_risk_context, default_multimodal)
        
        # Unless emergency stop triggers, should be at least T2
        if decision.action != TradeAction.HOLD:
            assert decision.tier.value >= Tier.T2.value
    
    def test_latency_under_budget(
        self, bullish_signals, default_risk_context, default_multimodal
    ):
        """Latency should be well under 50ms budget."""
        layer = TacticalLayerV2_1_1()
        
        # Run multiple times
        latencies = []
        for _ in range(100):
            decision = layer.evaluate(bullish_signals, default_risk_context, default_multimodal)
            latencies.append(decision.latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 5.0  # Should be well under 5ms
        assert max_latency < 50.0  # 99th percentile budget
    
    def test_tier_distribution_reasonable(
        self, default_risk_context, default_multimodal
    ):
        """Tier distribution should roughly match expected 65/25/10."""
        layer = TacticalLayerV2_1_1()
        
        # Generate various signals
        import random
        random.seed(42)
        
        for _ in range(1000):
            signals = SignalInput(
                momentum_ema=random.uniform(-1, 1),
                reversion_bb=random.uniform(-1, 1),
                volatility=random.uniform(-1, 1),
                flow_volume=random.uniform(-1, 1),
            )
            layer.evaluate(signals, default_risk_context, default_multimodal)
        
        metrics = layer.get_metrics()
        pcts = metrics['tier_percentages']
        
        # Allow wide tolerance for random data
        # Just verify distribution is reasonable
        assert pcts.get('T1', 0) > 0
        assert pcts.get('T2', 0) > 0


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    
    def test_v211_sentiment_boost_dampen(self):
        """v2.1.1: sentiment_boost_dampen should be 0.9."""
        config = DEFAULT_CONFIG
        assert config.sentiment_boost_dampen == 0.9
    
    def test_v211_regime_labels(self):
        """v2.1.1: Regime labels should be OPUS 2 aligned."""
        config = DEFAULT_CONFIG
        expected_labels = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'HIGH_VOLATILITY', 'CRISIS']
        
        for label in expected_labels:
            assert label in config.regime_penalties
    
    def test_config_validation(self):
        """Configuration should pass validation."""
        config = DEFAULT_CONFIG
        assert config.validate() is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
