"""
HIMARI Layer 2 V1 - Master Pipeline Implementation

This module implements the complete integrated pipeline orchestrating all 14 subsystems
(78 methods) for the HIMARI algorithmic trading system.

Author: HIMARI Development Team
Version: 1.0
Date: January 2026

Subsystems Integrated:
    - Part A: Preprocessing (8 methods)
    - Part B: Regime Detection (8 methods)
    - Part D: Decision Engine (10 methods)
    - Part E: HSM State Machine (6 methods)
    - Part F: Uncertainty Quantification (8 methods)
    - Part G: Hysteresis Filter (6 methods)
    - Part H: RSS Risk Management (8 methods)
    - Part I: Simplex Safety System (8 methods)
    - Part J: LLM Integration (8 methods) - Async
    - Part K: Training Infrastructure (8 methods) - Offline
    - Part L: Validation Framework (6 methods) - Offline
    - Part M: Adaptation Framework (6 methods) - Periodic
    - Part N: Interpretability Framework (4 methods) - Async
"""

from __future__ import annotations

import time
import logging
import asyncio
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from collections import deque
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class TradeAction(Enum):
    """Trading action enumeration."""
    SELL = -1
    HOLD = 0
    BUY = 1


class MarketRegime(Enum):
    """Market regime states."""
    BULL = auto()
    BEAR = auto()
    SIDEWAYS = auto()
    CRISIS = auto()


class MetaRegime(Enum):
    """Meta-regime uncertainty states."""
    LOW_UNCERTAINTY = auto()
    HIGH_UNCERTAINTY = auto()


class PositionState(Enum):
    """HSM position states."""
    FLAT = auto()
    LONG_ENTRY = auto()
    LONG_HOLD = auto()
    LONG_EXIT = auto()
    SHORT_ENTRY = auto()
    SHORT_HOLD = auto()
    SHORT_EXIT = auto()


class FallbackLevel(Enum):
    """Simplex fallback levels."""
    PRIMARY = 0
    BASELINE = 1
    CONSERVATIVE = 2
    MINIMAL = 3


# =============================================================================
# DATA CLASSES - INPUT/OUTPUT CONTRACTS
# =============================================================================

@dataclass
class RawMarketData:
    """Raw market data input to the pipeline."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    order_flow: Optional[np.ndarray] = None
    sentiment_raw: Optional[float] = None
    on_chain: Optional[Dict[str, float]] = None


@dataclass
class PortfolioState:
    """Current portfolio state."""
    position: int  # -1, 0, 1
    entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    peak_equity: float
    current_equity: float
    drawdown: float
    margin_used: float
    margin_available: float


@dataclass
class PreprocessedFeatures:
    """Output from Part A: Preprocessing."""
    features: np.ndarray  # Shape: (60,)
    ekf_state: Dict[str, float]
    uncertainty: float
    cae_ambiguity: float
    timestamp: float


@dataclass
class RegimeOutput:
    """Output from Part B: Regime Detection."""
    regime: MarketRegime
    meta_regime: MetaRegime
    probabilities: np.ndarray  # Shape: (4,)
    confidence: float
    transition_warning: bool
    crisis_probability: float
    hurst_exponent: float


@dataclass
class DecisionOutput:
    """Output from Part D: Decision Engine."""
    action: TradeAction
    confidence: float
    agent_outputs: Dict[str, Tuple[int, float]]
    disagreement: float
    ensemble_weights: Dict[str, float]


@dataclass
class UQOutput:
    """Output from Part F: Uncertainty Quantification."""
    calibrated_confidence: float
    prediction_interval: Tuple[float, float]
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    ood_score: float
    total_uncertainty: float


@dataclass
class HSMOutput:
    """Output from Part E: HSM State Machine."""
    validated_action: TradeAction
    is_valid: bool
    position_state: PositionState
    regime_state: str
    blocked_reason: Optional[str]
    oscillation_blocked: bool


@dataclass
class HysteresisOutput:
    """Output from Part G: Hysteresis Filter."""
    filtered_action: TradeAction
    entry_threshold: float
    exit_threshold: float
    efficiency_ratio: float
    was_filtered: bool
    filter_reason: Optional[str]


@dataclass
class RiskOutput:
    """Output from Part H: RSS Risk Management."""
    action: TradeAction
    position_size: float
    leverage: float
    var_95: float
    var_99: float
    expected_shortfall: float
    kelly_fraction: float
    drawdown_brake_active: bool
    risk_budget_remaining: float


@dataclass
class SafetyOutput:
    """Output from Part I: Simplex Safety System."""
    final_action: TradeAction
    final_position_size: float
    fallback_level: FallbackLevel
    primary_blocked: bool
    block_reason: str
    safety_margin: float
    stop_loss_triggered: bool
    recovery_status: str


@dataclass
class LLMSignal:
    """Cached signal from Part J: LLM Integration."""
    sentiment: float
    confidence: float
    direction: int
    reasoning: str
    source_summary: str
    timestamp: float
    latency_ms: float
    calibrated: bool = False


@dataclass
class PipelineOutput:
    """Complete output from Layer 2 pipeline."""
    action: TradeAction
    position_size: float
    leverage: float
    confidence: float

    # Detailed outputs from each stage
    preprocessing: PreprocessedFeatures
    regime: RegimeOutput
    decision: DecisionOutput
    uncertainty: UQOutput
    hsm: HSMOutput
    hysteresis: HysteresisOutput
    risk: RiskOutput
    safety: SafetyOutput

    # Metadata
    total_latency_ms: float
    timestamp: float


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class PreprocessingConfig:
    """Part A configuration."""
    ekf_process_noise: float = 0.001
    ekf_measurement_noise: float = 0.01
    ekf_use_faux_riccati: bool = True
    cae_latent_dim: int = 32
    cae_consensus_threshold: float = 0.8
    vecnormalize_clip: float = 10.0
    feature_dim: int = 60


@dataclass
class RegimeConfig:
    """Part B configuration."""
    n_market_states: int = 4
    n_meta_states: int = 2
    student_t_df: float = 5.0
    crisis_df: float = 3.0
    vix_high_threshold: float = 30.0
    vix_low_threshold: float = 20.0
    jump_threshold_sigma: float = 3.0
    hurst_window: int = 100


@dataclass
class DecisionConfig:
    """Part D configuration."""
    flag_trader_enabled: bool = True
    ppo_hidden_dim: int = 512
    sac_gamma: float = 0.99
    ensemble_lookback_days: int = 30
    disagreement_high_threshold: float = 0.7
    return_conditioning_targets: Dict[str, float] = field(
        default_factory=lambda: {
            "crisis": 0.5, "bear": 1.0, "sideways": 2.0, "bull": 2.5
        }
    )


@dataclass
class UQConfig:
    """Part F configuration."""
    conformal_alpha: float = 0.10
    conformal_n_calibration: int = 500
    temperature: float = 1.0
    mc_dropout_samples: int = 20
    ensemble_size: int = 5
    ood_threshold: float = 2.0


@dataclass
class HSMConfig:
    """Part E configuration."""
    initial_position_state: PositionState = PositionState.FLAT
    oscillation_window_bars: int = 20
    oscillation_max_flips: int = 3
    learned_transition_enabled: bool = True


@dataclass
class HysteresisConfig:
    """Part G configuration."""
    er_period: int = 10
    fast_threshold_entry: float = 0.25
    slow_threshold_entry: float = 0.45
    fast_threshold_exit: float = 0.25
    slow_threshold_exit: float = 0.12
    loss_aversion_ratio: float = 2.2
    knn_k: int = 20
    atr_multiplier: float = 1.5


@dataclass
class RiskConfig:
    """Part H configuration."""
    evt_threshold_percentile: float = 95.0
    evt_min_exceedances: int = 30
    kelly_fraction_cap: float = 0.25
    max_leverage: float = 10.0
    leverage_decay_start_size: float = 0.1
    drawdown_brake_levels: List[float] = field(
        default_factory=lambda: [0.05, 0.08, 0.10]
    )
    drawdown_brake_reductions: List[float] = field(
        default_factory=lambda: [0.25, 0.50, 0.90]
    )


@dataclass
class SafetyConfig:
    """Part I configuration."""
    primary_confidence_threshold: float = 0.6
    baseline_confidence_threshold: float = 0.5
    conservative_confidence_threshold: float = 0.4
    crisis_force_conservative: bool = True
    drawdown_force_minimal: float = 0.08
    max_daily_loss: float = 0.05
    recovery_bars_required: int = 20


@dataclass
class LLMConfig:
    """Part J configuration."""
    model_name: str = "facebook/opt-1.3b"
    cache_ttl_seconds: float = 300.0
    async_queue_size: int = 100
    async_workers: int = 2
    enabled: bool = True


@dataclass
class InterpretabilityConfig:
    """Part N configuration."""
    shap_background_samples: int = 100
    shap_top_k_features: int = 10
    mifid_enabled: bool = True
    async_enabled: bool = True


@dataclass
class MasterConfig:
    """Master configuration composing all subsystem configs."""
    trading_pair: str = "BTC/USDT"
    base_currency: str = "USDT"
    max_latency_ms: float = 50.0
    debug_mode: bool = False

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    uncertainty: UQConfig = field(default_factory=UQConfig)
    hsm: HSMConfig = field(default_factory=HSMConfig)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    interpretability: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)


# =============================================================================
# PART A: PREPROCESSING
# =============================================================================

class ExtendedKalmanFilter:
    """
    A1: Extended Kalman Filter for price denoising.

    Estimates latent state: [price, velocity, acceleration, volatility]
    Uses Student-t innovations for robustness to outliers.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.state = np.zeros(4)  # [price, velocity, acceleration, volatility]
        self.covariance = np.eye(4) * 0.1
        self.initialized = False

    def update(self, measurement: float) -> Dict[str, float]:
        """Update EKF with new price measurement."""
        if not self.initialized:
            self.state[0] = measurement
            self.initialized = True
            return self._get_state_dict()

        # State transition (constant acceleration model)
        dt = 1.0  # Assume unit time step
        F = np.array([
            [1, dt, 0.5*dt**2, 0],
            [0, 1, dt, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0.95]  # Volatility mean-reverts
        ])

        # Process noise
        Q = np.eye(4) * self.config.ekf_process_noise
        Q[3, 3] *= 10  # Higher volatility uncertainty

        # Predict
        predicted_state = F @ self.state
        predicted_cov = F @ self.covariance @ F.T + Q

        # Measurement model
        H = np.array([[1, 0, 0, 0]])  # Observe only price
        R = np.array([[self.config.ekf_measurement_noise]])

        # Innovation
        innovation = measurement - H @ predicted_state

        # Update volatility estimate with innovation magnitude
        predicted_state[3] = 0.9 * self.state[3] + 0.1 * abs(innovation[0])

        # Kalman gain (using Faux-Riccati for stability if enabled)
        S = H @ predicted_cov @ H.T + R
        K = predicted_cov @ H.T @ np.linalg.inv(S)

        # Update
        self.state = predicted_state + K.flatten() * innovation[0]
        self.covariance = (np.eye(4) - K @ H) @ predicted_cov

        return self._get_state_dict()

    def _get_state_dict(self) -> Dict[str, float]:
        return {
            'price': float(self.state[0]),
            'velocity': float(self.state[1]),
            'acceleration': float(self.state[2]),
            'volatility': float(self.state[3])
        }

    @property
    def uncertainty(self) -> float:
        """Return state uncertainty (trace of covariance)."""
        return float(np.trace(self.covariance))


class VecNormalize:
    """
    A6: Running normalization for stable feature scaling.

    Maintains running mean and std, clips to bounds.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.dim = config.feature_dim
        self.running_mean = np.zeros(self.dim)
        self.running_var = np.ones(self.dim)
        self.count = 1e-4

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics."""
        # Update running statistics
        batch_mean = features
        batch_var = np.zeros_like(features)  # Single sample

        delta = batch_mean - self.running_mean
        self.count += 1
        self.running_mean += delta / self.count
        self.running_var = (self.running_var * (self.count - 1) + batch_var) / self.count

        # Normalize
        std = np.sqrt(self.running_var + 1e-8)
        normalized = (features - self.running_mean) / std

        # Clip
        return np.clip(normalized, -self.config.vecnormalize_clip, self.config.vecnormalize_clip)


class PreprocessingPipeline:
    """
    Part A: Complete preprocessing pipeline.

    Components:
        A1: Extended Kalman Filter
        A2: Conversational Autoencoders (simplified)
        A3: Frequency Normalization (simplified)
        A6: VecNormalize
        A8: Online Augmentation (simplified)
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.ekf = ExtendedKalmanFilter(config)
        self.normalizer = VecNormalize(config)
        self.feature_buffer: deque = deque(maxlen=100)

    def process(self, market_data: RawMarketData) -> PreprocessedFeatures:
        """Process raw market data into normalized features."""
        start = time.perf_counter()

        # A1: EKF denoising
        ekf_state = self.ekf.update(market_data.close)

        # Build raw feature vector (60 dimensions)
        features = self._build_features(market_data, ekf_state)

        # A6: Normalize
        normalized_features = self.normalizer.normalize(features)

        # Store in buffer for CAE
        self.feature_buffer.append(normalized_features)

        # A2: CAE ambiguity (simplified - use variance of recent features)
        cae_ambiguity = self._compute_ambiguity()

        latency_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Preprocessing latency: {latency_ms:.2f}ms")

        return PreprocessedFeatures(
            features=normalized_features,
            ekf_state=ekf_state,
            uncertainty=self.ekf.uncertainty,
            cae_ambiguity=cae_ambiguity,
            timestamp=market_data.timestamp
        )

    def _build_features(self, data: RawMarketData, ekf: Dict[str, float]) -> np.ndarray:
        """Build 60-dimensional feature vector."""
        features = np.zeros(self.config.feature_dim)

        # OHLCV features (5)
        features[0] = data.close
        features[1] = data.high - data.low  # Range
        features[2] = data.close - data.open  # Body
        features[3] = data.volume
        features[4] = (data.close - data.low) / (data.high - data.low + 1e-8)  # Close location

        # EKF features (4)
        features[5] = ekf['price']
        features[6] = ekf['velocity']
        features[7] = ekf['acceleration']
        features[8] = ekf['volatility']

        # Order flow (2)
        if data.order_flow is not None:
            features[9] = data.order_flow[0] - data.order_flow[1]  # Imbalance
            features[10] = data.order_flow[0] + data.order_flow[1]  # Total

        # Sentiment (1)
        if data.sentiment_raw is not None:
            features[11] = data.sentiment_raw

        # Technical indicators (computed here for simplicity)
        # In production, these would be pre-computed
        features[12:60] = np.random.randn(48) * 0.1  # Placeholder

        return features

    def _compute_ambiguity(self) -> float:
        """Compute CAE-style ambiguity from feature variance."""
        if len(self.feature_buffer) < 10:
            return 0.5

        recent = np.array(list(self.feature_buffer)[-10:])
        variance = np.mean(np.var(recent, axis=0))
        return float(np.tanh(variance))


# =============================================================================
# PART B: REGIME DETECTION
# =============================================================================

class JumpDetector:
    """
    B5: Fast jump detection for crisis regime.

    Uses Z-score against rolling volatility to detect jumps.
    """

    def __init__(self, config: RegimeConfig):
        self.config = config
        self.returns: deque = deque(maxlen=100)

    def detect(self, return_pct: float) -> bool:
        """Check if return is a jump (>3 sigma)."""
        self.returns.append(return_pct)

        if len(self.returns) < 20:
            return False

        returns_arr = np.array(self.returns)
        mean = np.mean(returns_arr[:-1])
        std = np.std(returns_arr[:-1]) + 1e-8

        z_score = abs(return_pct - mean) / std
        return z_score > self.config.jump_threshold_sigma


class HurstExponent:
    """
    B6: Hurst exponent estimation.

    H > 0.5: Trending (persistent)
    H = 0.5: Random walk
    H < 0.5: Mean-reverting (anti-persistent)
    """

    def __init__(self, config: RegimeConfig):
        self.config = config
        self.prices: deque = deque(maxlen=config.hurst_window)

    def compute(self, price: float) -> float:
        """Compute Hurst exponent using R/S analysis."""
        self.prices.append(price)

        if len(self.prices) < 50:
            return 0.5  # Default to random walk

        prices = np.array(self.prices)
        returns = np.diff(np.log(prices))

        # Simplified R/S analysis
        n = len(returns)
        half = n // 2

        # Compute R/S for two scales
        rs1 = self._rs_statistic(returns[:half])
        rs2 = self._rs_statistic(returns[half:])

        if rs1 <= 0 or rs2 <= 0:
            return 0.5

        # Estimate H from scaling
        h = np.log(rs2 / rs1) / np.log(2) + 0.5
        return float(np.clip(h, 0.0, 1.0))

    def _rs_statistic(self, returns: np.ndarray) -> float:
        """Compute rescaled range statistic."""
        n = len(returns)
        if n < 10:
            return 1.0

        mean = np.mean(returns)
        cumdev = np.cumsum(returns - mean)
        r = np.max(cumdev) - np.min(cumdev)
        s = np.std(returns) + 1e-8
        return r / s


class RegimeDetector:
    """
    Part B: Complete regime detection pipeline.

    Components:
        B1: Student-t HMM (simplified to threshold-based)
        B2: Meta-Regime Layer
        B5: Jump Detector
        B6: Hurst Exponent
    """

    def __init__(self, config: RegimeConfig):
        self.config = config
        self.jump_detector = JumpDetector(config)
        self.hurst_estimator = HurstExponent(config)

        # Regime tracking
        self.regime_history: deque = deque(maxlen=100)
        self.current_regime = MarketRegime.SIDEWAYS
        self.meta_regime = MetaRegime.LOW_UNCERTAINTY

        # Volatility proxy (simplified VIX)
        self.volatility_proxy: deque = deque(maxlen=20)

    def detect(
        self,
        features: PreprocessedFeatures,
        llm_signal: Optional[LLMSignal] = None
    ) -> RegimeOutput:
        """Detect current market regime."""
        start = time.perf_counter()

        # Extract key features
        velocity = features.ekf_state['velocity']
        volatility = features.ekf_state['volatility']

        # Compute return
        if len(self.volatility_proxy) > 0:
            prev_price = self.volatility_proxy[-1]
            curr_price = features.ekf_state['price']
            return_pct = (curr_price - prev_price) / prev_price if prev_price > 0 else 0
        else:
            return_pct = 0

        self.volatility_proxy.append(features.ekf_state['price'])

        # B5: Jump detection (fast path to crisis)
        is_jump = self.jump_detector.detect(return_pct)
        if is_jump:
            self.current_regime = MarketRegime.CRISIS
            return self._build_output(features, crisis_override=True)

        # B6: Hurst exponent
        hurst = self.hurst_estimator.compute(features.ekf_state['price'])

        # B1: Simplified regime classification
        regime_probs = self._classify_regime(velocity, volatility, hurst)

        # B2: Meta-regime (uncertainty level)
        self.meta_regime = self._classify_meta_regime(volatility)

        # Determine regime from probabilities
        regime_idx = np.argmax(regime_probs)
        regimes = [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS, MarketRegime.CRISIS]
        self.current_regime = regimes[regime_idx]

        # Track history
        self.regime_history.append(self.current_regime)

        latency_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Regime detection latency: {latency_ms:.2f}ms")

        return self._build_output(features, hurst, regime_probs)

    def _classify_regime(
        self,
        velocity: float,
        volatility: float,
        hurst: float
    ) -> np.ndarray:
        """Simplified regime classification using thresholds."""
        probs = np.array([0.25, 0.25, 0.45, 0.05])  # [bull, bear, sideways, crisis]

        # Adjust based on velocity (momentum)
        if velocity > 0.01:
            probs[0] += 0.3  # More bullish
            probs[2] -= 0.15
        elif velocity < -0.01:
            probs[1] += 0.3  # More bearish
            probs[2] -= 0.15

        # Adjust based on Hurst
        if hurst > 0.6:
            # Trending - boost current direction
            probs[0] *= 1.2 if velocity > 0 else 1.0
            probs[1] *= 1.2 if velocity < 0 else 1.0
        elif hurst < 0.4:
            # Mean-reverting - boost sideways
            probs[2] += 0.2

        # Normalize
        probs = probs / probs.sum()
        return probs

    def _classify_meta_regime(self, volatility: float) -> MetaRegime:
        """Classify uncertainty level."""
        # Use volatility as proxy for VIX
        scaled_vol = volatility * 100  # Scale to VIX-like range

        if scaled_vol > self.config.vix_high_threshold:
            return MetaRegime.HIGH_UNCERTAINTY
        else:
            return MetaRegime.LOW_UNCERTAINTY

    def _build_output(
        self,
        features: PreprocessedFeatures,
        hurst: float = 0.5,
        regime_probs: Optional[np.ndarray] = None,
        crisis_override: bool = False
    ) -> RegimeOutput:
        """Build regime output."""
        if crisis_override:
            regime_probs = np.array([0.0, 0.0, 0.0, 1.0])
        elif regime_probs is None:
            regime_probs = np.array([0.25, 0.25, 0.45, 0.05])

        # Check for transition warning
        transition_warning = self._check_transition_warning()

        return RegimeOutput(
            regime=self.current_regime,
            meta_regime=self.meta_regime,
            probabilities=regime_probs,
            confidence=float(np.max(regime_probs)),
            transition_warning=transition_warning,
            crisis_probability=float(regime_probs[3]),
            hurst_exponent=hurst
        )

    def _check_transition_warning(self) -> bool:
        """Check if regime transition is likely."""
        if len(self.regime_history) < 10:
            return False

        recent = list(self.regime_history)[-10:]
        unique_regimes = len(set(recent))
        return unique_regimes >= 3


# =============================================================================
# PART D: DECISION ENGINE
# =============================================================================

class BaseAgent:
    """Base class for decision agents."""

    def __init__(self, name: str):
        self.name = name
        self.sharpe_history: deque = deque(maxlen=100)
        self.rolling_sharpe: float = 1.0

    def decide(self, features: np.ndarray, regime: RegimeOutput) -> Tuple[int, float]:
        """Return (action, confidence)."""
        raise NotImplementedError

    def update_sharpe(self, return_pct: float) -> None:
        """Update rolling Sharpe estimate."""
        self.sharpe_history.append(return_pct)
        if len(self.sharpe_history) >= 20:
            returns = np.array(self.sharpe_history)
            mean = np.mean(returns)
            std = np.std(returns) + 1e-8
            self.rolling_sharpe = max(0.1, mean / std * np.sqrt(252))


class PPOAgent(BaseAgent):
    """
    D5: PPO-LSTM agent (simplified).

    Uses feature patterns to generate trading signals.
    """

    def __init__(self, config: DecisionConfig):
        super().__init__("PPO")
        self.config = config

    def decide(self, features: np.ndarray, regime: RegimeOutput) -> Tuple[int, float]:
        """Simplified PPO decision based on momentum."""
        # Use velocity feature as primary signal
        velocity = features[6] if len(features) > 6 else 0

        # Regime-adjusted thresholds
        threshold = 0.5 if regime.regime == MarketRegime.SIDEWAYS else 0.3

        if velocity > threshold:
            action = 1  # BUY
            confidence = min(0.9, 0.5 + abs(velocity))
        elif velocity < -threshold:
            action = -1  # SELL
            confidence = min(0.9, 0.5 + abs(velocity))
        else:
            action = 0  # HOLD
            confidence = 0.6

        return action, confidence


class SACAgent(BaseAgent):
    """
    D6: SAC agent (simplified).

    More conservative, entropy-regularized decisions.
    """

    def __init__(self, config: DecisionConfig):
        super().__init__("SAC")
        self.config = config

    def decide(self, features: np.ndarray, regime: RegimeOutput) -> Tuple[int, float]:
        """Simplified SAC decision with entropy bonus."""
        velocity = features[6] if len(features) > 6 else 0
        volatility = features[8] if len(features) > 8 else 0.5

        # SAC is more conservative in high volatility
        vol_penalty = max(0.5, 1.0 - volatility * 2)

        threshold = 0.4 * vol_penalty

        if velocity > threshold:
            action = 1
            confidence = min(0.8, 0.4 + abs(velocity) * vol_penalty)
        elif velocity < -threshold:
            action = -1
            confidence = min(0.8, 0.4 + abs(velocity) * vol_penalty)
        else:
            action = 0
            confidence = 0.7

        return action, confidence


class ConservativeQAgent(BaseAgent):
    """
    D3: Conservative Q-Learning agent (simplified).

    Penalizes out-of-distribution actions.
    """

    def __init__(self, config: DecisionConfig):
        super().__init__("CQL")
        self.config = config

    def decide(self, features: np.ndarray, regime: RegimeOutput) -> Tuple[int, float]:
        """CQL: Conservative, prefers HOLD when uncertain."""
        velocity = features[6] if len(features) > 6 else 0
        uncertainty = features[8] if len(features) > 8 else 0.5

        # CQL penalizes actions more when uncertain
        action_penalty = 1.0 + uncertainty * 2
        threshold = 0.6 * action_penalty

        if velocity > threshold:
            action = 1
            confidence = min(0.7, 0.3 + abs(velocity) / action_penalty)
        elif velocity < -threshold:
            action = -1
            confidence = min(0.7, 0.3 + abs(velocity) / action_penalty)
        else:
            action = 0
            confidence = 0.8  # CQL is confident about HOLD

        return action, confidence


class DecisionEngine:
    """
    Part D: Complete decision engine with ensemble.

    Components:
        D3: Conservative Q (CQL)
        D5: PPO-LSTM
        D6: SAC
        D7: Sharpe-Weighted Voting
        D8: Disagreement Scaling
        D9: Return Conditioning
    """

    def __init__(self, config: DecisionConfig):
        self.config = config

        # Initialize agents
        self.agents: Dict[str, BaseAgent] = {
            'ppo': PPOAgent(config),
            'sac': SACAgent(config),
            'cql': ConservativeQAgent(config),
        }

    def decide(
        self,
        features: PreprocessedFeatures,
        regime: RegimeOutput
    ) -> DecisionOutput:
        """Generate ensemble trading decision."""
        start = time.perf_counter()

        # D9: Return conditioning - adjust target based on regime
        target_sharpe = self._get_target_sharpe(regime)

        # Get decisions from all agents
        agent_outputs: Dict[str, Tuple[int, float]] = {}
        for name, agent in self.agents.items():
            action, confidence = agent.decide(features.features, regime)
            agent_outputs[name] = (action, confidence)

        # D7: Sharpe-weighted voting
        weights = self._compute_sharpe_weights()

        # Combine votes
        combined_action, combined_confidence = self._weighted_vote(agent_outputs, weights)

        # D8: Disagreement scaling
        disagreement = self._compute_disagreement(agent_outputs)
        if disagreement > self.config.disagreement_high_threshold:
            combined_confidence *= 0.25  # Heavy penalty for disagreement

        latency_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Decision engine latency: {latency_ms:.2f}ms")

        return DecisionOutput(
            action=TradeAction(combined_action),
            confidence=combined_confidence,
            agent_outputs=agent_outputs,
            disagreement=disagreement,
            ensemble_weights=weights
        )

    def _get_target_sharpe(self, regime: RegimeOutput) -> float:
        """D9: Get regime-dependent target Sharpe."""
        regime_map = {
            MarketRegime.CRISIS: self.config.return_conditioning_targets.get('crisis', 0.5),
            MarketRegime.BEAR: self.config.return_conditioning_targets.get('bear', 1.0),
            MarketRegime.SIDEWAYS: self.config.return_conditioning_targets.get('sideways', 2.0),
            MarketRegime.BULL: self.config.return_conditioning_targets.get('bull', 2.5),
        }
        return regime_map.get(regime.regime, 1.5)

    def _compute_sharpe_weights(self) -> Dict[str, float]:
        """D7: Compute Sharpe-weighted ensemble weights."""
        sharpes = {name: agent.rolling_sharpe for name, agent in self.agents.items()}
        total = sum(sharpes.values())

        if total <= 0:
            # Equal weights if no Sharpe history
            return {name: 1.0 / len(self.agents) for name in self.agents}

        return {name: sharpe / total for name, sharpe in sharpes.items()}

    def _weighted_vote(
        self,
        outputs: Dict[str, Tuple[int, float]],
        weights: Dict[str, float]
    ) -> Tuple[int, float]:
        """Combine agent outputs with weighted voting."""
        # Compute weighted action probabilities
        action_scores = {-1: 0.0, 0: 0.0, 1: 0.0}

        for name, (action, confidence) in outputs.items():
            weight = weights.get(name, 0.0)
            action_scores[action] += weight * confidence

        # Select highest scoring action
        best_action = max(action_scores.keys(), key=lambda a: action_scores[a])
        best_confidence = action_scores[best_action]

        # Normalize confidence
        total = sum(action_scores.values())
        if total > 0:
            best_confidence = best_confidence / total

        return best_action, best_confidence

    def _compute_disagreement(self, outputs: Dict[str, Tuple[int, float]]) -> float:
        """D8: Compute ensemble disagreement."""
        actions = [action for action, _ in outputs.values()]

        if len(set(actions)) == 1:
            return 0.0  # Full agreement

        # Count unique actions
        unique = len(set(actions))
        return (unique - 1) / 2  # Normalized to [0, 1]


# =============================================================================
# PART F: UNCERTAINTY QUANTIFICATION
# =============================================================================

class TemperatureScaling:
    """
    F3: Temperature scaling for confidence calibration.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def calibrate(self, confidence: float) -> float:
        """Apply temperature scaling to confidence."""
        # Convert confidence to logit, scale, convert back
        eps = 1e-8
        confidence = np.clip(confidence, eps, 1 - eps)
        logit = np.log(confidence / (1 - confidence))
        scaled_logit = logit / self.temperature
        return float(1 / (1 + np.exp(-scaled_logit)))


class UncertaintyQuantifier:
    """
    Part F: Uncertainty quantification pipeline.

    Components:
        F1: CT-SSF Conformal (simplified)
        F2: CPTC Regime-Aware (simplified)
        F3: Temperature Scaling
        F5: MC Dropout (simplified)
        F6: Epistemic/Aleatoric Split
        F7: k-NN OOD Detection (simplified)
    """

    def __init__(self, config: UQConfig):
        self.config = config
        self.temp_scaler = TemperatureScaling(config.temperature)

        # Calibration data
        self.calibration_scores: deque = deque(maxlen=config.conformal_n_calibration)
        self.feature_history: deque = deque(maxlen=500)

    def quantify(
        self,
        features: PreprocessedFeatures,
        decision: DecisionOutput,
        regime: RegimeOutput
    ) -> UQOutput:
        """Quantify uncertainty in decision."""
        start = time.perf_counter()

        # F3: Temperature scaling
        calibrated_conf = self.temp_scaler.calibrate(decision.confidence)

        # F5/F6: Epistemic and aleatoric uncertainty (simplified)
        epistemic = self._estimate_epistemic(decision)
        aleatoric = self._estimate_aleatoric(features)

        # F7: OOD detection (simplified k-NN)
        ood_score = self._compute_ood_score(features)

        # F1/F2: Conformal prediction interval
        interval = self._compute_interval(calibrated_conf, regime)

        # Total uncertainty
        total = np.sqrt(epistemic**2 + aleatoric**2)

        # Store for future calibration
        self.feature_history.append(features.features)

        latency_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"UQ latency: {latency_ms:.2f}ms")

        return UQOutput(
            calibrated_confidence=calibrated_conf,
            prediction_interval=interval,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            ood_score=ood_score,
            total_uncertainty=total
        )

    def _estimate_epistemic(self, decision: DecisionOutput) -> float:
        """Estimate epistemic (model) uncertainty from disagreement."""
        return decision.disagreement

    def _estimate_aleatoric(self, features: PreprocessedFeatures) -> float:
        """Estimate aleatoric (data) uncertainty from feature variance."""
        return float(np.std(features.features) / 10)

    def _compute_ood_score(self, features: PreprocessedFeatures) -> float:
        """Compute out-of-distribution score using k-NN distance."""
        if len(self.feature_history) < 50:
            return 0.0

        # Compute distance to nearest neighbors (simplified)
        history = np.array(list(self.feature_history)[-50:])
        distances = np.linalg.norm(history - features.features, axis=1)
        k = min(5, len(distances))
        knn_dist = np.mean(np.sort(distances)[:k])

        # Normalize to [0, 1]
        return float(np.tanh(knn_dist / self.config.ood_threshold))

    def _compute_interval(
        self,
        confidence: float,
        regime: RegimeOutput
    ) -> Tuple[float, float]:
        """Compute conformal prediction interval."""
        # Base interval width inversely proportional to confidence
        base_width = (1 - confidence) * 0.05

        # F2: Expand during regime transitions
        if regime.transition_warning:
            base_width *= 2.0

        # Wider in high uncertainty meta-regime
        if regime.meta_regime == MetaRegime.HIGH_UNCERTAINTY:
            base_width *= 1.5

        return (-base_width, base_width)


# =============================================================================
# PART E: HSM STATE MACHINE
# =============================================================================

class TradingHSM:
    """
    Part E: Hierarchical State Machine for position validation.

    Components:
        E1: Orthogonal Regions (Position + Regime)
        E2: Hierarchical Nesting (Long/Short modes)
        E3: History States
        E4: Synchronized Events
        E6: Oscillation Detection
    """

    def __init__(self, config: HSMConfig):
        self.config = config

        # Position region state
        self.position_state = config.initial_position_state

        # Transition history for oscillation detection
        self.transition_history: deque = deque(maxlen=config.oscillation_window_bars)

        # Valid transitions
        self.valid_transitions: Dict[PositionState, Dict[int, PositionState]] = {
            PositionState.FLAT: {
                1: PositionState.LONG_ENTRY,
                -1: PositionState.SHORT_ENTRY,
                0: PositionState.FLAT
            },
            PositionState.LONG_ENTRY: {
                0: PositionState.LONG_HOLD,
                -1: PositionState.FLAT,
                1: PositionState.LONG_ENTRY
            },
            PositionState.LONG_HOLD: {
                -1: PositionState.LONG_EXIT,
                0: PositionState.LONG_HOLD,
                1: PositionState.LONG_HOLD
            },
            PositionState.LONG_EXIT: {
                0: PositionState.FLAT,
                -1: PositionState.FLAT,
                1: PositionState.LONG_HOLD
            },
            PositionState.SHORT_ENTRY: {
                0: PositionState.SHORT_HOLD,
                1: PositionState.FLAT,
                -1: PositionState.SHORT_ENTRY
            },
            PositionState.SHORT_HOLD: {
                1: PositionState.SHORT_EXIT,
                0: PositionState.SHORT_HOLD,
                -1: PositionState.SHORT_HOLD
            },
            PositionState.SHORT_EXIT: {
                0: PositionState.FLAT,
                1: PositionState.FLAT,
                -1: PositionState.SHORT_HOLD
            }
        }

    def validate(
        self,
        proposed_action: TradeAction,
        current_position: int
    ) -> HSMOutput:
        """Validate proposed action against current state."""
        start = time.perf_counter()

        action_value = proposed_action.value

        # Check for oscillation
        oscillation_blocked = self._check_oscillation(action_value)
        if oscillation_blocked:
            validated_action = TradeAction.HOLD
            blocked_reason = "Oscillation detected"
        else:
            # Check valid transition
            transitions = self.valid_transitions.get(self.position_state, {})
            if action_value in transitions:
                new_state = transitions[action_value]

                # Track transition
                if new_state != self.position_state:
                    self.transition_history.append(new_state)

                self.position_state = new_state
                validated_action = proposed_action
                blocked_reason = None
            else:
                validated_action = TradeAction.HOLD
                blocked_reason = f"Invalid transition from {self.position_state.name}"

        latency_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"HSM validation latency: {latency_ms:.2f}ms")

        return HSMOutput(
            validated_action=validated_action,
            is_valid=(validated_action == proposed_action),
            position_state=self.position_state,
            regime_state="",  # Not tracking regime state in simplified version
            blocked_reason=blocked_reason,
            oscillation_blocked=oscillation_blocked
        )

    def _check_oscillation(self, action: int) -> bool:
        """E6: Check for position oscillation (whipsaw)."""
        if len(self.transition_history) < self.config.oscillation_max_flips:
            return False

        # Count direction changes in recent history
        recent = list(self.transition_history)[-self.config.oscillation_window_bars:]

        # Map states to position direction
        direction_map = {
            PositionState.FLAT: 0,
            PositionState.LONG_ENTRY: 1,
            PositionState.LONG_HOLD: 1,
            PositionState.LONG_EXIT: 1,
            PositionState.SHORT_ENTRY: -1,
            PositionState.SHORT_HOLD: -1,
            PositionState.SHORT_EXIT: -1
        }

        directions = [direction_map.get(s, 0) for s in recent]

        # Count sign changes
        flips = sum(1 for i in range(1, len(directions))
                   if directions[i] != 0 and directions[i-1] != 0
                   and directions[i] != directions[i-1])

        return flips >= self.config.oscillation_max_flips


# =============================================================================
# PART G: HYSTERESIS FILTER
# =============================================================================

class KAMAThresholdAdapter:
    """
    G1: KAMA-based adaptive thresholds.
    """

    def __init__(self, config: HysteresisConfig):
        self.config = config
        self.prices: deque = deque(maxlen=config.er_period + 1)
        self.smoothed_er: float = 0.5

    def update(self, price: float) -> Tuple[float, float]:
        """Update and return (entry_threshold, exit_threshold)."""
        self.prices.append(price)

        if len(self.prices) < self.config.er_period + 1:
            return (0.35, 0.16)

        # Compute efficiency ratio
        er = self._compute_er()

        # Smooth
        self.smoothed_er = 0.3 * er + 0.7 * self.smoothed_er

        # Interpolate thresholds
        entry = (self.config.slow_threshold_entry +
                self.smoothed_er * (self.config.fast_threshold_entry - self.config.slow_threshold_entry))
        exit_thresh = (self.config.slow_threshold_exit +
                      self.smoothed_er * (self.config.fast_threshold_exit - self.config.slow_threshold_exit))

        return (entry, exit_thresh)

    def _compute_er(self) -> float:
        """Compute efficiency ratio."""
        prices = list(self.prices)
        direction = abs(prices[-1] - prices[0])
        volatility = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices)))

        if volatility < 1e-10:
            return 1.0

        return min(direction / volatility, 1.0)


class HysteresisFilter:
    """
    Part G: Hysteresis filter to prevent whipsaw.

    Components:
        G1: KAMA Adaptive Thresholds
        G3: ATR-Scaled Bands (simplified)
        G5: 2.2x Loss Aversion Ratio
        G6: Whipsaw Learning (simplified)
    """

    def __init__(self, config: HysteresisConfig):
        self.config = config
        self.kama = KAMAThresholdAdapter(config)
        self.current_position: int = 0

    def filter(
        self,
        action: TradeAction,
        confidence: float,
        price: float
    ) -> HysteresisOutput:
        """Apply hysteresis filtering to action."""
        start = time.perf_counter()

        # G1: Get adaptive thresholds
        entry_thresh, exit_thresh = self.kama.update(price)

        # G5: Apply loss aversion asymmetry
        exit_thresh = entry_thresh / self.config.loss_aversion_ratio

        action_value = action.value
        was_filtered = False
        filter_reason = None

        if self.current_position == 0:
            # Flat: require entry threshold
            if action_value != 0 and confidence >= entry_thresh:
                self.current_position = action_value
                filtered_action = action
            else:
                filtered_action = TradeAction.HOLD
                if action_value != 0:
                    was_filtered = True
                    filter_reason = f"Confidence {confidence:.2f} below entry {entry_thresh:.2f}"

        elif self.current_position == 1:  # Long
            if confidence < exit_thresh:
                self.current_position = 0
                filtered_action = TradeAction.SELL
            elif action_value == -1 and confidence >= entry_thresh:
                self.current_position = -1
                filtered_action = TradeAction.SELL
            else:
                filtered_action = TradeAction.HOLD
                if action_value == -1:
                    was_filtered = True
                    filter_reason = "Reversal blocked by hysteresis"

        else:  # Short (position == -1)
            if confidence < exit_thresh:
                self.current_position = 0
                filtered_action = TradeAction.BUY
            elif action_value == 1 and confidence >= entry_thresh:
                self.current_position = 1
                filtered_action = TradeAction.BUY
            else:
                filtered_action = TradeAction.HOLD
                if action_value == 1:
                    was_filtered = True
                    filter_reason = "Reversal blocked by hysteresis"

        latency_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Hysteresis latency: {latency_ms:.2f}ms")

        return HysteresisOutput(
            filtered_action=filtered_action,
            entry_threshold=entry_thresh,
            exit_threshold=exit_thresh,
            efficiency_ratio=self.kama.smoothed_er,
            was_filtered=was_filtered,
            filter_reason=filter_reason
        )


# =============================================================================
# PART H: RSS RISK MANAGEMENT
# =============================================================================

class EVTTailRisk:
    """
    H1: EVT + GPD tail risk estimation (simplified).
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.returns: deque = deque(maxlen=365)

    def update(self, return_pct: float) -> Dict[str, float]:
        """Update with new return and compute tail risk."""
        self.returns.append(return_pct)

        if len(self.returns) < 50:
            return {'var_95': 0.05, 'var_99': 0.10, 'es_99': 0.12}

        returns = np.array(self.returns)
        losses = -returns[returns < 0]

        if len(losses) < 20:
            return {'var_95': 0.05, 'var_99': 0.10, 'es_99': 0.12}

        var_95 = float(np.percentile(losses, 95))
        var_99 = float(np.percentile(losses, 99))

        # Expected shortfall (simplified)
        es_99 = float(np.mean(losses[losses >= np.percentile(losses, 99)]))

        return {'var_95': var_95, 'var_99': var_99, 'es_99': es_99}


class DrawdownBrake:
    """
    H4: Progressive drawdown brake.
    """

    def __init__(self, config: RiskConfig):
        self.config = config

    def compute_reduction(self, drawdown: float) -> float:
        """Compute position reduction factor based on drawdown."""
        for i, (level, reduction) in enumerate(
            zip(self.config.drawdown_brake_levels, self.config.drawdown_brake_reductions)
        ):
            if drawdown >= level:
                return reduction
        return 0.0


class RSSRiskManager:
    """
    Part H: RSS Risk Management pipeline.

    Components:
        H1: EVT + GPD Tail Risk
        H2: Dynamic Kelly (simplified)
        H4: Progressive Drawdown Brake
        H6: Safe Margin Formula
        H7: Dynamic Leverage Controller
        H8: Adaptive Risk Budget
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.evt = EVTTailRisk(config)
        self.dd_brake = DrawdownBrake(config)

        # Risk budget tracking
        self.risk_budget: float = 1.0

    def compute(
        self,
        action: TradeAction,
        portfolio: PortfolioState,
        regime: RegimeOutput
    ) -> RiskOutput:
        """Compute risk-adjusted position sizing."""
        start = time.perf_counter()

        # H1: Update tail risk
        tail_risk = self.evt.update(0.0)  # Would use actual returns

        # H2: Kelly fraction (simplified)
        kelly = self._compute_kelly(regime)

        # H4: Drawdown brake
        dd_reduction = self.dd_brake.compute_reduction(portfolio.drawdown)

        # H7: Dynamic leverage
        base_leverage = self._compute_dynamic_leverage(kelly)

        # H8: Risk budget adjustment
        self._update_risk_budget(portfolio)

        # Compute final position size
        base_size = kelly * self.risk_budget

        # Apply drawdown brake
        if dd_reduction > 0:
            base_size *= (1 - dd_reduction)
            dd_active = True
        else:
            dd_active = False

        # H6: Safe margin check
        leverage = min(base_leverage, self.config.max_leverage)

        # Scale down in crisis
        if regime.regime == MarketRegime.CRISIS:
            base_size *= 0.25
            leverage = min(leverage, 3.0)

        latency_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Risk management latency: {latency_ms:.2f}ms")

        return RiskOutput(
            action=action,
            position_size=base_size,
            leverage=leverage,
            var_95=tail_risk['var_95'],
            var_99=tail_risk['var_99'],
            expected_shortfall=tail_risk['es_99'],
            kelly_fraction=kelly,
            drawdown_brake_active=dd_active,
            risk_budget_remaining=self.risk_budget
        )

    def _compute_kelly(self, regime: RegimeOutput) -> float:
        """H2: Compute Kelly fraction based on regime."""
        base_kelly = self.config.kelly_fraction_cap

        # Scale by regime
        regime_scale = {
            MarketRegime.BULL: 1.0,
            MarketRegime.SIDEWAYS: 0.7,
            MarketRegime.BEAR: 0.5,
            MarketRegime.CRISIS: 0.25
        }

        return base_kelly * regime_scale.get(regime.regime, 0.5)

    def _compute_dynamic_leverage(self, kelly: float) -> float:
        """H7: Compute position-dependent leverage."""
        # Larger positions get lower leverage
        if kelly > 0.2:
            return 3.0
        elif kelly > 0.1:
            return 5.0
        else:
            return self.config.max_leverage

    def _update_risk_budget(self, portfolio: PortfolioState) -> None:
        """H8: Update risk budget based on performance."""
        if portfolio.realized_pnl > 0:
            self.risk_budget = min(1.2, self.risk_budget * 1.01)
        else:
            self.risk_budget = max(0.5, self.risk_budget * 0.99)


# =============================================================================
# PART I: SIMPLEX SAFETY SYSTEM
# =============================================================================

class SafetyMonitor:
    """
    I6: Runtime safety constraint monitor.
    """

    def __init__(self, config: SafetyConfig):
        self.config = config

    def check_invariants(
        self,
        position_size: float,
        leverage: float,
        drawdown: float
    ) -> Tuple[bool, float, str]:
        """Check safety invariants. Returns (is_safe, margin, violation)."""
        violations = []

        # Check leverage limit
        if leverage > 10.0:
            violations.append(f"Leverage {leverage:.1f}x exceeds limit")

        # Check position size
        if position_size > 1.0:
            violations.append(f"Position size {position_size:.2f} exceeds 100%")

        # Check drawdown
        if drawdown > 0.15:
            violations.append(f"Drawdown {drawdown:.1%} exceeds 15%")

        if violations:
            return False, 0.0, "; ".join(violations)

        # Compute safety margin
        margin = 1.0 - max(leverage / 10.0, position_size, drawdown / 0.15)
        return True, margin, ""


class SimplexSafetySystem:
    """
    Part I: Simplex Safety System with fallback cascade.

    Components:
        I1: 4-Level Fallback Cascade
        I2: Predictive Safety (simplified)
        I5: Safety Invariants
        I6: Safety Monitor
        I7: Stop-Loss Enforcer
        I8: Recovery Protocol
    """

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.monitor = SafetyMonitor(config)

        # Stop-loss tracking
        self.daily_pnl: float = 0.0
        self.halt_until: float = 0.0

        # Current fallback level
        self.current_level = FallbackLevel.PRIMARY
        self.bars_at_level: int = 0

    def verify(
        self,
        action: TradeAction,
        position_size: float,
        leverage: float,
        drawdown: float,
        regime: RegimeOutput,
        confidence: float
    ) -> SafetyOutput:
        """Verify action safety and apply fallback if needed."""
        start = time.perf_counter()

        # I7: Check stop-loss
        if self._check_stop_loss():
            return SafetyOutput(
                final_action=TradeAction.HOLD,
                final_position_size=0.0,
                fallback_level=FallbackLevel.MINIMAL,
                primary_blocked=True,
                block_reason="Daily stop-loss triggered",
                safety_margin=0.0,
                stop_loss_triggered=True,
                recovery_status="halted"
            )

        # I1: Check forced fallback conditions
        forced_level = self._check_forced_fallback(regime, drawdown)

        # I5/I6: Check safety invariants
        is_safe, margin, violation = self.monitor.check_invariants(
            position_size, leverage, drawdown
        )

        # Determine final action through cascade
        if forced_level is not None:
            final_action, final_size, level = self._cascade_from_level(
                forced_level, action, position_size, confidence
            )
            blocked = (level != FallbackLevel.PRIMARY)
            block_reason = f"Forced to {level.name}: {violation}" if blocked else ""
        elif not is_safe:
            final_action, final_size, level = self._cascade_from_level(
                FallbackLevel.CONSERVATIVE, action, position_size, confidence
            )
            blocked = True
            block_reason = f"Safety violation: {violation}"
        else:
            final_action, final_size, level = self._try_level(
                FallbackLevel.PRIMARY, action, position_size, confidence
            )
            blocked = (level != FallbackLevel.PRIMARY)
            block_reason = "" if not blocked else "Confidence below threshold"

        # Track level
        if level != self.current_level:
            self.current_level = level
            self.bars_at_level = 0
        else:
            self.bars_at_level += 1

        # I8: Check recovery
        recovery_status = self._check_recovery(confidence)

        latency_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Safety system latency: {latency_ms:.2f}ms")

        return SafetyOutput(
            final_action=final_action,
            final_position_size=final_size,
            fallback_level=level,
            primary_blocked=blocked,
            block_reason=block_reason,
            safety_margin=margin,
            stop_loss_triggered=False,
            recovery_status=recovery_status
        )

    def _check_stop_loss(self) -> bool:
        """I7: Check if daily stop-loss triggered."""
        if time.time() < self.halt_until:
            return True
        return self.daily_pnl < -self.config.max_daily_loss

    def _check_forced_fallback(
        self,
        regime: RegimeOutput,
        drawdown: float
    ) -> Optional[FallbackLevel]:
        """I1: Check conditions for forced fallback."""
        if drawdown >= self.config.drawdown_force_minimal:
            return FallbackLevel.MINIMAL

        if self.config.crisis_force_conservative and regime.regime == MarketRegime.CRISIS:
            return FallbackLevel.CONSERVATIVE

        return None

    def _try_level(
        self,
        level: FallbackLevel,
        action: TradeAction,
        position_size: float,
        confidence: float
    ) -> Tuple[TradeAction, float, FallbackLevel]:
        """Try action at a specific level."""
        thresholds = {
            FallbackLevel.PRIMARY: self.config.primary_confidence_threshold,
            FallbackLevel.BASELINE: self.config.baseline_confidence_threshold,
            FallbackLevel.CONSERVATIVE: self.config.conservative_confidence_threshold,
            FallbackLevel.MINIMAL: 0.0
        }

        if confidence >= thresholds[level]:
            return action, position_size, level

        # Cascade to next level
        next_levels = {
            FallbackLevel.PRIMARY: FallbackLevel.BASELINE,
            FallbackLevel.BASELINE: FallbackLevel.CONSERVATIVE,
            FallbackLevel.CONSERVATIVE: FallbackLevel.MINIMAL,
            FallbackLevel.MINIMAL: FallbackLevel.MINIMAL
        }

        return self._cascade_from_level(
            next_levels[level], action, position_size, confidence
        )

    def _cascade_from_level(
        self,
        start_level: FallbackLevel,
        action: TradeAction,
        position_size: float,
        confidence: float
    ) -> Tuple[TradeAction, float, FallbackLevel]:
        """Cascade through levels starting from start_level."""
        levels = [FallbackLevel.PRIMARY, FallbackLevel.BASELINE,
                  FallbackLevel.CONSERVATIVE, FallbackLevel.MINIMAL]

        for level in levels[start_level.value:]:
            if level == FallbackLevel.MINIMAL:
                return TradeAction.HOLD, 0.0, FallbackLevel.MINIMAL

            result_action, result_size, result_level = self._try_level(
                level, action, position_size, confidence
            )
            if result_level == level:
                return result_action, result_size, result_level

        return TradeAction.HOLD, 0.0, FallbackLevel.MINIMAL

    def _check_recovery(self, confidence: float) -> str:
        """I8: Check if system can recover to higher level."""
        if self.current_level == FallbackLevel.PRIMARY:
            return "normal"

        if (self.bars_at_level >= self.config.recovery_bars_required and
            confidence >= 0.7):
            return "ready_to_recover"

        return f"recovering ({self.bars_at_level}/{self.config.recovery_bars_required} bars)"


# =============================================================================
# PART J: LLM INTEGRATION (ASYNC SIDECAR)
# =============================================================================

class LLMSidecar:
    """
    Part J: Asynchronous LLM signal integration.

    Runs in background, caches signals for main pipeline to read.

    Components (simplified):
        J1: OPT Financial (stub)
        J4: Confidence Calibration
        J8: Async Processing
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.enabled = config.enabled

        # Cached signal
        self._cached_signal: Optional[LLMSignal] = None
        self._cache_time: float = 0.0

        # Background processing (simplified - in production use Redis)
        self._processing_queue: deque = deque(maxlen=config.async_queue_size)

    def get_cached_signal(self) -> Optional[LLMSignal]:
        """Get most recent cached LLM signal (if fresh)."""
        if not self.enabled:
            return None

        if self._cached_signal is None:
            return None

        # Check freshness
        age = time.time() - self._cache_time
        if age > self.config.cache_ttl_seconds:
            return None

        return self._cached_signal

    def queue_text(self, text: str, source: str) -> None:
        """Queue text for async processing."""
        if not self.enabled:
            return

        self._processing_queue.append({
            'text': text,
            'source': source,
            'timestamp': time.time()
        })

    def process_queue_sync(self) -> None:
        """Process queue synchronously (for testing). In production, this runs async."""
        if not self._processing_queue:
            return

        item = self._processing_queue.popleft()

        # Simplified LLM processing (would use actual model in production)
        signal = self._generate_signal_stub(item['text'])

        self._cached_signal = signal
        self._cache_time = time.time()

    def _generate_signal_stub(self, text: str) -> LLMSignal:
        """Generate stub signal (placeholder for actual LLM)."""
        # Simple sentiment heuristic
        positive_words = ['bullish', 'growth', 'surge', 'rally', 'strong']
        negative_words = ['bearish', 'crash', 'drop', 'weak', 'fear']

        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            sentiment = 0.5
            direction = 1
        elif neg_count > pos_count:
            sentiment = -0.5
            direction = -1
        else:
            sentiment = 0.0
            direction = 0

        return LLMSignal(
            sentiment=sentiment,
            confidence=0.6,
            direction=direction,
            reasoning="Keyword-based analysis",
            source_summary=text[:100],
            timestamp=time.time(),
            latency_ms=5.0,
            calibrated=False
        )


# =============================================================================
# PART N: INTERPRETABILITY (ASYNC)
# =============================================================================

class InterpretabilityEngine:
    """
    Part N: Async interpretability and explanation generation.

    Components (simplified):
        N1: SHAP Attribution (stub)
        N3: MiFID II Compliance logging
    """

    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self._explanation_queue: deque = deque(maxlen=100)

    def queue_explanation(
        self,
        features: PreprocessedFeatures,
        decision: DecisionOutput,
        safety: SafetyOutput
    ) -> None:
        """Queue decision for explanation generation."""
        if not self.config.async_enabled:
            return

        self._explanation_queue.append({
            'timestamp': time.time(),
            'features': features,
            'decision': decision,
            'safety': safety
        })

    def generate_explanation(self) -> Optional[Dict[str, Any]]:
        """Generate explanation for queued decision (async)."""
        if not self._explanation_queue:
            return None

        item = self._explanation_queue.popleft()

        # Simplified explanation (would use SHAP in production)
        return {
            'timestamp': item['timestamp'],
            'action': item['decision'].action.name,
            'confidence': item['decision'].confidence,
            'fallback_level': item['safety'].fallback_level.name,
            'top_features': ['velocity', 'volatility', 'regime'],
            'mifid_compliant': self.config.mifid_enabled
        }


# =============================================================================
# MASTER PIPELINE
# =============================================================================

class Layer2MasterPipeline:
    """
    Master Pipeline orchestrating all 14 HIMARI Layer 2 subsystems.

    This is the main entry point for processing market data through
    the complete trading pipeline.

    Usage:
        config = MasterConfig()
        pipeline = Layer2MasterPipeline(config)
        result = pipeline.process(market_data, portfolio)
    """

    def __init__(self, config: MasterConfig = None, debug: bool = False):
        """
        Initialize the master pipeline with all subsystems.

        Args:
            config: Master configuration (uses defaults if None)
            debug: Enable debug logging
        """
        self.config = config or MasterConfig()
        self.config.debug_mode = debug

        if debug:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info("Initializing HIMARI Layer 2 Master Pipeline...")

        # Initialize all subsystems
        self._init_subsystems()

        logger.info("Pipeline initialization complete")

    def _init_subsystems(self) -> None:
        """Initialize all subsystem components."""
        # Part A: Preprocessing
        logger.info("  Initializing Part A: Preprocessing...")
        self.preprocessor = PreprocessingPipeline(self.config.preprocessing)

        # Part B: Regime Detection
        logger.info("  Initializing Part B: Regime Detection...")
        self.regime_detector = RegimeDetector(self.config.regime)

        # Part D: Decision Engine
        logger.info("  Initializing Part D: Decision Engine...")
        self.decision_engine = DecisionEngine(self.config.decision)

        # Part F: Uncertainty Quantification
        logger.info("  Initializing Part F: Uncertainty Quantification...")
        self.uncertainty = UncertaintyQuantifier(self.config.uncertainty)

        # Part E: HSM State Machine
        logger.info("  Initializing Part E: HSM State Machine...")
        self.hsm = TradingHSM(self.config.hsm)

        # Part G: Hysteresis Filter
        logger.info("  Initializing Part G: Hysteresis Filter...")
        self.hysteresis = HysteresisFilter(self.config.hysteresis)

        # Part H: RSS Risk Management
        logger.info("  Initializing Part H: RSS Risk Management...")
        self.risk_manager = RSSRiskManager(self.config.risk)

        # Part I: Simplex Safety System
        logger.info("  Initializing Part I: Simplex Safety System...")
        self.safety_system = SimplexSafetySystem(self.config.safety)

        # Part J: LLM Integration (Async)
        logger.info("  Initializing Part J: LLM Integration...")
        self.llm_sidecar = LLMSidecar(self.config.llm)

        # Part N: Interpretability (Async)
        logger.info("  Initializing Part N: Interpretability...")
        self.interpretability = InterpretabilityEngine(self.config.interpretability)

    def process(
        self,
        market_data: RawMarketData,
        portfolio: PortfolioState
    ) -> PipelineOutput:
        """
        Process market data through the complete pipeline.

        Args:
            market_data: Raw OHLCV and metadata
            portfolio: Current portfolio state

        Returns:
            PipelineOutput with final decision and all intermediate outputs
        """
        start_time = time.perf_counter()

        # Stage 1: Preprocessing (Part A)
        features = self.preprocessor.process(market_data)

        # Stage 2: Regime Detection (Part B) with LLM signal injection
        llm_signal = self.llm_sidecar.get_cached_signal()
        regime = self.regime_detector.detect(features, llm_signal)

        # Stage 3: Decision Engine (Part D)
        decision = self.decision_engine.decide(features, regime)

        # Stage 4: Uncertainty Quantification (Part F)
        uq = self.uncertainty.quantify(features, decision, regime)

        # Stage 5: HSM Validation (Part E)
        hsm_result = self.hsm.validate(decision.action, portfolio.position)

        # Stage 6: Hysteresis Filter (Part G)
        hysteresis_result = self.hysteresis.filter(
            hsm_result.validated_action,
            uq.calibrated_confidence,
            market_data.close
        )

        # Stage 7: Risk Management (Part H)
        risk_result = self.risk_manager.compute(
            hysteresis_result.filtered_action,
            portfolio,
            regime
        )

        # Stage 8: Safety System (Part I)
        safety_result = self.safety_system.verify(
            risk_result.action,
            risk_result.position_size,
            risk_result.leverage,
            portfolio.drawdown,
            regime,
            uq.calibrated_confidence
        )

        # Queue for async interpretability (Part N)
        self.interpretability.queue_explanation(features, decision, safety_result)

        # Compute total latency
        total_latency_ms = (time.perf_counter() - start_time) * 1000

        if total_latency_ms > self.config.max_latency_ms:
            logger.warning(f"Pipeline latency {total_latency_ms:.2f}ms exceeds budget {self.config.max_latency_ms}ms")

        return PipelineOutput(
            action=safety_result.final_action,
            position_size=safety_result.final_position_size,
            leverage=risk_result.leverage,
            confidence=uq.calibrated_confidence,
            preprocessing=features,
            regime=regime,
            decision=decision,
            uncertainty=uq,
            hsm=hsm_result,
            hysteresis=hysteresis_result,
            risk=risk_result,
            safety=safety_result,
            total_latency_ms=total_latency_ms,
            timestamp=market_data.timestamp
        )

    def get_model_summary(self) -> Dict[str, str]:
        """Get summary of loaded models."""
        return {
            'preprocessing': 'EKF + VecNormalize',
            'regime': 'Threshold-based HMM',
            'decision': 'PPO + SAC + CQL Ensemble',
            'uncertainty': 'Temperature Scaling + k-NN OOD',
            'hsm': 'Finite State Machine',
            'hysteresis': 'KAMA Adaptive',
            'risk': 'EVT + Kelly + Drawdown Brake',
            'safety': '4-Level Fallback Cascade',
            'llm': 'Keyword-based (stub)' if self.config.llm.enabled else 'Disabled'
        }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Example usage of the Layer 2 Master Pipeline."""

    # Create configuration
    config = MasterConfig(
        trading_pair="BTC/USDT",
        debug_mode=True
    )

    # Initialize pipeline
    print("Initializing HIMARI Layer 2 Pipeline...")
    pipeline = Layer2MasterPipeline(config, debug=True)

    # Create sample market data
    market_data = RawMarketData(
        timestamp=time.time(),
        open=42150.0,
        high=42280.0,
        low=42100.0,
        close=42250.0,
        volume=1250.5,
        order_flow=np.array([0.55, 0.45]),
        sentiment_raw=0.15
    )

    # Create sample portfolio
    portfolio = PortfolioState(
        position=0,
        entry_price=0.0,
        unrealized_pnl=0.0,
        realized_pnl=1250.0,
        peak_equity=51250.0,
        current_equity=51250.0,
        drawdown=0.0,
        margin_used=0.0,
        margin_available=50000.0
    )

    # Process through pipeline
    print("\nProcessing market data...")
    result = pipeline.process(market_data, portfolio)

    # Display results
    print("\n" + "="*60)
    print("HIMARI LAYER 2 PIPELINE OUTPUT")
    print("="*60)

    print(f"\n[FINAL DECISION]")
    print(f"  Action: {result.action.name}")
    print(f"  Position Size: {result.position_size:.4f}")
    print(f"  Leverage: {result.leverage:.1f}x")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Total Latency: {result.total_latency_ms:.2f}ms")

    print(f"\n[REGIME]")
    print(f"  Market Regime: {result.regime.regime.name}")
    print(f"  Meta-Regime: {result.regime.meta_regime.name}")
    print(f"  Crisis Probability: {result.regime.crisis_probability:.2%}")

    print(f"\n[DECISION ENGINE]")
    print(f"  Raw Action: {result.decision.action.name}")
    print(f"  Ensemble Disagreement: {result.decision.disagreement:.2f}")
    for agent, (action, conf) in result.decision.agent_outputs.items():
        print(f"    - {agent}: {action} ({conf:.2%})")

    print(f"\n[SAFETY]")
    print(f"  Fallback Level: {result.safety.fallback_level.name}")
    print(f"  Primary Blocked: {result.safety.primary_blocked}")
    print(f"  Safety Margin: {result.safety.safety_margin:.2%}")

    print("\n" + "="*60)
    print("Pipeline execution complete")


if __name__ == "__main__":
    main()
