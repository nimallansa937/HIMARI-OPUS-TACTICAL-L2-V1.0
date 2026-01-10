# HIMARI Layer 2 V1 - Master Integration Guide

**Document Version:** 1.0
**Date:** January 2026
**Purpose:** Complete system integration documentation for all 14 subsystems (78 methods)
**Target Audience:** Developers implementing the full HIMARI Layer 2 trading system

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Master Architecture Overview](#2-master-architecture-overview)
3. [Interface Contracts](#3-interface-contracts)
4. [Master Pipeline Implementation](#4-master-pipeline-implementation)
5. [Data Flow Documentation](#5-data-flow-documentation)
6. [Configuration Integration](#6-configuration-integration)
7. [Complete Working Example](#7-complete-working-example)
8. [Sequence Diagrams](#8-sequence-diagrams)
9. [Integration Checklist](#9-integration-checklist)
10. [Troubleshooting Guide](#10-troubleshooting-guide)

---

## 1. Executive Summary

### System Overview

HIMARI Layer 2 V1 is a comprehensive algorithmic trading system consisting of **14 subsystems** implementing **78 methods** across the complete trading pipeline. The system processes raw market data through preprocessing, regime detection, decision making, risk management, and safety validation to produce final trading signals.

### Subsystem Summary

| Part | Name | Methods | Key Function | Latency Budget |
|------|------|---------|--------------|----------------|
| A | Preprocessing | 8 (A1-A8) | Data cleaning, denoising, augmentation | <5ms |
| B | Regime Detection | 8 (B1-B8) | Market state classification | <5ms |
| C | Multi-Timeframe Fusion | -- | Feature aggregation (implicit in D) | -- |
| D | Decision Engine | 10 (D1-D10) | Trading action generation | <30ms |
| E | HSM State Machine | 6 (E1-E6) | State validation and transitions | <1ms |
| F | Uncertainty Quantification | 8 (F1-F8) | Confidence calibration | <8ms |
| G | Hysteresis Filter | 6 (G1-G6) | Anti-whipsaw filtering | <1ms |
| H | RSS Risk Management | 8 (H1-H8) | Position sizing and risk limits | <3ms |
| I | Simplex Safety System | 8 (I1-I8) | Fallback cascade and safety | <2.5ms |
| J | LLM Integration | 8 (J1-J8) | Async sentiment analysis | Async (non-blocking) |
| K | Training Infrastructure | 8 (K1-K8) | Model training pipelines | Offline |
| L | Validation Framework | 6 (L1-L6) | Backtesting and validation | Offline |
| M | Adaptation Framework | 6 (M1-M6) | Online learning and drift detection | <5s periodic |
| N | Interpretability Framework | 4 (N1-N4) | Explainability and compliance | Async (non-blocking) |

**Total: 78 methods across 14 subsystems**

### Performance Targets

| Metric | Target | Achieved By |
|--------|--------|-------------|
| Total Latency | <50ms | Critical path optimization |
| Sharpe Ratio | >1.5 | Ensemble decision engine |
| Max Drawdown | <15% | RSS + Simplex safety |
| Whipsaw Reduction | >30% | Hysteresis filtering |
| Calibration (ECE) | <0.05 | Temperature scaling |

---

## 2. Master Architecture Overview

### 2.1 Complete System Data Flow Diagram

```
                            RAW MARKET DATA
                                  |
                    +-------------+-------------+
                    |                           |
              SYNCHRONOUS PATH              ASYNC SIDECAR
              (Critical 50ms)               (Background)
                    |                           |
                    v                           v
    +=======================================+   +=======================+
    |     PART A: PREPROCESSING             |   | PART J: LLM          |
    | +-----------------------------------+ |   | INTEGRATION          |
    | | A1: Extended Kalman Filter   <1ms | |   | +-------------------+ |
    | | A2: Conversational AE        ~2ms | |   | | J1: OPT Financial | |
    | | A3: Frequency Normalization <0.5ms| |   | | J2: Trading-R1    | |
    | | A6: VecNormalize            <0.1ms| |   | | J3: RAG + FAISS   | |
    | | A8: Online Augmentation      ~1ms | |   | | J5: FinBERT       | |
    | +-----------------------------------+ |   | +-------------------+ |
    | Output: 60-dim feature vector         |   | Output: LLM signals  |
    +=======================================+   | (cached in Redis)    |
                    |                           +=======================+
                    v                                     |
    +=======================================+             |
    |     PART B: REGIME DETECTION          |<------------+
    | +-----------------------------------+ |   (inject cached signal)
    | | B1: Student-t AH-HMM         ~1ms | |
    | | B2: Meta-Regime Layer       ~0.5ms| |
    | | B3: Causal Info Geometry     ~2ms | |
    | | B5: Jump Detector           <0.1ms| |
    | | B6: Hurst Exponent          ~0.5ms| |
    | | B8: ADWIN Drift             ~0.3ms| |
    | +-----------------------------------+ |
    | Output: RegimeOutput                  |
    |   - regime: BULL/BEAR/SIDEWAYS/CRISIS|
    |   - meta_regime: LOW/HIGH_UNCERTAINTY|
    |   - confidence: [0,1]                 |
    |   - crisis_probability: [0,1]         |
    +=======================================+
                    |
                    v
    +=======================================+
    |     PART D: DECISION ENGINE           |
    | +-----------------------------------+ |
    | | D9: Return Conditioning     <0.1ms| |
    | | D1: FLAG-TRADER (135M)       ~15ms| |
    | | D2: Critic-Guided DT         ~25ms| |
    | | D3: Conservative Q (CQL)      ~8ms| |
    | | D5: PPO-LSTM (25M)            ~5ms| |
    | | D6: SAC Agent                 ~4ms| |
    | | D7: Sharpe-Weighted Voting  <0.5ms| |
    | | D8: Disagreement Scaling    <0.5ms| |
    | +-----------------------------------+ |
    | Output: DecisionOutput                |
    |   - action: BUY/HOLD/SELL            |
    |   - confidence: [0,1]                 |
    |   - agent_outputs: Dict               |
    +=======================================+
                    |
                    v
    +=======================================+
    |     PART F: UNCERTAINTY QUANTIF.      |
    | +-----------------------------------+ |
    | | F1: CT-SSF Conformal         ~5ms | |
    | | F2: CPTC Regime-Aware        ~2ms | |
    | | F3: Temperature Scaling     <0.1ms| |
    | | F4: Deep Ensemble           ~15ms | |
    | | F5: MC Dropout               ~3ms | |
    | | F6: Epistemic/Aleatoric      ~1ms | |
    | | F7: k-NN OOD Detection       ~2ms | |
    | +-----------------------------------+ |
    | Output: UncertaintyOutput             |
    |   - calibrated_confidence: [0,1]      |
    |   - prediction_interval: (lo, hi)     |
    |   - epistemic_uncertainty: [0,1]      |
    |   - ood_score: [0,1]                  |
    +=======================================+
                    |
                    v
    +=======================================+
    |     PART E: HSM STATE MACHINE         |
    | +-----------------------------------+ |
    | | E1: Orthogonal Regions      <0.1ms| |
    | | E2: Hierarchical Nesting    <0.1ms| |
    | | E3: History States          <0.1ms| |
    | | E4: Synchronized Events     <0.1ms| |
    | | E5: Learned Transitions      ~1ms | |
    | | E6: Oscillation Detection   <0.1ms| |
    | +-----------------------------------+ |
    | Output: ValidatedAction               |
    |   - action: validated BUY/HOLD/SELL  |
    |   - is_valid: bool                    |
    |   - current_state: PositionState      |
    +=======================================+
                    |
                    v
    +=======================================+
    |     PART G: HYSTERESIS FILTER         |
    | +-----------------------------------+ |
    | | G1: KAMA Adaptive           ~0.1ms| |
    | | G2: KNN Pattern Match       ~0.2ms| |
    | | G3: ATR-Scaled Bands       ~0.05ms| |
    | | G4: Meta-Learned k         ~0.05ms| |
    | | G5: 2.2x Loss Aversion    ~0.02ms| |
    | | G6: Whipsaw Learning       ~0.08ms| |
    | +-----------------------------------+ |
    | Output: FilteredAction                |
    |   - action: filtered BUY/HOLD/SELL   |
    |   - entry_threshold: float            |
    |   - exit_threshold: float             |
    +=======================================+
                    |
                    v
    +=======================================+
    |     PART H: RSS RISK MANAGEMENT       |
    | +-----------------------------------+ |
    | | H1: EVT + GPD Tail Risk     ~0.3ms| |
    | | H2: DDPG-TiDE Kelly         ~0.5ms| |
    | | H3: DCC-GARCH Correlation   ~0.4ms| |
    | | H4: Progressive DD Brake    ~0.1ms| |
    | | H5: Portfolio VaR           ~0.6ms| |
    | | H6: Safe Margin Formula     ~0.2ms| |
    | | H7: Dynamic Leverage        ~0.2ms| |
    | | H8: Adaptive Risk Budget    ~0.2ms| |
    | +-----------------------------------+ |
    | Output: RiskAdjustedAction            |
    |   - action: BUY/HOLD/SELL            |
    |   - position_size: float              |
    |   - leverage: float                   |
    |   - var_99: float                     |
    +=======================================+
                    |
                    v
    +=======================================+
    |     PART I: SIMPLEX SAFETY SYSTEM     |
    | +-----------------------------------+ |
    | | I1: 4-Level Fallback        ~0.8ms| |
    | | I2: Predictive Safety       ~0.5ms| |
    | | I3: Formal Verification     ~0.2ms| |
    | | I4: Reachability Analysis   ~0.2ms| |
    | | I5: Safety Invariants       ~0.3ms| |
    | | I6: Safety Monitor          ~0.1ms| |
    | | I7: Stop-Loss Enforcer      ~0.1ms| |
    | | I8: Recovery Protocol       ~0.1ms| |
    | +-----------------------------------+ |
    | Output: SafeAction                    |
    |   - action: FINAL BUY/HOLD/SELL      |
    |   - fallback_level: 0-3               |
    |   - safety_margin: float              |
    +=======================================+
                    |
                    v
              FINAL OUTPUT
              - action: int (-1, 0, +1)
              - position_size: float
              - confidence: float
              - metadata: Dict

                    |
                    v (async logging)
    +=======================================+
    |     PART N: INTERPRETABILITY          |
    | +-----------------------------------+ |
    | | N1: SHAP Attribution              | |
    | | N2: DiCE Counterfactual           | |
    | | N3: MiFID II Compliance           | |
    | | N4: Attention Visualization       | |
    | +-----------------------------------+ |
    | Output: Explanation (async)           |
    +=======================================+


            OFFLINE TRAINING PIPELINE
            (Not in critical path)

    +=======================================+
    |     PART K: TRAINING INFRASTRUCTURE   |
    | +-----------------------------------+ |
    | | K1: 3-Stage Curriculum            | |
    | | K2: MAML Meta-Learning            | |
    | | K3: Causal Data Augmentation      | |
    | | K4: Multi-Task Learning           | |
    | | K5: Adversarial Training          | |
    | | K6: FGSM/PGD Robustness           | |
    | | K7: Reward Shaping                | |
    | | K8: Rare Event Synthesis          | |
    | +-----------------------------------+ |
    +=======================================+
                    |
                    v
    +=======================================+
    |     PART L: VALIDATION FRAMEWORK      |
    | +-----------------------------------+ |
    | | L1: CPCV Cross-Validation         | |
    | | L2: Walk-Forward Optimization     | |
    | | L3: LOBFrame Market Simulation    | |
    | | L4: Statistical Significance      | |
    | | L5: Out-of-Sample Regime Test     | |
    | | L6: Production Shadow Testing     | |
    | +-----------------------------------+ |
    +=======================================+
                    |
                    v
    +=======================================+
    |     PART M: ADAPTATION FRAMEWORK      |
    | +-----------------------------------+ |
    | | M1: Adaptive Memory Realignment   | |
    | | M2: Shadow A/B Testing            | |
    | | M3: Multi-Timescale Learning      | |
    | | M4: EWC + Progressive NNs         | |
    | | M5: Concept Drift Detection       | |
    | | M6: Incremental Updates           | |
    | +-----------------------------------+ |
    +=======================================+
```

### 2.2 Synchronous vs Asynchronous Components

#### Critical Path (Synchronous - Must complete in <50ms)
1. **Part A**: Preprocessing (EKF, CAE, VecNormalize)
2. **Part B**: Regime Detection (HMM, Meta-Regime)
3. **Part D**: Decision Engine (Ensemble agents)
4. **Part F**: Uncertainty Quantification
5. **Part E**: HSM State Validation
6. **Part G**: Hysteresis Filter
7. **Part H**: RSS Risk Management
8. **Part I**: Simplex Safety System

#### Background Processing (Asynchronous)
1. **Part J**: LLM Integration - Runs continuously, caches signals
2. **Part N**: Interpretability - Generates explanations post-decision

#### Offline Processing
1. **Part K**: Training Infrastructure - Model retraining
2. **Part L**: Validation Framework - Backtesting
3. **Part M**: Adaptation Framework - Periodic model updates

### 2.3 Critical Decision Paths

```
NORMAL OPERATION FLOW:
Features -> Regime -> Decision -> UQ -> HSM -> Hysteresis -> Risk -> Safety -> Execute

CRISIS MODE FLOW (Fast Path):
Jump Detected -> Crisis Regime -> Force HOLD -> Skip ensemble -> Immediate safety check

REGIME CHANGE FLOW:
Regime transition warning -> Reduce confidence -> Widen hysteresis -> Scale down positions

LLM INJECTION FLOW:
Background: News -> LLM -> Parse -> Cache
Main path: Read cache -> Inject into features -> Continue normal flow
```

---

## 3. Interface Contracts

### 3.1 Part A: Preprocessing

**Input:**
```python
@dataclass
class RawMarketData:
    timestamp: float            # Unix timestamp
    open: float                 # OHLCV
    high: float
    low: float
    close: float
    volume: float
    order_flow: Optional[np.ndarray]    # Bid/ask imbalance
    sentiment_raw: Optional[float]       # -1 to 1
    on_chain: Optional[Dict[str, float]] # On-chain metrics
```

**Output:**
```python
@dataclass
class PreprocessedFeatures:
    features: np.ndarray        # Shape: (60,) - normalized feature vector
    ekf_state: Dict[str, float] # price, velocity, acceleration, volatility
    uncertainty: float          # EKF state uncertainty
    cae_ambiguity: float        # Consensus ambiguity from autoencoders
    timestamp: float
```

### 3.2 Part B: Regime Detection

**Input:** `PreprocessedFeatures`

**Output:**
```python
@dataclass
class RegimeOutput:
    regime: MarketRegime        # BULL, BEAR, SIDEWAYS, CRISIS
    meta_regime: MetaRegime     # LOW_UNCERTAINTY, HIGH_UNCERTAINTY
    probabilities: np.ndarray   # Shape: (4,) - posterior over regimes
    confidence: float           # Max probability
    transition_warning: bool    # True if regime change likely
    crisis_probability: float   # P(Crisis) specifically
    hurst_exponent: float       # Trend vs mean-reversion
```

### 3.3 Part D: Decision Engine

**Input:**
```python
@dataclass
class DecisionInput:
    features: np.ndarray        # 60-dim or 256-dim embedding
    regime: RegimeOutput
    target_sharpe: float        # Return conditioning target
    llm_signal: Optional[LLMSignal]  # From Part J cache
```

**Output:**
```python
@dataclass
class DecisionOutput:
    action: TradeAction         # BUY, HOLD, SELL (-1, 0, 1)
    confidence: float           # Ensemble confidence
    agent_outputs: Dict[str, Tuple[int, float]]  # Per-agent decisions
    disagreement: float         # Ensemble disagreement metric
    ensemble_weights: Dict[str, float]  # Current Sharpe-weighted votes
```

### 3.4 Part E: HSM State Machine

**Input:**
```python
@dataclass
class HSMInput:
    proposed_action: TradeAction
    current_position: int       # -1, 0, 1
    regime: RegimeOutput
    timestamp: float
```

**Output:**
```python
@dataclass
class HSMOutput:
    validated_action: TradeAction
    is_valid: bool
    position_state: PositionState   # FLAT, LONG_ENTRY, LONG_HOLD, etc.
    regime_state: RegimeState
    blocked_reason: Optional[str]
    oscillation_blocked: bool
```

### 3.5 Part F: Uncertainty Quantification

**Input:**
```python
@dataclass
class UQInput:
    features: np.ndarray
    decision: DecisionOutput
    regime: RegimeOutput
    model_outputs: Dict         # Raw neural network outputs
```

**Output:**
```python
@dataclass
class UQOutput:
    calibrated_confidence: float    # Temperature-scaled
    prediction_interval: Tuple[float, float]  # Conformal interval
    epistemic_uncertainty: float    # Model uncertainty
    aleatoric_uncertainty: float    # Data uncertainty
    ood_score: float               # Out-of-distribution score
    total_uncertainty: float       # Combined measure
```

### 3.6 Part G: Hysteresis Filter

**Input:**
```python
@dataclass
class HysteresisInput:
    action: TradeAction
    confidence: float
    current_position: int
    price: float
    regime: RegimeOutput
```

**Output:**
```python
@dataclass
class HysteresisOutput:
    filtered_action: TradeAction
    entry_threshold: float
    exit_threshold: float
    efficiency_ratio: float
    was_filtered: bool
    filter_reason: Optional[str]
```

### 3.7 Part H: RSS Risk Management

**Input:**
```python
@dataclass
class RiskInput:
    action: TradeAction
    confidence: float
    position_size_proposed: float
    current_portfolio: PortfolioState
    volatility: float
    regime: RegimeOutput
    correlation_matrix: Optional[np.ndarray]
```

**Output:**
```python
@dataclass
class RiskOutput:
    action: TradeAction
    position_size: float        # Risk-adjusted size
    leverage: float             # Dynamic leverage
    var_95: float              # 95% VaR
    var_99: float              # 99% VaR (EVT)
    expected_shortfall: float
    kelly_fraction: float
    drawdown_brake_active: bool
    risk_budget_remaining: float
```

### 3.8 Part I: Simplex Safety System

**Input:**
```python
@dataclass
class SafetyInput:
    action: TradeAction
    position_size: float
    leverage: float
    current_position: int
    drawdown: float
    regime: RegimeOutput
    risk_output: RiskOutput
```

**Output:**
```python
@dataclass
class SafetyOutput:
    final_action: TradeAction
    final_position_size: float
    fallback_level: FallbackLevel   # PRIMARY, BASELINE, CONSERVATIVE, MINIMAL
    primary_blocked: bool
    block_reason: str
    safety_margin: float
    stop_loss_triggered: bool
    recovery_status: str
```

### 3.9 Part J: LLM Integration (Async)

**Input:**
```python
@dataclass
class LLMInput:
    text: str                   # News, tweet, report
    source: str                 # "twitter", "news", "sec_filing"
    timestamp: float
```

**Output:**
```python
@dataclass
class LLMSignal:
    sentiment: float            # -1 to +1
    confidence: float           # 0 to 1
    direction: int              # -1, 0, +1
    reasoning: str
    source_summary: str
    timestamp: float
    latency_ms: float
    calibrated: bool            # Post J4 calibration
```

### 3.10 Complete Pipeline Interface

```python
@dataclass
class PipelineInput:
    """Complete input to Layer 2 pipeline."""
    market_data: RawMarketData
    portfolio_state: PortfolioState
    llm_cache: Optional[LLMSignal]

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
```

---

## 4. Master Pipeline Implementation

The complete master pipeline implementation is provided in `src/layer2_master_pipeline.py`.

### 4.1 Pipeline Architecture

```python
class Layer2MasterPipeline:
    """
    Master pipeline orchestrating all 14 subsystems.

    Manages:
    - Initialization of all components
    - Data flow between components
    - Error handling and fallbacks
    - Performance monitoring
    - Async LLM integration
    """
```

### 4.2 Key Implementation Patterns

#### Initialization Order
```python
def __init__(self, config: MasterConfig):
    # 1. Initialize configuration
    self.config = config

    # 2. Initialize preprocessing (Part A)
    self.preprocessor = PreprocessingPipeline(config.preprocessing)

    # 3. Initialize regime detection (Part B)
    self.regime_detector = RegimeDetector(config.regime)

    # 4. Initialize decision engine (Part D)
    self.decision_engine = DecisionEngine(config.decision)

    # 5. Initialize uncertainty quantification (Part F)
    self.uncertainty = UncertaintyQuantifier(config.uncertainty)

    # 6. Initialize HSM (Part E)
    self.hsm = TradingHSM(config.hsm)

    # 7. Initialize hysteresis filter (Part G)
    self.hysteresis = HysteresisFilter(config.hysteresis)

    # 8. Initialize risk management (Part H)
    self.risk_manager = RSSRiskManager(config.risk)

    # 9. Initialize safety system (Part I)
    self.safety_system = SimplexSafetySystem(config.safety)

    # 10. Initialize async LLM sidecar (Part J)
    self.llm_sidecar = LLMSidecar(config.llm)

    # 11. Initialize interpretability (Part N) - async
    self.interpretability = InterpretabilityEngine(config.interpretability)
```

#### Processing Flow
```python
def process(self, market_data: RawMarketData) -> PipelineOutput:
    start_time = time.perf_counter()

    # Stage 1: Preprocessing
    features = self.preprocessor.process(market_data)

    # Stage 2: Regime Detection (with LLM injection)
    llm_signal = self.llm_sidecar.get_cached_signal()
    regime = self.regime_detector.detect(features, llm_signal)

    # Stage 3: Decision Engine
    decision = self.decision_engine.decide(features, regime)

    # Stage 4: Uncertainty Quantification
    uq = self.uncertainty.quantify(features, decision, regime)

    # Stage 5: HSM Validation
    hsm_result = self.hsm.validate(decision.action, self.portfolio.position)

    # Stage 6: Hysteresis Filter
    filtered = self.hysteresis.filter(
        hsm_result.validated_action,
        uq.calibrated_confidence,
        market_data.close
    )

    # Stage 7: Risk Management
    risk = self.risk_manager.compute(
        filtered.filtered_action,
        self.portfolio,
        regime
    )

    # Stage 8: Safety System
    safe = self.safety_system.verify(
        risk.action,
        risk.position_size,
        self.portfolio.drawdown,
        regime
    )

    # Async: Queue for interpretability
    self.interpretability.queue_explanation(features, decision, safe)

    return PipelineOutput(
        action=safe.final_action,
        position_size=safe.final_position_size,
        confidence=uq.calibrated_confidence,
        total_latency_ms=(time.perf_counter() - start_time) * 1000,
        ...
    )
```

---

## 5. Data Flow Documentation

### 5.1 Step-by-Step Data Trace

#### Step 1: Raw Data Ingestion
```
Input: OHLCV bar from exchange API
       {timestamp, open, high, low, close, volume}

Processing:
  - Validate data integrity
  - Check for missing values
  - Convert to internal format

Output: RawMarketData object
```

#### Step 2: Preprocessing (Part A)
```
Input: RawMarketData

Processing:
  A1: EKF denoises price, extracts momentum/acceleration
      - State: [price, velocity, acceleration, volatility]
      - Output: denoised_price, uncertainty_estimate

  A2: CAE speaker-listener consensus
      - Two autoencoders must agree on representation
      - Output: consensus_features, ambiguity_score

  A3: Frequency domain normalization
      - FFT -> normalize spectra -> IFFT
      - Handles non-stationarity in frequency content

  A6: VecNormalize
      - Running mean/std Z-score normalization
      - Stable-Baselines3 compatible

Output: PreprocessedFeatures
        - features: np.ndarray (60,)
        - ekf_state: Dict
        - uncertainty: float
```

#### Step 3: Regime Detection (Part B)
```
Input: PreprocessedFeatures

Processing:
  B5: Jump detector (fast path)
      - Check if |return| > 3 sigma
      - If crisis: short-circuit to CRISIS regime

  B1: Student-t HMM forward step
      - Compute emission log-probs (Student-t, not Gaussian)
      - Forward algorithm: P(regime | observations)
      - Output: regime_probabilities (4,)

  B2: Meta-regime layer
      - Check VIX/DVOL proxy, EPU indicators
      - Classify: LOW_UNCERTAINTY or HIGH_UNCERTAINTY
      - Modifies transition matrix for B1

  B6: Hurst exponent
      - R/S analysis or DFA
      - H > 0.5: trending, H < 0.5: mean-reverting

Output: RegimeOutput
        - regime: BULL/BEAR/SIDEWAYS/CRISIS
        - confidence: float
        - crisis_probability: float
```

#### Step 4: Decision Engine (Part D)
```
Input: Features + RegimeOutput

Processing:
  D9: Prepend target_sharpe to observation
      - Based on regime: crisis=0.5, bear=1.0, sideways=2.0, bull=2.5

  Parallel execution of agents:
    D1: FLAG-TRADER (LLM backbone) -> action, confidence
    D2: CGDT (Decision Transformer) -> action, confidence
    D3: CQL (Conservative Q) -> action, confidence
    D5: PPO-LSTM -> action, confidence
    D6: SAC -> action, confidence

  D7: Sharpe-weighted voting
      - weights = rolling_sharpe / sum(rolling_sharpes)
      - combined_probs = sum(weight_i * prob_i)

  D8: Disagreement scaling
      - disagreement = variance of agent actions
      - if disagreement > 0.7: reduce confidence by 75%

Output: DecisionOutput
        - action: BUY/HOLD/SELL
        - confidence: float (0-1)
        - disagreement: float
```

#### Step 5: Uncertainty Quantification (Part F)
```
Input: Features + DecisionOutput + RegimeOutput

Processing:
  F4: Deep ensemble disagreement
      - variance across N ensemble models

  F5: MC Dropout
      - 10-30 forward passes with dropout enabled
      - epistemic_uncertainty = variance

  F1: CT-SSF conformal prediction
      - Latent-space non-conformity scores
      - Generate prediction interval

  F2: CPTC regime-aware
      - Expand intervals during regime transitions

  F3: Temperature scaling
      - Calibrate: confidence = softmax(logits/T)
      - T learned on validation set

Output: UQOutput
        - calibrated_confidence: float
        - epistemic_uncertainty: float
        - ood_score: float
```

#### Step 6: HSM Validation (Part E)
```
Input: ProposedAction + CurrentPosition

Processing:
  E1: Check position region
      - Is transition valid? (FLAT -> LONG_ENTRY? etc.)

  E2: Check regime region
      - Is action consistent with regime?

  E6: Oscillation detection
      - Has position flipped > N times in M bars?
      - If yes: block transition

  E5: Learned transition timing
      - ML model predicts if now is good time to transition

Output: HSMOutput
        - validated_action: TradeAction
        - is_valid: bool
        - blocked_reason: Optional[str]
```

#### Step 7: Hysteresis Filter (Part G)
```
Input: ValidatedAction + Confidence + Price

Processing:
  G1: KAMA adaptive thresholds
      - Efficiency ratio -> threshold interpolation
      - High ER: tight thresholds
      - Low ER: wide thresholds

  G3: ATR-scaled bands
      - entry_thresh *= (1 + ATR_ratio * scale)

  G5: 2.2x loss aversion
      - Exit threshold = entry_threshold / 2.2
      - Asymmetric: harder to exit than enter

  Decision:
      if current_position == 0:
          if confidence >= entry_threshold:
              allow entry
          else:
              force HOLD
      else:
          if confidence < exit_threshold:
              allow exit
          else:
              maintain position

Output: HysteresisOutput
        - filtered_action: TradeAction
        - entry_threshold, exit_threshold: float
```

#### Step 8: Risk Management (Part H)
```
Input: FilteredAction + Portfolio + Regime

Processing:
  H1: EVT tail risk
      - VaR_99 = GPD quantile (not Gaussian!)
      - ES_99 = expected shortfall

  H2: Dynamic Kelly fraction
      - f* = edge / variance
      - scale by regime: crisis -> 0.25x, bull -> 1.0x

  H4: Drawdown brake
      - if drawdown > 5%: reduce position by 25%
      - if drawdown > 8%: reduce by 50%
      - if drawdown > 10%: force minimal position

  H6: Safe margin formula
      - margin_required = position * price * (1/leverage)
      - ensure margin > k * VaR (k=3)

  H7: Dynamic leverage
      - small positions: up to 10x
      - large positions: decay to 3x

Output: RiskOutput
        - position_size: float (risk-adjusted)
        - leverage: float
        - var_99: float
```

#### Step 9: Safety System (Part I)
```
Input: RiskAdjustedAction + Drawdown + Regime

Processing:
  I5: Check safety invariants
      - leverage <= max_leverage
      - position_size * price <= max_notional
      - margin_remaining >= min_margin

  I2: Predictive safety (N-step)
      - Simulate N future steps
      - Check if invariants could be violated

  I1: Fallback cascade
      if primary_fails_check:
          try baseline (PPO-LSTM)
      if baseline_fails:
          try conservative (trend-following)
      if conservative_fails:
          use HOLD

  I7: Stop-loss enforcer
      - if daily_loss > max_daily_loss:
          halt all trading

Output: SafetyOutput
        - final_action: TradeAction
        - fallback_level: 0-3
        - safety_margin: float
```

### 5.2 How Regime Detection Affects Downstream

```
Regime = CRISIS:
  - Decision Engine: D9 sets target_sharpe = 0.5 (conservative)
  - Decision Engine: D3 (CQL) weight increases
  - HSM: Blocks new position entries
  - Hysteresis: Widens thresholds by 2x
  - Risk: Kelly fraction *= 0.25
  - Risk: Force drawdown brake
  - Safety: Force conservative fallback level

Regime = BULL:
  - Decision Engine: D9 sets target_sharpe = 2.5 (aggressive)
  - Decision Engine: D1 (FLAG-TRADER) weight increases
  - Hysteresis: Tightens thresholds (capture momentum)
  - Risk: Full Kelly allowed
  - Safety: Primary controller allowed

Meta-Regime = HIGH_UNCERTAINTY:
  - Regime transitions more likely
  - All position sizes reduced by 30%
  - Wider prediction intervals from F2
```

### 5.3 How Safety Layers Intercept and Modify

```
Layer 1: HSM (Part E)
  - Blocks invalid transitions (e.g., FLAT -> LONG_EXIT)
  - Blocks oscillation (frequent flipping)
  - Does NOT modify size, only action

Layer 2: Hysteresis (Part G)
  - Blocks low-confidence entries
  - Maintains positions despite noise
  - Modifies action to HOLD if below threshold

Layer 3: Risk Management (Part H)
  - Scales down position size based on:
    - Tail risk (EVT VaR)
    - Drawdown
    - Kelly fraction
  - Adjusts leverage

Layer 4: Safety System (Part I)
  - Final gate: verifies all invariants
  - Can override to safer fallback
  - Emergency stop-loss

Order matters: Each layer can only make things SAFER, never riskier.
```

---

## 6. Configuration Integration

### 6.1 Master Configuration Structure

```python
@dataclass
class MasterConfig:
    """Master configuration composing all subsystem configs."""

    # System-wide settings
    trading_pair: str = "BTC/USDT"
    base_currency: str = "USDT"
    max_latency_ms: float = 50.0

    # Subsystem configurations
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
```

### 6.2 Subsystem Config Examples

```python
@dataclass
class PreprocessingConfig:
    # Part A configuration
    ekf_process_noise: float = 0.001
    ekf_measurement_noise: float = 0.01
    ekf_use_faux_riccati: bool = True
    cae_latent_dim: int = 32
    cae_consensus_threshold: float = 0.8
    vecnormalize_clip: float = 10.0

@dataclass
class RegimeConfig:
    # Part B configuration
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
    # Part D configuration
    flag_trader_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    ppo_hidden_dim: int = 512
    sac_gamma: float = 0.99
    ensemble_lookback_days: int = 30
    disagreement_high_threshold: float = 0.7

@dataclass
class RiskConfig:
    # Part H configuration
    evt_threshold_percentile: float = 95.0
    evt_min_exceedances: int = 30
    kelly_fraction_cap: float = 0.25
    max_leverage: float = 10.0
    drawdown_brake_levels: List[float] = field(
        default_factory=lambda: [0.05, 0.08, 0.10]
    )

@dataclass
class SafetyConfig:
    # Part I configuration
    primary_confidence_threshold: float = 0.6
    baseline_confidence_threshold: float = 0.5
    crisis_force_conservative: bool = True
    drawdown_force_minimal: float = 0.08
    max_daily_loss: float = 0.05
```

### 6.3 Configuration Override Mechanisms

```python
# Method 1: YAML file override
config = MasterConfig.from_yaml("config/production.yaml")

# Method 2: Environment variables
config.risk.max_leverage = float(os.getenv("HIMARI_MAX_LEVERAGE", 10.0))

# Method 3: Runtime override
config.override({
    "risk.kelly_fraction_cap": 0.15,  # More conservative
    "safety.crisis_force_conservative": True
})

# Method 4: Regime-dependent override
config.apply_regime_overrides(current_regime="crisis", {
    "risk.max_leverage": 3.0,
    "decision.ensemble_weights": {"cql": 0.5, "flag_trader": 0.1}
})
```

### 6.4 Master Config File Template

```yaml
# config/master_config.yaml

system:
  trading_pair: "BTC/USDT"
  base_currency: "USDT"
  max_latency_ms: 50.0
  log_level: "INFO"

preprocessing:
  ekf:
    process_noise: 0.001
    measurement_noise: 0.01
    use_faux_riccati: true
  cae:
    latent_dim: 32
    consensus_threshold: 0.8
  vecnormalize:
    clip: 10.0

regime:
  hmm:
    n_states: 4
    student_t_df: 5.0
    crisis_df: 3.0
  meta_regime:
    vix_high: 30.0
    vix_low: 20.0
  jump_detector:
    threshold_sigma: 3.0

decision:
  flag_trader:
    model: "HuggingFaceTB/SmolLM2-135M-Instruct"
    lora_r: 16
    lora_alpha: 32
  ensemble:
    lookback_days: 30
    disagreement_threshold: 0.7
  return_conditioning:
    crisis_target: 0.5
    bull_target: 2.5

uncertainty:
  conformal:
    alpha: 0.10
    calibration_samples: 500
  temperature_scaling:
    enabled: true
  mc_dropout:
    n_samples: 20

hsm:
  position:
    initial_state: "FLAT"
  oscillation:
    window_bars: 20
    max_flips: 3

hysteresis:
  kama:
    er_period: 10
    fast_threshold: 0.25
    slow_threshold: 0.45
  loss_aversion_ratio: 2.2

risk:
  evt:
    threshold_percentile: 95.0
    min_exceedances: 30
  kelly:
    fraction_cap: 0.25
  drawdown:
    brake_levels: [0.05, 0.08, 0.10]
    brake_reductions: [0.25, 0.50, 0.90]
  leverage:
    max: 10.0
    decay_start_size: 0.1

safety:
  fallback:
    primary_threshold: 0.6
    baseline_threshold: 0.5
  crisis:
    force_conservative: true
  stop_loss:
    max_daily_loss: 0.05
    halt_duration_hours: 24

llm:
  model: "facebook/opt-1.3b"
  cache_ttl_seconds: 300
  async:
    queue_size: 100
    workers: 2

interpretability:
  shap:
    background_samples: 100
    top_k_features: 10
  mifid:
    enabled: true
    report_frequency: "daily"
```

---

## 7. Complete Working Example

### 7.1 End-to-End Code Example

```python
"""
Complete end-to-end example of HIMARI Layer 2 pipeline.

This example demonstrates:
1. System initialization
2. Processing a single market data bar
3. Getting final trading decision
4. Interpreting results
"""

import numpy as np
from datetime import datetime
from layer2_master_pipeline import (
    Layer2MasterPipeline,
    MasterConfig,
    RawMarketData,
    PortfolioState
)

# ============================================================================
# STEP 1: Initialize the pipeline
# ============================================================================

# Load configuration
config = MasterConfig.from_yaml("config/production.yaml")

# Create pipeline (loads all models, ~30 seconds on first run)
pipeline = Layer2MasterPipeline(config)

print("Pipeline initialized successfully")
print(f"Models loaded: {pipeline.get_model_summary()}")

# ============================================================================
# STEP 2: Prepare sample market data
# ============================================================================

# Simulated 5-minute bar data
market_data = RawMarketData(
    timestamp=datetime.now().timestamp(),
    open=42150.0,
    high=42280.0,
    low=42100.0,
    close=42250.0,
    volume=1250.5,
    order_flow=np.array([0.55, 0.45]),  # Buy/sell imbalance
    sentiment_raw=0.15,  # Slightly bullish
    on_chain={
        "exchange_netflow": -1500.0,  # Outflow (bullish)
        "whale_transactions": 12
    }
)

# Current portfolio state
portfolio = PortfolioState(
    position=0,  # Currently flat
    entry_price=0.0,
    unrealized_pnl=0.0,
    realized_pnl=1250.0,  # Cumulative
    peak_equity=51250.0,
    current_equity=51250.0,
    drawdown=0.0,
    margin_used=0.0,
    margin_available=50000.0
)

# ============================================================================
# STEP 3: Process through pipeline
# ============================================================================

result = pipeline.process(market_data, portfolio)

# ============================================================================
# STEP 4: Examine results at each stage
# ============================================================================

print("\n" + "="*60)
print("HIMARI LAYER 2 PIPELINE OUTPUT")
print("="*60)

# Final decision
print(f"\n[FINAL DECISION]")
print(f"  Action: {result.action.name}")
print(f"  Position Size: {result.position_size:.4f}")
print(f"  Confidence: {result.confidence:.2%}")
print(f"  Total Latency: {result.total_latency_ms:.2f}ms")

# Preprocessing output
print(f"\n[PREPROCESSING - Part A]")
print(f"  Feature vector shape: {result.preprocessing.features.shape}")
print(f"  EKF momentum: {result.preprocessing.ekf_state['velocity']:.4f}")
print(f"  EKF volatility: {result.preprocessing.ekf_state['volatility']:.4f}")
print(f"  CAE ambiguity: {result.preprocessing.cae_ambiguity:.2f}")

# Regime detection
print(f"\n[REGIME DETECTION - Part B]")
print(f"  Regime: {result.regime.regime.name}")
print(f"  Meta-regime: {result.regime.meta_regime.name}")
print(f"  Confidence: {result.regime.confidence:.2%}")
print(f"  Crisis probability: {result.regime.crisis_probability:.2%}")
print(f"  Transition warning: {result.regime.transition_warning}")

# Decision engine
print(f"\n[DECISION ENGINE - Part D]")
print(f"  Raw action: {result.decision.action.name}")
print(f"  Ensemble confidence: {result.decision.confidence:.2%}")
print(f"  Disagreement: {result.decision.disagreement:.2f}")
print(f"  Agent votes:")
for agent, (action, conf) in result.decision.agent_outputs.items():
    print(f"    - {agent}: {action} ({conf:.2%})")

# Uncertainty quantification
print(f"\n[UNCERTAINTY - Part F]")
print(f"  Calibrated confidence: {result.uncertainty.calibrated_confidence:.2%}")
print(f"  Epistemic UQ: {result.uncertainty.epistemic_uncertainty:.3f}")
print(f"  OOD score: {result.uncertainty.ood_score:.3f}")
print(f"  Prediction interval: [{result.uncertainty.prediction_interval[0]:.2%}, "
      f"{result.uncertainty.prediction_interval[1]:.2%}]")

# HSM validation
print(f"\n[HSM STATE MACHINE - Part E]")
print(f"  Position state: {result.hsm.position_state.name}")
print(f"  Action valid: {result.hsm.is_valid}")
print(f"  Oscillation blocked: {result.hsm.oscillation_blocked}")
if result.hsm.blocked_reason:
    print(f"  Block reason: {result.hsm.blocked_reason}")

# Hysteresis filter
print(f"\n[HYSTERESIS FILTER - Part G]")
print(f"  Filtered action: {result.hysteresis.filtered_action.name}")
print(f"  Entry threshold: {result.hysteresis.entry_threshold:.3f}")
print(f"  Exit threshold: {result.hysteresis.exit_threshold:.3f}")
print(f"  Efficiency ratio: {result.hysteresis.efficiency_ratio:.3f}")
print(f"  Was filtered: {result.hysteresis.was_filtered}")

# Risk management
print(f"\n[RISK MANAGEMENT - Part H]")
print(f"  Position size: {result.risk.position_size:.4f}")
print(f"  Leverage: {result.risk.leverage:.1f}x")
print(f"  VaR 99%: {result.risk.var_99:.2%}")
print(f"  Expected Shortfall: {result.risk.expected_shortfall:.2%}")
print(f"  Kelly fraction: {result.risk.kelly_fraction:.3f}")
print(f"  Drawdown brake active: {result.risk.drawdown_brake_active}")

# Safety system
print(f"\n[SAFETY SYSTEM - Part I]")
print(f"  Final action: {result.safety.final_action.name}")
print(f"  Fallback level: {result.safety.fallback_level.name}")
print(f"  Primary blocked: {result.safety.primary_blocked}")
if result.safety.primary_blocked:
    print(f"  Block reason: {result.safety.block_reason}")
print(f"  Safety margin: {result.safety.safety_margin:.2%}")
print(f"  Stop-loss triggered: {result.safety.stop_loss_triggered}")

print("\n" + "="*60)
```

### 7.2 Sample Data Structures at Each Stage

```python
# Stage 1: Raw Market Data Input
raw_data = {
    "timestamp": 1704067200.0,
    "open": 42150.0,
    "high": 42280.0,
    "low": 42100.0,
    "close": 42250.0,
    "volume": 1250.5
}

# Stage 2: Preprocessed Features (Part A output)
preprocessed = {
    "features": np.array([0.02, -0.15, 0.45, ...]),  # 60 dims
    "ekf_state": {
        "price": 42248.5,
        "velocity": 0.0015,
        "acceleration": -0.0002,
        "volatility": 0.018
    },
    "uncertainty": 0.0012,
    "cae_ambiguity": 0.15
}

# Stage 3: Regime Output (Part B output)
regime = {
    "regime": "BULL",  # MarketRegime enum
    "meta_regime": "LOW_UNCERTAINTY",
    "probabilities": [0.65, 0.15, 0.18, 0.02],  # [bull, bear, sideways, crisis]
    "confidence": 0.65,
    "transition_warning": False,
    "crisis_probability": 0.02,
    "hurst_exponent": 0.62
}

# Stage 4: Decision Output (Part D output)
decision = {
    "action": "BUY",  # TradeAction enum
    "confidence": 0.72,
    "agent_outputs": {
        "flag_trader": (1, 0.75),
        "cgdt": (1, 0.68),
        "cql": (0, 0.55),
        "ppo": (1, 0.70),
        "sac": (1, 0.65)
    },
    "disagreement": 0.25,
    "ensemble_weights": {
        "flag_trader": 0.28,
        "cgdt": 0.22,
        "cql": 0.15,
        "ppo": 0.18,
        "sac": 0.17
    }
}

# Stage 5: Uncertainty Output (Part F output)
uncertainty = {
    "calibrated_confidence": 0.68,
    "prediction_interval": (-0.015, 0.025),
    "epistemic_uncertainty": 0.12,
    "aleatoric_uncertainty": 0.08,
    "ood_score": 0.05,
    "total_uncertainty": 0.145
}

# Stage 6: HSM Output (Part E output)
hsm = {
    "validated_action": "BUY",
    "is_valid": True,
    "position_state": "FLAT",
    "regime_state": "TRENDING_UP",
    "blocked_reason": None,
    "oscillation_blocked": False
}

# Stage 7: Hysteresis Output (Part G output)
hysteresis = {
    "filtered_action": "BUY",
    "entry_threshold": 0.32,
    "exit_threshold": 0.14,
    "efficiency_ratio": 0.78,
    "was_filtered": False,
    "filter_reason": None
}

# Stage 8: Risk Output (Part H output)
risk = {
    "action": "BUY",
    "position_size": 0.085,  # 8.5% of capital
    "leverage": 5.0,
    "var_95": 0.032,
    "var_99": 0.058,
    "expected_shortfall": 0.072,
    "kelly_fraction": 0.12,
    "drawdown_brake_active": False,
    "risk_budget_remaining": 0.85
}

# Stage 9: Safety Output (Part I output)
safety = {
    "final_action": "BUY",
    "final_position_size": 0.085,
    "fallback_level": "PRIMARY",
    "primary_blocked": False,
    "block_reason": "",
    "safety_margin": 0.45,
    "stop_loss_triggered": False,
    "recovery_status": "normal"
}

# Final Pipeline Output
final_output = {
    "action": "BUY",
    "position_size": 0.085,
    "leverage": 5.0,
    "confidence": 0.68,
    "total_latency_ms": 38.5
}
```

---

## 8. Sequence Diagrams

### 8.1 Normal Operation Flow

```
Time ->

User        Pipeline    PartA      PartB       PartD       PartF       PartE       PartG       PartH       PartI
  |            |          |          |           |           |           |           |           |           |
  |--market--->|          |          |           |           |           |           |           |           |
  |  data      |          |          |           |           |           |           |           |           |
  |            |--OHLCV-->|          |           |           |           |           |           |           |
  |            |          |--EKF---->|           |           |           |           |           |           |
  |            |          |--CAE---->|           |           |           |           |           |           |
  |            |          |--Norm--->|           |           |           |           |           |           |
  |            |<-features-|         |           |           |           |           |           |           |
  |            |          |          |           |           |           |           |           |           |
  |            |-features----------->|           |           |           |           |           |           |
  |            |          |          |--HMM----->|           |           |           |           |           |
  |            |          |          |--Jump---->|           |           |           |           |           |
  |            |          |          |--Meta---->|           |           |           |           |           |
  |            |<---------regime-----|           |           |           |           |           |           |
  |            |          |          |           |           |           |           |           |           |
  |            |------------features+regime----->|           |           |           |           |           |
  |            |          |          |           |--agents-->|           |           |           |           |
  |            |          |          |           |--vote---->|           |           |           |           |
  |            |<---------------decision---------|           |           |           |           |           |
  |            |          |          |           |           |           |           |           |           |
  |            |------decision+features--------->|-ensemble->|           |           |           |           |
  |            |          |          |           |           |-conformal>|           |           |           |
  |            |          |          |           |           |-calibrate>|           |           |           |
  |            |<----------------uq_output-------|           |           |           |           |           |
  |            |          |          |           |           |           |           |           |           |
  |            |----------action+position--------|---------->|-validate->|           |           |           |
  |            |          |          |           |           |           |-oscil---->|           |           |
  |            |<--------------------------hsm_output--------|           |           |           |           |
  |            |          |          |           |           |           |           |           |           |
  |            |------------action+confidence+price----------|---------->|-KAMA----->|           |           |
  |            |          |          |           |           |           |           |-ATR------>|           |
  |            |<-------------------------------hysteresis_out-----------|           |           |           |
  |            |          |          |           |           |           |           |           |           |
  |            |----------------action+portfolio+regime------|-----------|---------->|-EVT------>|           |
  |            |          |          |           |           |           |           |           |-Kelly---->|
  |            |          |          |           |           |           |           |           |-DD_brake->|
  |            |<------------------------------------risk_output----------------------|           |           |
  |            |          |          |           |           |           |           |           |           |
  |            |------------------------action+size+drawdown+regime------|-----------|---------->|-invariants|
  |            |          |          |           |           |           |           |           |           |-fallback|
  |            |<------------------------------------------safety_output--------------------------|           |
  |            |          |          |           |           |           |           |           |           |
  |<-final-----|          |          |           |           |           |           |           |           |
  |  decision  |          |          |           |           |           |           |           |           |

Total time: ~38ms (within 50ms budget)
```

### 8.2 Emergency Safety Override Flow

```
Time ->

Market      PartA       PartB      PartD       PartI       Exchange
  |           |           |          |           |            |
  |---flash---|           |          |           |            |
  |   crash   |           |          |           |            |
  |           |           |          |           |            |
  |           |--process->|          |           |            |
  |           |           |          |           |            |
  |           |           |--JUMP!---|           |            |
  |           |           |  (B5)    |           |            |
  |           |           |          |           |            |
  |           |           |--CRISIS->|           |            |
  |           |           | (forced) |           |            |
  |           |           |          |           |            |
  |           |           |          |--skip-----|            |
  |           |           |          | ensemble  |            |
  |           |           |          |           |            |
  |           |           |          |--HOLD---->|            |
  |           |           |          | forced    |            |
  |           |           |          |           |            |
  |           |           |          |           |--I7:stop-->|
  |           |           |          |           |   loss     |
  |           |           |          |           |            |
  |           |           |          |           |<--confirm--|
  |           |           |          |           |            |
  |           |           |          |<--force---|            |
  |           |           |          |   FLAT    |            |
  |           |           |          |           |            |
  |           |<----------crisis_exit------------|            |
  |           |          (E4: synchronized)      |            |

Total time: ~5ms (fast path)
```

### 8.3 Regime Change Transition Flow

```
Time ->

PartB_HMM    PartB_Meta   PartD       PartG       PartH       PartM
  |            |            |           |           |           |
  |--observe-->|            |           |           |           |
  |   probs    |            |           |           |           |
  |            |            |           |           |           |
  |--trans-----|            |           |           |           |
  | warning    |            |           |           |           |
  | (>30%)     |            |           |           |           |
  |            |            |           |           |           |
  |            |--VIX rise->|           |           |           |
  |            |            |           |           |           |
  |            |--meta: LO->HI-------->|           |           |
  |            |   uncertainty         |           |           |
  |            |            |           |           |           |
  |            |            |--reduce---|           |           |
  |            |            | target    |           |           |
  |            |            | sharpe    |           |           |
  |            |            |           |           |           |
  |            |            |           |--widen--->|           |
  |            |            |           | thresholds|           |
  |            |            |           |           |           |
  |            |            |           |           |--scale--->|
  |            |            |           |           | down      |
  |            |            |           |           | positions |
  |            |            |           |           |           |
  |            |            |           |           |           |--M5: drift->|
  |            |            |           |           |           |   detected  |
  |            |            |           |           |           |             |
  |            |            |           |           |           |<--M1: prune-|
  |            |            |           |           |           |   old buffer|
  |            |            |           |           |           |             |
  |--new trans matrix (from meta_regime)----------->|           |             |
```

### 8.4 Async LLM Injection Flow

```
Time ->

News_Feed     PartJ_Queue   PartJ_LLM    Redis_Cache   Main_Pipeline
  |              |             |              |             |
  |--tweet------>|             |              |             |
  |              |             |              |             |
  |              |--dequeue--->|              |             |
  |              |             |              |             |
  |              |             |--analyze---->|             |
  |              |             | (~200ms)     |             |
  |              |             |              |             |
  |              |             |--cache------>|             |
  |              |             | signal       |             |
  |              |             |              |             |
  |              |             |              |             |--process-->
  |              |             |              |             | market bar
  |              |             |              |             |
  |              |             |              |<---read-----|
  |              |             |              | (<1ms)      |
  |              |             |              |             |
  |              |             |              |--signal---->|
  |              |             |              |             |
  |              |             |              |             |--inject into
  |              |             |              |             | regime input
  |              |             |              |             |
  |              |             |              |             |--continue-->
  |              |             |              |             | normal path

Note: LLM processing is completely async.
Main pipeline only reads cache (<1ms), never waits for LLM.
```

---

## 9. Integration Checklist

### 9.1 Prerequisites

- [ ] **Hardware Requirements**
  - [ ] GPU: NVIDIA RTX 4090 or A10 (24GB VRAM minimum)
  - [ ] RAM: 64GB minimum
  - [ ] Storage: 500GB SSD (for model weights and data)
  - [ ] Network: Low-latency exchange connectivity

- [ ] **Software Requirements**
  - [ ] Python 3.10+
  - [ ] PyTorch 2.0+ with CUDA support
  - [ ] Redis (for LLM signal cache)
  - [ ] Required packages (see requirements.txt)

### 9.2 Dependency Installation

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft accelerate
pip install stable-baselines3
pip install filterpy scipy scikit-learn
pip install numpy pandas

# LLM and NLP
pip install sentencepiece tokenizers
pip install shap

# Infrastructure
pip install redis aioredis
pip install pyyaml

# Optional: FAISS for vector similarity
pip install faiss-gpu
```

### 9.3 Model Weight Downloads

```bash
# FLAG-TRADER base model
huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct

# OPT Financial (if using Part J)
huggingface-cli download facebook/opt-1.3b

# FinancialBERT
huggingface-cli download yiyanghkust/finbert-tone

# Pre-trained HIMARI weights (if available)
# Download from internal model registry
```

### 9.4 Configuration Steps

1. [ ] Copy `config/master_config.template.yaml` to `config/production.yaml`
2. [ ] Set trading pair and exchange credentials
3. [ ] Adjust risk parameters for your capital
4. [ ] Configure LLM API keys (if using external APIs)
5. [ ] Set logging paths and levels
6. [ ] Configure Redis connection

### 9.5 Testing Integration Points

| Integration Point | Test Command | Expected Outcome |
|------------------|--------------|------------------|
| A -> B | `pytest tests/test_a_to_b.py` | Features flow to regime detector |
| B -> D | `pytest tests/test_b_to_d.py` | Regime affects decision |
| D -> F | `pytest tests/test_d_to_f.py` | UQ calibrates confidence |
| F -> E | `pytest tests/test_f_to_e.py` | HSM validates actions |
| E -> G | `pytest tests/test_e_to_g.py` | Hysteresis filters |
| G -> H | `pytest tests/test_g_to_h.py` | Risk sizes positions |
| H -> I | `pytest tests/test_h_to_i.py` | Safety verifies final |
| Full pipeline | `pytest tests/test_full_pipeline.py` | End-to-end pass |

### 9.6 Pre-Launch Verification

- [ ] **Latency Testing**
  - [ ] Average latency < 40ms
  - [ ] P99 latency < 50ms
  - [ ] No GC pauses > 5ms

- [ ] **Accuracy Testing**
  - [ ] Regime detection accuracy > 75% on holdout
  - [ ] Decision ensemble Sharpe > 1.0 on backtest
  - [ ] UQ calibration ECE < 0.05

- [ ] **Safety Testing**
  - [ ] Fallback cascade triggers correctly
  - [ ] Stop-loss enforces at threshold
  - [ ] Crisis mode activates on jump detection

- [ ] **Integration Testing**
  - [ ] LLM sidecar doesn't block main path
  - [ ] Redis cache works under load
  - [ ] Interpretability logging works

### 9.7 Monitoring Setup

```yaml
# Prometheus metrics to track
metrics:
  - himari_pipeline_latency_ms
  - himari_regime_distribution
  - himari_decision_confidence
  - himari_fallback_level
  - himari_position_size
  - himari_drawdown
  - himari_llm_cache_age_seconds
```

---

## 10. Troubleshooting Guide

### 10.1 Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| High latency | Pipeline > 50ms | Profile with `cProfile`, check GPU memory |
| False crisis detection | Too many CRISIS regimes | Increase `jump_threshold_sigma` |
| Over-conservative | Always HOLD | Lower `hysteresis.entry_threshold` |
| Position oscillation | Rapid flips | Enable `oscillation_detection` |
| LLM cache stale | Old signals | Check Redis, reduce `cache_ttl` |
| OOM on GPU | CUDA out of memory | Reduce `mc_dropout.n_samples` |

### 10.2 Debug Mode

```python
pipeline = Layer2MasterPipeline(config, debug=True)

# This enables:
# - Verbose logging at each stage
# - Timing breakdown per component
# - Feature value inspection
# - Intermediate decision traces
```

### 10.3 Contact and Support

For issues with specific components, refer to the individual part guides:
- Part A: `Part_A_Preprocessing_Complete.md`
- Part B: `Part_B_Regime_Detection_Complete.md`
- Part D: `Part_D_Decision_Engine_Complete.md`
- ... (etc.)

---

## Appendix A: Method Reference

| Part | ID | Method Name | Latency | Key Parameters |
|------|-----|-------------|---------|----------------|
| A | A1 | Extended Kalman Filter | <1ms | process_noise, measurement_noise |
| A | A2 | Conversational Autoencoders | ~2ms | latent_dim, consensus_threshold |
| A | A3 | Frequency Normalization | <0.5ms | window_size |
| A | A4 | TimeGAN Augmentation | Offline | n_synthetic |
| A | A5 | Tab-DDPM Diffusion | Offline | n_tail_events |
| A | A6 | VecNormalize | <0.1ms | clip |
| A | A7 | Orthogonal Init | N/A | gain |
| A | A8 | Online Augmentation | ~1ms | jitter_scale |
| B | B1 | Student-t AH-HMM | ~1ms | n_states, df |
| B | B2 | Meta-Regime Layer | ~0.5ms | vix_thresholds |
| B | B3 | Causal Info Geometry | ~2ms | spd_threshold |
| B | B4 | AEDL Meta-Learning | Offline | maml_lr |
| B | B5 | Jump Detector | <0.1ms | threshold_sigma |
| B | B6 | Hurst Exponent | ~0.5ms | window |
| B | B7 | Online Baum-Welch | ~1ms | learning_rate |
| B | B8 | ADWIN Drift | ~0.3ms | delta |
| D | D1 | FLAG-TRADER | ~15ms | lora_r, lora_alpha |
| D | D2 | Critic-Guided DT | ~25ms | context_length |
| D | D3 | Conservative Q | ~8ms | cql_alpha |
| D | D4 | rsLoRA | N/A | rank |
| D | D5 | PPO-LSTM | ~5ms | hidden_dim |
| D | D6 | SAC Agent | ~4ms | gamma, alpha |
| D | D7 | Sharpe-Weighted Vote | <0.5ms | lookback_days |
| D | D8 | Disagreement Scaling | <0.5ms | threshold |
| D | D9 | Return Conditioning | <0.1ms | regime_targets |
| D | D10 | FinRL-DT Pipeline | Offline | batch_size |
| E | E1 | Orthogonal Regions | <0.1ms | -- |
| E | E2 | Hierarchical Nesting | <0.1ms | -- |
| E | E3 | History States | <0.1ms | -- |
| E | E4 | Synchronized Events | <0.1ms | -- |
| E | E5 | Learned Transitions | ~1ms | model_path |
| E | E6 | Oscillation Detection | <0.1ms | window, max_flips |
| F | F1 | CT-SSF Conformal | ~5ms | alpha, n_cal |
| F | F2 | CPTC Regime-Aware | ~2ms | expansion_factor |
| F | F3 | Temperature Scaling | <0.1ms | temperature |
| F | F4 | Deep Ensemble | ~15ms | n_models |
| F | F5 | MC Dropout | ~3ms | n_samples |
| F | F6 | Epistemic/Aleatoric | ~1ms | -- |
| F | F7 | k-NN OOD | ~2ms | k, threshold |
| F | F8 | Predictive UQ | ~3ms | horizon |
| G | G1 | KAMA Adaptive | ~0.1ms | er_period |
| G | G2 | KNN Pattern Match | ~0.2ms | k |
| G | G3 | ATR-Scaled Bands | ~0.05ms | atr_mult |
| G | G4 | Meta-Learned k | ~0.05ms | -- |
| G | G5 | Loss Aversion | ~0.02ms | ratio (2.2) |
| G | G6 | Whipsaw Learning | ~0.08ms | learning_rate |
| H | H1 | EVT + GPD | ~0.3ms | threshold_pct |
| H | H2 | DDPG-TiDE Kelly | ~0.5ms | -- |
| H | H3 | DCC-GARCH | ~0.4ms | -- |
| H | H4 | DD Brake | ~0.1ms | brake_levels |
| H | H5 | Portfolio VaR | ~0.6ms | copula_type |
| H | H6 | Safe Margin | ~0.2ms | k_sigma |
| H | H7 | Dynamic Leverage | ~0.2ms | decay_curve |
| H | H8 | Adaptive Risk Budget | ~0.2ms | -- |
| I | I1 | 4-Level Fallback | ~0.8ms | thresholds |
| I | I2 | Predictive Safety | ~0.5ms | n_steps |
| I | I3 | Formal Verification | ~0.2ms | -- |
| I | I4 | Reachability | ~0.2ms | -- |
| I | I5 | Safety Invariants | ~0.3ms | -- |
| I | I6 | Safety Monitor | ~0.1ms | -- |
| I | I7 | Stop-Loss Enforcer | ~0.1ms | max_daily_loss |
| I | I8 | Recovery Protocol | ~0.1ms | recovery_bars |
| J | J1 | OPT Financial | Async | model_size |
| J | J2 | Trading-R1 | Async | -- |
| J | J3 | RAG + FAISS | Async | index_size |
| J | J4 | LLM Calibration | Async | -- |
| J | J5 | FinancialBERT | Async | -- |
| J | J6 | Signal Extraction | Async | -- |
| J | J7 | Event Classification | Async | -- |
| J | J8 | Async Processing | -- | queue_size |
| K | K1-K8 | Training Methods | Offline | -- |
| L | L1-L6 | Validation Methods | Offline | -- |
| M | M1-M6 | Adaptation Methods | Periodic | -- |
| N | N1-N4 | Interpretability | Async | -- |

---

*Document generated by HIMARI Layer 2 Integration Team*
*Last updated: January 2026*
