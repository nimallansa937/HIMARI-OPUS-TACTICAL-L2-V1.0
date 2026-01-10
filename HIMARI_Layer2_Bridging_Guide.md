# HIMARI OPUS 2: Layer 2 Bridging Guide
## Mapping Existing Code to Developer Guide Subsystems

**Purpose:** Bridge between existing `HIMARI-OPUS-LAYER-2-TACTICAL-main` codebase and the new `HIMARI_Layer2_LLM_TRANSFORMER_Developer_Guide.md`

**For:** AI IDE Agents (Cursor, Windsurf, Aider, Claude Code)

---

# QUICK REFERENCE

| Subsystem | Existing File | Action | New Location |
|-----------|--------------|--------|--------------|
| A: Data Preprocessing | — | **CREATE** | `src/preprocessing/` |
| B: Regime Detection | `detectors/regime_detector.py` | **MODIFY** | Keep location |
| C: Multi-Timeframe | — | **CREATE** | `src/multi_timeframe/` |
| D: Decision Engine | `layers/baseline.py` | **REPLACE** | `src/decision_engine/` |
| E: HSM State Machine | `governance/tier_router.py` | **MODIFY** | Keep + add `src/state_machine/` |
| F: Uncertainty | — | **CREATE** | `src/uncertainty/` |
| G: Hysteresis | `detectors/regime_detector.py` (partial) | **EXTRACT** | `src/hysteresis/` |
| H: RSS Risk | `layers/cascade.py` | **MODIFY** | Keep + add `src/risk_management/` |
| I: Simplex Safety | `layers/emergency.py` | **MODIFY** | Keep location |
| J: LLM Integration | `detectors/sentiment_processor.py` | **MODIFY** | Keep + add `src/llm_integration/` |
| K: Training | — | **CREATE** | `src/training/` |
| L: Validation | `validation/` | **MODIFY** | Keep location |
| M: Adaptation | — | **CREATE** | `src/adaptation/` |
| N: Interpretability | — | **CREATE** | `src/interpretability/` |

---

# PART 1: FILES TO MODIFY

## 1.1 `detectors/regime_detector.py`

**Current:** Simple 5-state heuristic regime detector with hysteresis
**Target:** HMM 4-state with Hurst gating and jump detection

### Modifications Required:

```python
# ADD these imports
from hmmlearn import hmm
import numpy as np

# ADD HMM configuration dataclass (from Guide Part 5)
@dataclass
class HMMConfig:
    n_states: int = 4  # Bull, Bear, Sideways, Crisis
    covariance_type: str = 'full'
    n_iter: int = 100
    random_state: int = 42

# MODIFY RegimeDetector class:
# 1. Replace heuristic _infer_regime() with HMM forward algorithm
# 2. Add Hurst exponent gating (from Guide Part 5.2)
# 3. Add jump detection for crisis (threshold: 3σ moves)
# 4. Keep hysteresis logic but move to separate module

# ADD method:
def compute_hurst_exponent(self, series: np.ndarray, max_lag: int = 20) -> float:
    """R/S analysis for Hurst exponent"""
    # Implementation from Guide Part 5.2
    pass

# ADD method:
def detect_jump(self, returns: float, threshold_sigma: float = 3.0) -> bool:
    """Detect regime-breaking jumps"""
    pass
```

### Keep Unchanged:
- `RegimeState` dataclass
- `TRANSITION_PROBS` (update values for 4-state)
- `get_current_regime()`, `reset()`, `get_status()`

---

## 1.2 `layers/baseline.py`

**Current:** Weighted signal composite → action mapping
**Target:** PPO-LSTM Decision Transformer ensemble

### Modifications Required:

```python
# This file will be REPLACED but keep as fallback

# RENAME to: layers/baseline_fallback.py

# CREATE NEW: src/decision_engine/ppo_lstm.py
# - PPO-LSTM agent with 256 hidden units
# - Action space: BUY, HOLD, SELL (discrete)
# - State: 60D feature vector from Layer 1

# CREATE NEW: src/decision_engine/decision_transformer.py
# - Transformer architecture for trajectory modeling
# - Context length: 500 timesteps
# - Return conditioning for target Sharpe

# CREATE NEW: src/decision_engine/ensemble.py
# - Voting mechanism: PPO + SAC + DDQN
# - Disagreement-based confidence scaling
```

### Keep for Fallback:
- `BaselineComposite` class (rename to `FallbackComposite`)
- Used when GPU unavailable or model loading fails

---

## 1.3 `layers/cascade.py`

**Current:** Cascade risk dampening based on risk score
**Target:** RSS (Responsibility-Sensitive Safety) integration

### Modifications Required:

```python
# ADD RSS imports and logic
from src.risk_management.rss import RSSValidator

# MODIFY CascadeRiskGate class:
# 1. Add RSS proper response validation
# 2. Add minimum safe distance calculations
# 3. Keep existing dampening logic as secondary check

# ADD method:
def compute_safe_distance(self, position: float, volatility: float) -> float:
    """RSS-based minimum safe position distance"""
    pass

# ADD method:  
def validate_proper_response(self, action: TradeAction, risk_level: float) -> bool:
    """Ensure action is proper response to danger"""
    pass
```

---

## 1.4 `layers/emergency.py`

**Current:** Binary emergency stop with kill switch
**Target:** Simplex safety wrapper with fallback cascade

### Modifications Required:

```python
# MODIFY EmergencyStop class → SimplexSafetyWrapper

# ADD fallback cascade levels:
# Level 1: Decision Transformer (primary)
# Level 2: PPO-LSTM (if DT fails)
# Level 3: Rule-based (if PPO fails)
# Level 4: HOLD-only (if rules fail)

# ADD method:
def execute_with_fallback(self, action: TradeAction, confidence: float) -> Tuple[TradeAction, str]:
    """Execute action with automatic fallback on failure"""
    pass

# ADD method:
def verify_safety_invariant(self, action: TradeAction, state: Dict) -> bool:
    """Verify action maintains safety invariants"""
    # Max position size
    # Max drawdown rate
    # Min cash reserve
    pass
```

---

## 1.5 `detectors/sentiment_processor.py`

**Current:** Sentiment event detection with shock magnitude
**Target:** LLM artifact injection point

### Modifications Required:

```python
# ADD LLM integration imports
from src.llm_integration.artifact_injector import LLMArtifactInjector

# ADD to SentimentProcessor class:
self.llm_injector = LLMArtifactInjector(model_name="TheFinAI/FinLLaVA")

# ADD method:
def inject_llm_signal(self, news_text: str, chart_image: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Get LLM-derived trading signal from news/chart
    
    Returns:
        Dict with keys: sentiment_score, confidence, reasoning_embedding
    """
    pass

# MODIFY process() method:
# Add LLM signal to output alongside traditional sentiment
```

---

## 1.6 `governance/tier_router.py`

**Current:** Confidence-based tier routing
**Target:** HSM (Hierarchical State Machine) integration

### Modifications Required:

```python
# ADD HSM imports
from src.state_machine.hsm import HierarchicalStateMachine, TradingState

# ADD HSM instance to TierRouter:
self.hsm = HierarchicalStateMachine()

# MODIFY _route_to_tier() method:
# 1. Query HSM for current state
# 2. Apply state-specific routing rules
# 3. Consider HSM transition costs in tier assignment

# ADD method:
def update_hsm_state(self, market_state: Dict, action_taken: TradeAction) -> None:
    """Update HSM based on market and action"""
    pass
```

---

## 1.7 `validation/backtester.py`

**Current:** Basic PnL tracking backtester
**Target:** CPCV, DSR, Monte Carlo validation

### Modifications Required:

```python
# ADD imports
from validation.cpcv import CombinatorialPurgedCV
from validation.dsr import deflated_sharpe_ratio
from validation.monte_carlo import MonteCarloValidator

# ADD to Backtester class:
self.cpcv = CombinatorialPurgedCV(n_splits=7, embargo_pct=0.01)
self.monte_carlo = MonteCarloValidator(n_simulations=1000)

# ADD method:
def run_cpcv_validation(self, strategy, data) -> Dict[str, float]:
    """Run combinatorial purged cross-validation"""
    pass

# ADD method:
def compute_deflated_sharpe(self, returns: np.ndarray, n_trials: int) -> float:
    """Compute deflated Sharpe ratio accounting for multiple testing"""
    pass

# ADD method:
def monte_carlo_significance(self, strategy_returns: np.ndarray) -> float:
    """Monte Carlo p-value for strategy performance"""
    pass
```

---

## 1.8 `validation/metrics.py`

**Current:** Basic Sharpe, max drawdown, win rate
**Target:** Sortino, Calmar, deflated Sharpe, regime-specific metrics

### Modifications Required:

```python
# ADD new metric functions:

def sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
    """Sortino ratio using downside deviation"""
    pass

def calmar_ratio(returns: np.ndarray, max_drawdown: float) -> float:
    """Calmar ratio = annualized return / max drawdown"""
    pass

def deflated_sharpe_ratio(sharpe: float, n_trials: int, skewness: float, kurtosis: float) -> float:
    """DSR accounting for multiple testing bias"""
    pass

def regime_conditional_sharpe(returns: np.ndarray, regimes: np.ndarray) -> Dict[str, float]:
    """Sharpe ratio broken down by regime"""
    pass

def tail_ratio(returns: np.ndarray, percentile: float = 5.0) -> float:
    """Ratio of right tail to left tail"""
    pass
```

---

## 1.9 `core/config.py`

**Current:** Basic tactical config with thresholds
**Target:** All 14 subsystem configurations

### Modifications Required:

```python
# ADD configuration dataclasses for each subsystem:

@dataclass
class PreprocessingConfig:
    normalize_method: str = 'zscore'
    clip_outliers: bool = True
    outlier_threshold: float = 5.0

@dataclass
class HMMConfig:
    n_states: int = 4
    covariance_type: str = 'full'
    # ... (from Guide Part 5)

@dataclass
class MultiTimeframeConfig:
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '1h', '4h'])
    fusion_method: str = 'attention'

@dataclass
class DecisionEngineConfig:
    model_type: str = 'ppo_lstm'  # or 'decision_transformer', 'ensemble'
    hidden_size: int = 256
    lstm_layers: int = 2
    # ... (from Guide Part 7)

@dataclass
class HSMConfig:
    states: List[str] = field(default_factory=lambda: ['OBSERVE', 'POSITION', 'EXIT'])
    transition_costs: Dict[str, float] = None

@dataclass  
class UncertaintyConfig:
    method: str = 'mc_dropout'
    n_samples: int = 10
    dropout_rate: float = 0.1

@dataclass
class HysteresisConfig:
    entry_threshold: float = 0.6
    exit_threshold: float = 0.4
    min_hold_periods: int = 3

@dataclass
class RSSConfig:
    max_position_pct: float = 0.1
    min_safe_distance: float = 0.02

@dataclass
class SimplexConfig:
    fallback_levels: int = 4
    safety_invariants: List[str] = field(default_factory=lambda: ['max_position', 'max_dd_rate'])

@dataclass
class LLMConfig:
    model_name: str = 'TheFinAI/FinLLaVA'
    max_tokens: int = 256
    temperature: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    total_timesteps: int = 1_000_000
    # ... (from Guide Part 14)

@dataclass
class ValidationConfig:
    cpcv_splits: int = 7
    embargo_pct: float = 0.01
    monte_carlo_sims: int = 1000

@dataclass
class AdaptationConfig:
    drift_detection_method: str = 'adwin'
    online_learning_rate: float = 1e-4
    replay_buffer_size: int = 10000

@dataclass
class InterpretabilityConfig:
    shap_samples: int = 100
    lime_samples: int = 500

# MODIFY TacticalConfig to include all sub-configs:
@dataclass
class TacticalConfig:
    # ... existing fields ...
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    multi_timeframe: MultiTimeframeConfig = field(default_factory=MultiTimeframeConfig)
    decision_engine: DecisionEngineConfig = field(default_factory=DecisionEngineConfig)
    hsm: HSMConfig = field(default_factory=HSMConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    rss: RSSConfig = field(default_factory=RSSConfig)
    simplex: SimplexConfig = field(default_factory=SimplexConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    interpretability: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)
```

---

## 1.10 `core/types.py`

**Current:** TradeAction, Tier, RegimeLabel enums
**Target:** Add dataclasses for all subsystem outputs

### Modifications Required:

```python
# ADD new dataclasses:

@dataclass
class HMMState:
    regime: RegimeLabel
    confidence: float
    state_probabilities: Dict[str, float]
    hurst_exponent: float
    is_trending: bool

@dataclass
class UncertaintyResult:
    mean_action: TradeAction
    action_probabilities: Dict[TradeAction, float]
    epistemic_uncertainty: float
    aleatoric_uncertainty: float

@dataclass
class HysteresisState:
    signal_value: float
    smoothed_value: float
    is_active: bool
    hold_counter: int

@dataclass
class RSSValidation:
    is_safe: bool
    safe_distance: float
    proper_response: bool
    risk_attribution: Dict[str, float]

@dataclass
class LLMSignal:
    sentiment_score: float
    confidence: float
    reasoning: str
    embedding: Optional[np.ndarray] = None

@dataclass
class EnsembleVote:
    actions: Dict[str, TradeAction]  # model_name → action
    confidences: Dict[str, float]
    disagreement: float
    final_action: TradeAction
    final_confidence: float
```

---

## 1.11 `tactical_layer.py`

**Current:** 4-level subsumption orchestrator
**Target:** 14-subsystem orchestrator

### Modifications Required:

```python
# RENAME class: TacticalLayerV2_1_1 → TacticalLayerV3

# ADD imports for all new subsystems
from src.preprocessing import Preprocessor
from src.multi_timeframe import MultiTimeframeFusion
from src.decision_engine import DecisionEngineEnsemble
from src.state_machine import HierarchicalStateMachine
from src.uncertainty import UncertaintyQuantifier
from src.hysteresis import HysteresisFilter
from src.risk_management import RSSValidator
from src.simplex_safety import SimplexWrapper
from src.llm_integration import LLMArtifactInjector
from src.adaptation import OnlineLearner
from src.interpretability import SHAPExplainer

# MODIFY __init__:
def __init__(self, config: TacticalConfig = None):
    self.config = config or DEFAULT_CONFIG
    
    # Subsystem A: Data Preprocessing
    self.preprocessor = Preprocessor(config.preprocessing)
    
    # Subsystem B: Regime Detection (modified existing)
    self.regime_detector = RegimeDetector(config.hmm)
    
    # Subsystem C: Multi-Timeframe Fusion
    self.mtf_fusion = MultiTimeframeFusion(config.multi_timeframe)
    
    # Subsystem D: Decision Engine
    self.decision_engine = DecisionEngineEnsemble(config.decision_engine)
    
    # Subsystem E: HSM State Machine
    self.hsm = HierarchicalStateMachine(config.hsm)
    
    # Subsystem F: Uncertainty Quantification
    self.uncertainty = UncertaintyQuantifier(config.uncertainty)
    
    # Subsystem G: Hysteresis Filter
    self.hysteresis = HysteresisFilter(config.hysteresis)
    
    # Subsystem H: RSS Risk Management
    self.rss = RSSValidator(config.rss)
    
    # Subsystem I: Simplex Safety (modified existing)
    self.simplex = SimplexWrapper(config.simplex)
    
    # Subsystem J: LLM Integration
    self.llm_injector = LLMArtifactInjector(config.llm)
    
    # Subsystem K: Training (offline, not in runtime)
    # Subsystem L: Validation (offline, not in runtime)
    
    # Subsystem M: Adaptation
    self.adapter = OnlineLearner(config.adaptation)
    
    # Subsystem N: Interpretability
    self.explainer = SHAPExplainer(config.interpretability)
    
    # Keep existing layers as fallback
    self.emergency_stop = EmergencyStop(config)  # Part of Simplex now
    self.baseline_fallback = BaselineComposite(config)

# MODIFY evaluate() method:
def evaluate(self, features_60d: np.ndarray, risk_context: RiskContext, 
             multimodal: MultimodalInput) -> TacticalDecision:
    """
    New evaluation flow:
    1. Preprocess features (A)
    2. Detect regime (B)
    3. Fuse multi-timeframe (C)
    4. Get LLM signal (J)
    5. Query decision engine (D)
    6. Quantify uncertainty (F)
    7. Apply hysteresis (G)
    8. Validate RSS (H)
    9. Update HSM state (E)
    10. Apply Simplex safety (I)
    11. Adapt online (M)
    12. Generate explanation (N)
    """
    pass
```

---

# PART 2: FILES TO CREATE

## 2.1 Directory Structure

```
HIMARI-OPUS-LAYER-2-TACTICAL-main/
├── src/                              # NEW DIRECTORY
│   ├── __init__.py
│   ├── preprocessing/                # Subsystem A
│   │   ├── __init__.py
│   │   ├── normalizer.py
│   │   ├── feature_scaler.py
│   │   └── monte_carlo_augment.py
│   ├── multi_timeframe/             # Subsystem C
│   │   ├── __init__.py
│   │   ├── aggregator.py
│   │   └── fusion.py
│   ├── decision_engine/             # Subsystem D
│   │   ├── __init__.py
│   │   ├── ppo_lstm.py
│   │   ├── decision_transformer.py
│   │   ├── sac_agent.py
│   │   ├── ddqn_agent.py
│   │   └── ensemble.py
│   ├── state_machine/               # Subsystem E
│   │   ├── __init__.py
│   │   └── hsm.py
│   ├── uncertainty/                 # Subsystem F
│   │   ├── __init__.py
│   │   ├── mc_dropout.py
│   │   ├── ensemble_disagreement.py
│   │   └── bayesian.py
│   ├── hysteresis/                  # Subsystem G
│   │   ├── __init__.py
│   │   └── filter.py
│   ├── risk_management/             # Subsystem H
│   │   ├── __init__.py
│   │   └── rss.py
│   ├── llm_integration/             # Subsystem J
│   │   ├── __init__.py
│   │   └── artifact_injector.py
│   ├── training/                    # Subsystem K
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── adversarial_selfplay.py
│   │   └── curriculum.py
│   ├── adaptation/                  # Subsystem M
│   │   ├── __init__.py
│   │   ├── drift_detector.py
│   │   └── online_learning.py
│   └── interpretability/            # Subsystem N
│       ├── __init__.py
│       ├── shap_explainer.py
│       └── lime_explainer.py
├── docker/                          # NEW DIRECTORY
│   ├── Dockerfile.arm64
│   └── Dockerfile.x86
├── scripts/                         # NEW DIRECTORY
│   ├── arm64_compatibility_test.sh
│   ├── train.sh
│   ├── export_onnx.py
│   ├── quantize.py
│   └── benchmark.sh
├── configs/                         # NEW DIRECTORY
│   ├── training.yaml
│   ├── inference.yaml
│   └── validation.yaml
├── tests/
│   ├── integration/                 # NEW SUBDIRECTORY
│   │   └── test_pipeline_e2e.py
│   └── unit/                        # NEW SUBDIRECTORY
│       ├── test_preprocessing.py
│       ├── test_decision_engine.py
│       └── test_hsm.py
└── data/
    └── models/                      # NEW DIRECTORY (for checkpoints)
```

---

## 2.2 Key New Files (Implementation Stubs)

### `src/decision_engine/ppo_lstm.py`

```python
"""
Subsystem D: PPO-LSTM Decision Agent
Reference: Guide Part 7

Purpose: Core RL agent for tactical decisions
Input: 60D feature vector from Layer 1
Output: TradeAction with confidence
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTMExtractor(BaseFeaturesExtractor):
    """Custom LSTM feature extractor for PPO"""
    
    def __init__(self, observation_space, features_dim=256, lstm_hidden=256, lstm_layers=2):
        super().__init__(observation_space, features_dim)
        # Implementation from Guide Part 7.1
        pass

class PPOLSTMAgent:
    """PPO agent with LSTM policy network"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def train(self, env, total_timesteps: int):
        """Train PPO-LSTM agent"""
        pass
    
    def predict(self, observation: np.ndarray) -> Tuple[TradeAction, float]:
        """Get action and confidence for observation"""
        pass
    
    def save(self, path: str):
        """Save model checkpoint"""
        pass
    
    def load(self, path: str):
        """Load model checkpoint"""
        pass
```

### `src/state_machine/hsm.py`

```python
"""
Subsystem E: Hierarchical State Machine
Reference: Guide Part 8

Purpose: Track trading state and enforce state-dependent rules
States: OBSERVE → ENTRY → POSITION → EXIT → OBSERVE
"""

from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass

class TradingState(Enum):
    OBSERVE = "observe"      # No position, watching
    ENTRY = "entry"          # Considering entry
    POSITION = "position"    # In position
    EXIT = "exit"            # Considering exit
    COOLDOWN = "cooldown"    # Post-exit cooldown

@dataclass
class HSMTransition:
    from_state: TradingState
    to_state: TradingState
    condition: str
    cost: float

class HierarchicalStateMachine:
    """Trading state machine with transition rules"""
    
    TRANSITIONS = {
        TradingState.OBSERVE: [TradingState.ENTRY],
        TradingState.ENTRY: [TradingState.OBSERVE, TradingState.POSITION],
        TradingState.POSITION: [TradingState.EXIT],
        TradingState.EXIT: [TradingState.POSITION, TradingState.COOLDOWN],
        TradingState.COOLDOWN: [TradingState.OBSERVE],
    }
    
    def __init__(self, config):
        self.config = config
        self.state = TradingState.OBSERVE
        self.state_entry_time = 0
        
    def can_transition(self, to_state: TradingState) -> bool:
        """Check if transition is allowed"""
        pass
    
    def transition(self, to_state: TradingState) -> bool:
        """Execute state transition"""
        pass
    
    def get_allowed_actions(self) -> List[TradeAction]:
        """Get actions allowed in current state"""
        pass
```

### `src/uncertainty/mc_dropout.py`

```python
"""
Subsystem F: Monte Carlo Dropout Uncertainty
Reference: Guide Part 9

Purpose: Quantify epistemic uncertainty via dropout at inference
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

class MCDropoutWrapper:
    """Wrapper for MC Dropout uncertainty estimation"""
    
    def __init__(self, model: nn.Module, n_samples: int = 10, dropout_rate: float = 0.1):
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Run multiple forward passes with dropout enabled
        
        Returns:
            mean_prediction: Average prediction
            uncertainty: Standard deviation of predictions
        """
        pass
    
    def get_epistemic_uncertainty(self, x: torch.Tensor) -> float:
        """Compute epistemic (model) uncertainty"""
        pass
```

### `src/training/trainer.py`

```python
"""
Subsystem K: Training Infrastructure
Reference: Guide Part 14

Purpose: Orchestrate training of all learnable components
"""

import yaml
from pathlib import Path
from typing import Dict, Any

class Layer2Trainer:
    """Master trainer for Layer 2 components"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        
    def prepare_data(self):
        """Step 1: Load and preprocess training data"""
        pass
    
    def generate_synthetic(self, multiplier: int = 10):
        """Step 2: Monte Carlo data augmentation"""
        pass
    
    def train_regime_detector(self):
        """Step 3: Train HMM regime detector"""
        pass
    
    def train_decision_engine(self):
        """Step 4: Train PPO-LSTM / Decision Transformer ensemble"""
        pass
    
    def adversarial_training(self):
        """Step 5: Adversarial self-play"""
        pass
    
    def validate(self):
        """Step 6: CPCV validation"""
        pass
    
    def run_full_pipeline(self):
        """Run complete training pipeline"""
        self.prepare_data()
        self.generate_synthetic()
        self.train_regime_detector()
        self.train_decision_engine()
        self.adversarial_training()
        self.validate()
```

---

# PART 3: IMPLEMENTATION ORDER

## Phase 1: Foundation (Week 1-2)
1. ✅ Update `core/config.py` with all subsystem configs
2. ✅ Update `core/types.py` with new dataclasses
3. ✅ Create `src/` directory structure
4. ✅ Create `configs/training.yaml`

## Phase 2: Regime & Preprocessing (Week 3-4)
5. ✅ Modify `detectors/regime_detector.py` → HMM 4-state
6. ✅ Create `src/preprocessing/normalizer.py`
7. ✅ Create `src/hysteresis/filter.py` (extract from regime)

## Phase 3: Decision Engine (Week 5-8)
8. ✅ Create `src/decision_engine/ppo_lstm.py`
9. ✅ Create `src/decision_engine/decision_transformer.py`
10. ✅ Create `src/decision_engine/ensemble.py`
11. ✅ Create `src/training/trainer.py`

## Phase 4: Safety & Risk (Week 9-10)
12. ✅ Modify `layers/cascade.py` → RSS integration
13. ✅ Modify `layers/emergency.py` → Simplex wrapper
14. ✅ Create `src/state_machine/hsm.py`

## Phase 5: Enhancement (Week 11-12)
15. ✅ Create `src/uncertainty/mc_dropout.py`
16. ✅ Modify `detectors/sentiment_processor.py` → LLM injection
17. ✅ Create `src/adaptation/online_learning.py`

## Phase 6: Validation & Integration (Week 13-16)
18. ✅ Modify `validation/backtester.py` → CPCV
19. ✅ Modify `validation/metrics.py` → new metrics
20. ✅ Modify `tactical_layer.py` → 14-subsystem orchestrator
21. ✅ Create `tests/integration/test_pipeline_e2e.py`
22. ✅ Create Docker images and scripts

---

# PART 4: COMPATIBILITY NOTES

## Layer 1 Interface
- **Input:** 60D feature vector from `FeatureVectorAssembler`
- **No changes needed** to Layer 1 output format
- Add sequence buffer for Decision Transformer context

## Data Infrastructure Interface
- **Redis:** Feature caching (existing) ✅
- **Kafka:** Stream consumption (existing) ✅
- **TimescaleDB:** Historical data (existing) ✅
- **Neo4j:** Add causal graph population for regime analysis

## GPU Requirements
- **Training:** GH200 (96GB) or H100 (80GB)
- **Inference:** A10 (24GB) sufficient
- **Fallback:** CPU with baseline composite

---

# PART 5: TESTING CHECKLIST

## Unit Tests (per subsystem)
- [ ] Preprocessing normalizes to [-1, 1]
- [ ] HMM converges on synthetic data
- [ ] PPO-LSTM learns CartPole (sanity check)
- [ ] HSM transitions correctly
- [ ] Hysteresis prevents oscillation
- [ ] RSS validates safe actions
- [ ] Simplex fallback cascade works

## Integration Tests
- [ ] Full pipeline latency <100ms
- [ ] Same input → same output (determinism)
- [ ] Fallback triggers on model failure
- [ ] Memory stable over 24h simulation

## Validation Tests
- [ ] CPCV Sharpe > 0.5 (7 folds)
- [ ] DSR significant at p<0.05
- [ ] Monte Carlo beats random (95% CI)
- [ ] Regime-conditional performance stable

---

**END OF BRIDGING GUIDE**

Document Version: 1.0
Created: December 2024
Target: AI IDE Agents
