"""
HIMARI Layer 2 - Hysteresis Filter Module
Subsystem G: Anti-Whipsaw Signal Filtering (6 Methods)

Components:
    - G1: KAMA Adaptive Thresholds
    - G2: KNN Pattern Matching
    - G3: ATR-Scaled Bands
    - G4: Meta-Learned k Values
    - G5: 2.2× Loss Aversion Ratio
    - G6: Whipsaw Learning
"""

# G1-G4: Core pipeline components
from .hysteresis_pipeline import (
    HysteresisFilter,
    HysteresisConfig,
    KAMAThresholdAdapter,
    KAMAConfig,
    KNNPatternMatcher,
    ATRBandCalculator,
    ATRBandConfig,
    MetaLearnedKSelector,
)

# G5: Loss Aversion (Separate Module)
from .loss_aversion import (
    LossAversionThresholds,
    LossAversionConfig,
    create_loss_aversion_thresholds
)

# G6: Whipsaw Learning (Separate Module)
from .whipsaw_learning import (
    WhipsawLearner,
    WhipsawConfig,
    WhipsawEvent,
    create_whipsaw_learner
)

# Legacy
try:
    from .filter import HysteresisFilter as LegacyHysteresisFilter
except ImportError:
    LegacyHysteresisFilter = None

__all__ = [
    # Core Pipeline
    'HysteresisFilter',
    'HysteresisConfig',
    # G1: KAMA
    'KAMAThresholdAdapter',
    'KAMAConfig',
    # G2: KNN
    'KNNPatternMatcher',
    # G3: ATR
    'ATRBandCalculator',
    'ATRBandConfig',
    # G4: Meta-learned k
    'MetaLearnedKSelector',
    # G5: Loss Aversion
    'LossAversionThresholds',
    'LossAversionConfig',
    'create_loss_aversion_thresholds',
    # G6: Whipsaw
    'WhipsawLearner',
    'WhipsawConfig',
    'WhipsawEvent',
    'create_whipsaw_learner',
]
