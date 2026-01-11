"""
HIMARI Layer 2 - Uncertainty Quantification Module (v5.0)
Subsystem F: Uncertainty (8 Methods)

Components:
    - F1: CT-SSF Latent Conformal
    - F2: CPTC Regime Change Points
    - F3: Temperature Scaling
    - F4: Deep Ensemble Disagreement
    - F5: MC Dropout
    - F6: Epistemic/Aleatoric Split
    - F7: Data Uncertainty (k-NN)
    - F8: Predictive Uncertainty
"""

# v5.0 Complete Pipeline
from .uq_pipeline import (
    CTSSF,
    CTSSFConfig,
    LatentEncoder,
    LatentDecoder,
    CPTC,
    CPTCConfig,
    TemperatureScaler,
    TemperatureScalingConfig,
    DeepEnsemble,
    MCDropout,
    UncertaintySplitter,
    KNNOODDetector,
    PredictiveUncertainty,
    UncertaintyQuantificationPipeline,
    UQPipelineConfig
)

# v5.0 Components (original files)
try:
    from .ct_ssf import CTSSF as CTSSFOriginal, CTSSFConfig as CTSSFConfigOriginal, create_ctssf
except ImportError:
    CTSSFOriginal = None
    CTSSFConfigOriginal = None
    create_ctssf = None

# v4.0 Legacy
try:
    from .quantifier import UncertaintyQuantifier, UncertaintyConfig, UncertaintyResult
except ImportError:
    UncertaintyQuantifier = None
    UncertaintyConfig = None
    UncertaintyResult = None

# MC Dropout (F5)
try:
    from .mc_dropout import MCDropoutWrapper, MCDropoutConfig, MCDropoutEnsemble
except ImportError:
    MCDropoutWrapper = None
    MCDropoutConfig = None
    MCDropoutEnsemble = None

__all__ = [
    # v5.0 Complete Pipeline
    'UncertaintyQuantificationPipeline',
    'UQPipelineConfig',
    # F1
    'CTSSF',
    'CTSSFConfig',
    'LatentEncoder',
    'LatentDecoder',
    # F2
    'CPTC',
    'CPTCConfig',
    # F3
    'TemperatureScaler',
    'TemperatureScalingConfig',
    # F4
    'DeepEnsemble',
    # F5
    'MCDropout',
    # F6
    'UncertaintySplitter',
    # F7
    'KNNOODDetector',
    # F8
    'PredictiveUncertainty',
    # Legacy
    'UncertaintyQuantifier',
    'UncertaintyConfig',
    'UncertaintyResult',
    'MCDropoutWrapper',
    'MCDropoutConfig',
    'MCDropoutEnsemble',
    'create_ctssf',
]
