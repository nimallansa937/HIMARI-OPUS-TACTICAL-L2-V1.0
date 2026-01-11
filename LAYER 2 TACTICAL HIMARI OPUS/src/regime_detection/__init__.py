"""
HIMARI Layer 2 - Regime Detection Module (v5.0)
Subsystem B: Regime Detection

Components (v5.0 - 8 methods architecture):
    - Student-t AH-HMM (B1): Fat-tailed emissions + Hierarchical structure - UPGRADED
    - Meta-Regime Layer (B2): VIX/EPU structural transitions - NEW
    - Causal Info Geometry (B3): SPD correlation manifolds - NEW
    - AEDL Meta-Learning (B4): Adaptive labeling via MAML - NEW
    - Jump Detection (B5): 3σ threshold crisis detection - KEPT
    - Hurst Exponent (B6): Trend vs mean-reversion classification - KEPT
    - Online Baum-Welch (B7): Incremental parameter adaptation - KEPT
    - ADWIN Drift (B8): Distribution shift detection - KEPT

Legacy components (v4.0 - backward compatibility):
    - HMM Detector: 4-state Gaussian Hidden Markov Model
"""

# v5.0 Config
from .config.ahhmm_config import (
    MetaRegime,
    MarketRegime,
    EmissionParams,
    AHHMMConfig,
    DEFAULT_AHHMM_CONFIG
)
from .config.meta_regime_config import (
    MetaRegimeConfig,
    DEFAULT_META_REGIME_CONFIG
)

# v5.0 Core Components
from .student_t_ahhmm import (
    StudentTAHHMM,
    AHHMMState,
    create_regime_detector_v5
)

# B2: Meta-Regime Layer
from .meta_regime import (
    MetaRegimeLayer,
    MetaRegimeOutput,
    IntegratedRegimeDetector
)

# B3: Causal Information Geometry
from .causal_info_geometry import (
    CausalInfoGeometry,
    CryptoCorrelationMonitor,
    CIGConfig,
    CIGOutput
)

# B4: AEDL Meta-Learning
from .aedl_meta_learning import (
    AEDL,
    AEDLConfig,
    RegimeClassifier,
    AdaptiveLabeler
)

# B5: Jump Detector
from .jump_detector import (
    JumpDetector,
    JumpDetectorConfig,
    JumpOutput
)

# B6: Hurst Exponent Gating
from .hurst_gating import (
    HurstExponentGating,
    HurstConfig,
    HurstOutput
)

# B7: Online Baum-Welch
from .online_baum_welch import (
    OnlineBaumWelch,
    OnlineBWConfig
)

# B8: ADWIN Drift Detection
from .adwin_drift import (
    ADWIN,
    ADWINConfig,
    ADWINOutput,
    ADWINBucket,
    MultiFeatureADWIN
)

# Pipeline Integration
from .pipeline import (
    RegimeDetectionPipeline,
    RegimePipelineOutput,
    create_regime_pipeline
)

# v4.0 Legacy (backward compatibility)
from .hmm_detector import (
    HMMRegimeDetector,
    HMMConfig,
    HMMState,
    RegimeLabel
)

__all__ = [
    # v5.0 Config
    'MetaRegime',
    'MarketRegime',
    'EmissionParams',
    'AHHMMConfig',
    'DEFAULT_AHHMM_CONFIG',
    'MetaRegimeConfig',
    'DEFAULT_META_REGIME_CONFIG',
    
    # v5.0 Core Components
    'StudentTAHHMM',
    'AHHMMState',
    'create_regime_detector_v5',
    
    # B2: Meta-Regime
    'MetaRegimeLayer',
    'MetaRegimeOutput',
    'IntegratedRegimeDetector',
    
    # B3: Causal Info Geometry
    'CausalInfoGeometry',
    'CryptoCorrelationMonitor',
    'CIGConfig',
    'CIGOutput',
    
    # B4: AEDL
    'AEDL',
    'AEDLConfig',
    'RegimeClassifier',
    'AdaptiveLabeler',
    
    # B5: Jump Detector
    'JumpDetector',
    'JumpDetectorConfig',
    'JumpOutput',
    
    # B6: Hurst Gating
    'HurstExponentGating',
    'HurstConfig',
    'HurstOutput',
    
    # B7: Online Baum-Welch
    'OnlineBaumWelch',
    'OnlineBWConfig',
    
    # B8: ADWIN
    'ADWIN',
    'ADWINConfig',
    'ADWINOutput',
    'ADWINBucket',
    'MultiFeatureADWIN',
    
    # Pipeline
    'RegimeDetectionPipeline',
    'RegimePipelineOutput',
    'create_regime_pipeline',
    
    # v4.0 Legacy
    'HMMRegimeDetector',
    'HMMConfig',
    'HMMState',
    'RegimeLabel'
]
