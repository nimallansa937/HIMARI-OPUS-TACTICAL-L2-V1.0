# ============================================================================
# HIMARI Layer 2 - Part A: Data Preprocessing
# Version: 5.0
# ============================================================================
"""
Part A: Data Preprocessing for HIMARI Layer 2

This package provides 8 preprocessing methods for transforming raw market data
into clean, normalized, augmented feature vectors:

A1: Extended Kalman Filter (EKF) - Non-linear denoising with 4D state
A2: Conversational Autoencoders (CAE) - Speaker-listener signal separation
A3: Frequency Domain Normalization - Adaptive spectral normalization
A4: TimeGAN Augmentation - Synthetic data generation
A5: Tab-DDPM Diffusion - Tail event synthesis
A6: VecNormalize - Running mean/std standardization
A7: Orthogonal Initialization - Weight initialization
A8: Online Augmentation - Runtime data augmentation

Total Sharpe Contribution: +0.15
"""

# A1: Extended Kalman Filter
from .ekf_denoiser import (
    EKFDenoiser,
    EKFConfig,
    EKFBatch,
    migrate_kalman_to_ekf
)

# A2: Conversational Autoencoders
from .conversational_ae import (
    ConversationalAutoencoder,
    CAEConfig,
    CAETrainer,
    CAEInference,
    AutoencoderLSTM,
    AutoencoderTransformer
)

# A3: Frequency Domain Normalization
from .freq_normalizer import (
    FrequencyDomainNormalizer,
    FreqNormConfig,
    MultiChannelFreqNormalizer,
    AdaptiveFreqNormalizer
)

# A4: TimeGAN Augmentation
from .timegan_augment import (
    TimeGAN,
    TimeGANConfig,
    augment_dataset_v5
)

# A5: Tab-DDPM Diffusion
from .tab_ddpm import (
    TabDDPM,
    TabDDPMConfig
)

# A6: VecNormalize
from .vec_normalize import (
    VecNormalize,
    VecNormalizeConfig,
    RunningMeanStd
)

# A7: Orthogonal Initialization
from .initialization import (
    orthogonal_init,
    init_weights,
    InitializedLinear,
    InitializedLSTM
)

# A8: Online Augmentation
from .online_augment import (
    OnlineAugmentor,
    OnlineAugmentConfig,
    AugmentationPipeline,
    create_default_augmentor,
    create_conservative_augmentor,
    create_aggressive_augmentor
)

# Pipeline Integration
from .pipeline import (
    PreprocessingPipeline,
    PreprocessingPipelineConfig
)

__version__ = '5.0.0'
__all__ = [
    # A1
    'EKFDenoiser', 'EKFConfig', 'EKFBatch', 'migrate_kalman_to_ekf',
    # A2
    'ConversationalAutoencoder', 'CAEConfig', 'CAETrainer', 'CAEInference',
    'AutoencoderLSTM', 'AutoencoderTransformer',
    # A3
    'FrequencyDomainNormalizer', 'FreqNormConfig', 
    'MultiChannelFreqNormalizer', 'AdaptiveFreqNormalizer',
    # A4
    'TimeGAN', 'TimeGANConfig', 'augment_dataset_v5',
    # A5
    'TabDDPM', 'TabDDPMConfig',
    # A6
    'VecNormalize', 'VecNormalizeConfig', 'RunningMeanStd',
    # A7
    'orthogonal_init', 'init_weights', 'InitializedLinear', 'InitializedLSTM',
    # A8
    'OnlineAugmentor', 'OnlineAugmentConfig', 'AugmentationPipeline',
    'create_default_augmentor', 'create_conservative_augmentor',
    'create_aggressive_augmentor',
    # Pipeline
    'PreprocessingPipeline', 'PreprocessingPipelineConfig'
]
