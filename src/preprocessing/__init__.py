"""
HIMARI Layer 2 - Preprocessing Package (v5.0)
Subsystem A: Data Preprocessing & Augmentation

Components (v5.0 - 8 methods architecture):
- A1: EKF Denoiser - Extended Kalman Filter with faux Riccati [UPGRADE]
- A2: Conversational AE - Speaker-listener denoising [NEW]
- A3: Freq Normalizer - Spectral domain normalization [NEW]
- A4: TimeGAN Augment - GAN-based synthetic data [UPGRADE]
- A5: Tab-DDPM - Diffusion model for tail events [NEW]
- A6: VecNormalize - Dynamic feature standardization [KEEP]
- A7: Orthogonal Init - Weight initialization [KEEP]
- A8: Online Augment - Real-time data augmentation [NEW]

Pipeline Integration:
- PreprocessingPipeline - Unified pipeline for all A1-A8 components

Legacy components (v4.0 - backward compatibility):
- Kalman Filter: Basic noise reduction
- Monte Carlo Augment: MJD/GARCH synthetic data
"""

# v5.0 Components - A1: Extended Kalman Filter
from .ekf_denoiser import EKFDenoiser, EKFConfig, EKFBatch, create_denoiser_from_kalman_params, migrate_kalman_to_ekf

# v5.0 Components - A2: Conversational Autoencoders
from .conversational_ae import ConversationalAutoencoder, CAEConfig, CAETrainer, CAEInference

# v5.0 Components - A3: Frequency Domain Normalization
from .freq_normalizer import FrequencyDomainNormalizer, FreqNormConfig, AdaptiveFreqNormalizer

# v5.0 Components - A4: TimeGAN Augmentation
from .timegan_augment import TimeGAN, TimeGANConfig, augment_dataset_v5

# v5.0 Components - A5: Tab-DDPM Diffusion (NEW)
from .tab_ddpm import TabDDPM, TabDDPMConfig

# v5.0 Components - A6: VecNormalize (KEEP)
from .vec_normalize import VecNormalize

# v5.0 Components - A7: Orthogonal Initialization (KEEP)
from .orthogonal_init import orthogonal_init, orthogonal_init_recursive, layer_init, OrthogonalLinear

# v5.0 Components - A8: Online Augmentation (NEW)
from .online_augment import (
    OnlineAugmentor, OnlineAugmentConfig, AugmentationPipeline,
    create_default_augmentor, create_conservative_augmentor, create_aggressive_augmentor
)

# v5.0 Pipeline Integration
from .pipeline import PreprocessingPipeline, PreprocessingPipelineConfig

# v4.0 Legacy (backward compatibility)
from .kalman_filter import TradingKalmanFilter
from .monte_carlo_augment import MonteCarloAugmenter, MJDParams, GARCHParams

__all__ = [
    # v5.0 A1: EKF
    "EKFDenoiser",
    "EKFConfig",
    "EKFBatch",
    "create_denoiser_from_kalman_params",
    "migrate_kalman_to_ekf",
    # v5.0 A2: CAE
    "ConversationalAutoencoder",
    "CAEConfig",
    "CAETrainer",
    "CAEInference",
    # v5.0 A3: Freq Norm
    "FrequencyDomainNormalizer",
    "FreqNormConfig",
    "AdaptiveFreqNormalizer",
    # v5.0 A4: TimeGAN
    "TimeGAN",
    "TimeGANConfig",
    "augment_dataset_v5",
    # v5.0 A5: Tab-DDPM
    "TabDDPM",
    "TabDDPMConfig",
    # v5.0 A6: VecNormalize
    "VecNormalize",
    # v5.0 A7: Orthogonal Init
    "orthogonal_init",
    "orthogonal_init_recursive",
    "layer_init",
    "OrthogonalLinear",
    # v5.0 A8: Online Augment
    "OnlineAugmentor",
    "OnlineAugmentConfig",
    "AugmentationPipeline",
    "create_default_augmentor",
    "create_conservative_augmentor",
    "create_aggressive_augmentor",
    # v5.0 Pipeline
    "PreprocessingPipeline",
    "PreprocessingPipelineConfig",
    # v4.0 Legacy
    "TradingKalmanFilter",
    "MonteCarloAugmenter",
    "MJDParams",
    "GARCHParams",
]
