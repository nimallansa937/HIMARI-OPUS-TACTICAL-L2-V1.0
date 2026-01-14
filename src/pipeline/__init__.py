"""
HIMARI Layer 2 Pipeline - Enriched Dataset Generation

Modules:
    - ekf_denoiser: Extended Kalman Filter for feature smoothing
    - regime_detector: HMM-based regime labeling (4 regimes, wraps StudentTAHHMM)
    - reward_shaper: Sortino-based reward computation
    - dataset_generator: Main pipeline for generating enriched datasets

Usage:
    # Generate enriched dataset
    python -m src.pipeline.dataset_generator --input data/raw.pkl --output data/enriched.pkl

    # Or programmatically:
    from src.pipeline import DatasetGenerator
    generator = DatasetGenerator()
    dataset = generator.generate("data/raw.pkl", "data/enriched.pkl")
"""

from .ekf_denoiser import EKFDenoiser, EKFState, MultiScaleEKF
from .regime_detector import RegimeDetector, RegimeLabel, RegimeOutput
from .reward_shaper import SortinoRewardShaper, BatchRewardShaper, RewardComponents
from .dataset_generator import (
    DatasetGenerator,
    EnrichedSample,
    DatasetMetadata,
    generate_enriched_dataset
)

__all__ = [
    # EKF
    'EKFDenoiser',
    'EKFState',
    'MultiScaleEKF',
    # Regime
    'RegimeDetector',
    'RegimeLabel',
    'RegimeOutput',
    # Reward
    'SortinoRewardShaper',
    'BatchRewardShaper',
    'RewardComponents',
    # Dataset
    'DatasetGenerator',
    'EnrichedSample',
    'DatasetMetadata',
    'generate_enriched_dataset',
]
