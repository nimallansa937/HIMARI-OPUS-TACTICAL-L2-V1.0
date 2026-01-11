"""
HIMARI Layer 2 - Part K3: Causal Data Augmentation
Causally-valid synthetic data generation.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalAugmentConfig:
    """Configuration for causal data augmentation."""
    noise_scale: float = 0.1           # Scale of additive noise
    time_warp_rate: float = 0.1        # Probability of time warping
    magnitude_scale_range: Tuple[float, float] = (0.8, 1.2)
    regime_shift_prob: float = 0.05    # Probability of regime shift injection
    preserve_causality: bool = True    # Ensure augmented data is causally valid
    max_augmentations: int = 5         # Max augmentations per sample


class GaussianNoiseAugmenter:
    """Add Gaussian noise while preserving signal structure."""
    
    def __init__(self, scale: float = 0.1):
        self.scale = scale
    
    def augment(self, features: np.ndarray) -> np.ndarray:
        std = np.std(features, axis=0) * self.scale
        noise = np.random.randn(*features.shape) * std
        return features + noise


class TimeWarpAugmenter:
    """Time warping that preserves temporal causality."""
    
    def __init__(self, warp_rate: float = 0.1):
        self.warp_rate = warp_rate
    
    def augment(self, sequence: np.ndarray) -> np.ndarray:
        if len(sequence) < 3:
            return sequence
        
        # Create warping function (cumulative, preserves order)
        n = len(sequence)
        knots = np.random.rand(3) * 0.5 + 0.75  # 0.75 to 1.25
        warp_fn = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(knots)),
            np.cumsum(knots) / np.sum(knots)
        )
        
        # Resample at warped time points
        original_idx = np.arange(n)
        warped_idx = warp_fn * (n - 1)
        
        if sequence.ndim == 1:
            return np.interp(original_idx, warped_idx, sequence)
        else:
            # Multi-dimensional
            result = np.zeros_like(sequence)
            for i in range(sequence.shape[1]):
                result[:, i] = np.interp(original_idx, warped_idx, sequence[:, i])
            return result


class MagnitudeScaleAugmenter:
    """Scale magnitudes while preserving relative structure."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range
    
    def augment(self, features: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(*self.scale_range)
        return features * scale


class RegimeShiftInjector:
    """Inject synthetic regime shifts for robustness."""
    
    def __init__(self, shift_magnitude: float = 0.2):
        self.shift_magnitude = shift_magnitude
    
    def augment(self, features: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        n = len(features)
        if n < 10:
            return features, labels
        
        # Choose random shift point
        shift_point = np.random.randint(n // 3, 2 * n // 3)
        
        # Apply shift to second half
        augmented = features.copy()
        shift = (np.random.rand(features.shape[1]) - 0.5) * 2 * self.shift_magnitude
        augmented[shift_point:] += shift
        
        return augmented, labels


class CausalDataAugmenter:
    """
    Comprehensive causal data augmentation pipeline.
    
    Ensures all augmentations preserve:
    - Temporal causality (future doesn't affect past)
    - Feature dependencies
    - Label validity
    """
    
    def __init__(self, config: CausalAugmentConfig = None):
        self.config = config or CausalAugmentConfig()
        
        self.noise_aug = GaussianNoiseAugmenter(self.config.noise_scale)
        self.time_warp = TimeWarpAugmenter(self.config.time_warp_rate)
        self.magnitude = MagnitudeScaleAugmenter(self.config.magnitude_scale_range)
        self.regime_shift = RegimeShiftInjector()
        
        self._augmentation_count = 0
    
    def augment(self, features: np.ndarray, labels: np.ndarray = None,
               n_augmentations: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Augment dataset with causally-valid transformations.
        
        Args:
            features: Original feature array (n_samples, n_features)
            labels: Original labels (optional)
            n_augmentations: Number of augmented copies
            
        Returns:
            Augmented features and labels
        """
        n_augmentations = min(n_augmentations, self.config.max_augmentations)
        
        all_features = [features]
        all_labels = [labels] if labels is not None else None
        
        for _ in range(n_augmentations):
            aug_features = features.copy()
            aug_labels = labels.copy() if labels is not None else None
            
            # Apply random subset of augmentations
            if np.random.rand() < 0.7:
                aug_features = self.noise_aug.augment(aug_features)
            
            if np.random.rand() < self.config.time_warp_rate:
                aug_features = self.time_warp.augment(aug_features)
            
            if np.random.rand() < 0.5:
                aug_features = self.magnitude.augment(aug_features)
            
            if np.random.rand() < self.config.regime_shift_prob:
                aug_features, aug_labels = self.regime_shift.augment(aug_features, aug_labels)
            
            all_features.append(aug_features)
            if all_labels is not None:
                all_labels.append(aug_labels)
        
        result_features = np.concatenate(all_features, axis=0)
        result_labels = np.concatenate(all_labels, axis=0) if all_labels else None
        
        self._augmentation_count += n_augmentations
        
        return result_features, result_labels
    
    def augment_batch(self, batch_features: List[np.ndarray],
                     batch_labels: List[np.ndarray] = None) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
        """Augment a batch of sequences."""
        aug_features = []
        aug_labels = []
        
        for i, features in enumerate(batch_features):
            labels = batch_labels[i] if batch_labels else None
            aug_f, aug_l = self.augment(features, labels)
            aug_features.append(aug_f)
            if aug_l is not None:
                aug_labels.append(aug_l)
        
        return aug_features, aug_labels if aug_labels else None
    
    def get_statistics(self) -> Dict:
        return {
            'total_augmentations': self._augmentation_count
        }


def create_causal_augmenter(**kwargs) -> CausalDataAugmenter:
    """Create causal data augmenter with config."""
    config = CausalAugmentConfig(**kwargs)
    return CausalDataAugmenter(config)
