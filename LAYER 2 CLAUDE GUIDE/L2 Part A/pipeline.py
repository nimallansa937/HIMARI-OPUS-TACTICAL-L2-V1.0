# ============================================================================
# FILE: pipeline.py
# PURPOSE: Unified preprocessing pipeline integrating all A1-A8 components
# ============================================================================

import numpy as np
import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .ekf_denoiser import EKFDenoiser, EKFConfig
from .conversational_ae import CAEInference, CAEConfig
from .freq_normalizer import FrequencyDomainNormalizer, FreqNormConfig
from .vec_normalize import VecNormalize, VecNormalizeConfig
from .online_augment import OnlineAugmentor, OnlineAugmentConfig


@dataclass
class PreprocessingPipelineConfig:
    """Complete preprocessing pipeline configuration."""
    ekf: EKFConfig = None
    cae: CAEConfig = None
    freq_norm: FreqNormConfig = None
    vec_norm: VecNormalizeConfig = None
    online_augment: OnlineAugmentConfig = None
    
    # Pipeline behavior
    use_ekf: bool = True
    use_cae: bool = True
    use_freq_norm: bool = True
    use_vec_norm: bool = True
    use_online_augment: bool = True
    
    # Model paths
    cae_model_path: str = 'models/cae_v5.pt'
    
    def __post_init__(self):
        self.ekf = self.ekf or EKFConfig()
        self.cae = self.cae or CAEConfig()
        self.freq_norm = self.freq_norm or FreqNormConfig()
        self.vec_norm = self.vec_norm or VecNormalizeConfig()
        self.online_augment = self.online_augment or OnlineAugmentConfig()


class PreprocessingPipeline:
    """
    Unified preprocessing pipeline for HIMARI Layer 2.
    
    Integrates all preprocessing components (A1-A8) into a single
    callable interface for both training and inference.
    
    Pipeline order (real-time):
    1. EKF denoising (A1)
    2. CAE consensus denoising (A2)
    3. Frequency normalization (A3)
    4. VecNormalize standardization (A6)
    5. Online augmentation (A8, optional during inference)
    
    Offline augmentation (A4, A5) runs separately during training data prep.
    Weight initialization (A7) is applied during model creation.
    """
    
    def __init__(
        self,
        config: Optional[PreprocessingPipelineConfig] = None,
        device: str = 'cuda'
    ):
        self.config = config or PreprocessingPipelineConfig()
        self.device = device
        
        # Initialize components
        if self.config.use_ekf:
            self.ekf = EKFDenoiser(self.config.ekf)
        
        if self.config.use_cae:
            self.cae = CAEInference(
                self.config.cae_model_path,
                self.config.cae,
                device
            )
        
        if self.config.use_freq_norm:
            self.freq_norm = FrequencyDomainNormalizer(self.config.freq_norm)
        
        if self.config.use_vec_norm:
            self.vec_norm = VecNormalize(
                (self.config.cae.input_dim,),
                self.config.vec_norm
            )
        
        if self.config.use_online_augment:
            self.online_augment = OnlineAugmentor(self.config.online_augment)
        
        # State
        self._regime_uncertainty: float = 0.5
    
    def set_regime_uncertainty(self, uncertainty: float) -> None:
        """Update regime uncertainty for adaptive components."""
        self._regime_uncertainty = uncertainty
        if self.config.use_online_augment:
            self.online_augment.set_regime_uncertainty(uncertainty)
    
    def process_market_data(
        self,
        price: float,
        volume: float,
        features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process single market data point through pipeline.
        
        Args:
            price: Raw price observation
            volume: Raw volume observation
            features: Full feature vector (shape: input_dim,)
            
        Returns:
            Dictionary with processed features and metadata
        """
        result = {
            'original_price': price,
            'original_features': features.copy()
        }
        
        # A1: EKF denoising
        if self.config.use_ekf:
            denoised_price, uncertainty = self.ekf.update(price, volume)
            result['ekf_price'] = denoised_price
            result['ekf_uncertainty'] = uncertainty
            result['ekf_momentum'] = self.ekf.get_momentum()
            result['ekf_volatility'] = self.ekf.get_volatility_estimate()
            
            # Update features with EKF outputs
            features = features.copy()
            features[0] = denoised_price  # Assuming price is feature 0
        
        # A2: CAE denoising
        if self.config.use_cae:
            cae_result = self.cae.update(features)
            result['cae_denoised'] = cae_result['denoised']
            result['cae_ambiguity'] = cae_result['ambiguity']
            result['cae_latent'] = cae_result['fused_latent']
            
            # Update regime uncertainty
            self.set_regime_uncertainty(cae_result['ambiguity'])
            features = cae_result['denoised']
        
        # A6: VecNormalize (A3 freq_norm is for batch processing)
        if self.config.use_vec_norm:
            features = self.vec_norm(features)
        
        # A8: Online augmentation (optional, typically during training)
        if self.config.use_online_augment:
            augmented = self.online_augment.augment(features)
            result['augmented_features'] = augmented
        
        result['processed_features'] = features
        result['regime_uncertainty'] = self._regime_uncertainty
        
        return result
    
    def process_batch(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        features: np.ndarray,
        apply_augmentation: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Process batch of data through pipeline.
        
        Args:
            prices: Price array (n_samples,)
            volumes: Volume array (n_samples,)
            features: Feature array (n_samples, input_dim)
            apply_augmentation: Whether to apply online augmentation
            
        Returns:
            Dictionary with processed batches
        """
        n_samples = len(prices)
        processed = np.zeros_like(features)
        
        for i in range(n_samples):
            result = self.process_market_data(prices[i], volumes[i], features[i])
            processed[i] = result['processed_features']
        
        if apply_augmentation and self.config.use_online_augment:
            processed = self.online_augment.augment_batch(processed)
        
        return {'features': processed}
    
    def reset(self) -> None:
        """Reset all stateful components."""
        if self.config.use_ekf:
            self.ekf.reset()
        if self.config.use_cae:
            self.cae.reset()
        if self.config.use_freq_norm:
            self.freq_norm.reset()
        if self.config.use_vec_norm:
            self.vec_norm.reset()
        self._regime_uncertainty = 0.5
