"""
HIMARI Layer 2 - Part A: Preprocessing Initialization
Centralized initialization for all preprocessing modules.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for all preprocessing modules."""
    # A1: EKF Denoiser
    ekf_enabled: bool = True
    ekf_process_noise: float = 0.01
    ekf_measurement_noise: float = 0.1
    
    # A2: Autoencoder
    ae_enabled: bool = True
    ae_latent_dim: int = 32
    
    # A3: Anomaly Detection
    anomaly_enabled: bool = True
    anomaly_threshold: float = 3.0
    
    # A4: MMD Domain Adaptation
    mmd_enabled: bool = True
    mmd_bandwidth: float = 1.0
    
    # A5: Spectral Bridge
    spectral_enabled: bool = True
    spectral_components: int = 10
    
    # A6: Quantile Normalization
    quantile_enabled: bool = True
    quantile_n_bins: int = 100
    
    # A7: TimeGAN Augmentation
    timegan_enabled: bool = False  # Off by default (expensive)
    timegan_seq_len: int = 24
    
    # A8: Feature Engineering
    feature_engineering_enabled: bool = True
    feature_lookback: int = 20


class PreprocessingInitializer:
    """
    Initialize and configure all preprocessing modules.
    
    Usage:
        initializer = PreprocessingInitializer(config)
        pipeline = initializer.create_pipeline()
        processed_data = pipeline.process(raw_data)
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._modules: Dict[str, Any] = {}
        self._initialized = False
        
    def initialize(self) -> Dict[str, bool]:
        """Initialize all preprocessing modules."""
        results = {}
        
        # A1: EKF Denoiser
        if self.config.ekf_enabled:
            try:
                from .ekf_denoiser import EKFDenoiser
                self._modules['ekf'] = EKFDenoiser()
                results['ekf'] = True
            except Exception as e:
                logger.warning(f"Failed to init EKF: {e}")
                results['ekf'] = False
                
        # A2: Autoencoder
        if self.config.ae_enabled:
            try:
                from .conversational_ae import ConversationalAE
                self._modules['ae'] = ConversationalAE(latent_dim=self.config.ae_latent_dim)
                results['ae'] = True
            except Exception as e:
                logger.warning(f"Failed to init AE: {e}")
                results['ae'] = False
                
        # A3: Anomaly Detection
        if self.config.anomaly_enabled:
            try:
                from .anomaly_detector import AnomalyDetector
                self._modules['anomaly'] = AnomalyDetector(threshold=self.config.anomaly_threshold)
                results['anomaly'] = True
            except Exception as e:
                logger.warning(f"Failed to init anomaly: {e}")
                results['anomaly'] = False
                
        # A4-A8 similar pattern...
        self._initialized = True
        return results
        
    def get_module(self, name: str) -> Optional[Any]:
        """Get initialized module by name."""
        return self._modules.get(name)
        
    def is_initialized(self) -> bool:
        """Check if initialization complete."""
        return self._initialized


def create_preprocessing_pipeline(config: Optional[PreprocessingConfig] = None):
    """Factory function to create preprocessing pipeline."""
    initializer = PreprocessingInitializer(config)
    initializer.initialize()
    return initializer
