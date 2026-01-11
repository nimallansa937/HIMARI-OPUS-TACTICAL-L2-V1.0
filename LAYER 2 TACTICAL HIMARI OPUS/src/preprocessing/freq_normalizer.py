"""
HIMARI Layer 2 - Frequency Domain Normalization for Non-Stationary Series
Subsystem A: Data Preprocessing (Method A3)

Purpose:
    Adapts key frequency components for non-stationary financial data where
    standard Z-score fails due to time-varying mean/variance.

Why Frequency Normalization?
    - Standard Z-score assumes stationarity (constant mean/variance)
    - Financial series have time-varying spectral characteristics
    - Adapting frequency components handles regime changes better

Mechanism:
    1. FFT transform input window
    2. Normalize amplitude spectrum by rolling mean/std per frequency
    3. Preserve phase (critical for reconstruction)
    4. Inverse FFT for normalized time-domain signal

Performance:
    - Handles distribution shift better than rolling Z-score
    - Preserves high-frequency patterns (important for HFT signals)
"""

import numpy as np
from scipy import fft
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class FreqNormConfig:
    """Frequency domain normalization configuration"""
    window_size: int = 256
    n_freq_components: int = 32  # Number of key frequencies to preserve
    adapt_rate: float = 0.1     # How fast to adapt to new distributions
    clip_amplitude: float = 5.0  # Max normalized amplitude clip


class FrequencyDomainNormalizer:
    """
    Frequency Domain Normalization for non-stationary time series.
    
    Normalizes the power spectrum while preserving phase information,
    adapting statistics exponentially for regime changes.
    
    Example:
        >>> config = FreqNormConfig(window_size=256, n_freq_components=32)
        >>> normalizer = FrequencyDomainNormalizer(config)
        >>> normalized = normalizer.normalize(raw_series)
    """
    
    def __init__(self, config: Optional[FreqNormConfig] = None):
        """
        Initialize frequency domain normalizer.
        
        Args:
            config: Configuration options. Uses defaults if None.
        """
        self.config = config or FreqNormConfig()
        self.freq_means: Optional[np.ndarray] = None
        self.freq_stds: Optional[np.ndarray] = None
        self.initialized = False
        
        logger.debug(
            f"FrequencyDomainNormalizer initialized: "
            f"n_freq={self.config.n_freq_components}, adapt_rate={self.config.adapt_rate}"
        )
    
    def reset(self):
        """Reset normalizer statistics"""
        self.freq_means = None
        self.freq_stds = None
        self.initialized = False
    
    def _initialize_stats(self, freq_amplitudes: np.ndarray):
        """Initialize frequency statistics from first window"""
        self.freq_means = freq_amplitudes.copy()
        self.freq_stds = np.ones_like(freq_amplitudes) * 0.1
        self.initialized = True
    
    def _update_stats(self, freq_amplitudes: np.ndarray):
        """Exponential moving average update of frequency statistics"""
        alpha = self.config.adapt_rate
        self.freq_means = alpha * freq_amplitudes + (1 - alpha) * self.freq_means
        variance = alpha * (freq_amplitudes - self.freq_means)**2 + \
                   (1 - alpha) * self.freq_stds**2
        self.freq_stds = np.sqrt(np.maximum(variance, 1e-8))
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize time series in frequency domain.
        
        Args:
            x: Time series of shape (window_size,)
            
        Returns:
            Normalized time series preserving temporal structure
        """
        # Handle short sequences
        if len(x) < 4:
            return x
        
        # FFT
        freq = fft.fft(x)
        amplitudes = np.abs(freq)
        phases = np.angle(freq)
        
        # Keep only key frequency components
        n_keep = min(self.config.n_freq_components, len(amplitudes) // 2)
        key_amplitudes = amplitudes[:n_keep]
        
        # Initialize or update statistics
        if not self.initialized:
            self._initialize_stats(key_amplitudes)
        else:
            self._update_stats(key_amplitudes)
        
        # Normalize amplitudes
        normalized_amplitudes = amplitudes.copy()
        normalized_key = (key_amplitudes - self.freq_means) / (self.freq_stds + 1e-8)
        
        # Clip extreme values
        normalized_key = np.clip(normalized_key, -self.config.clip_amplitude, self.config.clip_amplitude)
        normalized_amplitudes[:n_keep] = normalized_key
        
        # Mirror for negative frequencies (maintain conjugate symmetry)
        if len(amplitudes) > 2 * n_keep:
            normalized_amplitudes[-n_keep+1:] = normalized_key[-1:0:-1]
        
        # Reconstruct with normalized amplitudes and original phases
        freq_normalized = normalized_amplitudes * np.exp(1j * phases)
        x_normalized = np.real(fft.ifft(freq_normalized))
        
        return x_normalized
    
    def normalize_batch(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize batch of time series.
        
        Args:
            x: Batch of series (batch, seq_len)
            
        Returns:
            Normalized batch
        """
        return np.array([self.normalize(xi) for xi in x])
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize feature matrix where each column is a feature time series.
        
        Args:
            features: (seq_len, n_features) feature matrix
            
        Returns:
            Normalized feature matrix
        """
        n_features = features.shape[1] if features.ndim > 1 else 1
        
        if features.ndim == 1:
            return self.normalize(features)
        
        normalized = np.zeros_like(features)
        for i in range(n_features):
            # Reset between features to maintain independence
            self.reset()
            normalized[:, i] = self.normalize(features[:, i])
        
        return normalized
    
    def get_power_spectrum(self, x: np.ndarray) -> np.ndarray:
        """
        Get normalized power spectrum.
        
        Args:
            x: Input time series
            
        Returns:
            Power spectrum (amplitude squared)
        """
        freq = fft.fft(x)
        return np.abs(freq) ** 2
    
    def get_dominant_frequencies(self, x: np.ndarray, top_k: int = 5) -> np.ndarray:
        """
        Get indices of top-k dominant frequencies.
        
        Args:
            x: Input time series
            top_k: Number of frequencies to return
            
        Returns:
            Indices of dominant frequencies
        """
        power = self.get_power_spectrum(x)
        # Only consider positive frequencies
        half_len = len(power) // 2
        return np.argsort(power[:half_len])[-top_k:][::-1]


class AdaptiveFreqNormalizer:
    """
    Multi-channel adaptive frequency normalizer for feature vectors.
    
    Maintains separate statistics per feature channel for independent
    normalization with shared adaptation rate.
    """
    
    def __init__(self, n_features: int, config: Optional[FreqNormConfig] = None):
        """
        Initialize multi-channel normalizer.
        
        Args:
            n_features: Number of feature channels
            config: Configuration (shared across channels)
        """
        self.n_features = n_features
        self.config = config or FreqNormConfig()
        self.normalizers = [
            FrequencyDomainNormalizer(self.config) 
            for _ in range(n_features)
        ]
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize multi-channel features.
        
        Args:
            features: (seq_len, n_features) or (batch, seq_len, n_features)
            
        Returns:
            Normalized features
        """
        if features.ndim == 2:
            seq_len, n_feat = features.shape
            normalized = np.zeros_like(features)
            for i in range(min(n_feat, self.n_features)):
                normalized[:, i] = self.normalizers[i].normalize(features[:, i])
            return normalized
        
        elif features.ndim == 3:
            batch_size, seq_len, n_feat = features.shape
            normalized = np.zeros_like(features)
            for b in range(batch_size):
                for i in range(min(n_feat, self.n_features)):
                    normalized[b, :, i] = self.normalizers[i].normalize(features[b, :, i])
            return normalized
        
        else:
            raise ValueError(f"Expected 2D or 3D input, got {features.ndim}D")
    
    def reset(self):
        """Reset all channel normalizers"""
        for norm in self.normalizers:
            norm.reset()
