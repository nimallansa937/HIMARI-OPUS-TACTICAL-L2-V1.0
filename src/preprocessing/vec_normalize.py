"""
HIMARI Layer 2 - VecNormalize Wrapper
Subsystem A: Data Preprocessing

Purpose:
    Dynamic feature normalization with running mean/std.
    Essential for neural network training stability.

Theory:
    z = (x - μ) / σ
    Where μ and σ are computed as exponential moving averages.

Testing Criteria:
    - |mean(normalized)| < 0.1 after warmup
    - 0.8 < std(normalized) < 1.2 after warmup
    - No NaN or Inf values
"""

from typing import Optional
import numpy as np
from loguru import logger


class VecNormalize:
    """
    Running normalization with exponential moving statistics.
    
    Example:
        >>> normalizer = VecNormalize(dim=60, clip=10.0)
        >>> normalized = normalizer.normalize(features)
    """
    
    def __init__(
        self,
        dim: int,
        clip: float = 10.0,
        epsilon: float = 1e-8,
        decay: float = 0.99
    ):
        """
        Initialize VecNormalize.
        
        Args:
            dim: Number of features
            clip: Clip normalized values to [-clip, clip]
            epsilon: Added to std to prevent division by zero
            decay: EMA decay factor (higher = slower adaptation)
        """
        self.dim = dim
        self.clip = clip
        self.epsilon = epsilon
        self.decay = decay
        
        # Running statistics
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self.count = 0
        
        logger.debug(f"VecNormalize initialized: dim={dim}, clip={clip}")
    
    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with new observation(s).
        
        Args:
            x: Array of shape (dim,) or (batch, dim)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        if self.count == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            # Welford's online algorithm for stable variance
            delta = batch_mean - self.running_mean
            total_count = self.count + batch_count
            
            self.running_mean = self.running_mean + delta * batch_count / total_count
            
            m_a = self.running_var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
            
            self.running_var = M2 / total_count
        
        self.count += batch_count
    
    def normalize(
        self,
        x: np.ndarray,
        update: bool = True
    ) -> np.ndarray:
        """
        Normalize input features.
        
        Args:
            x: Array of shape (dim,) or (batch, dim)
            update: Whether to update running statistics
            
        Returns:
            Normalized array (same shape as input)
        """
        if update:
            self.update(x)
        
        # Normalize
        normalized = (x - self.running_mean) / (np.sqrt(self.running_var) + self.epsilon)
        
        # Clip
        normalized = np.clip(normalized, -self.clip, self.clip)
        
        return normalized
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Reverse normalization"""
        return x * (np.sqrt(self.running_var) + self.epsilon) + self.running_mean
    
    def get_state(self) -> dict:
        """Get normalizer state for serialization"""
        return {
            'running_mean': self.running_mean.copy(),
            'running_var': self.running_var.copy(),
            'count': self.count
        }
    
    def set_state(self, state: dict) -> None:
        """Restore normalizer state"""
        self.running_mean = state['running_mean'].copy()
        self.running_var = state['running_var'].copy()
        self.count = state['count']
    
    def reset(self) -> None:
        """Reset running statistics"""
        self.running_mean = np.zeros(self.dim)
        self.running_var = np.ones(self.dim)
        self.count = 0
