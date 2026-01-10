# ============================================================================
# FILE: vec_normalize.py
# PURPOSE: Dynamic normalization wrapper for RL environments
# STATUS: KEEP from v4.0
# LATENCY: <0.1ms per call
# ============================================================================

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class VecNormalizeConfig:
    """
    VecNormalize configuration.
    
    Attributes:
        clip_obs: Observation clipping threshold
        clip_reward: Reward clipping threshold
        gamma: Discount factor for return normalization
        epsilon: Small constant for numerical stability
    """
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    gamma: float = 0.99
    epsilon: float = 1e-8


class RunningMeanStd:
    """
    Running mean and standard deviation calculator.
    
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        
    def update(self, x: np.ndarray) -> None:
        """Update statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int
    ) -> None:
        """Update from batch moments using parallel algorithm."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


class VecNormalize:
    """
    Vectorized normalization wrapper.
    
    Maintains running statistics and normalizes observations/rewards.
    Compatible with Stable-Baselines3 interface.
    
    Why VecNormalize?
    - Neural networks train better with normalized inputs
    - Running statistics adapt to changing distributions
    - Handles scale differences between features
    
    Performance: Baseline component, enables other improvements
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        config: Optional[VecNormalizeConfig] = None
    ):
        self.config = config or VecNormalizeConfig()
        self.observation_shape = observation_shape
        
        # Running statistics
        self.obs_rms = RunningMeanStd(observation_shape)
        self.ret_rms = RunningMeanStd(())
        
        # Return tracking for reward normalization
        self._returns = 0.0
        
        # Mode flags
        self.training = True
        self.norm_obs = True
        self.norm_reward = True
        
    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation."""
        if self.training:
            self.obs_rms.update(obs.reshape(-1, *self.observation_shape))
        
        normalized = (obs - self.obs_rms.mean) / (self.obs_rms.std + self.config.epsilon)
        
        return np.clip(normalized, -self.config.clip_obs, self.config.clip_obs)
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize reward using discounted return statistics."""
        if self.training:
            self._returns = self._returns * self.config.gamma + reward
            self.ret_rms.update(np.array([self._returns]))
        
        normalized = reward / (self.ret_rms.std + self.config.epsilon)
        
        return np.clip(normalized, -self.config.clip_reward, self.config.clip_reward)
    
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation (main interface)."""
        return self.normalize_obs(obs)
    
    def reset(self) -> None:
        """Reset return tracking (call at episode start)."""
        self._returns = 0.0
    
    def set_training_mode(self, training: bool) -> None:
        """Set training mode (updates statistics only when training)."""
        self.training = training
    
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """Return current statistics for logging/checkpointing."""
        return {
            'obs_mean': self.obs_rms.mean.copy(),
            'obs_var': self.obs_rms.var.copy(),
            'obs_count': self.obs_rms.count,
            'ret_mean': float(self.ret_rms.mean),
            'ret_var': float(self.ret_rms.var)
        }
    
    def load_statistics(self, stats: Dict[str, np.ndarray]) -> None:
        """Load statistics from checkpoint."""
        self.obs_rms.mean = stats['obs_mean']
        self.obs_rms.var = stats['obs_var']
        self.obs_rms.count = stats['obs_count']
        self.ret_rms.mean = np.array(stats['ret_mean'])
        self.ret_rms.var = np.array(stats['ret_var'])
