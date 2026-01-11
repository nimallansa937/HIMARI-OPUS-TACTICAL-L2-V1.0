"""
HIMARI Layer 2 - Transformer-A2C Environment Adapter
Wraps TradingEnvironment to provide context windows for transformer input.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import logging

from src.environment.trading_env import TradingEnvironment, TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class TransformerEnvConfig:
    """Configuration for transformer environment wrapper."""
    context_length: int = 100      # Timesteps for transformer context
    feature_dim: int = 44          # Features per timestep
    normalize_returns: bool = True  # Whether to normalize market returns
    

class TransformerA2CEnv:
    """
    Environment wrapper for Transformer-A2C training.
    
    Converts single-step TradingEnvironment into context-window format
    suitable for transformer input.
    
    Key differences from base TradingEnvironment:
    - Returns context window: [context_length, feature_dim]
    - Returns market_return instead of reward (reward computed by reward function)
    - Handles context buffer management
    """
    
    def __init__(
        self,
        data: np.ndarray,
        prices: np.ndarray,
        config: Optional[TransformerEnvConfig] = None,
        trading_config: Optional[TradingConfig] = None,
    ):
        """
        Initialize transformer environment.
        
        Args:
            data: Feature vectors (num_samples, feature_dim)
            prices: Corresponding prices (num_samples,)
            config: Transformer environment configuration
            trading_config: Trading configuration for base environment
        """
        self.config = config or TransformerEnvConfig()
        self.trading_config = trading_config or TradingConfig()
        
        self.data = data
        self.prices = prices
        self.num_samples = len(data)
        
        # Ensure feature_dim matches data
        actual_feature_dim = data.shape[1]
        if actual_feature_dim != self.config.feature_dim:
            logger.warning(
                f"Feature dim mismatch: config={self.config.feature_dim}, "
                f"data={actual_feature_dim}. Updating config."
            )
            self.config.feature_dim = actual_feature_dim
        
        # Base trading environment for execution
        self._base_env = TradingEnvironment(data, prices, self.trading_config)
        
        # Context buffer for transformer input
        self._context_buffer: List[np.ndarray] = []
        self._current_step = 0
        self._prev_price = 0.0
        
        logger.info(
            f"TransformerA2CEnv initialized: {self.num_samples} samples, "
            f"context={self.config.context_length}, features={self.config.feature_dim}"
        )
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and return initial context window.
        
        Returns:
            state: Initial context window [context_length, feature_dim]
            info: Reset information
        """
        # Reset base environment
        base_obs = self._base_env.reset()
        
        # Initialize context buffer with zeros padded to context length
        self._context_buffer = []
        self._current_step = 0
        
        # Get first observation from data (not from base env, which includes extra features)
        first_obs = self.data[0]
        
        # Pad context buffer with first observation repeated
        for _ in range(self.config.context_length):
            self._context_buffer.append(first_obs.copy())
        
        self._prev_price = self.prices[0]
        
        # Build context window
        context = np.stack(self._context_buffer, axis=0)
        
        info = {
            "step": 0,
            "price": self.prices[0],
        }
        
        return context, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Trading action (0=FLAT, 1=LONG, 2=SHORT)
            
        Returns:
            state: Next context window [context_length, feature_dim]
            market_return: Price return this step (not reward!)
            done: Whether episode is finished
            info: Additional information
        """
        # Map action: 0=FLAT, 1=LONG, 2=SHORT → base env: 0=SELL, 1=HOLD, 2=BUY
        base_action = self._convert_action(action)
        
        # Step base environment
        base_obs, base_reward, done, base_info = self._base_env.step(base_action)
        
        self._current_step += 1
        
        # Calculate market return
        current_price = base_info.get("price", self.prices[min(self._current_step, self.num_samples - 1)])
        market_return = (current_price - self._prev_price) / self._prev_price
        self._prev_price = current_price
        
        # Update context buffer
        # FIX: Use previous step's observation to avoid look-ahead bias
        # After incrementing _current_step, we should use _current_step - 1
        # to get the most recent *completed* bar's features
        obs_idx = max(0, min(self._current_step - 1, self.num_samples - 1))
        new_obs = self.data[obs_idx]
        
        self._context_buffer.pop(0)
        self._context_buffer.append(new_obs.copy())
        
        # Build context window
        context = np.stack(self._context_buffer, axis=0)
        
        # Enhanced info
        info = {
            **base_info,
            "market_return": market_return,
            "step": self._current_step,
        }
        
        return context, market_return, done, info
    
    def _convert_action(self, action: int) -> int:
        """
        Convert Transformer-A2C action to base environment action.
        
        Transformer-A2C: 0=FLAT, 1=LONG, 2=SHORT
        Base env: 0=SELL, 1=HOLD, 2=BUY
        """
        # FLAT (0) → HOLD (1), LONG (1) → BUY (2), SHORT (2) → SELL (0)
        mapping = {0: 1, 1: 2, 2: 0}
        return mapping[action]
    
    @property
    def observation_shape(self) -> Tuple[int, int]:
        """Shape of observation: (context_length, feature_dim)"""
        return (self.config.context_length, self.config.feature_dim)
    
    @property
    def num_actions(self) -> int:
        """Number of discrete actions."""
        return 3


class WalkForwardSplitter:
    """
    Creates walk-forward train/val/test splits for time series data.
    
    Following the training guide:
    - Train: 2020-01-01 to 2023-06-30 (3.5 years)
    - Val: 2023-07-01 to 2024-03-31 (9 months)
    - Test: 2024-04-01 to 2024-12-31 (9 months)
    """
    
    def __init__(
        self,
        data: np.ndarray,
        prices: np.ndarray,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ):
        """
        Initialize splitter.
        
        Args:
            data: Full feature dataset
            prices: Full price array
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
        """
        self.data = data
        self.prices = prices
        self.num_samples = len(data)
        
        # Calculate split indices
        self.train_end = int(self.num_samples * train_ratio)
        self.val_end = int(self.num_samples * (train_ratio + val_ratio))
        
        logger.info(
            f"WalkForwardSplitter: train=[0:{self.train_end}], "
            f"val=[{self.train_end}:{self.val_end}], "
            f"test=[{self.val_end}:{self.num_samples}]"
        )
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data."""
        return self.data[:self.train_end], self.prices[:self.train_end]
    
    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get validation data."""
        return self.data[self.train_end:self.val_end], self.prices[self.train_end:self.val_end]
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data."""
        return self.data[self.val_end:], self.prices[self.val_end:]
    
    def create_envs(
        self,
        config: Optional[TransformerEnvConfig] = None,
        trading_config: Optional[TradingConfig] = None,
    ) -> Tuple[TransformerA2CEnv, TransformerA2CEnv, TransformerA2CEnv]:
        """
        Create train, val, and test environments.
        
        Returns:
            train_env, val_env, test_env
        """
        train_data, train_prices = self.get_train_data()
        val_data, val_prices = self.get_val_data()
        test_data, test_prices = self.get_test_data()
        
        train_env = TransformerA2CEnv(train_data, train_prices, config, trading_config)
        val_env = TransformerA2CEnv(val_data, val_prices, config, trading_config)
        test_env = TransformerA2CEnv(test_data, test_prices, config, trading_config)
        
        return train_env, val_env, test_env


# ==============================================================================
# Synthetic Data Generator (for testing)
# ==============================================================================

def create_synthetic_data(
    num_samples: int = 10000,
    feature_dim: int = 44,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic data for testing.
    
    Args:
        num_samples: Number of timesteps
        feature_dim: Number of features per timestep
        seed: Random seed
        
    Returns:
        data: [num_samples, feature_dim]
        prices: [num_samples]
    """
    np.random.seed(seed)
    
    # Generate random features
    data = np.random.randn(num_samples, feature_dim).astype(np.float32)
    
    # Generate price series with random walk + trend
    returns = np.random.normal(0.0001, 0.002, num_samples)  # Mean 0.01%/step, std 0.2%
    prices = 50000 * np.cumprod(1 + returns)  # Start at $50k (BTC-like)
    
    return data.astype(np.float32), prices.astype(np.float32)
