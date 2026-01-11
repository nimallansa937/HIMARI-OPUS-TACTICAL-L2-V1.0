"""
HIMARI Layer 2 - Monte Carlo Data Augmentation (MJD/GARCH)
Subsystem A: Data Preprocessing (Method A4)

Purpose:
    Generate synthetic price paths that preserve statistical properties
    of real cryptocurrency data while providing novel scenarios.
Theory:
    MERTON JUMP-DIFFUSION (MJD):
    dS/S = μdt + σdW + J*dN
    
    GARCH(1,1) for volatility:
    σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

Expected Performance:
    - 10x data augmentation multiplier
    - Synthetic paths pass statistical tests for fat tails, volatility clustering
    - +10-15% Sharpe improvement from training on augmented data
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class MJDParams:
    """Parameters for Merton Jump-Diffusion model"""
    mu: float = 0.0001  # Drift per period
    sigma: float = 0.02  # Continuous volatility
    jump_intensity: float = 12.0  # Jumps per year
    jump_mean: float = -0.02  # Average jump size
    jump_std: float = 0.05  # Jump size volatility


@dataclass
class GARCHParams:
    """Parameters for GARCH(1,1) model"""
    omega: float = 0.00001  # Long-run variance
    alpha: float = 0.1  # ARCH coefficient
    beta: float = 0.85  # GARCH coefficient


class MonteCarloAugmenter:
    """
    Generate synthetic price paths using MJD + GARCH.
    
    Example:
        >>> augmenter = MonteCarloAugmenter()
        >>> augmenter.fit(real_returns)
        >>> synthetic_paths = augmenter.generate_batch(n_paths=100, n_steps=1000)
    """
    
    def __init__(
        self,
        mjd_params: Optional[MJDParams] = None,
        garch_params: Optional[GARCHParams] = None,
        periods_per_year: int = 252 * 288  # 5-min bars
    ):
        self.mjd_params = mjd_params or MJDParams()
        self.garch_params = garch_params or  GARCHParams()
        self.periods_per_year = periods_per_year
        self._fitted = False
    
    def fit(self, returns: np.ndarray) -> "MonteCarloAugmenter":
        """Fit model parameters from historical returns"""
        sigma = np.std(returns)
        jumps = returns[np.abs(returns) > 3 * sigma]
        
        if len(jumps) > 10:
            jump_freq = len(jumps) / len(returns)
            self.mjd_params.jump_intensity = jump_freq * self.periods_per_year
            self.mjd_params.jump_mean = np.mean(jumps)
            self.mjd_params.jump_std = np.std(jumps)
        
        non_jumps = returns[np.abs(returns) <= 3 * sigma]
        self.mjd_params.mu = np.mean(non_jumps)
        self.mjd_params.sigma = np.std(non_jumps)
        
        self._fitted = True
        logger.info(f"MJD params fitted: μ={self.mjd_params.mu:.6f}, σ={self.mjd_params.sigma:.4f}")
        return self
    
    def generate_garch_volatility(self, n_steps: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate GARCH(1,1) volatility path"""
        if seed is not None:
            np.random.seed(seed)
        
        omega, alpha, beta = self.garch_params.omega, self.garch_params.alpha, self.garch_params.beta
        long_run_var = omega / (1 - alpha - beta)
        
        sigma2 = np.zeros(n_steps)
        sigma2[0] = long_run_var
        z = np.random.standard_normal(n_steps)
        
        for t in range(1, n_steps):
            epsilon2 = sigma2[t-1] * z[t-1]**2
            sigma2[t] = omega + alpha * epsilon2 + beta * sigma2[t-1]
        
        return np.sqrt(sigma2)
    
    def generate_mjd_garch_path(
        self,
        n_steps: int,
        initial_price: float = 100.0,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate price path with MJD dynamics and GARCH volatility"""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate GARCH volatility
        sigma_t = self.generate_garch_volatility(n_steps, seed=None)
        
        # MJD parameters
        mu = self.mjd_params.mu
        lambda_j = self.mjd_params.jump_intensity / self.periods_per_year
        mu_j, sigma_j = self.mjd_params.jump_mean, self.mjd_params.jump_std
        
        # Brownian motion with time-varying volatility
        z = np.random.standard_normal(n_steps)
        dW = sigma_t * z
        
        # Jumps
        n_jumps = np.random.poisson(lambda_j, n_steps)
        jump_sizes = np.zeros(n_steps)
        for i in range(n_steps):
            if n_jumps[i] > 0:
                jumps = np.random.normal(mu_j, sigma_j, n_jumps[i])
                jump_sizes[i] = np.sum(jumps)
        
        # Returns and prices
        returns = mu + dW + jump_sizes
        prices = initial_price * np.exp(np.cumsum(returns))
        prices = np.insert(prices, 0, initial_price)
        
        return prices, returns
    
    def generate_batch(
        self,
        n_paths: int,
        n_steps: int,
        initial_price: float = 100.0,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate batch of synthetic paths"""
        if seed is not None:
            np.random.seed(seed)
        
        all_prices = np.zeros((n_paths, n_steps + 1))
        all_returns = np.zeros((n_paths, n_steps))
        
        for i in range(n_paths):
            prices, returns = self.generate_mjd_garch_path(n_steps, initial_price, seed=None)
            all_prices[i] = prices
            all_returns[i] = returns
        
        logger.info(f"Generated {n_paths} synthetic paths of {n_steps} steps each")
        return all_prices, all_returns
    
    def augment_dataset(self, real_returns: np.ndarray, multiplier: int = 10) -> np.ndarray:
        """Augment real dataset with synthetic paths"""
        if not self._fitted:
            self.fit(real_returns)
        
        n_steps = len(real_returns)
        _, synthetic_returns = self.generate_batch(n_paths=multiplier, n_steps=n_steps)
        
        augmented = np.vstack([real_returns.reshape(1, -1), synthetic_returns])
        logger.info(f"Dataset augmented: 1 real + {multiplier} synthetic = {len(augmented)} total")
        return augmented
