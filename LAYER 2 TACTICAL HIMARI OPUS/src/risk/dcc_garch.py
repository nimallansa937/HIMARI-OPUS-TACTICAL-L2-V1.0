"""
HIMARI Layer 2 - DCC-GARCH Dynamic Correlation
Subsystem H: Risk Management (Method H3)

Purpose:
    Time-varying correlation estimation using Dynamic Conditional Correlation
    GARCH for proper diversification and risk aggregation.

Why DCC-GARCH?
    - Correlations spike during stress (exactly when diversification fails)
    - Static correlation underestimates tail dependence
    - DCC captures time-varying dependence structure
    - Critical for multi-asset portfolio risk

Architecture:
    - Univariate GARCH for each asset's volatility
    - DCC model for dynamic correlation matrix
    - Efficient two-stage estimation

Reference:
    - Engle (2002), "Dynamic Conditional Correlation: A Simple Class of 
      Multivariate GARCH Models"
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger


@dataclass
class DCCConfig:
    """DCC-GARCH configuration"""
    garch_p: int = 1             # GARCH lag for variance
    garch_q: int = 1             # ARCH lag for shocks
    dcc_a: float = 0.05          # DCC alpha (short-run persistence)
    dcc_b: float = 0.90          # DCC beta (long-run persistence)
    min_variance: float = 1e-6   # Floor for variance
    window_size: int = 252       # Rolling window for estimation
    rebalance_hours: int = 4     # Correlation update frequency


class DCCGARCH:
    """
    Dynamic Conditional Correlation GARCH.
    
    Two-stage estimation:
    1. Univariate GARCH(1,1) for each asset
    2. DCC for correlation dynamics
    
    Example:
        >>> config = DCCConfig()
        >>> dcc = DCCGARCH(config, n_assets=3)
        >>> dcc.fit(returns_matrix)
        >>> corr = dcc.get_correlation()
        >>> cov = dcc.get_covariance()
    """
    
    def __init__(self, config: Optional[DCCConfig] = None, n_assets: int = 2):
        self.config = config or DCCConfig()
        self.n_assets = n_assets
        
        # GARCH parameters per asset: [omega, alpha, beta]
        self.garch_params = np.zeros((n_assets, 3))
        
        # Current conditional variances
        self.h: Optional[np.ndarray] = None  # (n_assets,)
        
        # DCC parameters
        self.dcc_a = self.config.dcc_a
        self.dcc_b = self.config.dcc_b
        
        # Unconditional correlation matrix
        self.Q_bar: Optional[np.ndarray] = None
        
        # Current Qt matrix
        self.Qt: Optional[np.ndarray] = None
        
        self._fitted = False
        
        logger.debug(f"DCCGARCH initialized: n_assets={n_assets}")
    
    def _garch_variance(self, returns: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Compute GARCH(1,1) conditional variance series.
        
        h_t = omega + alpha * e_{t-1}^2 + beta * h_{t-1}
        
        Args:
            returns: (T,) return series for single asset
            params: [omega, alpha, beta]
            
        Returns:
            (T,) conditional variance series
        """
        omega, alpha, beta = params
        T = len(returns)
        
        # Initialize with unconditional variance
        unconditional = omega / (1 - alpha - beta + 1e-8)
        h = np.zeros(T)
        h[0] = max(unconditional, self.config.min_variance)
        
        for t in range(1, T):
            h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
            h[t] = max(h[t], self.config.min_variance)
        
        return h
    
    def _garch_negloglik(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Negative log-likelihood for GARCH(1,1)"""
        omega, alpha, beta = params
        
        # Constraints
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return np.inf
        
        h = self._garch_variance(returns, params)
        T = len(returns)
        
        # Log-likelihood
        ll = -0.5 * np.sum(np.log(h) + returns**2 / h)
        return -ll
    
    def fit_univariate_garch(self, returns: np.ndarray, asset_idx: int):
        """
        Fit GARCH(1,1) to single asset.
        
        Args:
            returns: (T,) return series
            asset_idx: Asset index in portfolio
        """
        # Initial parameters
        var = np.var(returns)
        init_omega = var * 0.1
        init_alpha = 0.1
        init_beta = 0.85
        
        result = minimize(
            self._garch_negloglik,
            x0=[init_omega, init_alpha, init_beta],
            args=(returns,),
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        if result.success:
            self.garch_params[asset_idx] = result.x
        else:
            # Fallback
            self.garch_params[asset_idx] = [init_omega, init_alpha, init_beta]
    
    def fit(self, returns: np.ndarray):
        """
        Fit DCC-GARCH model.
        
        Args:
            returns: (T, n_assets) return matrix
        """
        T, n = returns.shape
        assert n == self.n_assets
        
        # Stage 1: Univariate GARCH for each asset
        H = np.zeros((T, n))
        for i in range(n):
            self.fit_univariate_garch(returns[:, i], i)
            H[:, i] = self._garch_variance(returns[:, i], self.garch_params[i])
        
        # Standardized residuals
        epsilon = returns / np.sqrt(H + 1e-8)
        
        # Stage 2: DCC estimation
        # Unconditional correlation
        self.Q_bar = np.corrcoef(epsilon.T)
        
        # Store current values
        self.h = H[-1]
        self.Qt = self.Q_bar.copy()
        
        self._fitted = True
        
        logger.info(f"DCC-GARCH fitted on {T} observations, {n} assets")
    
    def update(self, new_returns: np.ndarray):
        """
        Update model with new observations.
        
        Args:
            new_returns: (n_assets,) new return observation
        """
        if not self._fitted:
            raise ValueError("Must fit before update")
        
        # Update GARCH variances
        for i in range(self.n_assets):
            omega, alpha, beta = self.garch_params[i]
            self.h[i] = omega + alpha * new_returns[i]**2 + beta * self.h[i]
            self.h[i] = max(self.h[i], self.config.min_variance)
        
        # Standardized residuals
        epsilon = new_returns / np.sqrt(self.h + 1e-8)
        
        # Update Qt (DCC dynamics)
        outer = np.outer(epsilon, epsilon)
        self.Qt = (1 - self.dcc_a - self.dcc_b) * self.Q_bar + \
                  self.dcc_a * outer + \
                  self.dcc_b * self.Qt
    
    def get_correlation(self) -> np.ndarray:
        """
        Get current correlation matrix.
        
        Returns:
            (n_assets, n_assets) correlation matrix
        """
        if self.Qt is None:
            return np.eye(self.n_assets)
        
        # Normalize Qt to get correlation Rt
        diag = np.sqrt(np.diag(self.Qt))
        diag[diag < 1e-8] = 1e-8
        
        R = self.Qt / np.outer(diag, diag)
        
        # Ensure valid correlation matrix
        np.fill_diagonal(R, 1.0)
        R = np.clip(R, -1, 1)
        
        return R
    
    def get_covariance(self) -> np.ndarray:
        """
        Get current covariance matrix.
        
        Cov = D_t * R_t * D_t
        where D_t = diag(sqrt(h))
        
        Returns:
            (n_assets, n_assets) covariance matrix
        """
        if self.h is None:
            return np.eye(self.n_assets) * 0.01
        
        R = self.get_correlation()
        D = np.diag(np.sqrt(self.h))
        
        return D @ R @ D
    
    def get_portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Get portfolio volatility given weights.
        
        Args:
            weights: (n_assets,) portfolio weights
            
        Returns:
            Portfolio standard deviation
        """
        cov = self.get_covariance()
        var = weights @ cov @ weights
        return np.sqrt(max(var, 0))
    
    def get_correlation_change(self) -> float:
        """
        Get recent correlation change indicator.
        
        High values indicate correlation regime shift.
        
        Returns:
            Frobenius norm of correlation change
        """
        if self.Qt is None or self.Q_bar is None:
            return 0.0
        
        R = self.get_correlation()
        return np.linalg.norm(R - self.Q_bar, 'fro')
    
    def detect_correlation_spike(self, threshold: float = 0.2) -> bool:
        """
        Detect if correlations have spiked (stress indicator).
        
        Args:
            threshold: Change threshold
            
        Returns:
            True if correlation spike detected
        """
        return self.get_correlation_change() > threshold


# Factory function
def create_dcc_garch(n_assets: int = 2) -> DCCGARCH:
    """Create DCC-GARCH model"""
    config = DCCConfig()
    return DCCGARCH(config, n_assets)
