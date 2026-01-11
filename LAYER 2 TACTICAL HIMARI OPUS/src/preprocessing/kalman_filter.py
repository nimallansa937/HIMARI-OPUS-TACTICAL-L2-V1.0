"""
HIMARI Layer 2 - Kalman Filter for Financial Time Series
Subsystem A: Data Preprocessing

Purpose:
    Optimal noise reduction for price/indicator signals while preserving genuine patterns.
    Implements constant-velocity Kalman filter with state: [level, trend]

Testing Criteria:
    - Filter output variance < input variance
    - Innovations are white noise (Ljung-Box test p > 0.05)
    - No systematic lag > 3 bars
"""

from typing import Optional, Tuple
import numpy as np
from filterpy.kalman import KalmanFilter as FilterPyKalman
from loguru import logger


class TradingKalmanFilter:
    """
    Kalman filter optimized for financial time series.
    
    Implements constant-velocity model:
    - State: [level, trend]
    - Observation: price/indicator value
    
    Example:
        >>> kf = TradingKalmanFilter(process_noise=1e-5, measurement_noise=1e-2)
        >>> filtered_price = kf.filter(raw_prices)
    """
    
    def __init__(
        self,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-2,
        initial_state_mean: Optional[float] = None,
        initial_state_covariance: float = 1.0
    ):
        """
        Initialize Kalman filter for trading signals.
        
        Args:
            process_noise: Q matrix diagonal. Higher = more responsive. Range: [1e-7, 1e-3]
            measurement_noise: R value. Higher = smoother output, more lag. Range: [1e-4, 1e-1]
            initial_state_mean: Starting value. If None, uses first observation.
            initial_state_covariance: Initial uncertainty in state estimate.
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        
        # Initialize filterpy Kalman filter
        self.kf = FilterPyKalman(dim_x=2, dim_z=1)
        
        # State transition matrix: [level, trend] -> [level + trend, trend]
        self.kf.F = np.array([
            [1., 1.],
            [0., 1.]
        ])
        
        # Measurement matrix: observe level only
        self.kf.H = np.array([[1., 0.]])
        
        # Process noise covariance
        self.kf.Q = np.array([
            [process_noise, 0.],
            [0., process_noise]
        ])
        
        # Measurement noise covariance
        self.kf.R = np.array([[measurement_noise]])
        
        # Initial state
        self.kf.x = np.array([[0.], [0.]])
        self.kf.P = np.eye(2) * initial_state_covariance
        
        self._initialized = False
        
        logger.debug(
            f"KalmanFilter initialized: Q={process_noise:.2e}, R={measurement_noise:.2e}"
        )
    
    def reset(self, initial_value: Optional[float] = None):
        """Reset filter state"""
        if initial_value is not None:
            self.kf.x = np.array([[initial_value], [0.]])
        else:
            self.kf.x = np.array([[0.], [0.]])
        self.kf.P = np.eye(2) * self.initial_state_covariance
        self._initialized = False
    
    def update(self, observation: float) -> Tuple[float, float]:
        """
        Process single observation and return filtered value.
        
        Args:
            observation: Raw observation value
            
        Returns:
            Tuple of (filtered_value, uncertainty)
        """
        if not self._initialized:
            self.kf.x = np.array([[observation], [0.]])
            self._initialized = True
            return observation, self.initial_state_covariance
        
        # Predict
        self.kf.predict()
        
        # Update
        self.kf.update(np.array([[observation]]))
        
        filtered_value = self.kf.x[0, 0]
        uncertainty = self.kf.P[0, 0]
        
        return filtered_value, uncertainty
    
    def filter(self, observations: np.ndarray) -> np.ndarray:
        """
        Filter entire sequence of observations.
        
        Args:
            observations: 1D array of raw observations
            
        Returns:
            1D array of filtered values
        """
        self.reset(observations[0] if len(observations) > 0 else None)
        
        filtered = np.zeros_like(observations)
        
        for i, obs in enumerate(observations):
            filtered[i], _ = self.update(obs)
        
        return filtered
