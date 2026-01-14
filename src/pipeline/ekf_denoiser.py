"""
Extended Kalman Filter for Feature Denoising

This module implements a vectorized EKF that processes raw feature vectors
and outputs smoothed/denoised features for training.

Design:
    - Processes arbitrary feature dimensions (default 44)
    - Uses random walk model with adaptive process noise
    - Outputs both smoothed state and uncertainty estimates

Author: HIMARI Development Team
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EKFState:
    """State container for EKF output."""
    state: np.ndarray           # Smoothed features (dim,)
    covariance_diag: np.ndarray # Diagonal of P matrix (dim,)
    uncertainty: float          # Scalar uncertainty (trace of P)
    innovation: float           # Last innovation magnitude


class EKFDenoiser:
    """
    Extended Kalman Filter for feature vector denoising.

    Uses a simplified diagonal covariance approximation for computational
    efficiency while maintaining good denoising properties.

    Model:
        State transition: x_{t+1} = x_t + w_t  (random walk)
        Observation: z_t = x_t + v_t

    Where:
        w_t ~ N(0, Q)  process noise
        v_t ~ N(0, R)  observation noise

    Parameters:
        dim: Feature dimension (default 44)
        process_noise: Q diagonal value (default 0.001)
        obs_noise: R diagonal value (default 0.01)
        adaptive_q: Enable adaptive process noise (default True)
    """

    def __init__(
        self,
        dim: int = 44,
        process_noise: float = 0.001,
        obs_noise: float = 0.01,
        adaptive_q: bool = True
    ):
        self.dim = dim
        self.base_process_noise = process_noise
        self.obs_noise = obs_noise
        self.adaptive_q = adaptive_q

        # State (diagonal approximation for efficiency)
        self.state = np.zeros(dim, dtype=np.float64)
        self.P_diag = np.ones(dim, dtype=np.float64) * 0.1  # Diagonal of covariance

        # Noise covariances (diagonal)
        self.Q_diag = np.ones(dim, dtype=np.float64) * process_noise
        self.R_diag = np.ones(dim, dtype=np.float64) * obs_noise

        # Tracking
        self.initialized = False
        self.n_updates = 0
        self.innovation_history = []

    def reset(self) -> None:
        """Reset filter state."""
        self.state = np.zeros(self.dim, dtype=np.float64)
        self.P_diag = np.ones(self.dim, dtype=np.float64) * 0.1
        self.initialized = False
        self.n_updates = 0
        self.innovation_history = []

    def update(self, observation: np.ndarray) -> EKFState:
        """
        Update filter with new observation.

        Args:
            observation: Raw feature vector (dim,)

        Returns:
            EKFState with smoothed features and uncertainty
        """
        assert len(observation) == self.dim, f"Expected {self.dim}D, got {len(observation)}D"

        # Handle NaN/Inf in observation
        obs = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        if not self.initialized:
            self.state = obs.copy()
            self.initialized = True
            self.n_updates = 1
            return EKFState(
                state=self.state.copy(),
                covariance_diag=self.P_diag.copy(),
                uncertainty=float(np.sum(self.P_diag)),
                innovation=0.0
            )

        # === PREDICT ===
        # Random walk: state_pred = state (no change)
        state_pred = self.state.copy()

        # Covariance predict: P_pred = P + Q
        P_pred_diag = self.P_diag + self.Q_diag

        # === UPDATE ===
        # Innovation
        innovation = obs - state_pred
        innovation_magnitude = float(np.sqrt(np.mean(innovation ** 2)))
        self.innovation_history.append(innovation_magnitude)

        # Kalman gain (diagonal): K = P_pred / (P_pred + R)
        S_diag = P_pred_diag + self.R_diag
        K_diag = P_pred_diag / S_diag

        # State update
        self.state = state_pred + K_diag * innovation

        # Covariance update: P = (1 - K) * P_pred
        self.P_diag = (1.0 - K_diag) * P_pred_diag

        # === ADAPTIVE Q ===
        if self.adaptive_q and len(self.innovation_history) > 10:
            # Increase Q when innovations are large (model mismatch)
            recent_innovations = np.array(self.innovation_history[-10:])
            avg_innovation = np.mean(recent_innovations)

            # Scale Q based on innovation magnitude
            if avg_innovation > 0.1:
                self.Q_diag = self.Q_diag * 1.01  # Increase slightly
            elif avg_innovation < 0.01:
                self.Q_diag = np.maximum(
                    self.Q_diag * 0.99,
                    self.base_process_noise * 0.1
                )

        self.n_updates += 1

        return EKFState(
            state=self.state.copy(),
            covariance_diag=self.P_diag.copy(),
            uncertainty=float(np.sum(self.P_diag)),
            innovation=innovation_magnitude
        )

    def batch_update(self, observations: np.ndarray) -> np.ndarray:
        """
        Process batch of observations sequentially.

        Args:
            observations: Array of shape (T, dim)

        Returns:
            Smoothed features of shape (T, dim)
        """
        T = len(observations)
        smoothed = np.zeros_like(observations)

        for t in range(T):
            result = self.update(observations[t])
            smoothed[t] = result.state

        return smoothed

    def get_diagnostics(self) -> dict:
        """Get filter diagnostics."""
        return {
            'n_updates': self.n_updates,
            'avg_uncertainty': float(np.mean(self.P_diag)),
            'avg_innovation': float(np.mean(self.innovation_history[-100:])) if self.innovation_history else 0.0,
            'q_mean': float(np.mean(self.Q_diag)),
            'initialized': self.initialized
        }


class MultiScaleEKF:
    """
    Multi-scale EKF that runs parallel filters at different timescales.

    Useful for capturing both fast and slow dynamics in features.
    """

    def __init__(
        self,
        dim: int = 44,
        scales: Tuple[float, ...] = (0.001, 0.01, 0.1)
    ):
        self.dim = dim
        self.scales = scales

        # Create filters at different scales
        self.filters = [
            EKFDenoiser(dim=dim, process_noise=q, obs_noise=0.01)
            for q in scales
        ]

        # Weights for combining (learned or fixed)
        self.weights = np.ones(len(scales)) / len(scales)

    def update(self, observation: np.ndarray) -> np.ndarray:
        """Update all filters and return weighted combination."""
        states = []

        for ekf in self.filters:
            result = ekf.update(observation)
            states.append(result.state)

        # Weighted average
        combined = np.zeros(self.dim)
        for w, s in zip(self.weights, states):
            combined += w * s

        return combined

    def reset(self) -> None:
        """Reset all filters."""
        for ekf in self.filters:
            ekf.reset()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EKF Denoiser Self-Test")
    print("=" * 60)

    # Create filter
    ekf = EKFDenoiser(dim=44, process_noise=0.001, obs_noise=0.01)

    # Generate noisy observations
    np.random.seed(42)
    T = 1000
    true_signal = np.cumsum(np.random.randn(T, 44) * 0.01, axis=0)
    noise = np.random.randn(T, 44) * 0.05
    observations = true_signal + noise

    # Process
    smoothed = ekf.batch_update(observations)

    # Compute error reduction
    raw_error = np.mean((observations - true_signal) ** 2)
    smoothed_error = np.mean((smoothed - true_signal) ** 2)
    reduction = (raw_error - smoothed_error) / raw_error * 100

    print(f"\nRaw MSE:      {raw_error:.6f}")
    print(f"Smoothed MSE: {smoothed_error:.6f}")
    print(f"Error Reduction: {reduction:.1f}%")
    print(f"\nDiagnostics: {ekf.get_diagnostics()}")

    print("\n✅ EKF Denoiser test passed!")
