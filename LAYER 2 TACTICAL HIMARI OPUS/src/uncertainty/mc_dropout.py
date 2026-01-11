"""
HIMARI Layer 2 - MC Dropout Uncertainty
Subsystem F: Monte Carlo Dropout (F1 alternative)

Purpose:
    Bayesian uncertainty estimation via dropout at test time.

Method:
    F1.5: MC Dropout - Enable dropout during inference, sample N times

Expected Performance:
    - Comparable to ensemble uncertainty
    - 10× faster than full ensemble
    - Works with single model

Reference:
    - Gal & Ghahramani "Dropout as a Bayesian Approximation" (2016)
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class MCDropoutConfig:
    """Configuration for MC Dropout"""
    n_samples: int = 20  # Number of forward passes
    dropout_rate: float = 0.1  # Dropout probability
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MCDropoutWrapper(nn.Module):
    """
    Wraps any PyTorch model to enable MC Dropout uncertainty estimation.

    Example:
        >>> base_model = PPOLSTMAgent(...)
        >>> mc_model = MCDropoutWrapper(base_model, dropout_rate=0.1)
        >>> mean, std = mc_model.predict_with_uncertainty(state, n_samples=20)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[MCDropoutConfig] = None
    ):
        super().__init__()
        self.model = model
        self.config = config or MCDropoutConfig()

        # Add dropout layers if not present
        self._add_dropout_to_model()

        logger.info(f"MC Dropout initialized with {self.config.n_samples} samples, "
                   f"dropout_rate={self.config.dropout_rate}")

    def _add_dropout_to_model(self):
        """
        Recursively add dropout after each layer if not present.

        For models that already have dropout, this does nothing.
        """
        def add_dropout_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, (nn.Linear, nn.LSTM, nn.GRU)):
                    # Check if next module is already dropout
                    children_list = list(module.children())
                    idx = children_list.index(child)
                    if idx + 1 < len(children_list):
                        next_module = children_list[idx + 1]
                        if not isinstance(next_module, nn.Dropout):
                            # Insert dropout
                            setattr(module, f"{name}_dropout",
                                   nn.Dropout(p=self.config.dropout_rate))
                add_dropout_recursive(child)

        add_dropout_recursive(self.model)

    def enable_dropout(self):
        """Enable dropout at test time"""
        def enable_dropout_recursive(module):
            for m in module.modules():
                if isinstance(m, nn.Dropout):
                    m.train()  # Force dropout to be active

        enable_dropout_recursive(self.model)

    def forward(self, x):
        """Standard forward pass"""
        return self.model(x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict with MC Dropout uncertainty estimation.

        Args:
            x: Input tensor
            n_samples: Number of stochastic forward passes

        Returns:
            mean: Mean prediction across samples
            std: Standard deviation (epistemic uncertainty)
        """
        n_samples = n_samples or self.config.n_samples

        # Enable dropout
        self.enable_dropout()

        # Collect samples
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(x)
                predictions.append(pred.cpu().numpy())

        # Compute statistics
        predictions = np.array(predictions)  # (n_samples, batch_size, output_dim)

        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        return mean, std

    def predict_action_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: Optional[int] = None
    ) -> tuple[int, float]:
        """
        Predict action with uncertainty for trading.

        Args:
            x: Input state
            n_samples: Number of samples

        Returns:
            action: Most likely action
            uncertainty: Standard deviation of action probabilities
        """
        mean, std = self.predict_with_uncertainty(x, n_samples)

        # Get action from mean prediction
        if mean.ndim > 1:
            mean = mean[0]  # Take first sample if batched
            std = std[0]

        action = int(np.argmax(mean))
        uncertainty = float(np.mean(std))  # Average std across action dims

        return action, uncertainty

    def calibrate_uncertainty(
        self,
        calibration_data: list[tuple[torch.Tensor, int]],
        n_samples: Optional[int] = None
    ) -> float:
        """
        Calibrate uncertainty estimates on validation set.

        Args:
            calibration_data: List of (state, true_action) pairs
            n_samples: Number of MC samples

        Returns:
            calibration_error: Mean absolute error between uncertainty and error
        """
        uncertainties = []
        errors = []

        for state, true_action in calibration_data:
            mean, std = self.predict_with_uncertainty(state, n_samples)

            if mean.ndim > 1:
                mean = mean[0]
                std = std[0]

            pred_action = int(np.argmax(mean))
            uncertainty = float(np.mean(std))
            error = float(pred_action != true_action)

            uncertainties.append(uncertainty)
            errors.append(error)

        # Compute correlation between uncertainty and error
        uncertainties = np.array(uncertainties)
        errors = np.array(errors)

        # Normalize to [0, 1]
        if np.std(uncertainties) > 0:
            uncertainties = (uncertainties - np.min(uncertainties)) / \
                           (np.max(uncertainties) - np.min(uncertainties) + 1e-8)

        calibration_error = np.mean(np.abs(uncertainties - errors))

        logger.info(f"MC Dropout calibration error: {calibration_error:.4f}")
        return calibration_error


class MCDropoutEnsemble:
    """
    Use MC Dropout as alternative to full ensemble.

    Wraps multiple models with MC Dropout for hybrid uncertainty.
    """

    def __init__(
        self,
        models: list[nn.Module],
        config: Optional[MCDropoutConfig] = None
    ):
        self.config = config or MCDropoutConfig()
        self.mc_models = [MCDropoutWrapper(m, self.config) for m in models]

        logger.info(f"MC Dropout Ensemble with {len(models)} models")

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with both epistemic (inter-model) and aleatoric (intra-model) uncertainty.

        Args:
            x: Input tensor
            n_samples: MC samples per model

        Returns:
            mean: Overall mean prediction
            epistemic: Inter-model variance
            aleatoric: Intra-model variance (MC Dropout)
        """
        all_means = []
        all_stds = []

        for mc_model in self.mc_models:
            mean, std = mc_model.predict_with_uncertainty(x, n_samples)
            all_means.append(mean)
            all_stds.append(std)

        all_means = np.array(all_means)  # (n_models, batch_size, output_dim)
        all_stds = np.array(all_stds)

        # Overall mean
        grand_mean = np.mean(all_means, axis=0)

        # Epistemic uncertainty: variance across models
        epistemic = np.var(all_means, axis=0)

        # Aleatoric uncertainty: average intra-model variance
        aleatoric = np.mean(all_stds ** 2, axis=0)

        return grand_mean, epistemic, aleatoric


__all__ = ['MCDropoutWrapper', 'MCDropoutConfig', 'MCDropoutEnsemble']
