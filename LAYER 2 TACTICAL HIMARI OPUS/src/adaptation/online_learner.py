"""
HIMARI Layer 2 - Adaptation Framework
Subsystem M: Online Learning (Methods M1-M5)

Purpose:
    Continuous model adaptation without catastrophic forgetting.

Methods:
    M1: Continuous Online Learning - Never-freeze updates every 2 weeks
    M2: Elastic Weight Consolidation - Prevent forgetting via Fisher information
    M3: Concept Drift Detection - KL divergence, AUC-ROC slope, validation loss monitoring
    M4: HMM → MAML Trigger - Regime-triggered meta-learning adaptation
    M5: Fallback Safety - Return to baseline if adapted model fails

Expected Performance:
    - Adaptation lag: <3 days for regime changes
    - Forgetting rate: <5% on historical regimes
    - Drift detection accuracy: >85%
"""

from typing import Optional, Dict, Callable, List
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from loguru import logger


@dataclass
class AdaptationConfig:
    """Configuration for online adaptation"""
    # Method M1: Continuous learning
    online_learning_enabled: bool = True
    update_interval_days: int = 14  # Update every 2 weeks
    min_samples_for_update: int = 1000

    # Method M2: EWC
    ewc_enabled: bool = True
    ewc_lambda: float = 0.4  # Weight on EWC penalty
    fisher_samples: int = 200  # Samples for Fisher information

    # Method M3: Drift detection
    drift_detection_enabled: bool = True
    kl_threshold: float = 0.15  # KL divergence threshold for drift
    validation_loss_threshold: float = 0.05  # 5% increase triggers drift
    drift_window: int = 7  # Days to monitor

    # Method M4: MAML
    maml_enabled: bool = True
    maml_inner_lr: float = 0.001
    maml_inner_steps: int = 5
    maml_adaptation_samples: int = 300

    # Method M5: Fallback
    fallback_confidence_threshold: float = 0.6
    fallback_window: int = 100  # Consecutive low-confidence predictions

    # Replay buffer for online learning
    replay_buffer_size: int = 10000
    online_learning_rate: float = 1e-4


@dataclass
class DriftDetectionResult:
    """Result of drift detection (Method M3)"""
    drift_detected: bool
    kl_divergence: float
    validation_loss_change: float
    auc_roc_slope: float
    timestamp: str = ""


class OnlineLearner:
    """
    Continuous adaptation framework.

    Example:
        >>> learner = OnlineLearner(model)
        >>> learner.detect_drift(new_predictions, new_labels)
        >>> learner.adapt_to_regime(regime_samples)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[AdaptationConfig] = None
    ):
        self.model = model
        self.config = config or AdaptationConfig()

        # Method M1: Replay buffer
        self.replay_buffer = deque(maxlen=self.config.replay_buffer_size)

        # Method M2: EWC - Fisher information matrix
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

        # Method M3: Drift detection
        self.validation_loss_history = deque(maxlen=30)  # 30 days
        self.prediction_history = deque(maxlen=1000)
        self.label_history = deque(maxlen=1000)

        # Method M5: Fallback state
        self.baseline_model_state: Optional[Dict] = None
        self.low_confidence_count = 0
        self.using_fallback = False

        # Tracking
        self.days_since_update = 0
        self.total_updates = 0

        logger.info("Online Learner initialized")

    def add_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Add experience to replay buffer (Method M1)"""
        self.replay_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        })

    def compute_fisher_information(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix for EWC (Method M2).

        Fisher information measures importance of each parameter for current task.

        Args:
            data_loader: DataLoader with representative samples

        Returns:
            Dict mapping parameter name to Fisher information
        """
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        self.model.eval()
        num_samples = 0

        for batch_idx, (states, labels) in enumerate(data_loader):
            if batch_idx >= self.config.fisher_samples:
                break

            # Forward pass
            outputs = self.model(states)
            log_probs = torch.log_softmax(outputs, dim=-1)

            # Sample from output distribution
            sampled_actions = torch.multinomial(log_probs.exp(), 1).squeeze()

            # Compute log likelihood
            log_likelihood = log_probs.gather(1, sampled_actions.unsqueeze(1)).sum()

            # Backward
            self.model.zero_grad()
            log_likelihood.backward()

            # Accumulate squared gradients (Fisher = E[grad^2])
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            num_samples += states.size(0)

        # Average
        for name in fisher:
            fisher[name] /= num_samples

        self.model.train()
        logger.info(f"Computed Fisher information from {num_samples} samples")
        return fisher

    def consolidate_knowledge(self, data_loader: torch.utils.data.DataLoader):
        """
        Consolidate current knowledge via EWC (Method M2).

        Call this before adapting to new regime to prevent forgetting.
        """
        # Compute Fisher information
        self.fisher_dict = self.compute_fisher_information(data_loader)

        # Store current optimal parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()

        logger.info("Knowledge consolidated via EWC")

    def ewc_penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty term (Method M2).

        Returns:
            EWC loss penalty
        """
        if not self.fisher_dict:
            return torch.tensor(0.0)

        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict and param.requires_grad:
                # Penalize deviations weighted by Fisher information
                penalty += (self.fisher_dict[name] * (param - self.optimal_params[name]) ** 2).sum()

        return self.config.ewc_lambda * penalty / 2.0

    def detect_drift(
        self,
        recent_predictions: np.ndarray,
        recent_labels: np.ndarray,
        validation_loss: Optional[float] = None
    ) -> DriftDetectionResult:
        """
        Detect concept drift (Method M3).

        Args:
            recent_predictions: Recent model predictions
            recent_labels: Recent ground truth labels
            validation_loss: Current validation loss

        Returns:
            DriftDetectionResult with drift status and metrics
        """
        # Update history
        self.prediction_history.extend(recent_predictions)
        self.label_history.extend(recent_labels)
        if validation_loss is not None:
            self.validation_loss_history.append(validation_loss)

        drift_detected = False
        kl_div = 0.0
        val_loss_change = 0.0
        auc_slope = 0.0

        # Check 1: KL divergence on prediction distributions
        if len(self.prediction_history) >= 200:
            old_preds = np.array(list(self.prediction_history)[:100])
            new_preds = np.array(list(self.prediction_history)[-100:])

            # Compute empirical distributions
            old_dist = np.histogram(old_preds, bins=10, density=True)[0] + 1e-10
            new_dist = np.histogram(new_preds, bins=10, density=True)[0] + 1e-10

            # KL divergence
            kl_div = float(np.sum(new_dist * np.log(new_dist / old_dist)))

            if kl_div > self.config.kl_threshold:
                drift_detected = True
                logger.warning(f"Drift detected via KL divergence: {kl_div:.3f}")

        # Check 2: Validation loss trend
        if len(self.validation_loss_history) >= self.config.drift_window:
            recent_window = list(self.validation_loss_history)[-self.config.drift_window:]
            baseline_window = list(self.validation_loss_history)[:self.config.drift_window]

            recent_mean = np.mean(recent_window)
            baseline_mean = np.mean(baseline_window)

            val_loss_change = (recent_mean - baseline_mean) / baseline_mean

            if val_loss_change > self.config.validation_loss_threshold:
                drift_detected = True
                logger.warning(f"Drift detected via validation loss: +{val_loss_change:.1%}")

        # Check 3: AUC-ROC slope (bonus metric)
        # Simplified: would compute AUC over time and check if declining
        # For now, placeholder
        auc_slope = 0.0

        result = DriftDetectionResult(
            drift_detected=drift_detected,
            kl_divergence=kl_div,
            validation_loss_change=val_loss_change,
            auc_roc_slope=auc_slope
        )

        return result

    def maml_adapt(
        self,
        adaptation_data: List[tuple],
        optimizer: torch.optim.Optimizer
    ):
        """
        MAML-style rapid adaptation (Method M4).

        Args:
            adaptation_data: List of (state, action, reward) tuples from new regime
            optimizer: Optimizer for inner loop
        """
        if len(adaptation_data) < self.config.maml_adaptation_samples:
            logger.warning(f"Insufficient data for MAML: {len(adaptation_data)} < {self.config.maml_adaptation_samples}")
            return

        logger.info(f"MAML adaptation on {len(adaptation_data)} samples")

        # Inner loop: K gradient steps on adaptation data
        for step in range(self.config.maml_inner_steps):
            # Sample batch
            batch_indices = np.random.choice(len(adaptation_data), size=32, replace=False)
            batch = [adaptation_data[i] for i in batch_indices]

            states = torch.tensor([x[0] for x in batch], dtype=torch.float32)
            actions = torch.tensor([x[1] for x in batch], dtype=torch.long)
            rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)

            # Forward pass
            outputs = self.model(states)
            loss = nn.CrossEntropyLoss()(outputs, actions)

            # Add EWC penalty if enabled
            if self.config.ewc_enabled:
                loss += self.ewc_penalty()

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Inner loop update with smaller learning rate
            for param in self.model.parameters():
                if param.grad is not None:
                    param.data -= self.config.maml_inner_lr * param.grad

        logger.info("MAML adaptation complete")

    def online_update(self, optimizer: torch.optim.Optimizer):
        """
        Perform online update from replay buffer (Method M1).

        Args:
            optimizer: Optimizer for updates
        """
        if len(self.replay_buffer) < self.config.min_samples_for_update:
            logger.debug(f"Insufficient samples for update: {len(self.replay_buffer)}")
            return

        logger.info(f"Online update with {len(self.replay_buffer)} samples")

        # Sample from replay buffer
        batch_size = min(64, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), size=batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        states = torch.tensor([x['state'] for x in batch], dtype=torch.float32)
        actions = torch.tensor([x['action'] for x in batch], dtype=torch.long)

        # Forward pass
        outputs = self.model(states)
        loss = nn.CrossEntropyLoss()(outputs, actions)

        # Add EWC penalty (Method M2)
        if self.config.ewc_enabled:
            loss += self.ewc_penalty()

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.total_updates += 1
        logger.info(f"Online update #{self.total_updates} complete, loss={loss.item():.4f}")

    def check_fallback_condition(self, confidence: float) -> bool:
        """
        Check if should fallback to baseline (Method M5).

        Args:
            confidence: Model's confidence in current prediction

        Returns:
            True if should use fallback
        """
        if confidence < self.config.fallback_confidence_threshold:
            self.low_confidence_count += 1
        else:
            self.low_confidence_count = 0

        if self.low_confidence_count >= self.config.fallback_window:
            if not self.using_fallback:
                logger.warning(f"Fallback triggered: {self.low_confidence_count} consecutive low-confidence predictions")
                self.using_fallback = True
            return True

        if self.using_fallback and confidence > self.config.fallback_confidence_threshold:
            logger.info("Fallback restored: confidence recovered")
            self.using_fallback = False

        return False

    def save_baseline(self):
        """Save current model as baseline for fallback (Method M5)"""
        self.baseline_model_state = self.model.state_dict().copy()
        logger.info("Baseline model saved")

    def restore_baseline(self):
        """Restore baseline model (Method M5)"""
        if self.baseline_model_state is not None:
            self.model.load_state_dict(self.baseline_model_state)
            logger.info("Baseline model restored")
        else:
            logger.warning("No baseline model to restore")


__all__ = ['OnlineLearner', 'AdaptationConfig', 'DriftDetectionResult']
