"""
HIMARI Layer 2 - Part K: Advanced Training Methods
Complete implementation of all 8 advanced training techniques.

Methods:
- K1: 3-Stage Curriculum Learning
- K2: MAML Meta-Learning
- K3: Causal Data Augmentation
- K4: Multi-Task Learning
- K5: Adversarial Training
- K6: FGSM/PGD Robustness
- K7: Reward Shaping
- K8: Rare Event Synthesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# K1: 3-Stage Curriculum Learning
# ============================================================================

class CurriculumLearningScheduler:
    """
    3-Stage curriculum: Easy → Medium → Hard samples.
    Gradually increases task difficulty based on performance.
    """

    def __init__(self,
                 stage1_epochs: int = 10,
                 stage2_epochs: int = 15,
                 stage3_epochs: int = 25,
                 performance_threshold: float = 0.7):
        """
        Initialize curriculum scheduler.

        Args:
            stage1_epochs: Epochs for easy samples
            stage2_epochs: Epochs for medium samples
            stage3_epochs: Epochs for hard samples
            performance_threshold: Performance required to advance stages
        """
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage3_epochs = stage3_epochs
        self.performance_threshold = performance_threshold

        self.current_epoch = 0
        self.current_stage = 1
        self.performance_history = []

        logger.info(f"Curriculum: Stage 1={stage1_epochs}ep, Stage 2={stage2_epochs}ep, Stage 3={stage3_epochs}ep")

    def get_stage(self, epoch: int) -> int:
        """Get current curriculum stage."""
        if epoch < self.stage1_epochs:
            return 1
        elif epoch < self.stage1_epochs + self.stage2_epochs:
            return 2
        else:
            return 3

    def get_difficulty_mask(self, difficulties: np.ndarray, stage: int) -> np.ndarray:
        """
        Get sample mask for current stage.

        Args:
            difficulties: Sample difficulty scores (0-1)
            stage: Current stage (1, 2, or 3)

        Returns:
            Boolean mask for samples to include
        """
        if stage == 1:  # Easy: bottom 33%
            threshold = np.percentile(difficulties, 33)
            mask = difficulties <= threshold
        elif stage == 2:  # Medium: middle 33%
            low = np.percentile(difficulties, 33)
            high = np.percentile(difficulties, 67)
            mask = (difficulties > low) & (difficulties <= high)
        else:  # Hard: top 33%
            threshold = np.percentile(difficulties, 67)
            mask = difficulties > threshold

        return mask

    def update(self, epoch: int, performance: float):
        """Update curriculum based on performance."""
        self.current_epoch = epoch
        self.performance_history.append(performance)
        self.current_stage = self.get_stage(epoch)

        logger.info(f"Curriculum: Epoch {epoch}, Stage {self.current_stage}, Performance {performance:.4f}")


# ============================================================================
# K2: MAML Meta-Learning
# ============================================================================

class MAMLTrainer:
    """
    Model-Agnostic Meta-Learning for fast adaptation to new regimes.
    """

    def __init__(self,
                 model: nn.Module,
                 inner_lr: float = 0.01,
                 outer_lr: float = 0.001,
                 num_inner_steps: int = 5):
        """
        Initialize MAML trainer.

        Args:
            model: Model to meta-train
            inner_lr: Learning rate for inner loop (task adaptation)
            outer_lr: Learning rate for outer loop (meta-update)
            num_inner_steps: Number of gradient steps in inner loop
        """
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

        # Outer optimizer
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)

        logger.info(f"MAML initialized: inner_lr={inner_lr}, outer_lr={outer_lr}, inner_steps={num_inner_steps}")

    def inner_loop(self, support_data: Tuple, model_copy: nn.Module) -> nn.Module:
        """
        Inner loop: Adapt to task using support set.

        Args:
            support_data: (features, labels) for task adaptation
            model_copy: Copy of model to adapt

        Returns:
            Adapted model
        """
        features, labels = support_data

        for _ in range(self.num_inner_steps):
            # Forward pass
            outputs = model_copy(features)
            loss = F.cross_entropy(outputs, labels)

            # Compute gradients and update
            grads = torch.autograd.grad(loss, model_copy.parameters(), create_graph=True)

            # Manual SGD update
            for param, grad in zip(model_copy.parameters(), grads):
                param = param - self.inner_lr * grad

        return model_copy

    def meta_update(self, tasks: List[Tuple[Tuple, Tuple]]):
        """
        Meta-update using multiple tasks.

        Args:
            tasks: List of (support_set, query_set) tuples
        """
        self.meta_optimizer.zero_grad()

        meta_losses = []

        for support_set, query_set in tasks:
            # Create copy of model
            model_copy = self._copy_model()

            # Inner loop: adapt to task
            adapted_model = self.inner_loop(support_set, model_copy)

            # Compute loss on query set
            query_features, query_labels = query_set
            query_outputs = adapted_model(query_features)
            query_loss = F.cross_entropy(query_outputs, query_labels)

            meta_losses.append(query_loss)

        # Meta-loss: average over tasks
        meta_loss = torch.mean(torch.stack(meta_losses))

        # Meta-update
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def _copy_model(self) -> nn.Module:
        """Create a copy of the model."""
        model_copy = type(self.model)(*self.model.__init__.__code__.co_varnames[1:])
        model_copy.load_state_dict(self.model.state_dict())
        return model_copy


# ============================================================================
# K3: Causal Data Augmentation
# ============================================================================

class CausalAugmenter:
    """
    Causally-valid data augmentation that preserves causal structure.
    """

    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std

    def augment(self, features: np.ndarray, preserve_causality: bool = True) -> np.ndarray:
        """
        Augment data while preserving causal relationships.

        Args:
            features: Input features (num_samples, feature_dim)
            preserve_causality: If True, only augment exogenous variables

        Returns:
            Augmented features
        """
        augmented = features.copy()

        if preserve_causality:
            # Only add noise to exogenous features (e.g., first 20 features)
            exogenous_dim = min(20, features.shape[1])
            noise = np.random.randn(features.shape[0], exogenous_dim) * self.noise_std
            augmented[:, :exogenous_dim] += noise
        else:
            # Add noise to all features
            noise = np.random.randn(*features.shape) * self.noise_std
            augmented += noise

        return augmented


# ============================================================================
# K4: Multi-Task Learning
# ============================================================================

class MultiTaskLearner:
    """
    Multi-task learning: jointly train on action prediction + auxiliary tasks.
    """

    def __init__(self,
                 shared_encoder: nn.Module,
                 task_heads: Dict[str, nn.Module],
                 task_weights: Optional[Dict[str, float]] = None):
        """
        Initialize multi-task learner.

        Args:
            shared_encoder: Shared feature encoder
            task_heads: Dictionary of task-specific heads
            task_weights: Optional task loss weights
        """
        self.shared_encoder = shared_encoder
        self.task_heads = task_heads
        self.task_weights = task_weights or {task: 1.0 for task in task_heads}

        logger.info(f"Multi-task learning: {len(task_heads)} tasks - {list(task_heads.keys())}")

    def compute_multi_task_loss(self,
                                 features: torch.Tensor,
                                 targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute multi-task loss.

        Args:
            features: Input features
            targets: Dictionary of task targets

        Returns:
            total_loss: Weighted sum of task losses
            losses: Dictionary of individual task losses
        """
        # Shared encoding
        encoded = self.shared_encoder(features)

        # Compute loss for each task
        task_losses = {}
        for task_name, task_head in self.task_heads.items():
            if task_name not in targets:
                continue

            task_output = task_head(encoded)
            task_target = targets[task_name]

            # Task-specific loss
            if task_name == 'action':
                loss = F.cross_entropy(task_output, task_target)
            elif task_name == 'return':
                loss = F.mse_loss(task_output.squeeze(), task_target)
            elif task_name == 'volatility':
                loss = F.mse_loss(task_output.squeeze(), task_target)
            else:
                loss = F.mse_loss(task_output, task_target)

            task_losses[task_name] = loss * self.task_weights[task_name]

        # Total loss
        total_loss = sum(task_losses.values())

        return total_loss, {k: v.item() for k, v in task_losses.items()}


# ============================================================================
# K5: Adversarial Training
# ============================================================================

class AdversarialTrainer:
    """
    Adversarial training for robustness to worst-case perturbations.
    """

    def __init__(self, epsilon: float = 0.01, alpha: float = 0.005, num_steps: int = 5):
        """
        Initialize adversarial trainer.

        Args:
            epsilon: Maximum perturbation bound
            alpha: Step size for adversarial perturbation
            num_steps: Number of adversarial steps
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    def generate_adversarial_example(self,
                                     model: nn.Module,
                                     features: torch.Tensor,
                                     labels: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial example using PGD.

        Args:
            model: Model to attack
            features: Clean features
            labels: True labels

        Returns:
            Adversarial features
        """
        adv_features = features.clone().detach()
        adv_features.requires_grad = True

        for _ in range(self.num_steps):
            # Forward pass
            outputs = model(adv_features)
            loss = F.cross_entropy(outputs, labels)

            # Compute gradient
            grad = torch.autograd.grad(loss, adv_features)[0]

            # PGD step
            adv_features = adv_features.detach() + self.alpha * grad.sign()

            # Project to epsilon ball
            perturbation = torch.clamp(adv_features - features, -self.epsilon, self.epsilon)
            adv_features = features + perturbation

        return adv_features.detach()

    def adversarial_train_step(self,
                               model: nn.Module,
                               features: torch.Tensor,
                               labels: torch.Tensor,
                               optimizer: torch.optim.Optimizer) -> float:
        """
        Single adversarial training step.

        Args:
            model: Model to train
            features: Clean features
            labels: Labels
            optimizer: Optimizer

        Returns:
            Loss value
        """
        # Generate adversarial examples
        adv_features = self.generate_adversarial_example(model, features, labels)

        # Train on adversarial examples
        optimizer.zero_grad()
        outputs = model(adv_features)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        return loss.item()


# ============================================================================
# K6: FGSM/PGD Robustness Testing
# ============================================================================

def fgsm_attack(model: nn.Module,
                features: torch.Tensor,
                labels: torch.Tensor,
                epsilon: float = 0.01) -> torch.Tensor:
    """
    Fast Gradient Sign Method attack.

    Args:
        model: Model to attack
        features: Clean features
        labels: True labels
        epsilon: Perturbation magnitude

    Returns:
        Adversarial features
    """
    adv_features = features.clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(adv_features)
    loss = F.cross_entropy(outputs, labels)

    # Backward pass
    loss.backward()

    # FGSM perturbation
    perturbation = epsilon * adv_features.grad.sign()
    adv_features = adv_features + perturbation

    return adv_features.detach()


# ============================================================================
# K7: Reward Shaping
# ============================================================================

class RewardShaper:
    """
    Reward shaping for better learning signal.
    """

    def __init__(self,
                 sharpe_weight: float = 1.0,
                 drawdown_penalty: float = 0.5,
                 turnover_penalty: float = 0.1):
        """
        Initialize reward shaper.

        Args:
            sharpe_weight: Weight for Sharpe ratio component
            drawdown_penalty: Penalty for drawdown
            turnover_penalty: Penalty for excessive trading
        """
        self.sharpe_weight = sharpe_weight
        self.drawdown_penalty = drawdown_penalty
        self.turnover_penalty = turnover_penalty

    def shape_reward(self,
                     raw_reward: float,
                     portfolio_value: float,
                     peak_value: float,
                     num_trades: int) -> float:
        """
        Shape raw reward with multiple objectives.

        Args:
            raw_reward: Raw PnL reward
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value
            num_trades: Number of trades

        Returns:
            Shaped reward
        """
        # Sharpe-like component
        sharpe_component = raw_reward * self.sharpe_weight

        # Drawdown penalty
        if peak_value > 0:
            drawdown = (peak_value - portfolio_value) / peak_value
            drawdown_component = -drawdown * self.drawdown_penalty
        else:
            drawdown_component = 0.0

        # Turnover penalty (encourage low-frequency trading)
        turnover_component = -num_trades * self.turnover_penalty

        shaped_reward = sharpe_component + drawdown_component + turnover_component

        return shaped_reward


# ============================================================================
# K8: Rare Event Synthesis
# ============================================================================

class RareEventSynthesizer:
    """
    Synthesize rare events (crashes, flash crashes) for robust training.
    """

    def __init__(self, rare_event_prob: float = 0.01):
        self.rare_event_prob = rare_event_prob

    def synthesize_crash(self, features: np.ndarray, severity: float = 3.0) -> np.ndarray:
        """
        Synthesize market crash in features.

        Args:
            features: Clean features
            severity: Crash severity (standard deviations)

        Returns:
            Features with synthetic crash
        """
        crash_features = features.copy()

        # Price crash: large negative price movement
        crash_features[:, 0] -= severity * np.std(features[:, 0])

        # Volatility spike
        if features.shape[1] > 10:
            crash_features[:, 10] *= severity

        # Volume surge
        if features.shape[1] > 5:
            crash_features[:, 5] *= severity / 2

        return crash_features

    def augment_with_rare_events(self,
                                  features: np.ndarray,
                                  labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment dataset with synthetic rare events.

        Args:
            features: Clean features
            labels: Clean labels

        Returns:
            Augmented features and labels
        """
        num_rare_events = int(len(features) * self.rare_event_prob)

        rare_features = []
        rare_labels = []

        for _ in range(num_rare_events):
            # Random sample
            idx = np.random.randint(len(features))

            # Synthesize crash
            crash_sample = self.synthesize_crash(features[idx:idx+1])

            rare_features.append(crash_sample)
            rare_labels.append(0)  # Label as SELL during crash

        # Concatenate
        augmented_features = np.concatenate([features] + rare_features, axis=0)
        augmented_labels = np.concatenate([labels] + rare_labels, axis=0)

        logger.info(f"Synthesized {num_rare_events} rare events")

        return augmented_features, augmented_labels


# ============================================================================
# Complete Part K Pipeline
# ============================================================================

class PartKTrainer:
    """
    Complete Part K training pipeline integrating all 8 methods.
    """

    def __init__(self,
                 enable_curriculum: bool = True,
                 enable_maml: bool = False,  # Expensive
                 enable_causal_aug: bool = True,
                 enable_multitask: bool = False,  # Requires task heads
                 enable_adversarial: bool = True,
                 enable_reward_shaping: bool = True,
                 enable_rare_events: bool = True):
        """
        Initialize Part K trainer.

        Args:
            enable_curriculum: Enable curriculum learning
            enable_maml: Enable MAML meta-learning
            enable_causal_aug: Enable causal augmentation
            enable_multitask: Enable multi-task learning
            enable_adversarial: Enable adversarial training
            enable_reward_shaping: Enable reward shaping
            enable_rare_events: Enable rare event synthesis
        """
        self.enable_curriculum = enable_curriculum
        self.enable_maml = enable_maml
        self.enable_causal_aug = enable_causal_aug
        self.enable_multitask = enable_multitask
        self.enable_adversarial = enable_adversarial
        self.enable_reward_shaping = enable_reward_shaping
        self.enable_rare_events = enable_rare_events

        # Initialize components
        if enable_curriculum:
            self.curriculum = CurriculumLearningScheduler()

        if enable_causal_aug:
            self.causal_augmenter = CausalAugmenter()

        if enable_adversarial:
            self.adversarial_trainer = AdversarialTrainer()

        if enable_reward_shaping:
            self.reward_shaper = RewardShaper()

        if enable_rare_events:
            self.rare_event_synthesizer = RareEventSynthesizer()

        logger.info(f"Part K Trainer initialized: "
                   f"Curriculum={enable_curriculum}, MAML={enable_maml}, "
                   f"CausalAug={enable_causal_aug}, MultiTask={enable_multitask}, "
                   f"Adversarial={enable_adversarial}, RewardShaping={enable_reward_shaping}, "
                   f"RareEvents={enable_rare_events}")
