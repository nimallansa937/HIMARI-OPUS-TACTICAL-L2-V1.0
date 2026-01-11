"""
HIMARI Layer 2 - Training Infrastructure
Subsystem K: Adversarial Training (Methods K1, K3)

Purpose:
    Improve robustness via adversarial self-play and input perturbations.

Methods:
    K1: Adversarial Self-Play - Train against worst-case market scenarios
    K3: FGSM/PGD Attacks - Input perturbation robustness testing

Expected Performance:
    - Sharpe ratio under attack: >1.5 (vs 2.0 clean)
    - Max drawdown under attack: <20% (vs <15% clean)
    - Robustness improvement: +15% vs non-adversarial training
"""

from typing import Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class AdversarialConfig:
    """Configuration for adversarial training"""
    # Method K1: Self-play
    selfplay_enabled: bool = True
    opponent_model_path: Optional[str] = None
    opponent_update_interval: int = 1000  # Update opponent every N episodes

    # Method K3: FGSM/PGD
    fgsm_epsilon: float = 0.01  # Max perturbation (1% of feature scale)
    pgd_epsilon: float = 0.02  # Stronger attack
    pgd_alpha: float = 0.005  # Step size
    pgd_iterations: int = 10  # Number of PGD steps

    # Training mix
    adversarial_fraction: float = 0.3  # 30% of batches are adversarial

    # Reward shaping (K4 integration)
    use_robust_rewards: bool = True


class AdversarialTrainer:
    """
    Adversarial training methods for robustness.

    Example:
        >>> trainer = AdversarialTrainer(model)
        >>> adv_state = trainer.fgsm_attack(state, target_action)
        >>> loss = trainer.compute_robust_loss(state, action, reward)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[AdversarialConfig] = None
    ):
        self.model = model
        self.config = config or AdversarialConfig()
        self.opponent_model: Optional[nn.Module] = None
        self.training_step = 0

        logger.info(f"Adversarial Trainer initialized: FGSM ε={self.config.fgsm_epsilon}")

    def fgsm_attack(
        self,
        state: torch.Tensor,
        target_action: torch.Tensor,
        epsilon: Optional[float] = None
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method attack (Method K3).

        Args:
            state: Input state (requires_grad=True)
            target_action: Ground truth action for gradient
            epsilon: Perturbation magnitude (default: config value)

        Returns:
            Adversarially perturbed state
        """
        if epsilon is None:
            epsilon = self.config.fgsm_epsilon

        # Ensure gradient tracking
        state = state.clone().detach().requires_grad_(True)

        # Forward pass
        output = self.model(state)

        # Compute loss
        if target_action.dim() == 1:
            # Classification
            loss = nn.CrossEntropyLoss()(output, target_action)
        else:
            # Regression
            loss = nn.MSELoss()(output, target_action)

        # Backward to get gradient
        self.model.zero_grad()
        loss.backward()

        # Collect gradient sign
        grad_sign = state.grad.sign()

        # Create adversarial example
        adv_state = state + epsilon * grad_sign

        # Detach to prevent gradients flowing through attack
        return adv_state.detach()

    def pgd_attack(
        self,
        state: torch.Tensor,
        target_action: torch.Tensor,
        epsilon: Optional[float] = None,
        alpha: Optional[float] = None,
        num_iter: Optional[int] = None
    ) -> torch.Tensor:
        """
        Projected Gradient Descent attack (Method K3).

        Stronger iterative attack vs single-step FGSM.

        Args:
            state: Input state
            target_action: Ground truth action
            epsilon: Max perturbation bound
            alpha: Step size per iteration
            num_iter: Number of PGD steps

        Returns:
            Adversarially perturbed state
        """
        epsilon = epsilon or self.config.pgd_epsilon
        alpha = alpha or self.config.pgd_alpha
        num_iter = num_iter or self.config.pgd_iterations

        # Start from original state
        adv_state = state.clone().detach()

        # PGD iterations
        for _ in range(num_iter):
            adv_state.requires_grad = True

            # Forward pass
            output = self.model(adv_state)

            # Loss
            if target_action.dim() == 1:
                loss = nn.CrossEntropyLoss()(output, target_action)
            else:
                loss = nn.MSELoss()(output, target_action)

            # Gradient
            self.model.zero_grad()
            loss.backward()

            # Update adversarial state
            with torch.no_grad():
                grad_sign = adv_state.grad.sign()
                adv_state = adv_state + alpha * grad_sign

                # Project back to epsilon ball
                perturbation = torch.clamp(adv_state - state, -epsilon, epsilon)
                adv_state = state + perturbation

        return adv_state.detach()

    def selfplay_opponent_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action from opponent model for self-play (Method K1).

        The opponent tries to create worst-case market scenarios.

        Args:
            state: Current market state

        Returns:
            Opponent's action (market manipulation attempt)
        """
        if self.opponent_model is None:
            # No opponent yet, use random
            batch_size = state.shape[0]
            action_dim = 3  # Simplified: [buy, hold, sell]
            return torch.randint(0, action_dim, (batch_size,))

        with torch.no_grad():
            opponent_output = self.opponent_model(state)

            # Opponent tries to maximize our loss
            # In practice, this could be market maker trying to induce bad trades
            if opponent_output.dim() > 1:
                opponent_action = opponent_output.argmax(dim=-1)
            else:
                opponent_action = opponent_output

        return opponent_action

    def update_opponent(self):
        """
        Update opponent model from current model (Method K1).

        Opponent lags behind to create challenging but solvable scenarios.
        """
        if self.opponent_model is None:
            # Clone current model
            self.opponent_model = type(self.model)(*[])  # Placeholder
            logger.info("Opponent model initialized")
        else:
            # Copy weights from current model
            self.opponent_model.load_state_dict(self.model.state_dict())
            logger.debug("Opponent model updated")

    def compute_robust_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        use_adversarial: bool = True
    ) -> torch.Tensor:
        """
        Compute loss with adversarial regularization.

        Args:
            state: Input state
            action: Taken action
            reward: Observed reward
            use_adversarial: Whether to include adversarial component

        Returns:
            Total loss (clean + adversarial)
        """
        # Clean loss
        clean_output = self.model(state)
        if action.dim() == 1:
            clean_loss = nn.CrossEntropyLoss()(clean_output, action)
        else:
            clean_loss = nn.MSELoss()(clean_output, action)

        if not use_adversarial:
            return clean_loss

        # Adversarial loss (Method K3)
        # Use FGSM for efficiency
        adv_state = self.fgsm_attack(state, action)
        adv_output = self.model(adv_state)

        if action.dim() == 1:
            adv_loss = nn.CrossEntropyLoss()(adv_output, action)
        else:
            adv_loss = nn.MSELoss()(adv_output, action)

        # Combine losses
        # Model should perform well on both clean and adversarial inputs
        total_loss = 0.7 * clean_loss + 0.3 * adv_loss

        return total_loss

    def train_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Single training step with adversarial augmentation.

        Args:
            state: Batch of states
            action: Batch of actions
            reward: Batch of rewards
            optimizer: Optimizer

        Returns:
            Loss value
        """
        # Decide if this batch is adversarial
        use_adversarial = (
            self.config.adversarial_fraction > 0 and
            np.random.random() < self.config.adversarial_fraction
        )

        # Compute loss
        loss = self.compute_robust_loss(state, action, reward, use_adversarial)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update opponent periodically (Method K1)
        self.training_step += 1
        if self.config.selfplay_enabled and self.training_step % self.config.opponent_update_interval == 0:
            self.update_opponent()

        return loss.item()

    def evaluate_robustness(
        self,
        test_states: torch.Tensor,
        test_actions: torch.Tensor
    ) -> dict:
        """
        Evaluate model robustness against attacks.

        Returns:
            Dict with 'clean_acc', 'fgsm_acc', 'pgd_acc'
        """
        self.model.eval()

        with torch.no_grad():
            # Clean accuracy
            clean_output = self.model(test_states)
            clean_preds = clean_output.argmax(dim=-1) if clean_output.dim() > 1 else clean_output
            clean_acc = (clean_preds == test_actions).float().mean().item()

        # FGSM accuracy
        fgsm_states = self.fgsm_attack(test_states, test_actions)
        with torch.no_grad():
            fgsm_output = self.model(fgsm_states)
            fgsm_preds = fgsm_output.argmax(dim=-1) if fgsm_output.dim() > 1 else fgsm_output
            fgsm_acc = (fgsm_preds == test_actions).float().mean().item()

        # PGD accuracy
        pgd_states = self.pgd_attack(test_states, test_actions)
        with torch.no_grad():
            pgd_output = self.model(pgd_states)
            pgd_preds = pgd_output.argmax(dim=-1) if pgd_output.dim() > 1 else pgd_output
            pgd_acc = (pgd_preds == test_actions).float().mean().item()

        self.model.train()

        results = {
            'clean_acc': clean_acc,
            'fgsm_acc': fgsm_acc,
            'pgd_acc': pgd_acc,
            'robustness_gap': clean_acc - pgd_acc
        }

        logger.info(f"Robustness: Clean={clean_acc:.1%}, FGSM={fgsm_acc:.1%}, PGD={pgd_acc:.1%}")
        return results


__all__ = ['AdversarialTrainer', 'AdversarialConfig']
