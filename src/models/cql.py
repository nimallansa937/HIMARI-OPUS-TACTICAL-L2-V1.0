"""
HIMARI Layer 2 - Conservative Q-Learning (CQL)
Offline RL agent that learns from static datasets without environment interaction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """Q-network for CQL."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Q-values for each action (batch_size, action_dim)
        """
        return self.network(state)


class CQLAgent(nn.Module):
    """
    Conservative Q-Learning (CQL) Agent.

    Paper: "Conservative Q-Learning for Offline Reinforcement Learning"
    Features:
    - Double Q-learning to reduce overestimation
    - Conservative loss to penalize out-of-distribution actions
    - Target network for stable learning
    - Suitable for offline learning from static datasets
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int = 3,
                 hidden_dim: int = 256,
                 alpha: float = 2.0,
                 gamma: float = 0.99,
                 tau: float = 0.005):
        """
        Initialize CQL agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions (SELL, HOLD, BUY)
            hidden_dim: Hidden layer dimension
            alpha: CQL conservative penalty coefficient
            gamma: Discount factor
            tau: Target network soft update rate
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        # Q-networks (double Q-learning)
        self.q_network1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_network2 = QNetwork(state_dim, action_dim, hidden_dim)

        # Target networks
        self.q_target1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_target2 = QNetwork(state_dim, action_dim, hidden_dim)

        # Initialize targets
        self.q_target1.load_state_dict(self.q_network1.state_dict())
        self.q_target2.load_state_dict(self.q_network2.state_dict())

        # Freeze target networks
        for param in self.q_target1.parameters():
            param.requires_grad = False
        for param in self.q_target2.parameters():
            param.requires_grad = False

        logger.info(f"CQL initialized: state_dim={state_dim}, alpha={alpha}, hidden_dim={hidden_dim}")
        logger.info(f"Q-network params: {sum(p.numel() for p in self.q_network1.parameters()):,}")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for action selection.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Action logits (batch_size, action_dim)
        """
        # Use minimum Q-value from both networks (conservative)
        q1 = self.q_network1(state)
        q2 = self.q_network2(state)
        q_values = torch.min(q1, q2)
        return q_values

    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: State tensor (1, state_dim)
            epsilon: Exploration rate

        Returns:
            Selected action index
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            q_values = self.forward(state)
            action = q_values.argmax(dim=-1).item()

        return action

    def compute_cql_loss(self,
                         states: torch.Tensor,
                         actions: torch.Tensor,
                         rewards: torch.Tensor,
                         next_states: torch.Tensor,
                         dones: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute CQL loss with conservative penalty.

        Args:
            states: State batch (batch_size, state_dim)
            actions: Action batch (batch_size,)
            rewards: Reward batch (batch_size,)
            next_states: Next state batch (batch_size, state_dim)
            dones: Done flags (batch_size,)

        Returns:
            loss: Total CQL loss
            info: Dictionary with loss components
        """
        batch_size = states.shape[0]

        # Current Q-values
        q1_pred = self.q_network1(states)
        q2_pred = self.q_network2(states)

        # Q-values for taken actions
        q1_taken = q1_pred.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_taken = q2_pred.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (double Q-learning)
        with torch.no_grad():
            # Select actions using current networks
            next_q1 = self.q_network1(next_states)
            next_q2 = self.q_network2(next_states)
            next_actions = torch.min(next_q1, next_q2).argmax(dim=1)

            # Evaluate actions using target networks
            target_q1 = self.q_target1(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q2 = self.q_target2(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = torch.min(target_q1, target_q2)

            # Bellman backup
            q_target = rewards + (1 - dones) * self.gamma * target_q

        # TD loss (standard Q-learning loss)
        td_loss1 = F.mse_loss(q1_taken, q_target)
        td_loss2 = F.mse_loss(q2_taken, q_target)
        td_loss = td_loss1 + td_loss2

        # Conservative penalty (CQL regularization)
        # Penalize Q-values for all actions, reward Q-values for dataset actions
        logsumexp_q1 = torch.logsumexp(q1_pred, dim=1)
        logsumexp_q2 = torch.logsumexp(q2_pred, dim=1)

        cql_loss1 = (logsumexp_q1 - q1_taken).mean()
        cql_loss2 = (logsumexp_q2 - q2_taken).mean()
        cql_loss = cql_loss1 + cql_loss2

        # Total loss
        total_loss = td_loss + self.alpha * cql_loss

        # Info dict
        info = {
            'loss': total_loss.item(),
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'q1_mean': q1_taken.mean().item(),
            'q2_mean': q2_taken.mean().item(),
            'q_target_mean': q_target.mean().item()
        }

        return total_loss, info

    def update_target_networks(self):
        """Soft update of target networks."""
        for param, target_param in zip(self.q_network1.parameters(), self.q_target1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q_network2.parameters(), self.q_target2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_step(self,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   rewards: torch.Tensor,
                   next_states: torch.Tensor,
                   dones: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> dict:
        """
        Single training step.

        Args:
            states: State batch
            actions: Action batch
            rewards: Reward batch
            next_states: Next state batch
            dones: Done flags
            optimizer: Optimizer

        Returns:
            Dictionary with training metrics
        """
        # Compute loss
        loss, info = self.compute_cql_loss(states, actions, rewards, next_states, dones)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        # Update target networks
        self.update_target_networks()

        return info

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'q_network1': self.q_network1.state_dict(),
            'q_network2': self.q_network2.state_dict(),
            'q_target1': self.q_target1.state_dict(),
            'q_target2': self.q_target2.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'tau': self.tau
            }
        }
        torch.save(checkpoint, path)
        logger.info(f"CQL model saved: {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.q_network1.load_state_dict(checkpoint['q_network1'])
        self.q_network2.load_state_dict(checkpoint['q_network2'])
        self.q_target1.load_state_dict(checkpoint['q_target1'])
        self.q_target2.load_state_dict(checkpoint['q_target2'])
        logger.info(f"CQL model loaded: {path}")


def create_cql_agent(state_dim: int,
                     action_dim: int = 3,
                     hidden_dim: int = 256,
                     alpha: float = 2.0) -> CQLAgent:
    """
    Factory function to create CQL agent.

    Args:
        state_dim: Dimension of state space
        action_dim: Number of actions
        hidden_dim: Hidden layer dimension
        alpha: CQL alpha parameter

    Returns:
        Initialized CQL agent
    """
    return CQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        alpha=alpha
    )
