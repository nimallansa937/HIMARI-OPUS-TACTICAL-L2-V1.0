"""
HIMARI Layer 2 - PPO-LSTM
Proximal Policy Optimization with LSTM for sequential trading decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ActorCriticLSTM(nn.Module):
    """
    Actor-Critic network with LSTM for PPO.

    Architecture:
    - LSTM layer for temporal modeling
    - Actor head for policy (action probabilities)
    - Critic head for value estimation
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int = 3,
                 hidden_dim: int = 128,
                 num_layers: int = 2):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    param.data.zero_()

    def forward(self,
                state: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple:
        """
        Forward pass.

        Args:
            state: State tensor (batch_size, seq_len, state_dim) or (batch_size, state_dim)
            hidden: Optional LSTM hidden state tuple (h, c)

        Returns:
            action_logits: Action logits (batch_size, action_dim)
            value: State value estimate (batch_size, 1)
            hidden: Updated LSTM hidden state
        """
        # Add sequence dimension if needed
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (batch_size, 1, state_dim)

        # Feature extraction
        features = self.feature_net(state)  # (batch_size, seq_len, hidden_dim)

        # LSTM forward
        lstm_out, hidden = self.lstm(features, hidden)  # (batch_size, seq_len, hidden_dim)

        # Take last timestep for actor-critic
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Actor and critic outputs
        action_logits = self.actor(lstm_out)  # (batch_size, action_dim)
        value = self.critic(lstm_out)  # (batch_size, 1)

        return action_logits, value, hidden

    def get_action(self,
                   state: torch.Tensor,
                   hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                   deterministic: bool = False) -> Tuple:
        """
        Sample action from policy.

        Args:
            state: State tensor
            hidden: LSTM hidden state
            deterministic: If True, select argmax action

        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
            hidden: Updated hidden state
        """
        action_logits, value, hidden = self.forward(state, hidden)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, value, hidden

    def evaluate_actions(self,
                         states: torch.Tensor,
                         actions: torch.Tensor) -> Tuple:
        """
        Evaluate actions for PPO loss computation.

        Args:
            states: State batch (batch_size, seq_len, state_dim)
            actions: Action batch (batch_size,)

        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy
        """
        action_logits, values, _ = self.forward(states)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy

    def init_hidden(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)


class PPOLSTMAgent(nn.Module):
    """
    PPO-LSTM Agent for online reinforcement learning.

    Features:
    - Proximal Policy Optimization (PPO) for stable training
    - LSTM for temporal sequence modeling
    - Generalized Advantage Estimation (GAE)
    - Clipped objective for conservative updates
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int = 3,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        """
        Initialize PPO-LSTM agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Actor-Critic network
        self.ac_network = ActorCriticLSTM(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.ac_network.parameters(), lr=lr)

        logger.info(f"PPO-LSTM initialized: state_dim={state_dim}, hidden_dim={hidden_dim}")
        logger.info(f"Network params: {sum(p.numel() for p in self.ac_network.parameters()):,}")

    def forward(self, state: torch.Tensor, hidden=None):
        """Forward pass through actor-critic network."""
        return self.ac_network(state, hidden)

    def select_action(self,
                      state: torch.Tensor,
                      hidden=None,
                      deterministic: bool = False) -> Tuple:
        """
        Select action using current policy.

        Args:
            state: Current state
            hidden: LSTM hidden state
            deterministic: If True, use argmax

        Returns:
            action, log_prob, value, hidden
        """
        with torch.no_grad():
            return self.ac_network.get_action(state, hidden, deterministic)

    def compute_gae(self,
                    rewards: torch.Tensor,
                    values: torch.Tensor,
                    dones: torch.Tensor,
                    next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Reward sequence (T,)
            values: Value estimates (T,)
            dones: Done flags (T,)
            next_value: Next state value (scalar)

        Returns:
            advantages: Advantage estimates (T,)
            returns: Discounted returns (T,)
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        next_val = next_value

        for t in reversed(range(T)):
            if t == T - 1:
                next_value_t = next_val
            else:
                next_value_t = values[t + 1]

            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(self,
               states: torch.Tensor,
               actions: torch.Tensor,
               old_log_probs: torch.Tensor,
               advantages: torch.Tensor,
               returns: torch.Tensor,
               num_epochs: int = 10,
               batch_size: int = 64) -> dict:
        """
        Update policy using PPO.

        Args:
            states: State batch (num_steps, state_dim)
            actions: Action batch (num_steps,)
            old_log_probs: Old log probabilities (num_steps,)
            advantages: Advantage estimates (num_steps,)
            returns: Discounted returns (num_steps,)
            num_epochs: Number of update epochs
            batch_size: Mini-batch size

        Returns:
            Dictionary with training metrics
        """
        num_steps = states.shape[0]
        indices = np.arange(num_steps)

        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': []
        }

        for epoch in range(num_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_steps, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Mini-batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Normalize advantages
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # Evaluate actions
                log_probs, values, entropy = self.ac_network.evaluate_actions(batch_states, batch_actions)

                # Ratio for PPO
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), max_norm=0.5)
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    kl = (batch_old_log_probs - log_probs).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(entropy.mean().item())
                metrics['total_loss'].append(loss.item())
                metrics['kl_divergence'].append(kl.item())
                metrics['clip_fraction'].append(clip_fraction.item())

        # Average metrics
        return {k: np.mean(v) for k, v in metrics.items()}

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'ac_network': self.ac_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef
            }
        }
        torch.save(checkpoint, path)
        logger.info(f"PPO-LSTM model saved: {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.ac_network.load_state_dict(checkpoint['ac_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"PPO-LSTM model loaded: {path}")


def create_ppo_lstm_agent(state_dim: int,
                          action_dim: int = 3,
                          hidden_dim: int = 128,
                          num_layers: int = 2) -> PPOLSTMAgent:
    """
    Factory function to create PPO-LSTM agent.

    Args:
        state_dim: Dimension of state space
        action_dim: Number of actions
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers

    Returns:
        Initialized PPO-LSTM agent
    """
    return PPOLSTMAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
