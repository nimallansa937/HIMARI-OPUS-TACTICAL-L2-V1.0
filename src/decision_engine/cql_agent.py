"""
HIMARI Layer 2 - Conservative Q-Learning (CQL) Agent
Subsystem D: Decision Engine (Method D3)

Purpose:
    Offline RL fallback agent that learns conservative Q-values to avoid
    overestimation on out-of-distribution actions.

Why CQL?
    - Standard Q-learning overestimates OOD actions
    - CQL adds penalty for Q-values on unseen actions
    - More reliable in offline/batch RL settings
    - Serves as stable fallback when primary models fail

Architecture:
    - Dual Q-networks for clipped double Q-learning
    - CQL penalty on Q-values for regularization
    - Soft actor-critic style policy extraction

Reference:
    - Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np
from loguru import logger


@dataclass
class CQLConfig:
    """Conservative Q-Learning configuration"""
    state_dim: int = 60
    action_dim: int = 3          # BUY, HOLD, SELL
    hidden_dim: int = 256
    n_layers: int = 3
    learning_rate: float = 3e-4
    gamma: float = 0.99          # Discount factor
    tau: float = 0.005           # Target network update rate
    cql_alpha: float = 1.0       # CQL regularization weight
    cql_temp: float = 1.0        # Temperature for CQL logsumexp
    min_q_weight: float = 5.0    # Weight for min Q penalty
    num_random: int = 10         # Number of random actions for CQL
    with_lagrange: bool = False  # Use Lagrange multiplier for CQL alpha


class QNetwork(nn.Module):
    """Q-Network for discrete actions"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions"""
        return self.network(state)


class CQLAgent(nn.Module):
    """
    Conservative Q-Learning Agent for offline RL.
    
    Learns conservative Q-values by penalizing high Q-values on
    out-of-distribution actions, making it suitable as a stable
    fallback policy.
    
    Example:
        >>> config = CQLConfig(state_dim=60, action_dim=3)
        >>> agent = CQLAgent(config)
        >>> action = agent.select_action(state)
    """
    
    def __init__(self, config: CQLConfig, device: str = 'cuda'):
        super().__init__()
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Dual Q-networks for clipped double Q
        self.q1 = QNetwork(
            config.state_dim, config.action_dim, config.hidden_dim, config.n_layers
        ).to(self.device)
        self.q2 = QNetwork(
            config.state_dim, config.action_dim, config.hidden_dim, config.n_layers
        ).to(self.device)
        
        # Target networks
        self.q1_target = QNetwork(
            config.state_dim, config.action_dim, config.hidden_dim, config.n_layers
        ).to(self.device)
        self.q2_target = QNetwork(
            config.state_dim, config.action_dim, config.hidden_dim, config.n_layers
        ).to(self.device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=config.learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=config.learning_rate)
        
        # CQL alpha (can be learned with Lagrange)
        if config.with_lagrange:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
            self.target_action_gap = 5.0
        else:
            self.log_alpha = None
        
        logger.debug(
            f"CQLAgent initialized: state_dim={config.state_dim}, "
            f"action_dim={config.action_dim}, cql_alpha={config.cql_alpha}"
        )
    
    @property
    def cql_alpha(self) -> float:
        if self.log_alpha is not None:
            return torch.exp(self.log_alpha).item()
        return self.config.cql_alpha
    
    def select_action(self, state: torch.Tensor, deterministic: bool = True) -> int:
        """
        Select action based on Q-values.
        
        Args:
            state: (state_dim,) or (1, state_dim) state tensor
            deterministic: If True, select argmax; else sample
            
        Returns:
            action: Selected action index
        """
        self.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            
            # Use minimum of dual Q for conservatism
            q1 = self.q1(state)
            q2 = self.q2(state)
            q = torch.min(q1, q2)
            
            if deterministic:
                action = q.argmax(dim=-1).item()
            else:
                # Softmax sampling
                probs = F.softmax(q / self.config.cql_temp, dim=-1)
                action = torch.multinomial(probs, 1).item()
        
        return action
    
    def get_action_with_confidence(self, state: torch.Tensor) -> Tuple[int, float]:
        """
        Get action with confidence score.
        
        Args:
            state: State tensor
            
        Returns:
            action: Selected action
            confidence: Softmax probability of selected action
        """
        self.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            
            q = torch.min(self.q1(state), self.q2(state))
            probs = F.softmax(q, dim=-1)
            action = q.argmax(dim=-1).item()
            confidence = probs[0, action].item()
        
        return action, confidence
    
    def compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CQL loss.
        
        Args:
            states: (batch, state_dim)
            actions: (batch,) action indices
            rewards: (batch,)
            next_states: (batch, state_dim)
            dones: (batch,) terminal flags
            
        Returns:
            Dict with loss components
        """
        batch_size = states.shape[0]
        
        # Current Q-values for taken actions
        q1_values = self.q1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_values = self.q2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_q = torch.min(next_q1, next_q2)
            next_v = next_q.max(dim=-1)[0]
            td_target = rewards + (1 - dones) * self.config.gamma * next_v
        
        # Standard TD loss
        td_loss1 = F.mse_loss(q1_values, td_target)
        td_loss2 = F.mse_loss(q2_values, td_target)
        
        # CQL penalty: push down Q-values on random actions
        # logsumexp trick for stable computation
        q1_all = self.q1(states)  # (batch, action_dim)
        q2_all = self.q2(states)
        
        # Log-sum-exp over actions (pushes down all Q-values)
        cql_logsumexp1 = torch.logsumexp(q1_all / self.config.cql_temp, dim=-1)
        cql_logsumexp2 = torch.logsumexp(q2_all / self.config.cql_temp, dim=-1)
        
        # CQL loss: logsumexp(Q) - Q(s, a_data)
        cql_loss1 = (cql_logsumexp1 - q1_values).mean()
        cql_loss2 = (cql_logsumexp2 - q2_values).mean()
        
        # Combine losses
        q1_loss = td_loss1 + self.config.min_q_weight * self.cql_alpha * cql_loss1
        q2_loss = td_loss2 + self.config.min_q_weight * self.cql_alpha * cql_loss2
        
        return {
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'td_loss': (td_loss1 + td_loss2) / 2,
            'cql_loss': (cql_loss1 + cql_loss2) / 2,
            'q1_mean': q1_values.mean(),
            'q2_mean': q2_values.mean()
        }
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform single update step.
        
        Args:
            states: (batch, state_dim)
            actions: (batch,)
            rewards: (batch,)
            next_states: (batch, state_dim)
            dones: (batch,)
            
        Returns:
            Dict with training metrics
        """
        self.train()
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute losses
        losses = self.compute_cql_loss(states, actions, rewards, next_states, dones)
        
        # Update Q1
        self.q1_optimizer.zero_grad()
        losses['q1_loss'].backward(retain_graph=True)
        self.q1_optimizer.step()
        
        # Update Q2
        self.q2_optimizer.zero_grad()
        losses['q2_loss'].backward()
        self.q2_optimizer.step()
        
        # Soft update targets
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        
        # Update Lagrange alpha if enabled
        if self.log_alpha is not None:
            alpha_loss = -self.log_alpha * (losses['cql_loss'].detach() - self.target_action_gap)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in losses.items()}
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"CQLAgent saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        logger.info(f"CQLAgent loaded from {path}")


# Factory function
def create_cql_agent(state_dim: int = 60, action_dim: int = 3, 
                     device: str = 'cuda') -> CQLAgent:
    """Create CQL agent with default configuration"""
    config = CQLConfig(state_dim=state_dim, action_dim=action_dim)
    return CQLAgent(config, device=device)
