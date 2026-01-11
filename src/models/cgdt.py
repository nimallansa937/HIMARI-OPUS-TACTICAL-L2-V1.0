"""
HIMARI Layer 2 - Critic-Guided Decision Transformer (CGDT)
Sequence-to-sequence transformer for offline RL with critic guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # Feedforward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class DecisionTransformer(nn.Module):
    """
    Decision Transformer core architecture.

    Processes sequences of (return-to-go, state, action) tuples.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 max_length: int = 64,
                 dropout: float = 0.1):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Embeddings for different input types
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Embedding(action_dim, hidden_dim)
        self.return_encoder = nn.Linear(1, hidden_dim)

        # Timestep embedding
        self.timestep_encoder = nn.Embedding(max_length, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=3 * max_length)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Output heads
        self.action_predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(f"DecisionTransformer: {hidden_dim}D, {num_layers} layers, {num_heads} heads")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor,
                returns_to_go: torch.Tensor,
                timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            states: State sequence (batch, seq_len, state_dim)
            actions: Action sequence (batch, seq_len)
            returns_to_go: Return-to-go sequence (batch, seq_len)
            timesteps: Timestep indices (batch, seq_len)

        Returns:
            Action logits (batch, seq_len, action_dim)
        """
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Encode inputs
        state_emb = self.state_encoder(states)  # (batch, seq_len, hidden_dim)
        action_emb = self.action_encoder(actions)  # (batch, seq_len, hidden_dim)
        return_emb = self.return_encoder(returns_to_go.unsqueeze(-1))  # (batch, seq_len, hidden_dim)

        # Add timestep embeddings
        timestep_emb = self.timestep_encoder(timesteps)
        state_emb = state_emb + timestep_emb
        action_emb = action_emb + timestep_emb
        return_emb = return_emb + timestep_emb

        # Interleave: [R_0, s_0, a_0, R_1, s_1, a_1, ...]
        sequence = torch.stack([return_emb, state_emb, action_emb], dim=2)  # (batch, seq_len, 3, hidden_dim)
        sequence = sequence.reshape(batch_size, 3 * seq_len, self.hidden_dim)  # (batch, 3*seq_len, hidden_dim)

        # Add positional encoding
        sequence = self.pos_encoding(sequence)

        # Create causal mask
        mask = self._generate_causal_mask(3 * seq_len, sequence.device)

        # Transformer forward
        for block in self.transformer_blocks:
            sequence = block(sequence, mask)

        # Extract state embeddings (every 3rd token starting from index 1)
        state_indices = torch.arange(1, 3 * seq_len, 3, device=sequence.device)
        state_embeddings = sequence[:, state_indices, :]  # (batch, seq_len, hidden_dim)

        # Predict actions
        action_logits = self.action_predictor(state_embeddings)  # (batch, seq_len, action_dim)

        return action_logits

    def _generate_causal_mask(self, size: int, device: str) -> torch.Tensor:
        """Generate causal mask for autoregressive modeling."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class CriticNetwork(nn.Module):
    """Critic network for value estimation (used for guidance)."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate state value."""
        return self.network(state)


class CGDTAgent(nn.Module):
    """
    Critic-Guided Decision Transformer Agent.

    Features:
    - Decision Transformer for sequence modeling
    - Critic network for value-based guidance
    - Combines autoregressive generation with value estimation
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int = 3,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 max_length: int = 64,
                 critic_weight: float = 0.1):
        """
        Initialize CGDT agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dim: Transformer hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_length: Maximum sequence length
            critic_weight: Weight for critic guidance loss
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length
        self.critic_weight = critic_weight

        # Decision Transformer
        self.dt = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_length
        )

        # Critic network
        self.critic = CriticNetwork(state_dim, hidden_dim)

        logger.info(f"CGDT initialized: {hidden_dim}D, {num_layers} layers")
        logger.info(f"Total params: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, states, actions, returns_to_go, timesteps):
        """Forward pass through Decision Transformer."""
        return self.dt(states, actions, returns_to_go, timesteps)

    def get_action(self,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   returns_to_go: torch.Tensor,
                   timesteps: torch.Tensor,
                   deterministic: bool = False) -> int:
        """
        Get next action given trajectory context.

        Args:
            states: State sequence (1, seq_len, state_dim)
            actions: Action sequence (1, seq_len)
            returns_to_go: Return-to-go sequence (1, seq_len)
            timesteps: Timestep indices (1, seq_len)
            deterministic: If True, use argmax

        Returns:
            Next action
        """
        with torch.no_grad():
            action_logits = self.dt(states, actions, returns_to_go, timesteps)
            last_logits = action_logits[:, -1, :]  # Take last timestep

            if deterministic:
                action = last_logits.argmax(dim=-1)
            else:
                probs = F.softmax(last_logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return action.item()

    def compute_loss(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     returns_to_go: torch.Tensor,
                     timesteps: torch.Tensor,
                     target_actions: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute CGDT loss with critic guidance.

        Args:
            states: State batch (batch, seq_len, state_dim)
            actions: Action batch (batch, seq_len)
            returns_to_go: Return-to-go batch (batch, seq_len)
            timesteps: Timestep batch (batch, seq_len)
            target_actions: Ground truth actions (batch, seq_len)

        Returns:
            loss: Total loss
            info: Loss components
        """
        # Decision Transformer loss (action prediction)
        action_logits = self.dt(states, actions, returns_to_go, timesteps)
        dt_loss = F.cross_entropy(
            action_logits.reshape(-1, self.action_dim),
            target_actions.reshape(-1)
        )

        # Critic loss (value estimation)
        values = self.critic(states).squeeze(-1)  # (batch, seq_len)
        critic_loss = F.mse_loss(values, returns_to_go)

        # Total loss
        total_loss = dt_loss + self.critic_weight * critic_loss

        # Info dict
        info = {
            'loss': total_loss.item(),
            'dt_loss': dt_loss.item(),
            'critic_loss': critic_loss.item(),
            'action_accuracy': (action_logits.argmax(dim=-1) == target_actions).float().mean().item()
        }

        return total_loss, info

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'dt': self.dt.state_dict(),
            'critic': self.critic.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'max_length': self.max_length,
                'critic_weight': self.critic_weight
            }
        }
        torch.save(checkpoint, path)
        logger.info(f"CGDT model saved: {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.dt.load_state_dict(checkpoint['dt'])
        self.critic.load_state_dict(checkpoint['critic'])
        logger.info(f"CGDT model loaded: {path}")


def create_cgdt_agent(state_dim: int,
                      action_dim: int = 3,
                      hidden_dim: int = 256,
                      num_layers: int = 6) -> CGDTAgent:
    """
    Factory function to create CGDT agent.

    Args:
        state_dim: Dimension of state space
        action_dim: Number of actions
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers

    Returns:
        Initialized CGDT agent
    """
    return CGDTAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
