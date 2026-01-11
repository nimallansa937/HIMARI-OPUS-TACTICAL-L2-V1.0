"""
HIMARI Layer 2 - FLAG-TRADER with LoRA
Large transformer model (135M parameters) with LoRA fine-tuning for trading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer.

    Adds trainable low-rank matrices to frozen weights.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize A with small random values, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA: x @ (W + BA * scaling)"""
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()

        # Frozen base layer
        self.base = nn.Linear(in_features, out_features)
        for param in self.base.parameters():
            param.requires_grad = False

        # LoRA adaptation
        self.lora = LoRALayer(in_features, out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: base(x) + lora(x)"""
        return self.base(x) + self.lora(x)


class MultiHeadAttentionWithLoRA(nn.Module):
    """Multi-head attention with LoRA on Q, K, V projections."""

    def __init__(self, d_model: int, num_heads: int, lora_rank: int = 16, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q, K, V projections with LoRA
        self.q_proj = LoRALinear(d_model, d_model, rank=lora_rank)
        self.k_proj = LoRALinear(d_model, d_model, rank=lora_rank)
        self.v_proj = LoRALinear(d_model, d_model, rank=lora_rank)

        # Output projection
        self.out_proj = LoRALinear(d_model, d_model, rank=lora_rank)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Q, K, V projections
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_out = attn_probs @ V

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_out)

        return output


class TransformerBlockWithLoRA(nn.Module):
    """Transformer block with LoRA-adapted layers."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, lora_rank: int = 16, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttentionWithLoRA(d_model, num_heads, lora_rank, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward with LoRA
        self.ffn = nn.Sequential(
            LoRALinear(d_model, dim_feedforward, rank=lora_rank),
            nn.GELU(),
            nn.Dropout(dropout),
            LoRALinear(dim_feedforward, d_model, rank=lora_rank),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)

        # Feedforward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class FLAGTRADERModel(nn.Module):
    """
    FLAG-TRADER: Large transformer model for trading.

    Architecture:
    - 135M parameters (base model size)
    - LoRA fine-tuning (only ~1-2M trainable params)
    - Context window: 64-256 timesteps
    - Instruction tuning format
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int = 3,
                 d_model: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 dim_feedforward: int = 3072,
                 max_length: int = 256,
                 lora_rank: int = 16,
                 dropout: float = 0.1):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_length = max_length

        # Input embedding
        self.state_embedding = nn.Linear(state_dim, d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, d_model))

        # Transformer blocks with LoRA
        self.transformer_blocks = nn.ModuleList([
            TransformerBlockWithLoRA(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                lora_rank=lora_rank,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, action_dim)

        # Initialize
        self._init_weights()

        # Calculate parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"FLAG-TRADER initialized: {d_model}D, {num_layers} layers, {num_heads} heads")
        logger.info(f"Total params: {total_params:,} ({total_params/1e6:.1f}M)")
        logger.info(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        logger.info(f"LoRA rank: {lora_rank}")

    def _init_weights(self):
        """Initialize weights."""
        # Positional embeddings
        nn.init.normal_(self.pos_embedding, std=0.02)

        # Input/output layers
        nn.init.normal_(self.state_embedding.weight, std=0.02)
        nn.init.zeros_(self.state_embedding.bias)
        nn.init.normal_(self.action_head.weight, std=0.02)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            states: State sequence (batch, seq_len, state_dim)
            mask: Optional attention mask (batch, seq_len, seq_len)

        Returns:
            Action logits (batch, seq_len, action_dim)
        """
        batch_size, seq_len, _ = states.shape

        # Embed states
        x = self.state_embedding(states)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]

        # Transformer forward
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Output
        x = self.output_norm(x)
        action_logits = self.action_head(x)  # (batch, seq_len, action_dim)

        return action_logits

    def get_action(self, states: torch.Tensor, deterministic: bool = False) -> int:
        """
        Get next action.

        Args:
            states: State sequence (1, seq_len, state_dim)
            deterministic: If True, use argmax

        Returns:
            Action index
        """
        with torch.no_grad():
            action_logits = self.forward(states)
            last_logits = action_logits[:, -1, :]  # Last timestep

            if deterministic:
                action = last_logits.argmax(dim=-1)
            else:
                probs = F.softmax(last_logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return action.item()


class FLAGTRADERAgent(nn.Module):
    """
    FLAG-TRADER Agent with training interface.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int = 3,
                 d_model: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 lora_rank: int = 16,
                 max_length: int = 256):
        """
        Initialize FLAG-TRADER agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            d_model: Model hidden dimension (768 for ~135M params)
            num_layers: Number of transformer layers (12 for ~135M params)
            num_heads: Number of attention heads
            lora_rank: LoRA rank (lower = fewer trainable params)
            max_length: Maximum sequence length
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length

        # FLAG-TRADER model
        self.model = FLAGTRADERModel(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=4 * d_model,
            max_length=max_length,
            lora_rank=lora_rank
        )

    def forward(self, states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        return self.model(states, mask)

    def get_action(self, states: torch.Tensor, deterministic: bool = False) -> int:
        """Get next action."""
        return self.model.get_action(states, deterministic)

    def compute_loss(self, states: torch.Tensor, target_actions: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute supervised learning loss.

        Args:
            states: State batch (batch, seq_len, state_dim)
            target_actions: Target action batch (batch, seq_len)

        Returns:
            loss: Cross-entropy loss
            info: Loss info dict
        """
        action_logits = self.model(states)

        # Cross-entropy loss
        loss = F.cross_entropy(
            action_logits.reshape(-1, self.action_dim),
            target_actions.reshape(-1)
        )

        # Accuracy
        predictions = action_logits.argmax(dim=-1)
        accuracy = (predictions == target_actions).float().mean()

        info = {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }

        return loss, info

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model': self.model.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'max_length': self.max_length
            }
        }
        torch.save(checkpoint, path)
        logger.info(f"FLAG-TRADER model saved: {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        logger.info(f"FLAG-TRADER model loaded: {path}")


def create_flag_trader_agent(state_dim: int,
                             action_dim: int = 3,
                             model_size: str = "135M",
                             lora_rank: int = 16) -> FLAGTRADERAgent:
    """
    Factory function to create FLAG-TRADER agent.

    Args:
        state_dim: Dimension of state space
        action_dim: Number of actions
        model_size: Model size ("135M", "350M", "1B")
        lora_rank: LoRA rank

    Returns:
        Initialized FLAG-TRADER agent
    """
    # Model configurations
    configs = {
        "135M": {"d_model": 768, "num_layers": 12, "num_heads": 12},
        "350M": {"d_model": 1024, "num_layers": 24, "num_heads": 16},
        "1B": {"d_model": 1536, "num_layers": 36, "num_heads": 24}
    }

    if model_size not in configs:
        logger.warning(f"Unknown model size {model_size}, using 135M")
        model_size = "135M"

    config = configs[model_size]

    return FLAGTRADERAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        lora_rank=lora_rank
    )
