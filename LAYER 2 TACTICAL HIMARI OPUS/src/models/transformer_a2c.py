"""
HIMARI Layer 2 - Transformer-A2C Model
Tactical Decision Model following HIMARI_Layer2_Transformer_A2C_Training_Guide.md

Based on:
- TACR (Kim et al. 2023): Transformer for RL trading
- TFT (Lim et al. 2019): Temporal Fusion Transformer
- A2C: Advantage Actor-Critic

Architecture:
    Input → Transformer Encoder (4 blocks) → [Actor Head, Critic Head]
    
Output:
    - action: FLAT (0), LONG (1), SHORT (2)
    - confidence: softmax probability of chosen action
    - value: V(s) estimate for training
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class TransformerA2CConfig:
    """
    Hyperparameters tuned based on TACR paper + HIMARI constraints.
    
    KEY PRINCIPLE: Conservative defaults to prevent overfitting.
    """
    
    # === MODEL ARCHITECTURE ===
    input_dim: int = 44           # Features per timestep
    hidden_dim: int = 256         # Transformer hidden size
    num_heads: int = 8            # Multi-head attention
    num_layers: int = 4           # Transformer blocks
    context_length: int = 100     # Timesteps (8.3 hours @ 5-min)
    
    # === REGULARIZATION (CRITICAL) ===
    dropout: float = 0.2          # Higher than typical (prevent overfit)
    attention_dropout: float = 0.1
    weight_decay: float = 1e-4    # L2 regularization
    max_grad_norm: float = 0.5    # Gradient clipping
    
    # === A2C PARAMETERS ===
    actor_lr: float = 1e-4        # Conservative (not 3e-4)
    critic_lr: float = 3e-4       # Critic can learn faster
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE parameter
    entropy_coef: float = 0.07    # Exploration bonus (increased from 0.01 to prevent collapse)
    value_coef: float = 0.5       # Critic loss weight
    
    # === TRAINING SCHEDULE ===
    batch_size: int = 64
    rollout_steps: int = 2048     # Steps before update
    max_steps: int = 500_000      # Soft limit (early stop likely)
    
    # === EARLY STOPPING ===
    patience: int = 3             # Validation checks before stop
    min_improvement: float = 0.01 # Minimum Sharpe improvement
    val_frequency: int = 25_000   # Steps between validation
    
    # === CHECKPOINTING ===
    checkpoint_frequency: int = 50_000
    keep_best_n: int = 5          # Keep top N checkpoints by val_sharpe


# ==============================================================================
# Positional Encoding
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Learned positional encoding for time series."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ==============================================================================
# Transformer Encoder Block
# ==============================================================================

class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with pre-norm."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)
        
        # Pre-norm feedforward
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out
        
        return x


# ==============================================================================
# Tactical Transformer Encoder
# ==============================================================================

class TacticalTransformerEncoder(nn.Module):
    """
    Transformer encoder for Layer 2 tactical decisions.
    
    Based on TFT (Lim et al. 2019) and TACR (Kim et al. 2023).
    """
    
    def __init__(self, config: TransformerA2CConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.hidden_dim,
            max_len=config.context_length + 50,
            dropout=config.dropout,
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.hidden_dim,
                num_heads=config.num_heads,
                d_ff=config.hidden_dim * 4,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, input_dim]
            
        Returns:
            encoded: [batch, hidden_dim] (last timestep representation)
        """
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Return last timestep (for decision making)
        return x[:, -1, :]


# ==============================================================================
# Actor Head
# ==============================================================================

class ActorHead(nn.Module):
    """
    Actor network for discrete actions (LONG/SHORT/FLAT).
    
    Output: Probability distribution over actions.
    """
    
    def __init__(self, hidden_dim: int, num_actions: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        
        # Small initialization for output layer
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_dim] from encoder
            
        Returns:
            logits: [batch, num_actions] (unnormalized log probabilities)
        """
        return self.network(x)
    
    def get_action(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: [batch] selected action indices
            log_prob: [batch] log probability of selected action
            probs: [batch, num_actions] full probability distribution
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)
        
        return action, log_prob, probs


# ==============================================================================
# Critic Head
# ==============================================================================

class CriticHead(nn.Module):
    """
    Critic network for state value estimation.
    
    Output: V(s) estimate.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Standard initialization for value function
        nn.init.orthogonal_(self.network[-1].weight, gain=1.0)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_dim] from encoder
            
        Returns:
            value: [batch, 1] state value estimate
        """
        return self.network(x)


# ==============================================================================
# Complete Transformer-A2C Model
# ==============================================================================

class TransformerA2C(nn.Module):
    """
    Complete Transformer-A2C model for Layer 2 tactical decisions.
    
    Architecture:
        Input → Transformer Encoder → [Actor Head, Critic Head]
        
    Output:
        - action: FLAT (0), LONG (1), SHORT (2)
        - confidence: softmax probability of chosen action
        - value: V(s) estimate for training
    """
    
    def __init__(self, config: TransformerA2CConfig):
        super().__init__()
        self.config = config
        
        self.encoder = TacticalTransformerEncoder(config)
        self.actor = ActorHead(config.hidden_dim, num_actions=3, dropout=config.dropout * 0.5)
        self.critic = CriticHead(config.hidden_dim, dropout=config.dropout * 0.5)
        
        # Log model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"TransformerA2C initialized: {trainable_params:,} trainable parameters")
    
    def forward(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict:
        """
        Forward pass for training and inference.
        
        Args:
            states: [batch, seq_len, input_dim]
            deterministic: If True, take argmax action (for inference)
            
        Returns:
            dict with: action, log_prob, value, probs, confidence
        """
        # Encode sequence
        encoded = self.encoder(states)
        
        # Get action from actor
        action, log_prob, probs = self.actor.get_action(encoded, deterministic)
        
        # Get value from critic
        value = self.critic(encoded).squeeze(-1)
        
        # Confidence is the probability of the selected action
        confidence = probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "probs": probs,
            "confidence": confidence,
            "encoded": encoded,  # For debugging/analysis
        }
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for A2C update.
        
        Args:
            states: [batch, seq_len, input_dim]
            actions: [batch] action indices
            
        Returns:
            log_probs: [batch] log probability of actions
            values: [batch] state values
            entropy: [batch] policy entropy
        """
        encoded = self.encoder(states)
        
        # Get logits and values
        logits = self.actor(encoded)
        values = self.critic(encoded).squeeze(-1)
        
        # Compute log probs and entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)
        
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy()
        
        return log_probs, values, entropy
    
    @torch.no_grad()
    def predict(self, states: torch.Tensor) -> Dict:
        """
        Inference-only prediction (no gradients).
        
        For production use in online pipeline.
        """
        self.eval()
        return self.forward(states, deterministic=True)
    
    def get_config(self) -> Dict:
        """Return configuration as dictionary."""
        return asdict(self.config)


# ==============================================================================
# Factory Function
# ==============================================================================

def create_transformer_a2c(
    input_dim: int = 44,
    hidden_dim: int = 256,
    num_layers: int = 4,
    context_length: int = 100,
    device: str = "cuda",
) -> TransformerA2C:
    """
    Factory function to create TransformerA2C model.
    
    Args:
        input_dim: Feature dimension per timestep
        hidden_dim: Transformer hidden size
        num_layers: Number of transformer blocks
        context_length: Sequence length
        device: Device to place model on
        
    Returns:
        Initialized TransformerA2C model
    """
    config = TransformerA2CConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        context_length=context_length,
    )
    
    model = TransformerA2C(config)
    model = model.to(device)
    
    return model
