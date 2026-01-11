"""
HIMARI Layer 2 - Critic-Guided Decision Transformer (CGDT)
Subsystem D: Decision Engine (Method D2)

Purpose:
    Upgrade from basic Decision Transformer to Critic-Guided DT that uses
    a learned critic to weight action selection by advantage.

Why CGDT over basic DT?
    - DT naively conditions on returns without considering action quality
    - CGDT uses critic to estimate advantages and weight actions
    - Better credit assignment in sparse reward settings
    - Improved sample efficiency

Architecture:
    - Same GPT-style transformer backbone as DT
    - Added critic head for Q-value estimation
    - Advantage-weighted action selection

Reference:
    - Decision Transformer (Chen et al., 2021)
    - Critic-guided improvements for offline RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math
from loguru import logger


@dataclass
class CGDTConfig:
    """Critic-Guided Decision Transformer configuration"""
    state_dim: int = 60
    action_dim: int = 3          # BUY, HOLD, SELL
    hidden_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    context_length: int = 500    # K in DT paper
    target_return: float = 2.5   # Target Sharpe ratio
    max_timestep: int = 10000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    max_grad_norm: float = 0.25
    critic_coef: float = 0.5     # Weight for critic loss


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention"""
    
    def __init__(self, config: CGDTConfig):
        super().__init__()
        assert config.hidden_dim % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_dim // config.n_heads
        
        self.query = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_length * 3, config.context_length * 3))
                 .view(1, 1, config.context_length * 3, config.context_length * 3)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        q = self.query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    """Transformer block with attention + MLP"""
    
    def __init__(self, config: CGDTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CriticGuidedDecisionTransformer(nn.Module):
    """
    Critic-Guided Decision Transformer.
    
    Extends DT with:
    - Critic head for Q-value estimation
    - Advantage-weighted action selection
    - Improved credit assignment
    
    Example:
        >>> config = CGDTConfig(state_dim=60, action_dim=3)
        >>> cgdt = CriticGuidedDecisionTransformer(config)
        >>> action = cgdt.get_action(states, actions, returns_to_go)
    """
    
    def __init__(self, config: CGDTConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.state_embed = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_embed = nn.Embedding(config.action_dim, config.hidden_dim)
        self.return_embed = nn.Linear(1, config.hidden_dim)
        self.timestep_embed = nn.Embedding(config.max_timestep, config.hidden_dim)
        
        # Position embedding for sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, config.context_length * 3, config.hidden_dim))
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        
        # Output heads
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.return_head = nn.Linear(config.hidden_dim, 1)
        
        # CGDT addition: Critic head for Q-values
        self.critic_head = nn.Linear(config.hidden_dim, config.action_dim)
        
        self.apply(self._init_weights)
        
        logger.debug(
            f"CGDT initialized: {sum(p.numel() for p in self.parameters())/1e6:.1f}M params"
        )
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through CGDT.
        
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len) action indices
            returns_to_go: (batch, seq_len, 1) cumulative returns
            timesteps: (batch, seq_len) timestep indices
            
        Returns:
            action_preds: (batch, seq_len, action_dim) action logits
            return_preds: (batch, seq_len, 1) return predictions
            q_values: (batch, seq_len, action_dim) Q-value estimates
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        if timesteps is None:
            timesteps = torch.arange(seq_len, device=states.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed each modality
        state_embeds = self.state_embed(states)
        action_embeds = self.action_embed(actions)
        return_embeds = self.return_embed(returns_to_go)
        time_embeds = self.timestep_embed(timesteps)
        
        # Add timestep embeddings
        state_embeds = state_embeds + time_embeds
        action_embeds = action_embeds + time_embeds
        return_embeds = return_embeds + time_embeds
        
        # Interleave: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # Stack along sequence dimension
        stacked = torch.stack([return_embeds, state_embeds, action_embeds], dim=2)
        stacked = stacked.reshape(batch_size, seq_len * 3, self.config.hidden_dim)
        
        # Add positional embedding
        stacked = stacked + self.pos_embed[:, :seq_len * 3, :]
        stacked = self.dropout(stacked)
        
        # Transformer
        for block in self.blocks:
            stacked = block(stacked)
        
        stacked = self.ln_f(stacked)
        
        # Extract state positions (index 1, 4, 7, ...)
        state_reps = stacked[:, 1::3, :]  # (batch, seq_len, hidden)
        
        # Predict actions and returns from state representations
        action_preds = self.action_head(state_reps)
        return_preds = self.return_head(state_reps)
        
        # CGDT: Q-value predictions
        q_values = self.critic_head(state_reps)
        
        return action_preds, return_preds, q_values
    
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        use_critic: bool = True
    ) -> Tuple[int, float]:
        """
        Get action using critic-guided selection.
        
        Args:
            states: (1, seq_len, state_dim)
            actions: (1, seq_len)
            returns_to_go: (1, seq_len, 1)
            timesteps: (1, seq_len)
            temperature: Softmax temperature
            use_critic: Whether to use critic for action selection
            
        Returns:
            action: Selected action index
            confidence: Action probability
        """
        self.eval()
        with torch.no_grad():
            action_preds, _, q_values = self.forward(states, actions, returns_to_go, timesteps)
            
            # Get last timestep predictions
            action_logits = action_preds[:, -1, :]  # (1, action_dim)
            q = q_values[:, -1, :]  # (1, action_dim)
            
            if use_critic:
                # CGDT: Weight action logits by advantage
                # A(s,a) ≈ Q(s,a) - V(s), where V(s) = max Q(s,a)
                advantages = q - q.max(dim=-1, keepdim=True)[0]
                weighted_logits = action_logits + advantages
                probs = F.softmax(weighted_logits / temperature, dim=-1)
            else:
                probs = F.softmax(action_logits / temperature, dim=-1)
            
            action = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, action].item()
        
        return action, confidence
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        rewards: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CGDT training loss.
        
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len) target actions
            returns_to_go: (batch, seq_len, 1)
            rewards: (batch, seq_len) rewards for critic
            timesteps: (batch, seq_len)
            
        Returns:
            Dict with loss components
        """
        action_preds, return_preds, q_values = self.forward(
            states, actions, returns_to_go, timesteps
        )
        
        # Action prediction loss (cross-entropy)
        action_loss = F.cross_entropy(
            action_preds.view(-1, self.config.action_dim),
            actions.view(-1)
        )
        
        # Return prediction loss (MSE)
        return_loss = F.mse_loss(return_preds, returns_to_go)
        
        # Critic loss (TD target)
        # Q-target = r + γ * V(s')
        with torch.no_grad():
            next_values = q_values[:, 1:, :].max(dim=-1)[0]
            td_targets = rewards[:, :-1] + 0.99 * next_values
        
        # Get Q-values for taken actions
        q_taken = q_values[:, :-1, :].gather(
            2, actions[:, :-1].unsqueeze(-1)
        ).squeeze(-1)
        
        critic_loss = F.mse_loss(q_taken, td_targets)
        
        # Total loss
        total_loss = action_loss + return_loss + self.config.critic_coef * critic_loss
        
        return {
            'total': total_loss,
            'action': action_loss,
            'return': return_loss,
            'critic': critic_loss
        }


class CGDTTrainer:
    """Trainer for Critic-Guided Decision Transformer"""
    
    def __init__(self, model: CriticGuidedDecisionTransformer, config: CGDTConfig, 
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.step = 0
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        rewards: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns_to_go = returns_to_go.to(self.device)
        rewards = rewards.to(self.device)
        timesteps = timesteps.to(self.device)
        
        # Forward + loss
        losses = self.model.compute_loss(states, actions, returns_to_go, rewards, timesteps)
        
        # Backward
        self.optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        self.step += 1
        
        return {k: v.item() for k, v in losses.items()}
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'config': self.config
        }, path)
        logger.info(f"CGDT checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        logger.info(f"CGDT checkpoint loaded from {path}")
