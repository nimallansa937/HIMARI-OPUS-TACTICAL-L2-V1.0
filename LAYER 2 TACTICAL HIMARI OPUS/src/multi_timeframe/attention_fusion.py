"""
HIMARI Layer 2 - Hierarchical Cross-Attention Fusion
Subsystem C: Multi-Timeframe Fusion (Method C2)

Purpose:
    Fuse information from multiple timeframes using cross-attention.
    Allows model to dynamically weight importance of different timeframes.

Architecture:
    - Query: Current state
    - Keys/Values: Timeframe encodings
    - Multi-head attention with 8 heads
"""

import torch
import torch.nn as nn
import math
from typing import List
from loguru import logger


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention layer for timeframe fusion.
    
    Args:
        embed_dim: Embedding dimension (should match encoder hidden_dim)
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Scaling factor
        self.scale = math.sqrt(self.head_dim)
        
        logger.debug(f"CrossAttentionFusion: embed_dim={embed_dim}, num_heads={num_heads}")
    
    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: Query tensor (batch, query_len, embed_dim)
            keys: Key tensor (batch, key_len, embed_dim)
            values: Value tensor (batch, value_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Fused output (batch, query_len, embed_dim)
        """
        batch_size = query.shape[0]
        
        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(keys)
        V = self.v_proj(values)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Residual connection + layer norm
        output = self.layer_norm(query + output)
        
        return output


class HierarchicalTimeframeFusion(nn.Module):
    """
    Hierarchical fusion of multiple timeframes.
    
    Strategy:
        1. Fast timeframes (1m, 5m) → Short-term context
        2. Slow timeframes (1h, 4h) → Long-term context
        3. Cross-attend short-term to long-term
        4. Final fusion
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Attention layers
        self.fast_attention = CrossAttentionFusion(embed_dim, num_heads, dropout)
        self.slow_attention = CrossAttentionFusion(embed_dim, num_heads, dropout)
        self.cross_scale_attention = CrossAttentionFusion(embed_dim, num_heads, dropout)
        
        # Final MLP
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        logger.info("HierarchicalTimeframeFusion initialized")
    
    def forward(self, timeframe_encodings: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse timeframe encodings hierarchically.
        
        Args:
            timeframe_encodings: Dict mapping timeframe -> encoding (batch, embed_dim)
            
        Returns:
            Fused representation (batch, embed_dim)
        """
        # Add sequence dimension for attention
        fast_encs = []
        slow_encs = []
        
        for tf, enc in timeframe_encodings.items():
            enc = enc.unsqueeze(1)  # (batch, 1, embed_dim)
            if tf in ["1m", "5m"]:
                fast_encs.append(enc)
            elif tf in ["1h", "4h"]:
                slow_encs.append(enc)
        
        # Stack encodings
        if fast_encs:
            fast_stack = torch.cat(fast_encs, dim=1)  # (batch, n_fast, embed_dim)
        else:
            fast_stack = torch.zeros(enc.shape[0], 1, self.embed_dim, device=enc.device)
        
        if slow_encs:
            slow_stack = torch.cat(slow_encs, dim=1)  # (batch, n_slow, embed_dim)
        else:
            slow_stack = torch.zeros(enc.shape[0], 1, self.embed_dim, device=enc.device)
        
        # Self-attention within fast and slow
        fast_fused = self.fast_attention(fast_stack, fast_stack, fast_stack)
        slow_fused = self.slow_attention(slow_stack, slow_stack, slow_stack)
        
        # Cross-scale attention: fast queries slow
        cross_fused = self.cross_scale_attention(fast_fused, slow_fused, slow_fused)
        
        # Pool and feed-forward
        pooled = cross_fused.mean(dim=1)  # (batch, embed_dim)
        output = self.ffn(pooled)
        
        return output
