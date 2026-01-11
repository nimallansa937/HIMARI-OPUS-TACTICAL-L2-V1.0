"""
HIMARI Layer 2 - C6: Cross-Attention Fusion
============================================

Let me explain why multi-timeframe analysis requires more than simple
concatenation. HIMARI processes 9 timeframes simultaneously: tick, 1m, 5m,
15m, 30m, 1h, 4h, 1d, and weekly. Each captures different market dynamics:

- Tick/1m: Microstructure, order flow, immediate price action
- 5m/15m: Intraday patterns, session transitions
- 30m/1h: Trend development, support/resistance
- 4h/1d: Swing trades, major trend direction
- Weekly: Macro cycles, long-term sentiment

The naive approach—concatenating all timeframe representations—creates two
problems. First, it treats all timeframes equally, when really their relevance
varies with market conditions. During a breakout, short timeframes dominate;
during consolidation, longer timeframes matter more. Second, it ignores the
hierarchical structure where shorter timeframes are nested within longer ones.

The Cross-Attention Solution:
-----------------------------
Cross-attention allows each timeframe to selectively query information from
others. Think of it as asking questions:

- The 5m timeframe asks: "What's the 1h trend direction I should align with?"
- The 1h timeframe asks: "Is the 5m showing momentum that confirms my signal?"
- The 4h timeframe asks: "Are shorter timeframes showing reversal patterns?"

This is implemented as:
    CrossAttention(Q=timeframe_i, K=timeframe_j, V=timeframe_j)

Where timeframe_i queries timeframe_j for relevant information.

Hierarchical Structure:
-----------------------
We organize cross-attention in a hierarchy:

Level 1 (Local): Adjacent timeframes interact
    tick ↔ 1m, 1m ↔ 5m, 5m ↔ 15m, etc.

Level 2 (Regional): Skip-one connections
    tick ↔ 5m, 1m ↔ 15m, 5m ↔ 30m, etc.

Level 3 (Global): All timeframes attend to daily/weekly anchors
    All → 1d, All → weekly

This hierarchy reduces computational cost (not all pairs computed) while
capturing the essential multi-scale relationships.

For HIMARI, cross-attention fusion is the core mechanism that transforms
9 independent timeframe representations into a unified multi-scale view
that respects temporal hierarchy.

Performance Targets:
- Latency: <15ms for full cross-attention pass
- Memory: <400MB for 9 timeframes × 128-dim representations
- Quality: 20-30% improvement in regime detection vs single timeframe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TimeframeLevel(Enum):
    """Timeframe hierarchy levels."""
    MICRO = 0      # tick, 1m
    SHORT = 1      # 5m, 15m
    MEDIUM = 2     # 30m, 1h
    LONG = 3       # 4h, 1d
    MACRO = 4      # weekly+


# Mapping of timeframe names to hierarchy levels
TIMEFRAME_HIERARCHY = {
    'tick': TimeframeLevel.MICRO,
    '1m': TimeframeLevel.MICRO,
    '5m': TimeframeLevel.SHORT,
    '15m': TimeframeLevel.SHORT,
    '30m': TimeframeLevel.MEDIUM,
    '1h': TimeframeLevel.MEDIUM,
    '4h': TimeframeLevel.LONG,
    '1d': TimeframeLevel.LONG,
    'weekly': TimeframeLevel.MACRO,
}


@dataclass
class CrossAttentionConfig:
    """Configuration for Cross-Attention Fusion.
    
    Attributes:
        d_model: Model dimension for all timeframes
        n_heads: Number of attention heads
        dropout: Dropout rate
        timeframes: List of timeframe names in order
        use_hierarchical: Whether to use hierarchical attention structure
        n_fusion_layers: Number of cross-attention fusion layers
        use_gating: Whether to gate cross-attention outputs
    """
    d_model: int = 128
    n_heads: int = 4
    dropout: float = 0.1
    timeframes: List[str] = field(default_factory=lambda: [
        'tick', '1m', '5m', '15m', '30m', '1h', '4h', '1d', 'weekly'
    ])
    use_hierarchical: bool = True
    n_fusion_layers: int = 2
    use_gating: bool = True


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block between two timeframes.
    
    This is the fundamental building block: timeframe A queries timeframe B.
    The query timeframe learns what information to extract from the key/value
    timeframe based on its current state.
    
    For example, if query=5m and key/value=1h:
    - Query asks: "Given my current pattern, what 1h context is relevant?"
    - Keys encode: "Here are the aspects of 1h data you can query"
    - Values provide: "Here's the actual 1h information for each aspect"
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0
        
        # Separate projections for query and key/value sources
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention from query timeframe to key/value timeframe.
        
        Args:
            query: Query timeframe representation (batch, d_model)
            key_value: Key/value timeframe representation (batch, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Cross-attended representation (batch, d_model)
            attention: Attention weights (batch, n_heads, 1, 1)
        """
        batch_size = query.shape[0]
        
        # Add sequence dimension if not present (for single-vector representations)
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (batch, 1, d_model)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)  # (batch, 1, d_model)
        
        seq_q = query.shape[1]
        seq_kv = key_value.shape[1]
        
        # Project
        q = self.q_proj(query)    # (batch, seq_q, d_model)
        k = self.k_proj(key_value)  # (batch, seq_kv, d_model)
        v = self.v_proj(key_value)  # (batch, seq_kv, d_model)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_kv, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_q, self.d_model)
        output = self.out_proj(output)
        
        # Remove sequence dimension if we added it
        if seq_q == 1:
            output = output.squeeze(1)
        
        return output, attention


class GatedFusion(nn.Module):
    """
    Gated fusion of cross-attention output with original representation.
    
    Not all cross-attention information is useful. The gate learns when to
    incorporate cross-timeframe information and when to rely on the
    original single-timeframe representation.
    
    Gate(x, cross) = σ(W[x; cross] + b) * cross + (1 - σ(...)) * x
    
    This is similar to LSTM/GRU gating and allows the model to:
    - Ignore noisy cross-timeframe signals
    - Amplify strong cross-timeframe confirmations
    - Adapt gating based on market conditions
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # Gate computation from concatenated features
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        original: torch.Tensor,
        cross_attended: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gated combination of original and cross-attended representations.
        
        Args:
            original: Original timeframe representation (batch, d_model)
            cross_attended: Cross-attention output (batch, d_model)
            
        Returns:
            Gated fusion output (batch, d_model)
        """
        # Compute gate
        combined = torch.cat([original, cross_attended], dim=-1)
        gate = self.gate(combined)
        
        # Gated combination
        fused = gate * cross_attended + (1 - gate) * original
        
        # Project and normalize
        output = self.output_proj(fused)
        output = self.norm(original + output)  # Residual
        
        return output


class HierarchicalCrossAttention(nn.Module):
    """
    Hierarchical cross-attention structure for multi-timeframe fusion.
    
    The hierarchy is designed to match how traders analyze multiple timeframes:
    
    1. Local Connections (Adjacent timeframes):
       - tick ↔ 1m: Immediate price action context
       - 1m ↔ 5m: Short-term momentum
       - 5m ↔ 15m: Intraday trend
       - etc.
    
    2. Skip Connections (Every other timeframe):
       - tick ↔ 5m: Microstructure to pattern
       - 1m ↔ 15m: Minute to quarter-hour
       - etc.
    
    3. Anchor Connections (All to major timeframes):
       - All → 1d: Daily trend alignment
       - All → weekly: Macro context
    
    This reduces complexity from O(n²) pairwise attention to O(n) while
    capturing the essential cross-scale relationships.
    """
    
    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        
        self.config = config
        self.timeframes = config.timeframes
        n_tf = len(config.timeframes)
        
        # Define attention connections
        self.connections = self._build_connections()
        
        # Create cross-attention blocks for each connection
        self.cross_attention_blocks = nn.ModuleDict()
        for src, tgt in self.connections:
            key = f"{src}_to_{tgt}"
            self.cross_attention_blocks[key] = CrossAttentionBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
            )
        
        # Gated fusion for each timeframe
        if config.use_gating:
            self.gates = nn.ModuleDict({
                tf: GatedFusion(config.d_model)
                for tf in config.timeframes
            })
        else:
            self.gates = None
        
        # Final layer norm per timeframe
        self.norms = nn.ModuleDict({
            tf: nn.LayerNorm(config.d_model)
            for tf in config.timeframes
        })
    
    def _build_connections(self) -> List[Tuple[str, str]]:
        """Build list of (source, target) attention connections."""
        connections = []
        tfs = self.config.timeframes
        n = len(tfs)
        
        # Level 1: Adjacent connections (bidirectional)
        for i in range(n - 1):
            connections.append((tfs[i], tfs[i + 1]))
            connections.append((tfs[i + 1], tfs[i]))
        
        # Level 2: Skip-one connections
        for i in range(n - 2):
            connections.append((tfs[i], tfs[i + 2]))
            connections.append((tfs[i + 2], tfs[i]))
        
        # Level 3: Anchor connections (all to daily and weekly)
        anchor_tfs = ['1d', 'weekly']
        for anchor in anchor_tfs:
            if anchor in tfs:
                for tf in tfs:
                    if tf != anchor and (tf, anchor) not in connections:
                        connections.append((tf, anchor))
        
        return list(set(connections))  # Remove duplicates
    
    def forward(
        self,
        timeframe_reps: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Apply hierarchical cross-attention.
        
        Args:
            timeframe_reps: Dict mapping timeframe names to representations
                           Each value has shape (batch, d_model)
        
        Returns:
            Dict of enhanced timeframe representations
        """
        # Collect cross-attention outputs for each timeframe
        cross_outputs = {tf: [] for tf in self.timeframes if tf in timeframe_reps}
        attention_weights = {}
        
        # Apply all cross-attention connections
        for src, tgt in self.connections:
            if src not in timeframe_reps or tgt not in timeframe_reps:
                continue
            
            key = f"{src}_to_{tgt}"
            cross_block = self.cross_attention_blocks[key]
            
            # Source queries target
            cross_out, attn = cross_block(
                query=timeframe_reps[src],
                key_value=timeframe_reps[tgt],
            )
            
            cross_outputs[src].append(cross_out)
            attention_weights[key] = attn
        
        # Aggregate cross-attention outputs for each timeframe
        enhanced = {}
        for tf in timeframe_reps:
            original = timeframe_reps[tf]
            
            if cross_outputs.get(tf):
                # Average all incoming cross-attention
                cross_avg = torch.stack(cross_outputs[tf], dim=0).mean(dim=0)
                
                # Apply gating if enabled
                if self.gates is not None:
                    fused = self.gates[tf](original, cross_avg)
                else:
                    fused = original + cross_avg
            else:
                fused = original
            
            enhanced[tf] = self.norms[tf](fused)
        
        return enhanced


class MultiScaleFusion(nn.Module):
    """
    Final fusion of all timeframe representations into unified output.
    
    After cross-attention enrichment, we need to combine all timeframes
    into a single representation for downstream tasks. This is done via:
    
    1. Learned importance weights per timeframe
    2. Weighted pooling across timeframes
    3. Final projection
    
    The importance weights adapt based on market regime—during trends,
    longer timeframes get higher weights; during ranging, shorter timeframes
    dominate.
    """
    
    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        
        self.config = config
        n_timeframes = len(config.timeframes)
        
        # Learnable base importance per timeframe
        self.base_importance = nn.Parameter(
            torch.ones(n_timeframes) / n_timeframes
        )
        
        # Context-dependent importance adjustment
        self.importance_net = nn.Sequential(
            nn.Linear(config.d_model * n_timeframes, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, n_timeframes),
        )
        
        # Final fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        timeframe_reps: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse all timeframe representations.
        
        Args:
            timeframe_reps: Dict of timeframe representations
            
        Returns:
            fused: Unified representation (batch, d_model)
            importance: Timeframe importance weights (batch, n_timeframes)
        """
        # Stack representations in order
        ordered_reps = []
        for tf in self.config.timeframes:
            if tf in timeframe_reps:
                ordered_reps.append(timeframe_reps[tf])
            else:
                # Zero padding for missing timeframes
                batch_size = list(timeframe_reps.values())[0].shape[0]
                ordered_reps.append(
                    torch.zeros(batch_size, self.config.d_model,
                               device=list(timeframe_reps.values())[0].device)
                )
        
        # Stack: (batch, n_timeframes, d_model)
        stacked = torch.stack(ordered_reps, dim=1)
        batch_size = stacked.shape[0]
        
        # Compute context-dependent importance
        context = stacked.flatten(start_dim=1)  # (batch, n_tf * d_model)
        importance_adj = self.importance_net(context)  # (batch, n_tf)
        
        # Combine with base importance
        importance = F.softmax(
            self.base_importance.unsqueeze(0) + importance_adj, dim=-1
        )
        
        # Weighted combination
        # (batch, n_tf, 1) * (batch, n_tf, d_model) -> sum -> (batch, d_model)
        weighted = (importance.unsqueeze(-1) * stacked).sum(dim=1)
        
        # Final projection
        fused = self.fusion_proj(weighted)
        fused = self.norm(fused)
        
        return fused, importance


class CrossAttentionFusion(nn.Module):
    """
    Complete Cross-Attention Fusion module for HIMARI Layer 2.
    
    This module:
    1. Takes representations from all timeframes
    2. Applies hierarchical cross-attention for information exchange
    3. Fuses into a unified multi-scale representation
    
    The output captures both per-timeframe signals AND their interactions,
    enabling trading decisions that consider the full temporal context.
    """
    
    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        
        self.config = config
        
        # Multiple layers of cross-attention
        self.cross_attention_layers = nn.ModuleList([
            HierarchicalCrossAttention(config)
            for _ in range(config.n_fusion_layers)
        ])
        
        # Final fusion
        self.fusion = MultiScaleFusion(config)
    
    def forward(
        self,
        timeframe_reps: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Full cross-attention fusion pass.
        
        Args:
            timeframe_reps: Dict mapping timeframe names to representations
            
        Returns:
            Dict containing fused representation and per-timeframe enhanced reps
        """
        # Apply cross-attention layers
        current_reps = timeframe_reps
        for layer in self.cross_attention_layers:
            current_reps = layer(current_reps)
        
        # Final fusion
        fused, importance = self.fusion(current_reps)
        
        return {
            'fused': fused,
            'timeframe_importance': importance,
            'enhanced_timeframes': current_reps,
        }


class CrossAttentionFusionForTrading(nn.Module):
    """
    Cross-Attention Fusion wrapper for HIMARI trading use case.
    
    Adds trading-specific outputs and provides interface for integration
    with the Layer 2 decision engine.
    """
    
    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        
        self.cross_fusion = CrossAttentionFusion(config)
        self.config = config
        
        # Trading heads
        self.action_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 3),
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        # Regime detection head (uses multi-scale information)
        self.regime_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 6),  # 6 regime types
        )
    
    def forward(
        self,
        timeframe_reps: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for trading.
        
        Args:
            timeframe_reps: Dict of per-timeframe representations
            
        Returns:
            Dict with action logits, confidence, regime, importance, etc.
        """
        # Cross-attention fusion
        fusion_outputs = self.cross_fusion(timeframe_reps)
        fused = fusion_outputs['fused']
        
        # Generate trading outputs
        action_logits = self.action_head(fused)
        confidence = self.confidence_head(fused).squeeze(-1)
        regime_logits = self.regime_head(fused)
        
        return {
            'action_logits': action_logits,
            'confidence': confidence,
            'regime_logits': regime_logits,
            'representation': fused,
            'timeframe_importance': fusion_outputs['timeframe_importance'],
            'enhanced_timeframes': fusion_outputs['enhanced_timeframes'],
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_cross_attention_fusion_for_himari(
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    timeframes: Optional[List[str]] = None,
) -> CrossAttentionFusionForTrading:
    """
    Create Cross-Attention Fusion configured for HIMARI Layer 2.
    
    Default configuration for 9 timeframes with hierarchical attention
    and gated fusion.
    """
    if timeframes is None:
        timeframes = ['tick', '1m', '5m', '15m', '30m', '1h', '4h', '1d', 'weekly']
    
    config = CrossAttentionConfig(
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.1,
        timeframes=timeframes,
        use_hierarchical=True,
        n_fusion_layers=n_layers,
        use_gating=True,
    )
    
    return CrossAttentionFusionForTrading(config)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """Demonstrate Cross-Attention Fusion for HIMARI."""
    
    # Create model
    model = create_cross_attention_fusion_for_himari(
        d_model=128,
        n_heads=4,
        n_layers=2,
    )
    model.eval()
    
    # Example timeframe representations (would come from TFT, FEDformer, etc.)
    batch_size = 4
    timeframe_reps = {
        'tick': torch.randn(batch_size, 128),
        '1m': torch.randn(batch_size, 128),
        '5m': torch.randn(batch_size, 128),
        '15m': torch.randn(batch_size, 128),
        '30m': torch.randn(batch_size, 128),
        '1h': torch.randn(batch_size, 128),
        '4h': torch.randn(batch_size, 128),
        '1d': torch.randn(batch_size, 128),
        'weekly': torch.randn(batch_size, 128),
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(timeframe_reps)
    
    print("Cross-Attention Fusion Output Shapes:")
    print(f"  Action logits: {outputs['action_logits'].shape}")          # [4, 3]
    print(f"  Confidence: {outputs['confidence'].shape}")                # [4]
    print(f"  Regime logits: {outputs['regime_logits'].shape}")          # [4, 6]
    print(f"  Fused representation: {outputs['representation'].shape}")  # [4, 128]
    print(f"  Timeframe importance: {outputs['timeframe_importance'].shape}")  # [4, 9]
    
    # Interpretation
    importance = outputs['timeframe_importance'][0]
    timeframes = ['tick', '1m', '5m', '15m', '30m', '1h', '4h', '1d', 'weekly']
    print("\nTimeframe Importance (sample 0):")
    for tf, imp in zip(timeframes, importance.tolist()):
        print(f"  {tf:8s}: {imp:.3f}")
    
    # Show which timeframes dominate
    top_idx = torch.topk(importance, k=3).indices
    print(f"\nTop 3 timeframes: {[timeframes[i] for i in top_idx.tolist()]}")
    
    # Regime prediction
    regime_names = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE', 'BREAKOUT', 'REVERSAL']
    regimes = torch.argmax(outputs['regime_logits'], dim=-1)
    print(f"\nPredicted regimes: {[regime_names[r] for r in regimes.tolist()]}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    example_usage()
