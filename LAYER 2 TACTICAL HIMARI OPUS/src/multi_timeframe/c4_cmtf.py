"""
HIMARI Layer 2 - C4: CMTF (Coupled Matrix-Tensor Factorization)
================================================================

Let me explain a fundamental challenge in multi-source financial analysis.
HIMARI ingests data from radically different sources: OHLCV price data
(structured time series), sentiment scores (text-derived scalars), on-chain
metrics (blockchain transaction graphs), and order book microstructure
(high-frequency 2D data). These sources have different:

- Dimensionalities: Price is 1D, order books are 2D, transaction graphs are irregular
- Update frequencies: Prices update every 5 minutes, on-chain hourly, sentiment daily
- Noise characteristics: Price noise is market microstructure; sentiment noise is NLP errors
- Information content: Each source captures different market dynamics

The naive approach—concatenating all features into one vector—loses the
structural relationships within and across sources. Think of it like
averaging the R, G, and B channels of an image: you get a gray blob
that loses the spatial structure that made the image informative.

The CMTF Solution:
------------------
Coupled Matrix-Tensor Factorization provides a principled framework for
fusing heterogeneous data while preserving source-specific structure.
The core insight is that different data sources often share underlying
latent factors—a "fear" factor might manifest as:
- Falling prices (in OHLCV)
- Negative sentiment scores (in social data)
- Increased withdrawals (in on-chain data)
- Widening spreads (in order books)

CMTF finds these shared latent factors by decomposing each source into
factor matrices, with coupling constraints ensuring consistency across sources.

Mathematically:
- Price tensor X₁ ≈ U × S₁ × V₁ᵀ  (time × latent × features)
- Sentiment matrix X₂ ≈ U × S₂ × V₂ᵀ  (time × latent × sources)
- On-chain tensor X₃ ≈ U × S₃ × V₃ᵀ  (time × latent × metrics)

The key is that U (time dimension factors) is SHARED across all sources,
forcing the model to discover latent factors that explain all data sources
simultaneously. This acts as a strong regularizer and enables:
1. Transfer learning across modalities
2. Robust predictions when some sources are noisy
3. Interpretable factor structures

For HIMARI, CMTF serves as the multi-modal fusion layer that integrates
outputs from TFT (price), FEDformer (long-range), and ViT-LOB (microstructure)
before the decision engine.

Performance Targets:
- Latency: <10ms for factor projection
- Memory: Factor matrices cached for O(1) projection
- Quality: 15-25% variance explained by shared factors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class CMTFConfig:
    """Configuration for Coupled Matrix-Tensor Factorization.
    
    Attributes:
        n_latent: Number of shared latent factors
        n_source_specific: Number of source-specific factors per source
        n_sources: Number of data sources to fuse
        source_dims: Dictionary mapping source names to their dimensions
        regularization: Weight for factor regularization (prevents overfitting)
        coupling_strength: How strongly to enforce factor sharing (0-1)
        n_iterations: Number of ALS iterations for decomposition
        use_neural_coupling: Use neural networks for factor coupling (vs linear)
    """
    n_latent: int = 16           # Shared latent factors
    n_source_specific: int = 8    # Per-source additional factors
    n_sources: int = 4            # Price, sentiment, on-chain, orderbook
    source_dims: Dict[str, int] = field(default_factory=lambda: {
        'price': 64,      # TFT representation dimension
        'sentiment': 32,  # Sentiment feature dimension
        'onchain': 24,    # On-chain metric dimension
        'orderbook': 64,  # ViT-LOB representation dimension
    })
    regularization: float = 0.01
    coupling_strength: float = 0.7
    n_iterations: int = 10
    use_neural_coupling: bool = True


class FactorEncoder(nn.Module):
    """
    Neural network that projects a data source into latent factor space.
    
    Unlike classical tensor factorization which uses linear decomposition,
    this encoder applies nonlinear transformations. This is more expressive
    for capturing complex relationships in financial data.
    
    Architecture:
        Input (source_dim)
          ↓
        Linear → LayerNorm → GELU
          ↓
        Linear → LayerNorm → GELU
          ↓
        Shared factors (n_latent) + Source-specific factors (n_source_specific)
    """
    
    def __init__(
        self,
        input_dim: int,
        n_latent: int,
        n_source_specific: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or (input_dim + n_latent) // 2
        total_factors = n_latent + n_source_specific
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, total_factors),
        )
        
        self.n_latent = n_latent
        self.n_source_specific = n_source_specific
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent factors.
        
        Args:
            x: Input tensor (..., input_dim)
            
        Returns:
            shared_factors: Factors shared across sources (..., n_latent)
            specific_factors: Source-specific factors (..., n_source_specific)
        """
        factors = self.encoder(x)
        shared = factors[..., :self.n_latent]
        specific = factors[..., self.n_latent:]
        return shared, specific


class FactorDecoder(nn.Module):
    """
    Neural network that reconstructs source data from latent factors.
    
    Used for:
    1. Reconstruction loss (regularization during training)
    2. Generating cross-modal predictions
    3. Imputing missing data from one source using others
    """
    
    def __init__(
        self,
        output_dim: int,
        n_latent: int,
        n_source_specific: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        total_factors = n_latent + n_source_specific
        hidden_dim = hidden_dim or (output_dim + total_factors) // 2
        
        self.decoder = nn.Sequential(
            nn.Linear(total_factors, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        shared_factors: torch.Tensor,
        specific_factors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode factors back to source space.
        
        Args:
            shared_factors: Shared factors (..., n_latent)
            specific_factors: Source-specific factors (..., n_source_specific)
            
        Returns:
            Reconstructed data (..., output_dim)
        """
        factors = torch.cat([shared_factors, specific_factors], dim=-1)
        return self.decoder(factors)


class FactorCoupler(nn.Module):
    """
    Enforces consistency of shared factors across sources.
    
    When different sources encode to shared factor space, their representations
    should be similar (that's what makes them "shared"). The coupler learns
    a canonical form of the shared factors and projects each source's version
    toward this canonical form.
    
    Think of it as a "consensus mechanism" for latent factors.
    """
    
    def __init__(
        self,
        n_latent: int,
        n_sources: int,
        coupling_strength: float = 0.7,
    ):
        super().__init__()
        
        self.n_latent = n_latent
        self.n_sources = n_sources
        self.coupling_strength = coupling_strength
        
        # Learnable canonical factors (the "prototype")
        self.canonical = nn.Parameter(torch.zeros(n_latent))
        
        # Per-source projection to canonical space
        self.source_projections = nn.ModuleList([
            nn.Linear(n_latent, n_latent) for _ in range(n_sources)
        ])
        
        # Aggregation weights (learned importance per source)
        self.aggregation_weights = nn.Parameter(torch.ones(n_sources) / n_sources)
    
    def forward(
        self,
        source_factors: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Couple factors from multiple sources.
        
        Args:
            source_factors: List of [(..., n_latent)] tensors, one per source
            
        Returns:
            consensus: Aggregated shared factors (..., n_latent)
            aligned: List of source factors projected toward consensus
        """
        # Normalize aggregation weights
        weights = F.softmax(self.aggregation_weights, dim=0)
        
        # Project each source to canonical space
        projected = []
        for i, (factors, proj) in enumerate(zip(source_factors, self.source_projections)):
            projected.append(proj(factors))
        
        # Compute weighted consensus
        # Stack: (n_sources, batch, n_latent)
        stacked = torch.stack(projected, dim=0)
        weights_expanded = weights.view(-1, 1, 1)  # (n_sources, 1, 1)
        consensus = (stacked * weights_expanded).sum(dim=0)
        
        # Align each source toward consensus
        aligned = []
        for proj_factors in projected:
            # Blend: coupling_strength * consensus + (1 - coupling_strength) * source
            blended = (
                self.coupling_strength * consensus +
                (1 - self.coupling_strength) * proj_factors
            )
            aligned.append(blended)
        
        return consensus, aligned


class CMTF(nn.Module):
    """
    Complete Coupled Matrix-Tensor Factorization module.
    
    This is the main class that:
    1. Encodes each source to shared + specific factors
    2. Couples shared factors across sources
    3. Enables reconstruction and cross-modal prediction
    4. Outputs unified representation for downstream tasks
    
    The module is differentiable end-to-end and can be trained with
    backpropagation (unlike classical ALS-based tensor factorization).
    """
    
    def __init__(self, config: CMTFConfig):
        super().__init__()
        
        self.config = config
        self.source_names = list(config.source_dims.keys())
        
        # Per-source encoders
        self.encoders = nn.ModuleDict({
            name: FactorEncoder(
                input_dim=dim,
                n_latent=config.n_latent,
                n_source_specific=config.n_source_specific,
            )
            for name, dim in config.source_dims.items()
        })
        
        # Per-source decoders
        self.decoders = nn.ModuleDict({
            name: FactorDecoder(
                output_dim=dim,
                n_latent=config.n_latent,
                n_source_specific=config.n_source_specific,
            )
            for name, dim in config.source_dims.items()
        })
        
        # Factor coupler for shared factors
        if config.use_neural_coupling:
            self.coupler = FactorCoupler(
                n_latent=config.n_latent,
                n_sources=len(config.source_dims),
                coupling_strength=config.coupling_strength,
            )
        else:
            self.coupler = None
        
        # Final fusion layer
        total_factor_dim = config.n_latent + config.n_source_specific * len(config.source_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_factor_dim, config.n_latent * 2),
            nn.LayerNorm(config.n_latent * 2),
            nn.GELU(),
            nn.Linear(config.n_latent * 2, config.n_latent),
        )
    
    def encode_sources(
        self,
        sources: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Encode all available sources to latent factors.
        
        Args:
            sources: Dict mapping source names to tensors
            
        Returns:
            shared_factors: Dict of shared factors per source
            specific_factors: Dict of source-specific factors
        """
        shared = {}
        specific = {}
        
        for name, data in sources.items():
            if name in self.encoders:
                s, sp = self.encoders[name](data)
                shared[name] = s
                specific[name] = sp
        
        return shared, specific
    
    def couple_factors(
        self,
        shared_factors: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply factor coupling to align shared factors.
        
        Args:
            shared_factors: Dict of shared factors per source
            
        Returns:
            consensus: Aggregated consensus factors
            aligned: Dict of aligned factors per source
        """
        if self.coupler is None:
            # Simple averaging if no neural coupler
            factor_list = list(shared_factors.values())
            consensus = torch.stack(factor_list, dim=0).mean(dim=0)
            aligned = {name: consensus for name in shared_factors}
            return consensus, aligned
        
        # Order factors by source name for consistency
        factor_list = [shared_factors[name] for name in self.source_names if name in shared_factors]
        consensus, aligned_list = self.coupler(factor_list)
        
        aligned = {}
        idx = 0
        for name in self.source_names:
            if name in shared_factors:
                aligned[name] = aligned_list[idx]
                idx += 1
        
        return consensus, aligned
    
    def reconstruct(
        self,
        shared_factors: torch.Tensor,
        specific_factors: Dict[str, torch.Tensor],
        target_sources: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct source data from factors.
        
        Args:
            shared_factors: Consensus shared factors
            specific_factors: Dict of source-specific factors
            target_sources: Which sources to reconstruct (None = all)
            
        Returns:
            Dict of reconstructed data per source
        """
        if target_sources is None:
            target_sources = list(specific_factors.keys())
        
        reconstructed = {}
        for name in target_sources:
            if name in self.decoders and name in specific_factors:
                recon = self.decoders[name](shared_factors, specific_factors[name])
                reconstructed[name] = recon
        
        return reconstructed
    
    def forward(
        self,
        sources: Dict[str, torch.Tensor],
        return_reconstruction: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through CMTF.
        
        Args:
            sources: Dict mapping source names to input tensors
            return_reconstruction: Whether to return reconstructed sources
            
        Returns:
            Dict containing:
                - fused: Final fused representation
                - consensus: Consensus shared factors
                - reconstructed: Reconstructed sources (if requested)
                - coupling_loss: Factor alignment loss (for training)
        """
        # Encode all sources
        shared, specific = self.encode_sources(sources)
        
        # Couple shared factors
        consensus, aligned = self.couple_factors(shared)
        
        # Compute coupling loss (MSE between original and aligned factors)
        coupling_loss = 0.0
        for name in shared:
            if name in aligned:
                coupling_loss += F.mse_loss(shared[name], aligned[name])
        coupling_loss /= len(shared)
        
        # Concatenate all factors for fusion
        all_factors = [consensus]
        for name in self.source_names:
            if name in specific:
                all_factors.append(specific[name])
        
        concatenated = torch.cat(all_factors, dim=-1)
        fused = self.fusion(concatenated)
        
        outputs = {
            'fused': fused,
            'consensus': consensus,
            'aligned_factors': aligned,
            'specific_factors': specific,
            'coupling_loss': coupling_loss,
        }
        
        if return_reconstruction:
            outputs['reconstructed'] = self.reconstruct(consensus, specific)
        
        return outputs
    
    def compute_loss(
        self,
        sources: Dict[str, torch.Tensor],
        reconstruction_weight: float = 1.0,
        coupling_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total training loss.
        
        The loss consists of:
        1. Reconstruction loss: Each source should be reconstructable from its factors
        2. Coupling loss: Shared factors should be aligned across sources
        3. Regularization: Prevent factor magnitudes from exploding
        
        Args:
            sources: Dict of source data
            reconstruction_weight: Weight for reconstruction loss
            coupling_weight: Weight for coupling loss
            
        Returns:
            total_loss: Combined loss value
            loss_components: Dict of individual loss components
        """
        outputs = self.forward(sources, return_reconstruction=True)
        
        # Reconstruction loss
        recon_loss = 0.0
        n_sources = 0
        for name, original in sources.items():
            if name in outputs['reconstructed']:
                recon_loss += F.mse_loss(outputs['reconstructed'][name], original)
                n_sources += 1
        recon_loss /= max(n_sources, 1)
        
        # Coupling loss (already computed)
        coupling_loss = outputs['coupling_loss']
        
        # Regularization loss (L2 on factors)
        reg_loss = 0.0
        reg_loss += outputs['consensus'].pow(2).mean()
        for factors in outputs['specific_factors'].values():
            reg_loss += factors.pow(2).mean()
        reg_loss *= self.config.regularization
        
        # Total loss
        total_loss = (
            reconstruction_weight * recon_loss +
            coupling_weight * coupling_loss +
            reg_loss
        )
        
        return total_loss, {
            'reconstruction': recon_loss,
            'coupling': coupling_loss,
            'regularization': reg_loss,
            'total': total_loss,
        }


class CMTFForTrading(nn.Module):
    """
    CMTF wrapper for HIMARI trading use case.
    
    This module:
    1. Takes representations from multiple upstream models (TFT, FEDformer, ViT-LOB)
    2. Fuses them via CMTF into a unified representation
    3. Generates trading signals with proper uncertainty quantification
    
    The key insight is that CMTF discovers shared factors like "risk appetite"
    or "liquidity conditions" that manifest across all data sources. These
    factors are more stable and predictive than raw features.
    """
    
    def __init__(self, config: CMTFConfig):
        super().__init__()
        
        self.cmtf = CMTF(config)
        self.config = config
        
        # Trading heads operating on fused representation
        self.action_head = nn.Sequential(
            nn.Linear(config.n_latent, config.n_latent // 2),
            nn.GELU(),
            nn.Linear(config.n_latent // 2, 3),  # BUY, HOLD, SELL
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(config.n_latent, config.n_latent // 2),
            nn.GELU(),
            nn.Linear(config.n_latent // 2, 1),
            nn.Sigmoid(),
        )
        
        # Factor importance for interpretability
        self.factor_importance = nn.Linear(config.n_latent, config.n_latent)
    
    def forward(
        self,
        sources: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for trading.
        
        Args:
            sources: Dict with keys like 'price', 'sentiment', 'onchain', 'orderbook'
                    Each value is a tensor of shape (batch, source_dim)
        
        Returns:
            Dict containing action logits, confidence, factor importance, etc.
        """
        # CMTF forward
        cmtf_outputs = self.cmtf(sources, return_reconstruction=False)
        
        fused = cmtf_outputs['fused']
        consensus = cmtf_outputs['consensus']
        
        # Generate trading signals
        action_logits = self.action_head(fused)
        confidence = self.confidence_head(fused).squeeze(-1)
        
        # Factor importance (which latent factors drive the decision)
        importance_weights = F.softmax(self.factor_importance(consensus), dim=-1)
        
        return {
            'action_logits': action_logits,
            'confidence': confidence,
            'representation': fused,
            'consensus_factors': consensus,
            'factor_importance': importance_weights,
            'aligned_factors': cmtf_outputs['aligned_factors'],
            'specific_factors': cmtf_outputs['specific_factors'],
        }
    
    def get_cross_modal_prediction(
        self,
        available_sources: Dict[str, torch.Tensor],
        target_source: str,
    ) -> torch.Tensor:
        """
        Predict missing source data from available sources.
        
        This is useful when one data source is delayed or missing.
        For example, predict what on-chain metrics "should" be given
        the current price action and sentiment.
        
        Args:
            available_sources: Dict of available source data
            target_source: Name of source to predict
            
        Returns:
            Predicted data for target source
        """
        # Encode available sources
        shared, specific = self.cmtf.encode_sources(available_sources)
        
        # Get consensus from available sources
        consensus, _ = self.cmtf.couple_factors(shared)
        
        # Create zero-initialized specific factors for target
        batch_size = consensus.shape[0]
        target_specific = torch.zeros(
            batch_size, self.config.n_source_specific,
            device=consensus.device,
        )
        
        # Decode to target source
        predicted = self.cmtf.decoders[target_source](consensus, target_specific)
        
        return predicted


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_cmtf_for_himari(
    price_dim: int = 128,      # From TFT
    sentiment_dim: int = 32,
    onchain_dim: int = 24,
    orderbook_dim: int = 128,  # From ViT-LOB
    n_latent: int = 32,
) -> CMTFForTrading:
    """
    Create CMTF configured for HIMARI Layer 2.
    
    Default configuration assumes:
    - Price representation from TFT/FEDformer (128-dim)
    - Sentiment features from sentiment model (32-dim)
    - On-chain metrics from on-chain processor (24-dim)
    - Order book representation from ViT-LOB (128-dim)
    """
    config = CMTFConfig(
        n_latent=n_latent,
        n_source_specific=n_latent // 2,
        source_dims={
            'price': price_dim,
            'sentiment': sentiment_dim,
            'onchain': onchain_dim,
            'orderbook': orderbook_dim,
        },
        regularization=0.01,
        coupling_strength=0.7,
        use_neural_coupling=True,
    )
    
    return CMTFForTrading(config)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """Demonstrate CMTF for HIMARI."""
    
    # Create model
    model = create_cmtf_for_himari(
        price_dim=128,
        sentiment_dim=32,
        onchain_dim=24,
        orderbook_dim=128,
        n_latent=32,
    )
    model.eval()
    
    # Example source data (would come from TFT, sentiment model, etc.)
    batch_size = 4
    sources = {
        'price': torch.randn(batch_size, 128),      # TFT output
        'sentiment': torch.randn(batch_size, 32),   # Sentiment scores
        'onchain': torch.randn(batch_size, 24),     # On-chain metrics
        'orderbook': torch.randn(batch_size, 128),  # ViT-LOB output
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(sources)
    
    print("CMTF for Trading Output Shapes:")
    print(f"  Action logits: {outputs['action_logits'].shape}")        # [4, 3]
    print(f"  Confidence: {outputs['confidence'].shape}")              # [4]
    print(f"  Fused representation: {outputs['representation'].shape}")# [4, 32]
    print(f"  Consensus factors: {outputs['consensus_factors'].shape}")# [4, 32]
    print(f"  Factor importance: {outputs['factor_importance'].shape}")# [4, 32]
    
    # Interpretation
    actions = torch.argmax(outputs['action_logits'], dim=-1)
    action_names = ['SELL', 'HOLD', 'BUY']
    print(f"\nPredicted actions: {[action_names[a] for a in actions.tolist()]}")
    print(f"Confidence scores: {[f'{c:.3f}' for c in outputs['confidence'].tolist()]}")
    
    # Show top factors
    importance = outputs['factor_importance'][0]
    top_factors = torch.topk(importance, k=5)
    print(f"\nTop 5 factor indices: {top_factors.indices.tolist()}")
    print(f"Top 5 factor weights: {[f'{w:.3f}' for w in top_factors.values.tolist()]}")
    
    # Cross-modal prediction example
    print("\n--- Cross-Modal Prediction ---")
    # Predict on-chain from price + sentiment only
    partial_sources = {
        'price': sources['price'],
        'sentiment': sources['sentiment'],
    }
    predicted_onchain = model.get_cross_modal_prediction(partial_sources, 'onchain')
    print(f"Predicted on-chain shape: {predicted_onchain.shape}")  # [4, 24]
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    example_usage()
