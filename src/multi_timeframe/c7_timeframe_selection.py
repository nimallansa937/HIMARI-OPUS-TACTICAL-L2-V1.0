"""
HIMARI Layer 2 - C7: Timeframe Selection
=========================================

Not all timeframes are equally important at all times. During a trending market,
the 4-hour and daily timeframes dominate—short-term noise should be filtered.
During a ranging market, the 1-minute and 5-minute timeframes become primary—
catching mean-reversion opportunities before they disappear. During high
volatility, microstructure (tick, 1m) matters most for execution timing.

The static approach of equal-weighting all timeframes misses this dynamic.
The naive approach of learning fixed weights overfits to historical regime
distributions. What we need is CONDITIONAL weighting: given the current
market state, which timeframes should we trust?

Timeframe Selection Mechanism:
------------------------------
This module learns to predict timeframe importance weights conditioned on:
1. Current market regime (trend, range, volatile, quiet)
2. Recent performance of each timeframe's predictions
3. Cross-timeframe agreement/disagreement signals
4. Volatility structure across scales

The selection is "soft"—we output continuous weights that sum to 1, not
hard binary selections. This allows gradient flow and smooth transitions
between regime-appropriate weightings.

Scaleformer Integration:
------------------------
Scaleformer (2023) introduced iterative multi-scale refinement for time series.
Instead of processing scales independently then fusing, Scaleformer iteratively
refines predictions by alternating between scales:

    Initial: Predict from coarsest scale (daily)
    Refine 1: Update prediction using 4h information
    Refine 2: Update prediction using 1h information
    Refine 3: Update prediction using 5m information
    Final: Fine-tune with 1m microstructure

Each refinement step learns how much to trust the new scale vs. maintain
the previous estimate. This is implemented via learnable residual gates.

For HIMARI, we adapt this to timeframe selection:
- Start with a prior belief about timeframe importance (uniform or regime-based)
- Iteratively refine importance based on each timeframe's signal quality
- Output refined weights for final fusion

Performance Targets:
- Latency: <5ms for weight computation
- Adaptivity: Weights should shift by >20% between trend/range regimes
- Stability: No oscillation in weights during stable periods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    VOLATILE = 3
    QUIET = 4
    TRANSITIONING = 5


@dataclass
class TimeframeSelectionConfig:
    """Configuration for Timeframe Selection module.
    
    Attributes:
        d_model: Model dimension
        n_timeframes: Number of timeframes to weight
        n_regimes: Number of market regime types
        n_refinement_steps: Number of Scaleformer-style refinement iterations
        dropout: Dropout rate
        temperature: Softmax temperature for weight sharpening
        use_gumbel: Whether to use Gumbel-softmax for differentiable selection
        min_weight: Minimum weight per timeframe (prevents complete suppression)
    """
    d_model: int = 128
    n_timeframes: int = 5
    n_regimes: int = 6
    n_refinement_steps: int = 3
    dropout: float = 0.1
    temperature: float = 1.0
    use_gumbel: bool = False
    min_weight: float = 0.05  # Each timeframe gets at least 5%


class RegimeDetector(nn.Module):
    """
    Detect current market regime from multi-timeframe features.
    
    The regime detector takes features from all timeframes and outputs
    a soft distribution over regime types. This isn't a hard classification—
    markets often exhibit characteristics of multiple regimes simultaneously.
    
    Features considered:
    - Trend strength (ADX-like measure across scales)
    - Volatility ratio (short-term vs long-term)
    - Momentum agreement across timeframes
    - Volume profile changes
    """
    
    def __init__(
        self,
        d_model: int,
        n_timeframes: int,
        n_regimes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_regimes = n_regimes
        
        # Timeframe feature aggregation
        self.timeframe_encoder = nn.Sequential(
            nn.Linear(d_model * n_timeframes, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        
        # Regime classification
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_regimes),
        )
        
        # Learnable regime embeddings (for downstream conditioning)
        self.regime_embeddings = nn.Embedding(n_regimes, d_model)
    
    def forward(
        self,
        timeframe_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect market regime from timeframe features.
        
        Args:
            timeframe_features: List of (batch, d_model) tensors, one per timeframe
            
        Returns:
            regime_probs: Soft distribution over regimes (batch, n_regimes)
            regime_embedding: Weighted regime embedding (batch, d_model)
            aggregated: Aggregated timeframe features (batch, d_model)
        """
        # Concatenate timeframe features
        concat = torch.cat(timeframe_features, dim=-1)  # (batch, n_tf * d_model)
        
        # Aggregate
        aggregated = self.timeframe_encoder(concat)  # (batch, d_model)
        
        # Classify regime
        regime_logits = self.regime_head(aggregated)  # (batch, n_regimes)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # Compute weighted regime embedding
        # (batch, n_regimes) @ (n_regimes, d_model) → (batch, d_model)
        regime_embedding = torch.matmul(
            regime_probs,
            self.regime_embeddings.weight
        )
        
        return regime_probs, regime_embedding, aggregated


class TimeframeQualityEstimator(nn.Module):
    """
    Estimate signal quality for each timeframe.
    
    Signal quality is assessed based on:
    1. Internal consistency (do features agree with each other?)
    2. Predictability (is the pattern clear or noisy?)
    3. Recent accuracy (if we have labels, how well did predictions perform?)
    
    This is used to upweight timeframes with clean signals and downweight
    timeframes with noisy or contradictory information.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # Quality estimation network
        self.quality_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),  # Quality score in [0, 1]
        )
        
        # Uncertainty estimation (epistemic + aleatoric)
        self.uncertainty_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),  # mean, log_var
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate quality and uncertainty for a timeframe.
        
        Args:
            x: Timeframe representation (batch, d_model)
            
        Returns:
            quality: Quality score in [0, 1] (batch, 1)
            uncertainty: Uncertainty estimate (batch, 1)
        """
        quality = self.quality_net(x)
        
        uncertainty_params = self.uncertainty_net(x)
        # Use variance as uncertainty measure
        uncertainty = F.softplus(uncertainty_params[:, 1:2])
        
        return quality, uncertainty


class ScaleformerRefinement(nn.Module):
    """
    Scaleformer-style iterative refinement of timeframe weights.
    
    The core idea: start with a rough estimate of timeframe importance,
    then iteratively refine by incorporating information from each scale.
    
    Each refinement step:
    1. Takes current weight estimate and timeframe representation
    2. Computes a residual update
    3. Gates the update based on confidence
    4. Applies residual to get refined weights
    
    This allows the model to "change its mind" about timeframe importance
    as it processes more information, without committing too early.
    """
    
    def __init__(
        self,
        d_model: int,
        n_timeframes: int,
        n_steps: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_steps = n_steps
        self.n_timeframes = n_timeframes
        
        # Per-step refinement networks
        self.refinement_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model + n_timeframes, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_timeframes),
            )
            for _ in range(n_steps)
        ])
        
        # Per-step gating (controls how much to update)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model + n_timeframes, n_timeframes),
                nn.Sigmoid(),
            )
            for _ in range(n_steps)
        ])
        
        # Step embeddings (inform network which refinement step we're on)
        self.step_embeddings = nn.Embedding(n_steps, d_model)
    
    def forward(
        self,
        initial_weights: torch.Tensor,
        timeframe_features: List[torch.Tensor],
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Iteratively refine timeframe weights.
        
        Args:
            initial_weights: Starting weight distribution (batch, n_timeframes)
            timeframe_features: List of (batch, d_model) per timeframe
            context: Conditioning context, e.g., regime embedding (batch, d_model)
            
        Returns:
            refined_weights: Final refined weights (batch, n_timeframes)
            weight_history: List of weights at each step for visualization
        """
        weights = initial_weights
        weight_history = [weights.clone()]
        
        for step in range(self.n_steps):
            # Get step embedding
            step_embed = self.step_embeddings.weight[step]  # (d_model,)
            
            # Select timeframe feature for this step
            # Process from coarse to fine (reverse order)
            tf_idx = min(step, len(timeframe_features) - 1)
            tf_feature = timeframe_features[-(tf_idx + 1)]  # Start from coarsest
            
            # Combine context, step, and current weights
            combined = torch.cat([context + step_embed, weights], dim=-1)
            
            # Compute residual update
            residual = self.refinement_nets[step](combined)
            
            # Compute gate
            gate = self.gates[step](combined)
            
            # Apply gated residual
            weights = weights + gate * residual
            
            # Re-normalize to valid distribution
            weights = F.softmax(weights, dim=-1)
            
            weight_history.append(weights.clone())
        
        return weights, weight_history


class GumbelSoftmaxSelector(nn.Module):
    """
    Differentiable discrete selection using Gumbel-Softmax.
    
    Sometimes we want "hard" selection (pick one timeframe) rather than
    soft weighting. Gumbel-Softmax provides a differentiable approximation
    to categorical sampling:
    
    During training: Soft weights (gradients flow)
    During inference: Hard one-hot selection (discrete decision)
    
    The temperature parameter controls the trade-off:
    - High temperature → Soft, uniform-ish weights
    - Low temperature → Sharp, nearly one-hot weights
    """
    
    def __init__(self, temperature: float = 1.0, hard: bool = False):
        super().__init__()
        self.temperature = temperature
        self.hard = hard
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Gumbel-Softmax to logits.
        
        Args:
            logits: Unnormalized scores (batch, n_options)
            
        Returns:
            weights: Soft or hard selection weights (batch, n_options)
        """
        if self.training:
            # Sample Gumbel noise
            gumbels = -torch.empty_like(logits).exponential_().log()
            gumbels = (logits + gumbels) / self.temperature
            
            weights = F.softmax(gumbels, dim=-1)
            
            if self.hard:
                # Straight-through estimator
                hard_weights = F.one_hot(
                    weights.argmax(dim=-1),
                    num_classes=logits.shape[-1]
                ).float()
                weights = hard_weights - weights.detach() + weights
            
            return weights
        else:
            # During inference, use regular softmax or argmax
            if self.hard:
                return F.one_hot(
                    logits.argmax(dim=-1),
                    num_classes=logits.shape[-1]
                ).float()
            return F.softmax(logits / self.temperature, dim=-1)


class TimeframeSelector(nn.Module):
    """
    Complete timeframe selection module for HIMARI Layer 2.
    
    Combines:
    1. Regime detection for conditional weighting
    2. Quality estimation per timeframe
    3. Scaleformer-style iterative refinement
    4. Optional Gumbel-softmax for hard selection
    """
    
    def __init__(self, config: TimeframeSelectionConfig):
        super().__init__()
        
        self.config = config
        
        # Regime detector
        self.regime_detector = RegimeDetector(
            config.d_model,
            config.n_timeframes,
            config.n_regimes,
            config.dropout,
        )
        
        # Quality estimators (one per timeframe)
        self.quality_estimators = nn.ModuleList([
            TimeframeQualityEstimator(config.d_model, config.dropout)
            for _ in range(config.n_timeframes)
        ])
        
        # Scaleformer refinement
        self.refinement = ScaleformerRefinement(
            config.d_model,
            config.n_timeframes,
            config.n_refinement_steps,
            config.dropout,
        )
        
        # Regime-conditional prior weights
        self.regime_priors = nn.Parameter(
            torch.ones(config.n_regimes, config.n_timeframes) / config.n_timeframes
        )
        
        # Optional Gumbel selector
        if config.use_gumbel:
            self.gumbel = GumbelSoftmaxSelector(config.temperature, hard=True)
        else:
            self.gumbel = None
        
        # Final weight adjustment
        self.final_adjustment = nn.Sequential(
            nn.Linear(config.n_timeframes * 2, config.n_timeframes),
            nn.Softmax(dim=-1),
        )
    
    def forward(
        self,
        timeframe_representations: List[torch.Tensor],
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute timeframe selection weights.
        
        Args:
            timeframe_representations: List of (batch, d_model) tensors
            return_diagnostics: Whether to return intermediate values
            
        Returns:
            Dict with weights, regime info, quality scores, etc.
        """
        batch_size = timeframe_representations[0].shape[0]
        
        # Detect regime
        regime_probs, regime_embed, _ = self.regime_detector(
            timeframe_representations
        )
        
        # Get regime-conditional prior weights
        # (batch, n_regimes) @ (n_regimes, n_timeframes) → (batch, n_timeframes)
        prior_weights = torch.matmul(regime_probs, self.regime_priors)
        
        # Estimate quality per timeframe
        qualities = []
        uncertainties = []
        for i, (repr, estimator) in enumerate(
            zip(timeframe_representations, self.quality_estimators)
        ):
            q, u = estimator(repr)
            qualities.append(q)
            uncertainties.append(u)
        
        quality_scores = torch.cat(qualities, dim=-1)  # (batch, n_timeframes)
        uncertainty_scores = torch.cat(uncertainties, dim=-1)
        
        # Incorporate quality into prior
        quality_adjusted = prior_weights * quality_scores
        quality_adjusted = quality_adjusted / (quality_adjusted.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Scaleformer refinement
        refined_weights, weight_history = self.refinement(
            quality_adjusted,
            timeframe_representations,
            regime_embed,
        )
        
        # Optional Gumbel selection
        if self.gumbel is not None:
            # Convert weights to logits for Gumbel
            logits = torch.log(refined_weights + 1e-8)
            final_weights = self.gumbel(logits)
        else:
            final_weights = refined_weights
        
        # Ensure minimum weight constraint
        final_weights = final_weights.clamp(min=self.config.min_weight)
        final_weights = final_weights / final_weights.sum(dim=-1, keepdim=True)
        
        outputs = {
            'weights': final_weights,
            'regime_probs': regime_probs,
            'regime_embedding': regime_embed,
        }
        
        if return_diagnostics:
            outputs.update({
                'quality_scores': quality_scores,
                'uncertainty_scores': uncertainty_scores,
                'prior_weights': prior_weights,
                'weight_history': weight_history,
            })
        
        return outputs
    
    def get_regime_priors(self) -> torch.Tensor:
        """Return learned regime-conditional priors for inspection."""
        return F.softmax(self.regime_priors, dim=-1)


class TimeframeSelectorForTrading(nn.Module):
    """
    Timeframe selector wrapper with trading-specific outputs.
    
    Applies selection weights to timeframe representations and outputs
    a weighted combination for downstream trading decisions.
    """
    
    def __init__(self, config: TimeframeSelectionConfig):
        super().__init__()
        
        self.selector = TimeframeSelector(config)
        self.config = config
        
        # Weighted fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        
        # Trading heads
        self.action_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 3),
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        timeframe_representations: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Select timeframes and produce trading outputs.
        
        Args:
            timeframe_representations: List of (batch, d_model) per timeframe
            
        Returns:
            Dict with weighted representation, action logits, etc.
        """
        # Get selection weights
        selection = self.selector(timeframe_representations, return_diagnostics=True)
        weights = selection['weights']  # (batch, n_timeframes)
        
        # Stack representations
        stacked = torch.stack(timeframe_representations, dim=1)  # (batch, n_tf, d_model)
        
        # Apply weights
        weights_expanded = weights.unsqueeze(-1)  # (batch, n_tf, 1)
        weighted = (stacked * weights_expanded).sum(dim=1)  # (batch, d_model)
        
        # Fuse
        fused = self.fusion(weighted)
        
        # Trading outputs
        action_logits = self.action_head(fused)
        confidence = self.confidence_head(fused).squeeze(-1)
        
        return {
            'representation': fused,
            'action_logits': action_logits,
            'confidence': confidence,
            'timeframe_weights': weights,
            'regime_probs': selection['regime_probs'],
            'quality_scores': selection['quality_scores'],
            'weight_history': selection['weight_history'],
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_timeframe_selector(
    d_model: int = 128,
    n_timeframes: int = 5,
    n_refinement_steps: int = 3,
    use_gumbel: bool = False,
) -> TimeframeSelectorForTrading:
    """
    Create timeframe selector for HIMARI Layer 2.
    """
    config = TimeframeSelectionConfig(
        d_model=d_model,
        n_timeframes=n_timeframes,
        n_regimes=6,
        n_refinement_steps=n_refinement_steps,
        dropout=0.1,
        temperature=1.0,
        use_gumbel=use_gumbel,
        min_weight=0.05,
    )
    
    return TimeframeSelectorForTrading(config)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """Demonstrate timeframe selection for HIMARI."""
    
    # Create selector
    model = create_timeframe_selector(
        d_model=128,
        n_timeframes=5,
        n_refinement_steps=3,
        use_gumbel=False,
    )
    model.eval()
    
    # Example timeframe representations
    batch_size = 4
    timeframe_reprs = [
        torch.randn(batch_size, 128) for _ in range(5)
    ]
    
    with torch.no_grad():
        outputs = model(timeframe_reprs)
    
    print("Timeframe Selector Output Shapes:")
    print(f"  Representation: {outputs['representation'].shape}")      # [4, 128]
    print(f"  Action logits: {outputs['action_logits'].shape}")        # [4, 3]
    print(f"  Confidence: {outputs['confidence'].shape}")              # [4]
    print(f"  Timeframe weights: {outputs['timeframe_weights'].shape}")# [4, 5]
    print(f"  Regime probs: {outputs['regime_probs'].shape}")          # [4, 6]
    print(f"  Quality scores: {outputs['quality_scores'].shape}")      # [4, 5]
    
    # Show weights for first sample
    weights = outputs['timeframe_weights'][0]
    tf_names = ['1m', '5m', '1h', '4h', '1d']
    print(f"\nTimeframe weights (sample 0):")
    for name, w in zip(tf_names, weights.tolist()):
        print(f"  {name}: {w:.3f}")
    
    # Show regime probabilities
    regime_names = ['TREND_UP', 'TREND_DOWN', 'RANGE', 'VOLATILE', 'QUIET', 'TRANSITION']
    regime_probs = outputs['regime_probs'][0]
    print(f"\nRegime probabilities (sample 0):")
    for name, p in zip(regime_names, regime_probs.tolist()):
        print(f"  {name}: {p:.3f}")
    
    # Show weight refinement history
    print(f"\nWeight refinement history (sample 0, 5m timeframe):")
    for step, hist_weights in enumerate(outputs['weight_history']):
        print(f"  Step {step}: {hist_weights[0, 1].item():.3f}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    example_usage()
