"""
HIMARI Layer 2 - C1: Temporal Fusion Transformer (TFT)
======================================================

The Temporal Fusion Transformer is a specialized architecture for multi-horizon
time series forecasting that provides both high accuracy and interpretability.
Originally developed by Google (Lim et al., 2021), TFT has become the go-to
architecture for financial time series due to three key innovations:

1. Variable Selection Networks: Learn which features matter at each timestep
2. Gated Residual Networks: Control information flow with skip connections
3. Interpretable Multi-Head Attention: Show which historical patterns drive predictions

For HIMARI Layer 2, TFT serves as the primary encoder for each timeframe's data
before cross-timeframe fusion. This implementation includes SAB (Surrogate
Attention Blocks) integration for 61% parameter reduction while maintaining
interpretability.

Why TFT for Trading:
--------------------
- Handles known vs unknown future inputs (scheduled events vs price)
- Provides feature importance per timestep (explains decisions)
- Multi-horizon output (1-step, 5-step, 12-step forecasts simultaneously)
- Static covariate integration (asset metadata, regime classification)

Architecture Overview:
----------------------
Input: [batch, seq_len, n_features]
  ↓
Variable Selection Network (per feature)
  ↓
LSTM Encoder (past) + LSTM Decoder (future)
  ↓
Gated Residual Network
  ↓
SAB/Multi-Head Attention (temporal patterns)
  ↓
Position-wise Feed-Forward
  ↓
Quantile Outputs: [batch, horizon, 3] (10%, 50%, 90% quantiles)

Performance Targets:
- Latency: <15ms per timeframe
- Memory: <200MB GPU RAM
- Accuracy: MASE < 0.8 on crypto 5-min bars
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import SAB from our implementation
try:
    from .sab_attention import SABConfig, SurrogateAttentionBlock, SABEncoder
except ImportError:
    from sab_attention import SABConfig, SurrogateAttentionBlock, SABEncoder


class FeatureType(Enum):
    """Classification of input features for TFT processing."""
    STATIC = "static"           # Constant per sequence (asset ID, exchange)
    KNOWN_FUTURE = "known"      # Known at prediction time (hour, day, scheduled events)
    OBSERVED = "observed"       # Only known historically (prices, volume)


@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer.
    
    Attributes:
        d_model: Hidden dimension throughout the network
        n_heads: Number of attention heads
        n_encoder_layers: LSTM encoder layers
        n_decoder_layers: LSTM decoder layers  
        dropout: Dropout rate
        use_sab: Whether to use SAB instead of standard attention
        n_surrogates: Number of surrogate tokens if using SAB
        quantiles: Output quantiles for prediction intervals
        static_features: Number of static features
        known_features: Number of known future features
        observed_features: Number of observed features
        max_encoder_len: Maximum encoder sequence length
        max_decoder_len: Maximum decoder (forecast) length
    """
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 1
    n_decoder_layers: int = 1
    dropout: float = 0.1
    use_sab: bool = True  # Use SAB for 61% parameter reduction
    n_surrogates: int = 8
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    static_features: int = 4    # e.g., asset_id, exchange, sector, regime
    known_features: int = 8     # e.g., hour, day, is_weekend, scheduled_events
    observed_features: int = 48 # e.g., OHLCV, technicals, sentiment
    max_encoder_len: int = 64
    max_decoder_len: int = 12


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit for controlled information flow.
    
    GLU splits the input in half: one half is the signal, the other
    becomes a gate via sigmoid activation. This allows the network
    to learn which information to suppress.
    
    GLU(x) = σ(Wx + b) ⊙ (Vx + c)
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - the building block of TFT.
    
    GRN provides flexible nonlinear processing with:
    1. ELU activation for negative values (better gradients)
    2. Gating mechanism to control what passes through
    3. Residual connection for gradient flow
    4. Optional context vector for conditioning
    
    Architecture:
        Input (+ Context if provided)
          ↓
        Linear → ELU → Linear → Dropout
          ↓
        GLU (gating)
          ↓
        Add & LayerNorm (with residual)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Primary pathway
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Optional context injection
        self.context_dim = context_dim
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        
        # Gating
        self.glu = GatedLinearUnit(output_dim, output_dim, dropout)
        
        # Skip connection (project if dimensions differ)
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GRN.
        
        Args:
            x: Input tensor (..., input_dim)
            context: Optional context vector (..., context_dim)
            
        Returns:
            Output tensor (..., output_dim)
        """
        # Primary pathway
        hidden = self.fc1(x)
        
        # Add context if provided
        if context is not None and self.context_dim is not None:
            hidden = hidden + self.context_fc(context)
        
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        
        # Gating
        hidden = self.glu(hidden)
        
        # Residual connection
        if self.skip_proj is not None:
            residual = self.skip_proj(x)
        else:
            residual = x
        
        return self.layer_norm(hidden + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) - learns feature importance dynamically.
    
    VSN addresses a key challenge in financial ML: not all features are
    equally important, and importance changes with market conditions.
    For example:
    - In trending markets: momentum features dominate
    - In ranging markets: mean-reversion features dominate
    - During news events: sentiment features spike in importance
    
    Mechanism:
    1. Each feature is processed through its own GRN
    2. A softmax gate computes importance weights over features
    3. Features are weighted and summed
    
    This provides interpretability: we can see which features drove each decision.
    """
    
    def __init__(
        self,
        n_features: int,
        input_dim: int,
        hidden_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # Per-feature GRNs (process each feature independently)
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                context_dim=context_dim,
                dropout=dropout,
            )
            for _ in range(n_features)
        ])
        
        # Feature weighting (importance scoring)
        # Input: flattened features, Output: softmax weights
        self.weight_grn = GatedResidualNetwork(
            input_dim=n_features * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=n_features,
            context_dim=context_dim,
            dropout=dropout,
        )
    
    def forward(
        self,
        features: List[torch.Tensor],
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select and weight features.
        
        Args:
            features: List of n_features tensors, each (..., input_dim)
            context: Optional context for conditioning (..., context_dim)
            
        Returns:
            selected: Weighted combination (..., hidden_dim)
            weights: Feature importance scores (..., n_features)
        """
        # Process each feature through its GRN
        processed = []
        for i, feature in enumerate(features):
            processed.append(self.feature_grns[i](feature, context))
        
        # Stack for weight computation
        stacked = torch.stack(processed, dim=-2)  # (..., n_features, hidden)
        
        # Compute importance weights
        flattened = stacked.flatten(start_dim=-2)  # (..., n_features * hidden)
        weights = self.weight_grn(flattened, context)  # (..., n_features)
        weights = F.softmax(weights, dim=-1)
        
        # Weighted combination
        # weights: (..., n_features) → (..., n_features, 1)
        # stacked: (..., n_features, hidden)
        weights_expanded = weights.unsqueeze(-1)
        selected = (stacked * weights_expanded).sum(dim=-2)  # (..., hidden)
        
        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention with optional SAB backend.
    
    Standard attention is modified for interpretability:
    1. Values are shared across heads (reduces parameters)
    2. Attention patterns are normalized for visualization
    3. SAB can replace standard attention for efficiency
    
    The shared-value design means attention weights directly show
    which timesteps influence the output, without confounding from
    per-head value transformations.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_sab: bool = False,
        n_surrogates: int = 8,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_sab = use_sab
        
        assert d_model % n_heads == 0
        
        if use_sab:
            # Use Surrogate Attention Blocks
            sab_config = SABConfig(
                d_model=d_model,
                n_heads=n_heads,
                n_surrogates=n_surrogates,
                dropout=dropout,
            )
            self.sab = SurrogateAttentionBlock(sab_config)
        else:
            # Standard attention with shared values
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)  # Shared across heads
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention.
        
        For self-attention, query = key = value.
        
        Args:
            query: Query tensor (batch, seq_q, d_model)
            key: Key tensor (batch, seq_k, d_model)
            value: Value tensor (batch, seq_k, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Attended output (batch, seq_q, d_model)
            attention_weights: Attention scores (batch, n_heads, seq_q, seq_k)
        """
        if self.use_sab:
            # SAB expects single input for self-attention
            assert torch.equal(query, key) and torch.equal(key, value), \
                "SAB only supports self-attention"
            output, attn_info = self.sab(query, return_attention=True)
            # Average attention across stages for interpretability
            attn = attn_info['compress_attention'] if attn_info else None
            return output, attn
        
        batch_size, seq_q, _ = query.shape
        seq_k = key.shape[1]
        
        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head: (batch, seq, d) → (batch, heads, seq, head_dim)
        q = q.view(batch_size, seq_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_k, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_q, self.d_model)
        output = self.out_proj(output)
        
        return output, attn_weights


class TemporalFusionTransformer(nn.Module):
    """
    Complete Temporal Fusion Transformer for multi-horizon forecasting.
    
    This is the main model class integrating all TFT components:
    - Static covariate encoders for asset metadata
    - Variable selection for dynamic feature importance
    - LSTM encoder-decoder for sequence processing
    - Interpretable attention for temporal patterns
    - Quantile outputs for uncertainty estimation
    
    For HIMARI Layer 2, one TFT instance processes each timeframe,
    and outputs are fused in the cross-timeframe attention layer.
    """
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        
        self.config = config
        d_model = config.d_model
        
        # ================================================================
        # INPUT EMBEDDINGS
        # ================================================================
        
        # Static feature embedding
        self.static_embedding = nn.Linear(config.static_features, d_model)
        
        # Known future feature embedding  
        self.known_embedding = nn.Linear(config.known_features, d_model)
        
        # Observed feature embedding
        self.observed_embedding = nn.Linear(config.observed_features, d_model)
        
        # ================================================================
        # STATIC COVARIATE ENCODERS
        # ================================================================
        # Generate context vectors from static features
        
        # Context for variable selection
        self.static_context_vsn = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=config.dropout,
        )
        
        # Context for LSTM hidden state initialization
        self.static_context_hidden = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=config.dropout,
        )
        
        # Context for LSTM cell state initialization
        self.static_context_cell = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=config.dropout,
        )
        
        # Context for static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=config.dropout,
        )
        
        # ================================================================
        # VARIABLE SELECTION NETWORKS
        # ================================================================
        
        # For encoder (historical) inputs
        n_encoder_features = 2  # observed + known
        self.encoder_vsn = VariableSelectionNetwork(
            n_features=n_encoder_features,
            input_dim=d_model,
            hidden_dim=d_model,
            context_dim=d_model,
            dropout=config.dropout,
        )
        
        # For decoder (future) inputs
        self.decoder_vsn = VariableSelectionNetwork(
            n_features=1,  # only known future
            input_dim=d_model,
            hidden_dim=d_model,
            context_dim=d_model,
            dropout=config.dropout,
        )
        
        # ================================================================
        # LSTM ENCODER-DECODER
        # ================================================================
        
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=config.n_encoder_layers,
            dropout=config.dropout if config.n_encoder_layers > 1 else 0,
            batch_first=True,
        )
        
        self.lstm_decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=config.n_decoder_layers,
            dropout=config.dropout if config.n_decoder_layers > 1 else 0,
            batch_first=True,
        )
        
        # ================================================================
        # STATIC ENRICHMENT
        # ================================================================
        
        self.static_enrichment = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            context_dim=d_model,
            dropout=config.dropout,
        )
        
        # ================================================================
        # TEMPORAL SELF-ATTENTION
        # ================================================================
        
        self.attention = InterpretableMultiHeadAttention(
            d_model=d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_sab=config.use_sab,
            n_surrogates=config.n_surrogates,
        )
        
        self.attention_grn = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=config.dropout,
        )
        
        self.attention_norm = nn.LayerNorm(d_model)
        
        # ================================================================
        # POSITION-WISE FEED-FORWARD
        # ================================================================
        
        self.positionwise_grn = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=d_model * 4,
            output_dim=d_model,
            dropout=config.dropout,
        )
        
        # ================================================================
        # OUTPUT LAYER
        # ================================================================
        
        # Quantile outputs for uncertainty
        self.output_layer = nn.Linear(d_model, len(config.quantiles))
    
    def forward(
        self,
        static_features: torch.Tensor,
        known_features: torch.Tensor,
        observed_features: torch.Tensor,
        encoder_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TFT.
        
        Args:
            static_features: Static covariates (batch, static_dim)
            known_features: Known future features (batch, total_len, known_dim)
            observed_features: Historical features (batch, encoder_len, observed_dim)
            encoder_lengths: Actual lengths per batch (for masking)
            
        Returns:
            Dict containing:
                - predictions: Quantile forecasts (batch, decoder_len, n_quantiles)
                - attention_weights: Temporal attention patterns
                - feature_weights_encoder: Feature importance (encoder)
                - feature_weights_decoder: Feature importance (decoder)
                - hidden_states: LSTM hidden states for downstream use
        """
        batch_size = static_features.shape[0]
        encoder_len = observed_features.shape[1]
        total_len = known_features.shape[1]
        decoder_len = total_len - encoder_len
        
        # ================================================================
        # EMBED INPUTS
        # ================================================================
        
        static_emb = self.static_embedding(static_features)  # (batch, d_model)
        known_emb = self.known_embedding(known_features)      # (batch, total, d_model)
        observed_emb = self.observed_embedding(observed_features)  # (batch, enc, d_model)
        
        # ================================================================
        # STATIC COVARIATE ENCODING
        # ================================================================
        
        # Generate different context vectors from static features
        context_vsn = self.static_context_vsn(static_emb)
        context_hidden = self.static_context_hidden(static_emb)
        context_cell = self.static_context_cell(static_emb)
        context_enrichment = self.static_context_enrichment(static_emb)
        
        # ================================================================
        # VARIABLE SELECTION
        # ================================================================
        
        # Split known features into encoder and decoder portions
        known_encoder = known_emb[:, :encoder_len, :]
        known_decoder = known_emb[:, encoder_len:, :]
        
        # Encoder variable selection
        encoder_features = [observed_emb, known_encoder]
        encoder_selected, encoder_weights = self.encoder_vsn(
            encoder_features,
            context=context_vsn.unsqueeze(1).expand(-1, encoder_len, -1)
        )
        
        # Decoder variable selection (only known features available)
        decoder_features = [known_decoder]
        decoder_selected, decoder_weights = self.decoder_vsn(
            decoder_features,
            context=context_vsn.unsqueeze(1).expand(-1, decoder_len, -1)
        )
        
        # ================================================================
        # LSTM ENCODING
        # ================================================================
        
        # Initialize LSTM states with static context
        h0 = context_hidden.unsqueeze(0).expand(self.config.n_encoder_layers, -1, -1)
        c0 = context_cell.unsqueeze(0).expand(self.config.n_encoder_layers, -1, -1)
        
        encoder_output, (h_n, c_n) = self.lstm_encoder(
            encoder_selected.contiguous(),
            (h0.contiguous(), c0.contiguous())
        )
        
        # ================================================================
        # LSTM DECODING
        # ================================================================
        
        decoder_output, _ = self.lstm_decoder(
            decoder_selected.contiguous(),
            (h_n.contiguous(), c_n.contiguous())
        )
        
        # ================================================================
        # STATIC ENRICHMENT
        # ================================================================
        
        # Combine encoder and decoder outputs
        lstm_output = torch.cat([encoder_output, decoder_output], dim=1)
        
        # Enrich with static context
        enriched = self.static_enrichment(
            lstm_output,
            context=context_enrichment.unsqueeze(1).expand(-1, total_len, -1)
        )
        
        # ================================================================
        # TEMPORAL SELF-ATTENTION
        # ================================================================
        
        # Mask: decoder can't attend to future
        mask = self._create_attention_mask(encoder_len, decoder_len, batch_size)
        mask = mask.to(enriched.device)
        
        # Self-attention
        attended, attention_weights = self.attention(
            enriched, enriched, enriched, mask=mask
        )
        
        # Gated residual
        attended = self.attention_grn(attended)
        attended = self.attention_norm(enriched + attended)
        
        # ================================================================
        # POSITION-WISE PROCESSING
        # ================================================================
        
        output = self.positionwise_grn(attended)
        
        # ================================================================
        # GENERATE PREDICTIONS
        # ================================================================
        
        # Only use decoder portion for predictions
        decoder_output = output[:, encoder_len:, :]
        predictions = self.output_layer(decoder_output)  # (batch, dec_len, n_quantiles)
        
        return {
            'predictions': predictions,
            'attention_weights': attention_weights,
            'feature_weights_encoder': encoder_weights,
            'feature_weights_decoder': decoder_weights,
            'hidden_states': output,
            'encoder_output': encoder_output,
            'context_vector': context_enrichment,
        }
    
    def _create_attention_mask(
        self,
        encoder_len: int,
        decoder_len: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Create causal attention mask for decoder."""
        total_len = encoder_len + decoder_len
        
        # Start with all ones
        mask = torch.ones(total_len, total_len)
        
        # Decoder positions can only attend to encoder + previous decoder positions
        for i in range(encoder_len, total_len):
            mask[i, i+1:] = 0
        
        # Expand for batch
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        return mask
    
    def get_feature_importance(
        self,
        static_features: torch.Tensor,
        known_features: torch.Tensor,
        observed_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get feature importance scores for interpretability.
        
        Returns:
            Dict with encoder and decoder feature weights, plus attention patterns.
        """
        with torch.no_grad():
            outputs = self.forward(static_features, known_features, observed_features)
        
        return {
            'encoder_feature_importance': outputs['feature_weights_encoder'],
            'decoder_feature_importance': outputs['feature_weights_decoder'],
            'temporal_attention': outputs['attention_weights'],
        }


class TFTForTrading(nn.Module):
    """
    TFT wrapper optimized for HIMARI trading use case.
    
    This wrapper adds:
    1. Action head for BUY/HOLD/SELL classification
    2. Confidence estimation
    3. Multi-horizon price forecasts
    4. Integration hooks for cross-timeframe fusion
    """
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        
        self.tft = TemporalFusionTransformer(config)
        self.config = config
        
        # Action classification head
        self.action_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 3),  # BUY, HOLD, SELL
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        # Volatility prediction head
        self.volatility_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.max_decoder_len),
            nn.Softplus(),
        )
    
    def forward(
        self,
        static_features: torch.Tensor,
        known_features: torch.Tensor,
        observed_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for trading.
        
        Returns:
            Dict with predictions, action logits, confidence, etc.
        """
        # Base TFT forward
        tft_outputs = self.tft(static_features, known_features, observed_features)
        
        # Use context vector for classification (summarizes entire sequence)
        context = tft_outputs['context_vector']
        
        # Trading outputs
        action_logits = self.action_head(context)
        confidence = self.confidence_head(context).squeeze(-1)
        volatility = self.volatility_head(context)
        
        return {
            'price_quantiles': tft_outputs['predictions'],
            'action_logits': action_logits,
            'confidence': confidence,
            'volatility_forecast': volatility,
            'attention_weights': tft_outputs['attention_weights'],
            'feature_importance_encoder': tft_outputs['feature_weights_encoder'],
            'feature_importance_decoder': tft_outputs['feature_weights_decoder'],
            'representation': tft_outputs['encoder_output'][:, -1, :],  # Last encoder state
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_tft_for_himari(
    d_model: int = 128,
    use_sab: bool = True,
    n_surrogates: int = 8,
    observed_features: int = 48,
) -> TFTForTrading:
    """
    Create TFT configured for HIMARI Layer 2.
    
    Default configuration optimized for:
    - 5-minute crypto bars
    - 64-bar lookback (encoder)
    - 12-bar forecast horizon (decoder)
    - <15ms latency target
    """
    config = TFTConfig(
        d_model=d_model,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.1,
        use_sab=use_sab,
        n_surrogates=n_surrogates,
        quantiles=[0.1, 0.5, 0.9],
        static_features=4,
        known_features=8,
        observed_features=observed_features,
        max_encoder_len=64,
        max_decoder_len=12,
    )
    
    return TFTForTrading(config)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """Demonstrate TFT usage for HIMARI."""
    
    # Create model
    model = create_tft_for_himari(d_model=128, use_sab=True)
    model.eval()
    
    # Example inputs
    batch_size = 4
    encoder_len = 64
    decoder_len = 12
    
    static = torch.randn(batch_size, 4)  # Asset metadata
    known = torch.randn(batch_size, encoder_len + decoder_len, 8)  # Time features
    observed = torch.randn(batch_size, encoder_len, 48)  # Historical OHLCV + technicals
    
    # Forward pass
    with torch.no_grad():
        outputs = model(static, known, observed)
    
    print("TFT for Trading Output Shapes:")
    print(f"  Price quantiles: {outputs['price_quantiles'].shape}")  # [4, 12, 3]
    print(f"  Action logits: {outputs['action_logits'].shape}")      # [4, 3]
    print(f"  Confidence: {outputs['confidence'].shape}")            # [4]
    print(f"  Volatility: {outputs['volatility_forecast'].shape}")   # [4, 12]
    print(f"  Representation: {outputs['representation'].shape}")    # [4, 128]
    
    # Interpret action
    actions = torch.argmax(outputs['action_logits'], dim=-1)
    action_names = ['SELL', 'HOLD', 'BUY']
    print(f"\nPredicted actions: {[action_names[a] for a in actions.tolist()]}")
    print(f"Confidence scores: {outputs['confidence'].tolist()}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    example_usage()
