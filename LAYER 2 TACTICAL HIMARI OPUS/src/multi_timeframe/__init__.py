"""
HIMARI Layer 2 - Part C: Multi-Timeframe Fusion Components
==========================================================

This package contains the complete implementation of Layer 2's multi-timeframe
fusion subsystem. Part C is responsible for combining information across
9 timeframes (tick to daily) to produce unified trading signals.

Module Overview:
----------------

SAB (sab_attention.py):
    Surrogate Attention Blocks providing O(n) attention complexity.
    61% parameter reduction with 12.4% performance improvement.
    Foundation for C1 (TFT) and other transformer components.

C1 - TFT (c1_temporal_fusion_transformer.py):
    Temporal Fusion Transformer for multi-horizon forecasting.
    Interpretable attention with Variable Selection Networks.
    Primary encoder for short-medium term predictions.

C2 - FEDformer (c2_fedformer.py):
    Frequency Enhanced Decomposition Transformer.
    O(n log n) via Fourier/Wavelet attention.
    Captures long-range patterns (1h, 4h timeframes).

C3 - ViT-LOB (c3_vit_lob.py):
    Vision Transformer for Limit Order Book.
    Treats order book as image for microstructure analysis.
    Neural temporal encoding for tick-level processing.

C4 - CMTF (c4_cmtf.py):
    Coupled Matrix-Tensor Factorization.
    Multi-source fusion (price, sentiment, on-chain, order book).
    Discovers shared latent factors across data modalities.

C5 - PatchTST (c5_patchtst.py):
    Patch Time Series Transformer.
    Channel-independent processing with RevIN normalization.
    Efficient short-horizon encoding for 1m, 5m timeframes.

C6 - Cross-Attention (c6_cross_attention_fusion.py):
    Hierarchical cross-attention across timeframes.
    Bidirectional information flow (fine↔coarse).
    Core fusion mechanism before decision head.

C7 - Timeframe Selection (c7_timeframe_selection.py):
    Regime-conditional timeframe weighting.
    Scaleformer-style iterative refinement.
    Dynamic importance scoring based on market state.

C8 - Variable Selection (c8_variable_selection.py):
    GRN-based feature importance networks.
    Sparse attention for high-dimensional features.
    Interpretability through weight inspection.

Usage:
------
```python
from layer2_part_c import (
    # SAB Foundation
    create_sab_encoder,
    SABConfig,
    
    # Main Encoders
    create_tft_for_himari,
    create_fedformer_for_himari,
    create_vit_lob_for_himari,
    create_patchtst_for_himari,
    
    # Fusion Components
    create_cmtf_for_himari,
    create_cross_attention_fusion,
    create_timeframe_selector,
    create_variable_selector,
)

# Example: Create full Part C pipeline
tft = create_tft_for_himari(d_model=128, use_sab=True)
fedformer = create_fedformer_for_himari(d_model=128, seq_len=96)
vit_lob = create_vit_lob_for_himari(n_levels=10, n_timesteps=100)
patchtst = create_patchtst_for_himari(seq_len=512, patch_size=16)

cross_attn = create_cross_attention_fusion(d_model=128, n_timeframes=5)
selector = create_timeframe_selector(d_model=128, n_timeframes=5)
```

Performance Targets:
--------------------
- Total latency: <50ms (P99)
- Critical path: <25ms
- GPU memory: <2GB
- Sharpe contribution: +0.30 from multi-scale analysis

Architecture Integration:
-------------------------
Part C receives encoded representations from:
- Layer 1 signals (via Part A feature processing)
- Part B regime detection outputs

Part C outputs to:
- Layer 2 decision engine (Part D)
- Layer 3 position sizing module
"""

# =============================================================================
# SAB - Surrogate Attention Blocks
# =============================================================================
from .sab_attention import (
    SABConfig,
    SurrogateAttentionBlock,
    SABEncoder,
    create_sab_attention,
    create_sab_encoder,
)

# =============================================================================
# C1 - Temporal Fusion Transformer
# =============================================================================
from .c1_temporal_fusion_transformer import (
    TFTConfig,
    FeatureType,
    GatedLinearUnit,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
    TemporalFusionTransformer,
    TFTForTrading,
    create_tft_for_himari,
)

# =============================================================================
# C2 - FEDformer
# =============================================================================
from .c2_fedformer import (
    FEDformerConfig,
    FrequencyMode,
    MovingAverage,
    SeriesDecomposition,
    FourierBlock,
    WaveletBlock,
    FEDformerEncoderLayer,
    FEDformerDecoderLayer,
    FEDformer,
    FEDformerForTrading,
    create_fedformer_for_himari,
)

# =============================================================================
# C3 - ViT-LOB
# =============================================================================
from .c3_vit_lob import (
    ViTLOBConfig,
    NeuralTemporalEncoding,
    PatchEmbedding,
    LOBFeatureExtractor,
    ViTEncoder,
    ViTLOB,
    ViTLOBForTrading,
    create_lob_tensor,
    create_vit_lob_for_himari,
)

# =============================================================================
# C4 - CMTF
# =============================================================================
from .c4_cmtf import (
    CMTFConfig,
    FactorEncoder,
    FactorDecoder,
    FactorCoupler,
    CMTF,
    CMTFForTrading,
    create_cmtf_for_himari,
)

# =============================================================================
# C5 - PatchTST
# =============================================================================
from .c5_patchtst import (
    PatchTSTConfig,
    RevIN,
    PatchEmbedding,
    TransformerEncoderLayer,
    ChannelEncoder,
    PatchTST,
    PatchTSTForTrading,
    create_patchtst_for_himari,
)

# =============================================================================
# C6 - Cross-Attention Fusion
# =============================================================================
from .c6_cross_attention_fusion import (
    CrossAttentionConfig,
    TimeframeLevel,
    CrossAttentionBlock,
    GatedFusion,
    HierarchicalCrossAttention,
    CrossAttentionFusion,
    CrossAttentionFusionForTrading,
    create_cross_attention_fusion_for_himari,
)
# Create alias for backwards compatibility
CrossTimeframeFusion = CrossAttentionFusion
CrossAttentionLayer = CrossAttentionBlock
TemporalAlignmentModule = GatedFusion
create_cross_attention_fusion = create_cross_attention_fusion_for_himari

# =============================================================================
# C7 - Timeframe Selection
# =============================================================================
from .c7_timeframe_selection import (
    TimeframeSelectionConfig,
    MarketRegime,
    RegimeDetector,
    TimeframeQualityEstimator,
    ScaleformerRefinement,
    GumbelSoftmaxSelector,
    TimeframeSelector,
    TimeframeSelectorForTrading,
    create_timeframe_selector,
)

# =============================================================================
# C8 - Variable Selection
# =============================================================================
from .c8_variable_selection import (
    VariableSelectionConfig,
    FeatureEmbedding,
    SparseVariableSelection,
    VariableSelectorForTrading,
    MultiHeadVariableSelection,
    create_variable_selector,
)

# =============================================================================
# ALL EXPORTS
# =============================================================================
__all__ = [
    # SAB
    'SABConfig',
    'SurrogateAttentionBlock',
    'SABEncoder',
    'create_sab_attention',
    'create_sab_encoder',
    
    # C1 - TFT
    'TFTConfig',
    'FeatureType',
    'GatedLinearUnit',
    'GatedResidualNetwork',
    'VariableSelectionNetwork',
    'InterpretableMultiHeadAttention',
    'TemporalFusionTransformer',
    'TFTForTrading',
    'create_tft_for_himari',
    
    # C2 - FEDformer
    'FEDformerConfig',
    'FrequencyMode',
    'MovingAverage',
    'SeriesDecomposition',
    'FourierBlock',
    'WaveletBlock',
    'FEDformerEncoderLayer',
    'FEDformerDecoderLayer',
    'FEDformer',
    'FEDformerForTrading',
    'create_fedformer_for_himari',
    
    # C3 - ViT-LOB
    'ViTLOBConfig',
    'NeuralTemporalEncoding',
    'PatchEmbedding',
    'LOBFeatureExtractor',
    'ViTEncoder',
    'ViTLOB',
    'ViTLOBForTrading',
    'create_lob_tensor',
    'create_vit_lob_for_himari',
    
    # C4 - CMTF
    'CMTFConfig',
    'FactorEncoder',
    'FactorDecoder',
    'FactorCoupler',
    'CMTF',
    'CMTFForTrading',
    'create_cmtf_for_himari',
    
    # C5 - PatchTST
    'PatchTSTConfig',
    'RevIN',
    'PatchEmbedding',
    'TransformerEncoderLayer',
    'ChannelEncoder',
    'PatchTST',
    'PatchTSTForTrading',
    'create_patchtst_for_himari',
    
    # C6 - Cross-Attention
    'CrossAttentionConfig',
    'TimeframeLevel',
    'CrossAttentionLayer',
    'TemporalAlignmentModule',
    'HierarchicalCrossAttention',
    'CrossTimeframeFusion',
    'create_cross_attention_fusion',
    
    # C7 - Timeframe Selection
    'TimeframeSelectionConfig',
    'MarketRegime',
    'RegimeDetector',
    'TimeframeQualityEstimator',
    'ScaleformerRefinement',
    'GumbelSoftmaxSelector',
    'TimeframeSelector',
    'TimeframeSelectorForTrading',
    'create_timeframe_selector',
    
    # C8 - Variable Selection
    'VariableSelectionConfig',
    'FeatureEmbedding',
    'SparseVariableSelection',
    'VariableSelectorForTrading',
    'MultiHeadVariableSelection',
    'create_variable_selector',
]

__version__ = '0.1.0'
