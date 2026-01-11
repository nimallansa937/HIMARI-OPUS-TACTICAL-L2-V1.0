"""
HIMARI Layer 2 - Models Module
Contains baseline and advanced RL models for tactical decision making.
"""

from src.models.baseline_mlp import (
    BaselineMLP,
    BaselineMLPWithUncertainty,
    create_baseline_model
)

from src.models.transformer_a2c import (
    TransformerA2C,
    TransformerA2CConfig,
    TacticalTransformerEncoder,
    ActorHead,
    CriticHead,
    create_transformer_a2c
)

__all__ = [
    'BaselineMLP',
    'BaselineMLPWithUncertainty',
    'create_baseline_model',
    # Transformer-A2C
    'TransformerA2C',
    'TransformerA2CConfig',
    'TacticalTransformerEncoder',
    'ActorHead',
    'CriticHead',
    'create_transformer_a2c',
]
