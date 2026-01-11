"""
HIMARI Layer 2 - Adaptation Module
Subsystem M: Adaptation Framework (6 Methods)

Components:
    - M1: Online Learning
    - M2: Concept Drift Detection
    - M3: Model Ensemble Weighting
    - M4: Regime-Adaptive Parameters
    - M5: Experience Prioritization
    - M6: Forgetting Prevention
"""

from .adaptation_pipeline import (
    AdaptationPipeline,
    AdaptationConfig,
    OnlineLearner,
    OnlineLearningConfig,
    DriftDetector,
    AdaptiveEnsembleWeighter,
    RegimeAdaptiveParameters,
    ExperiencePrioritizer,
    ForgettingPreventer
)

__all__ = [
    'AdaptationPipeline',
    'AdaptationConfig',
    'OnlineLearner',
    'OnlineLearningConfig',
    'DriftDetector',
    'AdaptiveEnsembleWeighter',
    'RegimeAdaptiveParameters',
    'ExperiencePrioritizer',
    'ForgettingPreventer',
]
