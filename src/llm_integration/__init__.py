"""
HIMARI Layer 2 - LLM Integration Module
Subsystem J: LLM-Based Analysis (6 Methods)

Components:
    - J1: Market Narrative Generator
    - J2: Signal Explainer
    - J3: Risk Narrator
    - J4: Embedding Generator
    - J5: Context Aggregator
    - J6: LLM Decision Advisor
"""

from .llm_pipeline import (
    LLMIntegrationPipeline,
    LLMConfig,
    LLMDecisionAdvisor,
    MarketNarrativeGenerator,
    NarrativeConfig,
    SignalExplainer,
    RiskNarrator,
    TextEmbedder,
    ContextAggregator
)

__all__ = [
    'LLMIntegrationPipeline',
    'LLMConfig',
    'LLMDecisionAdvisor',
    'MarketNarrativeGenerator',
    'NarrativeConfig',
    'SignalExplainer',
    'RiskNarrator',
    'TextEmbedder',
    'ContextAggregator',
]
