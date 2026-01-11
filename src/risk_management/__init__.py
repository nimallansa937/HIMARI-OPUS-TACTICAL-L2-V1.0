"""
HIMARI Layer 2 - Risk Management Module
Subsystem H: RSS Risk Management (6 Methods)

Components:
    - H1: Kelly Criterion Position Sizing
    - H2: Volatility Targeting
    - H3: Drawdown Control
    - H4: Correlation-Based Risk Adjustment
    - H5: Regime-Adaptive Risk
    - H6: Stop Loss Manager
"""

from .rss_pipeline import (
    RSSRiskManager,
    RiskConfig,
    KellySizer,
    KellyConfig,
    VolatilityTargeting,
    VolTargetConfig,
    DrawdownController,
    DrawdownConfig,
    CorrelationRiskAdjuster,
    RegimeRiskScaler,
    StopLossManager,
    StopLossConfig
)

__all__ = [
    'RSSRiskManager',
    'RiskConfig',
    'KellySizer',
    'KellyConfig',
    'VolatilityTargeting',
    'VolTargetConfig',
    'DrawdownController',
    'DrawdownConfig',
    'CorrelationRiskAdjuster',
    'RegimeRiskScaler',
    'StopLossManager',
    'StopLossConfig',
]
