"""
HIMARI Layer 2 - Risk Management Module (v5.0)
Subsystem H: RSS Risk Management

Components (v5.0 - 78 methods architecture):
    - EVT-GPD (H1): Extreme Value Theory tail risk - NEW
    - Dynamic Kelly (H2): RL-based position sizing - NEW
    - DCC-GARCH (H3): Time-varying correlation - NEW
    - Safe Margin Formula (H6): Conservative margin calculation - KEPT
    - Dynamic Leverage Controller (H7): Adaptive leverage - KEPT
"""

from .evt_gpd import (
    EVTGPDRisk,
    EVTConfig,
    DynamicKellyFraction,
    create_evt_risk
)
from .dcc_garch import (
    DCCGARCH,
    DCCConfig,
    create_dcc_garch
)

__all__ = [
    # v5.0 Components
    'EVTGPDRisk',
    'EVTConfig',
    'DynamicKellyFraction',
    'create_evt_risk',
    'DCCGARCH',
    'DCCConfig',
    'create_dcc_garch'
]
