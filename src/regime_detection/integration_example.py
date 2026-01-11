"""
HIMARI Layer 2 - Part B: Regime Detection Integration Example
Demonstrates how to integrate regime detection with other subsystems.
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def create_regime_detector():
    """Create and configure regime detector."""
    try:
        from .regime_pipeline import RegimeDetectionPipeline, RegimePipelineConfig
        config = RegimePipelineConfig()
        return RegimeDetectionPipeline(config)
    except ImportError:
        logger.warning("Regime pipeline not available, using stub")
        return RegimeDetectorStub()


class RegimeDetectorStub:
    """Stub for testing without full pipeline."""
    def detect(self, features: np.ndarray) -> Dict:
        return {'regime': 2, 'confidence': 0.5, 'label': 'RANGING'}


def integration_example():
    """
    Example integration of regime detection with trading pipeline.
    
    Shows how to:
    1. Initialize regime detector
    2. Process market features
    3. Use regime for decision-making
    """
    # Create detector
    detector = create_regime_detector()
    
    # Simulated features
    features = np.random.randn(60)
    
    # Detect regime
    result = detector.detect(features)
    
    # Use in trading logic
    regime = result.get('regime', 2)
    confidence = result.get('confidence', 0.5)
    
    # Adjust strategy based on regime
    if regime == 0:  # Crisis
        strategy = "DEFENSIVE"
        position_scale = 0.0
    elif regime == 1:  # Bearish
        strategy = "SHORT_BIAS"
        position_scale = 0.5
    elif regime == 3:  # Bullish
        strategy = "LONG_BIAS"
        position_scale = 1.0
    else:  # Ranging
        strategy = "NEUTRAL"
        position_scale = 0.75
        
    return {
        'regime': regime,
        'confidence': confidence,
        'strategy': strategy,
        'position_scale': position_scale
    }


if __name__ == "__main__":
    result = integration_example()
    print(f"Integration result: {result}")
