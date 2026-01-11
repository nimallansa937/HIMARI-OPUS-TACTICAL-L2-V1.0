"""
HIMARI Layer 2 - Part F: Uncertainty Decomposition
F6: Split total uncertainty into epistemic and aleatoric components.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyDecomposition:
    """Decomposed uncertainty components."""
    total: float
    epistemic: float  # Model uncertainty (reducible)
    aleatoric: float  # Data noise (irreducible)
    ratio: float      # epistemic / total


class UncertaintySplitter:
    """
    Split total uncertainty into epistemic and aleatoric.
    
    Epistemic = variance of means (disagreement between models)
    Aleatoric = mean of variances (average within-model uncertainty)
    """
    
    def __init__(self):
        pass
        
    def decompose(
        self,
        ensemble_predictions: np.ndarray,
        ensemble_variances: Optional[np.ndarray] = None
    ) -> UncertaintyDecomposition:
        """
        Decompose uncertainty from ensemble.
        
        Args:
            ensemble_predictions: [n_members, n_samples, n_outputs]
            ensemble_variances: Optional per-member variances
        """
        # Epistemic: variance across members
        variance_across = np.var(ensemble_predictions, axis=0)
        epistemic = float(np.mean(variance_across))
        
        # Aleatoric: mean of variances
        if ensemble_variances is not None:
            aleatoric = float(np.mean(ensemble_variances))
        else:
            aleatoric = 0.0
            
        total = epistemic + aleatoric
        ratio = epistemic / total if total > 0 else 0.5
        
        return UncertaintyDecomposition(
            total=total,
            epistemic=epistemic,
            aleatoric=aleatoric,
            ratio=ratio
        )
        
    def interpret(self, decomp: UncertaintyDecomposition) -> str:
        """Interpret uncertainty decomposition."""
        if decomp.ratio > 0.7:
            return "HIGH_EPISTEMIC: Model uncertain, more data could help"
        elif decomp.ratio < 0.3:
            return "HIGH_ALEATORIC: Inherent noise, uncertainty irreducible"
        else:
            return "BALANCED: Both model and data uncertainty present"
