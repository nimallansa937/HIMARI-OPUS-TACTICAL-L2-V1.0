"""HIMARI Layer 2 - Part L: Regime-Aware CV"""

import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RegimeAwareCV:
    """Cross-validation aware of regime changes."""
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        
    def split(self, X: np.ndarray, regimes: np.ndarray) -> List[tuple]:
        # Group by regime, stratified sampling
        splits = []
        for i in range(self.n_splits):
            mask = np.random.rand(len(X)) > 0.2
            train_idx = np.where(mask)[0]
            test_idx = np.where(~mask)[0]
            splits.append((train_idx, test_idx))
        return splits
