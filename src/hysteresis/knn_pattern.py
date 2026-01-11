"""
HIMARI Layer 2 - Part G: KNN Pattern Matching
G2: Pattern-based whipsaw detection using k-nearest neighbors.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class KNNPatternConfig:
    """KNN pattern configuration."""
    k: int = 5
    max_patterns: int = 1000
    whipsaw_threshold: float = 0.6


class KNNPatternMatcher:
    """
    Pattern-based whipsaw detection using k-NN.
    Learns from historical whipsaw patterns.
    """
    
    def __init__(self, config: Optional[KNNPatternConfig] = None):
        self.config = config or KNNPatternConfig()
        self.patterns: List[np.ndarray] = []
        self.labels: List[int] = []  # 1 = whipsaw, 0 = valid
        self.knn = NearestNeighbors(n_neighbors=self.config.k)
        self._fitted = False
        
    def add_pattern(self, features: np.ndarray, was_whipsaw: bool) -> None:
        """Add pattern to memory."""
        self.patterns.append(features)
        self.labels.append(1 if was_whipsaw else 0)
        
        if len(self.patterns) > self.config.max_patterns:
            self.patterns.pop(0)
            self.labels.pop(0)
            
        if len(self.patterns) >= self.config.k:
            self._fit()
            
    def _fit(self) -> None:
        """Fit k-NN on patterns."""
        X = np.array(self.patterns)
        self.knn.fit(X)
        self._fitted = True
        
    def predict_whipsaw_probability(self, features: np.ndarray) -> Tuple[float, Dict]:
        """Predict probability of whipsaw."""
        if not self._fitted:
            return 0.0, {'fitted': False}
            
        features = features.reshape(1, -1)
        distances, indices = self.knn.kneighbors(features)
        
        neighbor_labels = [self.labels[i] for i in indices[0]]
        prob = np.mean(neighbor_labels)
        
        return prob, {
            'n_whipsaw_neighbors': sum(neighbor_labels),
            'avg_distance': float(np.mean(distances))
        }
        
    def should_block(self, features: np.ndarray) -> bool:
        """Check if signal should be blocked."""
        prob, _ = self.predict_whipsaw_probability(features)
        return prob > self.config.whipsaw_threshold
