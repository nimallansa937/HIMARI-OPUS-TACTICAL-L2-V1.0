"""
HIMARI Layer 2 - Part F: k-NN OOD Detection
F7: Out-of-distribution detection via k-NN in feature space.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class KNNOODConfig:
    """k-NN OOD detection configuration."""
    k: int = 10
    distance_threshold: float = 3.0
    use_mahalanobis: bool = False


class KNNOOD:
    """
    Out-of-distribution detection via k-NN distance.
    Large distance to training data indicates OOD input.
    """
    
    def __init__(self, config: Optional[KNNOODConfig] = None):
        self.config = config or KNNOODConfig()
        self.knn = NearestNeighbors(n_neighbors=self.config.k, metric='euclidean')
        self._fitted = False
        self.mean_distance = 0.0
        self.std_distance = 1.0
        
    def fit(self, X: np.ndarray) -> Dict[str, float]:
        """Fit k-NN on training features."""
        self.knn.fit(X)
        
        distances, _ = self.knn.kneighbors(X)
        mean_k_dist = np.mean(distances, axis=1)
        
        self.mean_distance = np.mean(mean_k_dist)
        self.std_distance = np.std(mean_k_dist) + 1e-8
        self._fitted = True
        
        return {
            'n_training': len(X),
            'mean_distance': self.mean_distance,
            'std_distance': self.std_distance
        }
        
    def check(self, x: np.ndarray) -> Tuple[bool, float, Dict]:
        """Check if input is OOD."""
        if not self._fitted:
            raise RuntimeError("KNNOOD not fitted")
            
        x = x.reshape(1, -1) if x.ndim == 1 else x
        distances, indices = self.knn.kneighbors(x)
        mean_dist = np.mean(distances)
        
        z_score = (mean_dist - self.mean_distance) / self.std_distance
        is_ood = z_score > self.config.distance_threshold
        
        return is_ood, float(z_score), {'raw_distance': mean_dist}
        
    def get_uncertainty(self, x: np.ndarray) -> float:
        """Get uncertainty score based on distance (0-1)."""
        _, z_score, _ = self.check(x)
        scaled = z_score / self.config.distance_threshold
        return 1 / (1 + np.exp(-scaled))
