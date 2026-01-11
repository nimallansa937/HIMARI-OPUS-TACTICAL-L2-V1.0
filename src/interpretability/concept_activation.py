"""HIMARI Layer 2 - Part N: Concept Activation, Counterfactual Explanation"""

import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ConceptActivation:
    """Concept Activation Vectors (CAVs) for interpretability."""
    def __init__(self):
        self.concepts: Dict[str, np.ndarray] = {}
        
    def add_concept(self, name: str, examples: List[np.ndarray]) -> None:
        # Compute average activation direction
        self.concepts[name] = np.mean(examples, axis=0)
        
    def score_concept(self, activations: np.ndarray, concept: str) -> float:
        if concept not in self.concepts:
            return 0.0
        return float(np.dot(activations, self.concepts[concept]) / 
                    (np.linalg.norm(activations) * np.linalg.norm(self.concepts[concept]) + 1e-8))


class CounterfactualExplanation:
    """Generate counterfactual explanations."""
    def __init__(self, model=None):
        self.model = model
        
    def generate(self, x: np.ndarray, target_action: str) -> Dict:
        # Find minimal change to flip prediction
        # Placeholder - actual impl uses optimization
        counterfactual = x.copy()
        changes = []
        
        for i in range(5):  # Try changing top 5 features
            idx = np.random.randint(len(x))
            old_val = counterfactual[idx]
            counterfactual[idx] *= 1.5
            changes.append({'feature': idx, 'old': old_val, 'new': counterfactual[idx]})
            
        return {
            'original': x,
            'counterfactual': counterfactual,
            'changes': changes,
            'target': target_action
        }
