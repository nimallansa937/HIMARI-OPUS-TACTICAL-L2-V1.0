"""HIMARI Layer 2 - Part J: Alignment Loss, Causal Intervention, Contextual Bandit"""

import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class AlignmentLoss:
    """Compute alignment loss between LLM and model decisions."""
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        
    def compute(self, llm_probs: np.ndarray, model_probs: np.ndarray) -> float:
        # KL divergence
        kl = np.sum(model_probs * np.log(model_probs / (llm_probs + 1e-8) + 1e-8))
        return float(kl)


class CausalIntervention:
    """Causal intervention for counterfactual analysis."""
    def __init__(self):
        self.interventions: List[Dict] = []
        
    def intervene(self, variable: str, value: float) -> None:
        self.interventions.append({"variable": variable, "value": value})
        
    def compute_effect(self, baseline: float, intervened: float) -> float:
        return intervened - baseline


class ContextualBandit:
    """Contextual bandit for action selection."""
    def __init__(self, n_actions: int = 3, alpha: float = 0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.rewards: Dict[int, List[float]] = {i: [] for i in range(n_actions)}
        
    def select_action(self, context: np.ndarray) -> int:
        ucb_values = []
        for a in range(self.n_actions):
            if len(self.rewards[a]) == 0:
                ucb_values.append(float('inf'))
            else:
                mean = np.mean(self.rewards[a])
                bonus = self.alpha * np.sqrt(2 * np.log(sum(len(r) for r in self.rewards.values())) / len(self.rewards[a]))
                ucb_values.append(mean + bonus)
        return int(np.argmax(ucb_values))
        
    def update(self, action: int, reward: float) -> None:
        self.rewards[action].append(reward)
