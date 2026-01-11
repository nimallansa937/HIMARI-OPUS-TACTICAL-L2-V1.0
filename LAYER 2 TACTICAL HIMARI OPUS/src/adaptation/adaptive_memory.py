"""HIMARI Layer 2 - Part M: Adaptation Components"""

import numpy as np
from collections import deque
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class AdaptiveMemory:
    """Adaptive experience memory with prioritization."""
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def add(self, experience: Dict, priority: float = 1.0) -> None:
        self.memory.append(experience)
        self.priorities.append(priority)
        
    def sample(self, n: int) -> List[Dict]:
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.memory), min(n, len(self.memory)), p=probs, replace=False)
        return [self.memory[i] for i in indices]


class ThompsonSampling:
    """Thompson Sampling for action selection."""
    def __init__(self, n_arms: int = 3):
        self.n_arms = n_arms
        self.alphas = np.ones(n_arms)
        self.betas = np.ones(n_arms)
        
    def select(self) -> int:
        samples = [np.random.beta(a, b) for a, b in zip(self.alphas, self.betas)]
        return int(np.argmax(samples))
        
    def update(self, arm: int, reward: float) -> None:
        if reward > 0:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1


class DualLearning:
    """Dual learning for model-based and model-free RL."""
    def __init__(self):
        self.model_based_weight = 0.5
        
    def blend(self, mb_action: np.ndarray, mf_action: np.ndarray) -> np.ndarray:
        return self.model_based_weight * mb_action + (1 - self.model_based_weight) * mf_action
        
    def update_weight(self, mb_error: float, mf_error: float) -> None:
        total = mb_error + mf_error
        if total > 0:
            self.model_based_weight = mf_error / total


class CounterfactualRegret:
    """Counterfactual regret minimization."""
    def __init__(self, n_actions: int = 3):
        self.regrets = np.zeros(n_actions)
        self.strategy = np.ones(n_actions) / n_actions
        
    def update_regret(self, action: int, utility: np.ndarray) -> None:
        for a in range(len(utility)):
            self.regrets[a] += utility[a] - utility[action]
        self.regrets = np.maximum(self.regrets, 0)
        total = sum(self.regrets)
        if total > 0:
            self.strategy = self.regrets / total
        else:
            self.strategy = np.ones(len(self.regrets)) / len(self.regrets)


class PageHinkley:
    """Page-Hinkley drift detection."""
    def __init__(self, threshold: float = 50, alpha: float = 0.005):
        self.threshold = threshold
        self.alpha = alpha
        self.m = 0
        self.M = 0
        self.sum = 0
        self.n = 0
        
    def update(self, x: float) -> bool:
        self.n += 1
        self.sum += x
        self.m = self.sum / self.n - self.alpha
        self.M = max(self.M, self.m)
        return self.M - self.m > self.threshold
