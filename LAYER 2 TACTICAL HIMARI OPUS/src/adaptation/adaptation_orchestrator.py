"""HIMARI Layer 2 - Part M: Adaptation Orchestrator"""

from typing import Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AdaptationOrchestrator:
    """Orchestrate all adaptation components."""
    def __init__(self):
        from .adaptive_memory import AdaptiveMemory, ThompsonSampling, PageHinkley
        self.memory = AdaptiveMemory()
        self.bandit = ThompsonSampling()
        self.drift_detector = PageHinkley()
        
    def adapt(self, state: Dict, reward: float) -> Dict:
        # Detect drift
        drift = self.drift_detector.update(reward)
        
        # Select action via Thompson Sampling
        action = self.bandit.select()
        
        # Update bandit
        self.bandit.update(action, reward)
        
        # Store experience
        self.memory.add({"state": state, "reward": reward})
        
        return {
            "action": action,
            "drift_detected": drift,
            "memory_size": len(self.memory.memory)
        }
