"""
HIMARI Layer 2 - Part M: Adaptation Framework
Online adaptation and continual learning.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, List
from collections import deque
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# M1: Online Learning
# ============================================================================

@dataclass
class OnlineLearningConfig:
    learning_rate: float = 1e-5
    batch_size: int = 32
    update_frequency: int = 100

class OnlineLearner:
    """Online learning for continuous model adaptation."""
    
    def __init__(self, model: nn.Module, config=None, device='cpu'):
        self.config = config or OnlineLearningConfig()
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        self.buffer = deque(maxlen=1000)
        self.update_count = 0
        
    def observe(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
        
        if len(self.buffer) >= self.config.batch_size and \
           self.update_count % self.config.update_frequency == 0:
            self._update()
            
        self.update_count += 1
        
    def _update(self):
        batch_idx = np.random.choice(len(self.buffer), self.config.batch_size)
        batch = [self.buffer[i] for i in batch_idx]
        
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        
        self.optimizer.zero_grad()
        # Generic loss (placeholder)
        predictions = self.model(states)
        if predictions.dim() > 1:
            predictions = predictions.mean(dim=-1)
        loss = torch.mean((predictions - rewards.unsqueeze(1)) ** 2)
        loss.backward()
        self.optimizer.step()


# ============================================================================
# M2: Concept Drift Detection
# ============================================================================

class DriftDetector:
    """Detects distribution shift in data."""
    
    def __init__(self, window_size: int = 500, threshold: float = 2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_stats = None
        self.current_window = deque(maxlen=window_size)
        
    def update(self, value: float) -> bool:
        self.current_window.append(value)
        
        if len(self.current_window) < self.window_size:
            return False
            
        current_mean = np.mean(self.current_window)
        current_std = np.std(self.current_window)
        
        if self.reference_stats is None:
            self.reference_stats = (current_mean, current_std)
            return False
            
        ref_mean, ref_std = self.reference_stats
        z_score = abs(current_mean - ref_mean) / (ref_std + 1e-8)
        
        drift_detected = z_score > self.threshold
        if drift_detected:
            logger.warning(f"Drift detected: z-score = {z_score:.2f}")
            
        return drift_detected
    
    def reset_reference(self):
        if len(self.current_window) >= self.window_size // 2:
            self.reference_stats = (np.mean(self.current_window), np.std(self.current_window))


# ============================================================================
# M3: Model Ensemble Weighting
# ============================================================================

class AdaptiveEnsembleWeighter:
    """Dynamically adjusts ensemble weights based on recent performance."""
    
    def __init__(self, model_names: List[str], window: int = 100):
        self.model_names = model_names
        self.window = window
        self.performance = {name: deque(maxlen=window) for name in model_names}
        self.weights = {name: 1.0 / len(model_names) for name in model_names}
        
    def update(self, model_name: str, reward: float):
        self.performance[model_name].append(reward)
        self._recompute_weights()
        
    def _recompute_weights(self):
        scores = {}
        for name in self.model_names:
            if len(self.performance[name]) > 10:
                scores[name] = np.mean(self.performance[name])
            else:
                scores[name] = 0.0
                
        # Softmax over scores
        max_score = max(scores.values())
        exp_scores = {k: np.exp((v - max_score) * 2) for k, v in scores.items()}
        total = sum(exp_scores.values())
        
        self.weights = {k: v / total for k, v in exp_scores.items()}
        
    def get_weights(self) -> Dict[str, float]:
        return self.weights


# ============================================================================
# M4: Regime-Adaptive Parameters
# ============================================================================

class RegimeAdaptiveParameters:
    """Adapts model parameters based on detected regime."""
    
    def __init__(self):
        self.regime_params = {
            0: {'learning_rate': 1e-4, 'exploration': 0.1, 'position_scale': 1.0},
            1: {'learning_rate': 1e-4, 'exploration': 0.1, 'position_scale': 0.8},
            2: {'learning_rate': 5e-5, 'exploration': 0.2, 'position_scale': 0.6},
            3: {'learning_rate': 1e-5, 'exploration': 0.3, 'position_scale': 0.3},
            4: {'learning_rate': 0, 'exploration': 0, 'position_scale': 0.0},
        }
        self.current_regime = 2
        
    def update_regime(self, regime: int):
        if regime != self.current_regime:
            logger.info(f"Regime change: {self.current_regime} -> {regime}")
            self.current_regime = regime
            
    def get_params(self) -> Dict:
        return self.regime_params.get(self.current_regime, self.regime_params[2])


# ============================================================================
# M5: Experience Prioritization
# ============================================================================

class ExperiencePrioritizer:
    """Prioritizes experiences for replay based on informativeness."""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.experiences = []
        self.priorities = []
        
    def add(self, experience, td_error: float):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        
        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
            self.priorities.append(priority)
        else:
            min_idx = np.argmin(self.priorities)
            if priority > self.priorities[min_idx]:
                self.experiences[min_idx] = experience
                self.priorities[min_idx] = priority
                
    def sample(self, batch_size: int):
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.experiences), batch_size, p=probs, replace=False)
        return [self.experiences[i] for i in indices], indices
    
    def update_priorities(self, indices, new_td_errors):
        for idx, td_error in zip(indices, new_td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha


# ============================================================================
# M6: Forgetting Prevention
# ============================================================================

class ForgettingPreventer:
    """Prevents catastrophic forgetting via elastic weight consolidation."""
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.optimal_params = {}
        
    def compute_fisher(self, data_loader):
        """Compute Fisher information matrix."""
        self.fisher = {}
        self.optimal_params = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param.data)
                self.optimal_params[name] = param.data.clone()
                
    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        loss = torch.tensor(0.0)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                fisher = self.fisher[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal) ** 2).sum()
                
        return self.lambda_ewc * loss


# ============================================================================
# Complete Adaptation Pipeline
# ============================================================================

@dataclass
class AdaptationConfig:
    online_lr: float = 1e-5
    drift_threshold: float = 2.0
    window_size: int = 500

class AdaptationPipeline:
    """Complete adaptation framework."""
    
    def __init__(self, model: nn.Module = None, model_names: List[str] = None,
                 config=None, device='cpu'):
        self.config = config or AdaptationConfig()
        
        if model is not None:
            self.online_learner = OnlineLearner(
                model, OnlineLearningConfig(learning_rate=self.config.online_lr), device
            )
        else:
            self.online_learner = None
            
        self.drift_detector = DriftDetector(
            window_size=self.config.window_size,
            threshold=self.config.drift_threshold
        )
        
        if model_names:
            self.ensemble_weighter = AdaptiveEnsembleWeighter(model_names)
        else:
            self.ensemble_weighter = None
            
        self.regime_params = RegimeAdaptiveParameters()
        self.prioritizer = ExperiencePrioritizer()
        
    def step(self, state, action, reward, next_state, regime: int = 2):
        """Process one adaptation step."""
        # Online learning
        if self.online_learner:
            self.online_learner.observe(state, action, reward, next_state)
        
        # Drift detection
        drift = self.drift_detector.update(reward)
        if drift:
            self.drift_detector.reset_reference()
            
        # Regime adaptation
        self.regime_params.update_regime(regime)
        
        return {
            'drift_detected': drift,
            'regime_params': self.regime_params.get_params(),
            'ensemble_weights': self.ensemble_weighter.get_weights() if self.ensemble_weighter else {}
        }
