"""
HIMARI Layer 2 - Part K2: MAML Meta-Learning
Fast regime adaptation via Model-Agnostic Meta-Learning.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


@dataclass
class MAMLConfig:
    """Configuration for MAML meta-learning."""
    inner_lr: float = 0.01               # Learning rate for task adaptation
    outer_lr: float = 0.001              # Learning rate for meta-update
    inner_steps: int = 5                 # Gradient steps for adaptation
    meta_batch_size: int = 4             # Tasks per meta-batch
    shots: int = 20                      # Samples per task for adaptation
    query_size: int = 20                 # Samples per task for evaluation
    first_order: bool = True             # Use first-order approximation


@dataclass
class MAMLTask:
    """A meta-learning task (regime-specific data)."""
    support_x: torch.Tensor     
    support_y: torch.Tensor     
    query_x: torch.Tensor       
    query_y: torch.Tensor       
    regime: str                 


class RegimeTaskGenerator:
    """Generate meta-learning tasks from regime-labeled data."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray,
                regimes: np.ndarray, config: MAMLConfig):
        self.features = features
        self.labels = labels
        self.regimes = regimes
        self.config = config
        
        self.regime_indices = {}
        unique_regimes = np.unique(regimes)
        for regime in unique_regimes:
            self.regime_indices[regime] = np.where(regimes == regime)[0]
        
        self.regime_list = list(self.regime_indices.keys())
        
    def sample_task(self, regime: str = None) -> MAMLTask:
        if regime is None:
            regime = np.random.choice(self.regime_list)
        
        indices = self.regime_indices[regime]
        total_needed = self.config.shots + self.config.query_size
        
        if len(indices) < total_needed:
            sampled = np.random.choice(indices, total_needed, replace=True)
        else:
            sampled = np.random.choice(indices, total_needed, replace=False)
        
        support_idx = sampled[:self.config.shots]
        query_idx = sampled[self.config.shots:]
        
        return MAMLTask(
            support_x=torch.tensor(self.features[support_idx], dtype=torch.float32),
            support_y=torch.tensor(self.labels[support_idx], dtype=torch.long),
            query_x=torch.tensor(self.features[query_idx], dtype=torch.float32),
            query_y=torch.tensor(self.labels[query_idx], dtype=torch.long),
            regime=str(regime)
        )
    
    def sample_batch(self, batch_size: int = None) -> List[MAMLTask]:
        batch_size = batch_size or self.config.meta_batch_size
        
        if batch_size <= len(self.regime_list):
            regimes = np.random.choice(self.regime_list, batch_size, replace=False)
        else:
            regimes = np.random.choice(self.regime_list, batch_size, replace=True)
        
        return [self.sample_task(r) for r in regimes]


class MAMLTrainer:
    """MAML training for rapid regime adaptation."""
    
    def __init__(self, model: nn.Module, task_generator: RegimeTaskGenerator,
                config: MAMLConfig = None, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.task_generator = task_generator
        self.config = config or MAMLConfig()
        
        self.meta_optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.outer_lr
        )
        self.criterion = nn.CrossEntropyLoss()
        
        self._meta_losses: List[float] = []
        self._adaptation_improvements: List[float] = []
        
    def inner_loop(self, task: MAMLTask,
                  params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        adapted_params = {k: v.clone() for k, v in params.items()}
        
        for step in range(self.config.inner_steps):
            outputs = self._forward_with_params(task.support_x.to(self.device), adapted_params)
            loss = self.criterion(outputs, task.support_y.to(self.device))
            
            grads = torch.autograd.grad(
                loss, adapted_params.values(),
                create_graph=not self.config.first_order
            )
            
            adapted_params = {
                k: v - self.config.inner_lr * g
                for (k, v), g in zip(adapted_params.items(), grads)
            }
        
        return adapted_params
    
    def outer_loop(self, tasks: List[MAMLTask]) -> float:
        self.meta_optimizer.zero_grad()
        params = {name: param for name, param in self.model.named_parameters()}
        
        total_loss = 0.0
        for task in tasks:
            adapted_params = self.inner_loop(task, params)
            query_outputs = self._forward_with_params(task.query_x.to(self.device), adapted_params)
            query_loss = self.criterion(query_outputs, task.query_y.to(self.device))
            total_loss += query_loss
        
        meta_loss = total_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()
        
        self._meta_losses.append(meta_loss.item())
        return meta_loss.item()
    
    def _forward_with_params(self, x: torch.Tensor,
                            params: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Simple functional forward for linear models
        # For real implementation, use higher-order methods
        return self.model(x)
    
    def train_step(self) -> float:
        tasks = self.task_generator.sample_batch()
        return self.outer_loop(tasks)
    
    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor,
             steps: int = None) -> nn.Module:
        """Adapt model to new data."""
        steps = steps or self.config.inner_steps
        adapted_model = deepcopy(self.model)
        adapted_model.train()
        
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.config.inner_lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            outputs = adapted_model(support_x.to(self.device))
            loss = self.criterion(outputs, support_y.to(self.device))
            loss.backward()
            optimizer.step()
        
        return adapted_model


def create_maml_trainer(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    regimes: np.ndarray,
    **kwargs
) -> MAMLTrainer:
    """Create MAML trainer with given data."""
    config = MAMLConfig(**kwargs)
    task_gen = RegimeTaskGenerator(features, labels, regimes, config)
    return MAMLTrainer(model, task_gen, config)
