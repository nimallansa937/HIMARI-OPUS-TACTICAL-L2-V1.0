"""
HIMARI Layer 2 - Part K4: Multi-Task Learning
Shared representation learning across related tasks.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task learning."""
    shared_hidden_dim: int = 256
    task_hidden_dim: int = 128
    n_shared_layers: int = 3
    task_weight_strategy: str = "uncertainty"  # "uniform", "uncertainty", "gradnorm"
    gradient_clip: float = 1.0
    task_dropout: float = 0.1


class SharedEncoder(nn.Module):
    """Shared encoder for multi-task learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        for i in range(n_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TaskHead(nn.Module):
    """Task-specific head."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MultiTaskModel(nn.Module):
    """
    Multi-task model with shared encoder and task-specific heads.
    
    Tasks:
    - Direction prediction (3-class: buy/hold/sell)
    - Volatility prediction (regression)
    - Regime classification (5-class)
    """
    
    def __init__(self, input_dim: int, config: MultiTaskConfig = None):
        super().__init__()
        self.config = config or MultiTaskConfig()
        
        # Shared encoder
        self.encoder = SharedEncoder(
            input_dim, 
            self.config.shared_hidden_dim,
            self.config.n_shared_layers
        )
        
        # Task heads
        self.direction_head = TaskHead(
            self.encoder.output_dim,
            self.config.task_hidden_dim,
            3  # buy/hold/sell
        )
        
        self.volatility_head = TaskHead(
            self.encoder.output_dim,
            self.config.task_hidden_dim,
            1  # regression
        )
        
        self.regime_head = TaskHead(
            self.encoder.output_dim,
            self.config.task_hidden_dim,
            5  # 5 regimes
        )
        
        # Task weights (learnable for uncertainty weighting)
        self.log_task_weights = nn.Parameter(torch.zeros(3))
    
    def forward(self, x: torch.Tensor, task: str = None) -> Dict[str, torch.Tensor]:
        shared = self.encoder(x)
        
        outputs = {}
        
        if task is None or task == 'direction':
            outputs['direction'] = self.direction_head(shared)
        
        if task is None or task == 'volatility':
            outputs['volatility'] = self.volatility_head(shared)
        
        if task is None or task == 'regime':
            outputs['regime'] = self.regime_head(shared)
        
        return outputs
    
    def get_task_weights(self) -> torch.Tensor:
        """Get normalized task weights."""
        return torch.softmax(self.log_task_weights, dim=0)


class MultiTaskTrainer:
    """Trainer for multi-task learning."""
    
    def __init__(self, model: MultiTaskModel, config: MultiTaskConfig = None, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = config or MultiTaskConfig()
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Loss functions
        self.direction_loss = nn.CrossEntropyLoss()
        self.volatility_loss = nn.MSELoss()
        self.regime_loss = nn.CrossEntropyLoss()
        
        self._losses: List[Dict] = []
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Compute weighted multi-task loss."""
        losses = {}
        
        if 'direction' in outputs and 'direction' in targets:
            losses['direction'] = self.direction_loss(
                outputs['direction'], targets['direction']
            )
        
        if 'volatility' in outputs and 'volatility' in targets:
            losses['volatility'] = self.volatility_loss(
                outputs['volatility'].squeeze(), targets['volatility']
            )
        
        if 'regime' in outputs and 'regime' in targets:
            losses['regime'] = self.regime_loss(
                outputs['regime'], targets['regime']
            )
        
        # Weighted combination
        if self.config.task_weight_strategy == "uncertainty":
            task_weights = self.model.get_task_weights()
            total_loss = sum(
                losses[task] / (2 * torch.exp(self.model.log_task_weights[i])) + 
                self.model.log_task_weights[i] / 2
                for i, task in enumerate(['direction', 'volatility', 'regime'])
                if task in losses
            )
        else:
            total_loss = sum(losses.values())
        
        return total_loss, {k: v.item() for k, v in losses.items()}
    
    def train_step(self, features: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(features.to(self.device))
        targets_device = {k: v.to(self.device) for k, v in targets.items()}
        
        total_loss, task_losses = self.compute_loss(outputs, targets_device)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        
        self._losses.append(task_losses)
        
        return {
            'total_loss': total_loss.item(),
            **task_losses
        }
    
    def get_loss_summary(self) -> Dict:
        if not self._losses:
            return {}
        
        summary = {}
        for task in ['direction', 'volatility', 'regime']:
            task_losses = [l.get(task, 0) for l in self._losses[-100:]]
            if task_losses:
                summary[f'{task}_avg'] = np.mean(task_losses)
        
        return summary


def create_multi_task_model(input_dim: int, **kwargs) -> MultiTaskModel:
    """Create multi-task model."""
    config = MultiTaskConfig(**kwargs)
    return MultiTaskModel(input_dim, config)
