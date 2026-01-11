"""
HIMARI Layer 2 - Part K1: 3-Stage Curriculum Learning
Progressive difficulty learning for trading models.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterator, Optional
from torch.utils.data import Dataset, Sampler
import logging

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    """Configuration for 3-Stage Curriculum Learning."""
    stage1_epochs: int = 10              # Epochs for easy stage
    stage2_epochs: int = 15              # Epochs for medium stage
    stage3_epochs: int = 25              # Epochs for hard stage
    difficulty_metric: str = "volatility"  # How to measure difficulty
    stage1_percentile: float = 0.3       # Bottom 30% easiest
    stage2_percentile: float = 0.7       # Middle 30-70%
    warmup_epochs: int = 2               # Warmup within each stage
    anti_curriculum_ratio: float = 0.1   # Hard samples in easy stage


@dataclass
class DifficultyScore:
    """Difficulty metrics for a training sample."""
    volatility: float          
    regime_stability: float    
    signal_clarity: float      
    noise_ratio: float         
    composite: float           


class DifficultyScorer:
    """Score training samples by difficulty."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'volatility': 0.3,
            'regime_stability': 0.25,
            'signal_clarity': 0.25,
            'noise_ratio': 0.2
        }
        self._volatility_stats = {'mean': 0.02, 'std': 0.015}
        
    def score(self, features: np.ndarray, returns: np.ndarray,
             window: int = 20) -> DifficultyScore:
        volatility = np.std(returns) if len(returns) > 1 else 0.02
        vol_normalized = (volatility - self._volatility_stats['mean']) / self._volatility_stats['std']
        vol_score = self._sigmoid(vol_normalized)
        
        if len(returns) > 5:
            sign_changes = np.sum(np.diff(np.sign(returns)) != 0) / (len(returns) - 1)
            stability = 1 - sign_changes
        else:
            stability = 0.5
        stability_score = 1 - stability
        
        momentum = np.mean(returns) if len(returns) > 0 else 0
        clarity = min(1.0, abs(momentum) / 0.02)
        clarity_score = 1 - clarity
        
        if len(returns) > 1 and abs(np.mean(returns)) > 1e-10:
            noise_ratio = np.std(returns) / abs(np.mean(returns))
            noise_ratio = min(5.0, noise_ratio) / 5.0
        else:
            noise_ratio = 0.5
        
        composite = (
            self.weights['volatility'] * vol_score +
            self.weights['regime_stability'] * stability_score +
            self.weights['signal_clarity'] * clarity_score +
            self.weights['noise_ratio'] * noise_ratio
        )
        
        return DifficultyScore(
            volatility=vol_score,
            regime_stability=stability_score,
            signal_clarity=clarity_score,
            noise_ratio=noise_ratio,
            composite=composite
        )
    
    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))
    
    def fit_statistics(self, all_returns: List[np.ndarray]) -> None:
        volatilities = [np.std(r) for r in all_returns if len(r) > 1]
        if volatilities:
            self._volatility_stats['mean'] = np.mean(volatilities)
            self._volatility_stats['std'] = np.std(volatilities) + 1e-8


class CurriculumDataset(Dataset):
    """Dataset wrapper that supports curriculum learning."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray,
                returns_windows: List[np.ndarray], config: CurriculumConfig = None):
        self.features = features
        self.labels = labels
        self.returns_windows = returns_windows
        self.config = config or CurriculumConfig()
        
        self.scorer = DifficultyScorer()
        self.scorer.fit_statistics(returns_windows)
        
        self.difficulty_scores = []
        for i, returns in enumerate(returns_windows):
            score = self.scorer.score(features[i], returns)
            self.difficulty_scores.append(score.composite)
        
        self.difficulty_scores = np.array(self.difficulty_scores)
        self._compute_stage_indices()
        self._current_stage = 1
        
    def _compute_stage_indices(self) -> None:
        sorted_indices = np.argsort(self.difficulty_scores)
        n = len(sorted_indices)
        
        stage1_cutoff = int(n * self.config.stage1_percentile)
        stage2_cutoff = int(n * self.config.stage2_percentile)
        
        self.stage_indices = {
            1: sorted_indices[:stage1_cutoff],
            2: sorted_indices[stage1_cutoff:stage2_cutoff],
            3: sorted_indices[stage2_cutoff:]
        }
        
        n_hard_in_easy = int(len(self.stage_indices[1]) * self.config.anti_curriculum_ratio)
        if n_hard_in_easy > 0:
            hard_samples = sorted_indices[-n_hard_in_easy:]
            self.stage_indices[1] = np.concatenate([self.stage_indices[1], hard_samples])
    
    def set_stage(self, stage: int) -> None:
        if stage not in [1, 2, 3]:
            raise ValueError("Stage must be 1, 2, or 3")
        self._current_stage = stage
    
    def get_current_indices(self) -> np.ndarray:
        if self._current_stage == 1:
            return self.stage_indices[1]
        elif self._current_stage == 2:
            return np.concatenate([self.stage_indices[1], self.stage_indices[2]])
        else:
            return np.arange(len(self.features))
    
    def __len__(self) -> int:
        return len(self.get_current_indices())
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        actual_idx = self.get_current_indices()[idx]
        return (
            torch.tensor(self.features[actual_idx], dtype=torch.float32),
            torch.tensor(self.labels[actual_idx], dtype=torch.long)
        )


class CurriculumTrainer:
    """Training loop with 3-stage curriculum."""
    
    def __init__(self, model: nn.Module, dataset: CurriculumDataset,
                optimizer: torch.optim.Optimizer, config: CurriculumConfig = None):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.config = config or CurriculumConfig()
        self._current_epoch = 0
        self._current_stage = 1
        self._stage_history: List[Dict] = []
        
    def get_current_stage(self, epoch: int) -> int:
        if epoch < self.config.stage1_epochs:
            return 1
        elif epoch < self.config.stage1_epochs + self.config.stage2_epochs:
            return 2
        else:
            return 3
    
    def train_epoch(self, epoch: int) -> Dict:
        new_stage = self.get_current_stage(epoch)
        if new_stage != self._current_stage:
            logger.info(f"Curriculum: Stage {self._current_stage} → Stage {new_stage}")
            self._current_stage = new_stage
        
        self.dataset.set_stage(self._current_stage)
        
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=64, shuffle=True
        )
        
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for features, labels in loader:
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(1, n_batches)
        
        metrics = {
            'epoch': epoch,
            'stage': self._current_stage,
            'loss': avg_loss,
            'n_samples': len(self.dataset)
        }
        self._stage_history.append(metrics)
        return metrics
    
    def get_curriculum_summary(self) -> Dict:
        stages = {}
        for stage in [1, 2, 3]:
            stage_metrics = [m for m in self._stage_history if m['stage'] == stage]
            if stage_metrics:
                stages[f'stage_{stage}'] = {
                    'epochs': len(stage_metrics),
                    'final_loss': stage_metrics[-1]['loss']
                }
        return stages
