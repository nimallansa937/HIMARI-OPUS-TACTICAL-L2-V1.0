"""
HIMARI Layer 2 - FinRL-DT Pipeline
Subsystem D: Decision Engine (Method D10)

Purpose:
    Training infrastructure for Decision Transformer models.
    Handles trajectory processing, return-to-go computation, and batch sampling.

Performance:
    Enables D1-D4 offline RL training
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
from collections import deque
import pickle


logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """Single trading trajectory."""
    states: np.ndarray           # (T, state_dim)
    actions: np.ndarray          # (T,) action indices
    rewards: np.ndarray          # (T,) per-step rewards
    returns_to_go: np.ndarray    # (T,) cumulative future returns
    timesteps: np.ndarray        # (T,) timestep indices
    
    def __len__(self) -> int:
        return len(self.states)


@dataclass
class TrajectoryDatasetConfig:
    """Configuration for trajectory dataset."""
    context_length: int = 100        # Sequence length for transformer
    gamma: float = 0.99              # Discount factor
    scale_rewards: bool = True       # Normalize rewards
    reward_scale: float = 100.0      # Reward scaling factor
    normalize_states: bool = True    # Z-score normalize states


class TrajectoryDataset(Dataset):
    """
    Dataset of trading trajectories for Decision Transformer training.
    
    Key responsibilities:
    1. Compute return-to-go for each timestep
    2. Handle trajectory segmentation for context windows
    3. Normalize states and scale rewards
    4. Efficient batch sampling for training
    
    The return-to-go is computed as:
        R_t = Σ_{t'>=t} γ^(t'-t) * r_{t'}
    
    This enables conditioning the model on desired future returns.
    """
    
    def __init__(self, config: Optional[TrajectoryDatasetConfig] = None):
        self.config = config or TrajectoryDatasetConfig()
        
        self.trajectories: List[Trajectory] = []
        self._total_samples = 0
        
        # For state normalization
        self._state_mean: Optional[np.ndarray] = None
        self._state_std: Optional[np.ndarray] = None
        
        # Index mapping: sample_idx → (traj_idx, start_idx)
        self._index_map: List[Tuple[int, int]] = []
        
    def _compute_returns_to_go(self, rewards: np.ndarray) -> np.ndarray:
        """
        Compute discounted return-to-go for each timestep.
        
        R_t = r_t + γ*R_{t+1}
        """
        returns_to_go = np.zeros_like(rewards)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.gamma * running_return
            returns_to_go[t] = running_return
        
        if self.config.scale_rewards:
            returns_to_go = returns_to_go / self.config.reward_scale
        
        return returns_to_go
    
    def add_trajectory(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray
    ) -> None:
        """
        Add a new trajectory to the dataset.
        
        Args:
            states: (T, state_dim) state observations
            actions: (T,) action indices (0, 1, 2 for SELL, HOLD, BUY)
            rewards: (T,) per-step rewards (typically returns)
        """
        assert len(states) == len(actions) == len(rewards)
        
        # Compute return-to-go
        returns_to_go = self._compute_returns_to_go(rewards)
        timesteps = np.arange(len(states))
        
        traj = Trajectory(
            states=states.astype(np.float32),
            actions=actions.astype(np.int64),
            rewards=rewards.astype(np.float32),
            returns_to_go=returns_to_go.astype(np.float32),
            timesteps=timesteps.astype(np.int64)
        )
        
        traj_idx = len(self.trajectories)
        self.trajectories.append(traj)
        
        # Update index map for random sampling
        for start in range(len(traj) - self.config.context_length + 1):
            self._index_map.append((traj_idx, start))
        
        self._total_samples = len(self._index_map)
        
    def compute_statistics(self) -> None:
        """Compute state normalization statistics from all trajectories."""
        if len(self.trajectories) == 0:
            return
        
        all_states = np.concatenate([t.states for t in self.trajectories], axis=0)
        self._state_mean = all_states.mean(axis=0)
        self._state_std = all_states.std(axis=0) + 1e-6
        
        logger.info(f"Computed statistics over {len(all_states)} samples")
    
    def _normalize_states(self, states: np.ndarray) -> np.ndarray:
        """Normalize states using pre-computed statistics."""
        if not self.config.normalize_states or self._state_mean is None:
            return states
        return (states - self._state_mean) / self._state_std
    
    def __len__(self) -> int:
        return self._total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, start_idx = self._index_map[idx]
        traj = self.trajectories[traj_idx]
        
        end_idx = start_idx + self.config.context_length
        
        states = self._normalize_states(traj.states[start_idx:end_idx])
        
        return {
            'states': torch.from_numpy(states),
            'actions': torch.from_numpy(traj.actions[start_idx:end_idx]),
            'rewards': torch.from_numpy(traj.rewards[start_idx:end_idx]),
            'returns_to_go': torch.from_numpy(traj.returns_to_go[start_idx:end_idx]),
            'timesteps': torch.from_numpy(traj.timesteps[start_idx:end_idx]),
            'attention_mask': torch.ones(self.config.context_length)
        }
    
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch for training."""
        indices = np.random.randint(0, len(self), size=batch_size)
        
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'returns_to_go': [],
            'timesteps': [],
            'attention_mask': []
        }
        
        for idx in indices:
            sample = self[idx]
            for key in batch:
                batch[key].append(sample[key])
        
        return {k: torch.stack(v) for k, v in batch.items()}
    
    def save(self, path: str) -> None:
        """Save dataset to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'trajectories': self.trajectories,
                'config': self.config,
                'state_mean': self._state_mean,
                'state_std': self._state_std
            }, f)
        logger.info(f"Dataset saved to {path}")
    
    def load(self, path: str) -> None:
        """Load dataset from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.trajectories = data['trajectories']
        self.config = data['config']
        self._state_mean = data['state_mean']
        self._state_std = data['state_std']
        
        # Rebuild index map
        self._index_map = []
        for traj_idx, traj in enumerate(self.trajectories):
            for start in range(len(traj) - self.config.context_length + 1):
                self._index_map.append((traj_idx, start))
        
        self._total_samples = len(self._index_map)
        logger.info(f"Dataset loaded: {len(self.trajectories)} trajectories, {self._total_samples} samples")


@dataclass
class FinRLDTPipelineConfig:
    """Configuration for FinRL-DT training pipeline."""
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    eval_interval: int = 10
    checkpoint_interval: int = 20
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000


class FinRLDTPipeline:
    """
    Complete training pipeline for Decision Transformer models.
    
    Responsibilities:
    1. Data loading and preprocessing
    2. Training loop with gradient clipping
    3. Evaluation and checkpointing
    4. Logging and progress tracking
    """
    
    def __init__(
        self, 
        model: nn.Module,
        dataset: TrajectoryDataset,
        config: Optional[FinRLDTPipelineConfig] = None,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.config = config or FinRLDTPipelineConfig()
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / self.config.warmup_steps)
        )
        
        self._step = 0
        self._best_loss = float('inf')
        self._history: Dict[str, List[float]] = {
            'train_loss': [],
            'eval_loss': []
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        self.optimizer.zero_grad()
        
        if hasattr(self.model, 'compute_loss'):
            loss_dict = self.model.compute_loss(**batch)
            loss = loss_dict.get('total', loss_dict.get('loss'))
        else:
            # Generic forward
            logits = self.model(
                states=batch['states'],
                actions=batch['actions'],
                returns_to_go=batch['returns_to_go'],
                timesteps=batch['timesteps']
            )
            loss = nn.functional.cross_entropy(
                logits.view(-1, 3), 
                batch['actions'].view(-1)
            )
            loss_dict = {'loss': loss.item()}
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        
        self._step += 1
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
    def train(self, num_steps: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Run training loop.
        
        Args:
            num_steps: Number of training steps (default: full epochs)
        """
        if num_steps is None:
            num_steps = len(self.dataset) // self.config.batch_size * self.config.num_epochs
        
        logger.info(f"Starting training for {num_steps} steps")
        
        for step in range(num_steps):
            batch = self.dataset.sample_batch(self.config.batch_size)
            losses = self.train_step(batch)
            
            self._history['train_loss'].append(losses.get('loss', losses.get('total', 0)))
            
            if step % 100 == 0:
                logger.info(f"Step {step}/{num_steps}: loss={losses}")
            
            if step % self.config.checkpoint_interval == 0 and step > 0:
                self._save_checkpoint(step)
        
        return self._history
    
    def _save_checkpoint(self, step: int) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'step': step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self._history
        }
        path = f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self._step = checkpoint['step']
        self._history = checkpoint['history']
        logger.info(f"Checkpoint loaded from {path}")


def create_dt_training_pipeline(
    model: nn.Module,
    trajectories: List[Dict],
    device: str = 'cuda'
) -> FinRLDTPipeline:
    """
    Factory function to create training pipeline.
    
    Args:
        model: Decision Transformer model
        trajectories: List of dicts with 'states', 'actions', 'rewards'
        device: Training device
    
    Returns:
        Configured training pipeline
    """
    dataset = TrajectoryDataset()
    
    for traj in trajectories:
        dataset.add_trajectory(
            states=np.array(traj['states']),
            actions=np.array(traj['actions']),
            rewards=np.array(traj['rewards'])
        )
    
    dataset.compute_statistics()
    
    return FinRLDTPipeline(model, dataset, device=device)
