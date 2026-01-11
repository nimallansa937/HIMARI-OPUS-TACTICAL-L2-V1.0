"""
HIMARI Layer 2 - Part K: Training Infrastructure
Complete training pipeline for all RL agents.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


from src.training.monitoring import TrainingMonitor
# ============================================================================
# K1: Experience Replay Buffer
# ============================================================================

@dataclass
class ReplayConfig:
    capacity: int = 100000
    batch_size: int = 256
    priority_alpha: float = 0.6
    priority_beta: float = 0.4

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay for sample-efficient learning."""
    
    def __init__(self, config=None):
        self.config = config or ReplayConfig()
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done, priority=1.0):
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.config.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority ** self.config.priority_alpha)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority ** self.config.priority_alpha
            
        self.position = (self.position + 1) % self.config.capacity
        
    def sample(self) -> Tuple:
        if len(self.buffer) < self.config.batch_size:
            return None
            
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), self.config.batch_size, p=probs)
        
        batch = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.config.priority_beta)
        weights /= weights.max()
        
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.config.priority_alpha
            
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# K2: Curriculum Learning Scheduler
# ============================================================================

class CurriculumScheduler:
    """Gradually increases task difficulty during training."""
    
    def __init__(self, stages: int = 5, warmup_steps: int = 1000):
        self.stages = stages
        self.warmup_steps = warmup_steps
        self.current_stage = 0
        self.step_count = 0
        self.stage_performance = []
        
    def step(self, performance: float):
        self.step_count += 1
        self.stage_performance.append(performance)
        
        # Advance stage if performance is good
        if len(self.stage_performance) >= self.warmup_steps:
            avg_perf = np.mean(self.stage_performance[-100:])
            if avg_perf > 0.7 and self.current_stage < self.stages - 1:
                self.current_stage += 1
                logger.info(f"Curriculum: Advanced to stage {self.current_stage}")
                
    def get_difficulty(self) -> float:
        """Returns difficulty multiplier [0.2, 1.0]."""
        return 0.2 + 0.8 * (self.current_stage / (self.stages - 1))
    
    def get_stage_config(self) -> Dict:
        """Returns stage-specific configuration."""
        return {
            'volatility_range': (0.01, 0.05 + 0.1 * self.current_stage),
            'noise_level': 0.1 * self.current_stage,
            'regime_complexity': self.current_stage + 1
        }


# ============================================================================
# K3: Multi-Agent Training Coordinator
# ============================================================================

class MultiAgentCoordinator:
    """Coordinates training across multiple agents."""
    
    def __init__(self, agents: Dict[str, nn.Module]):
        self.agents = agents
        self.optimizers = {
            name: optim.AdamW(agent.parameters(), lr=1e-4)
            for name, agent in agents.items()
        }
        self.losses = {name: [] for name in agents}
        self.best_checkpoints = {}
        
    def train_step(self, batch, agent_name: str) -> float:
        agent = self.agents[agent_name]
        optimizer = self.optimizers[agent_name]
        
        # Generic training step - specific implementation per agent type
        optimizer.zero_grad()
        # Loss computation would go here
        loss = torch.tensor(0.0, requires_grad=True)  # Placeholder
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        
        self.losses[agent_name].append(loss.item())
        return loss.item()
    
    def save_checkpoint(self, agent_name: str, path: str, metrics: Dict):
        agent = self.agents[agent_name]
        checkpoint = {
            'model_state': agent.state_dict(),
            'optimizer_state': self.optimizers[agent_name].state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, path)
        self.best_checkpoints[agent_name] = path
        
    def load_checkpoint(self, agent_name: str, path: str):
        checkpoint = torch.load(path)
        self.agents[agent_name].load_state_dict(checkpoint['model_state'])
        self.optimizers[agent_name].load_state_dict(checkpoint['optimizer_state'])


# ============================================================================
# K4: Hyperparameter Scheduler
# ============================================================================

class HyperparameterScheduler:
    """Dynamic hyperparameter scheduling during training."""
    
    def __init__(self, initial_lr: float = 1e-3, warmup_steps: int = 1000):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
    def get_lr(self) -> float:
        self.step_count += 1
        
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / 100000
            return self.initial_lr * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
    
    def get_exploration_rate(self) -> float:
        # Exponential decay of exploration
        return max(0.01, 1.0 * (0.995 ** self.step_count))


# ============================================================================
# K5: Distributed Training Manager
# ============================================================================

class DistributedTrainingManager:
    """Manages distributed training across multiple GPUs/nodes."""
    
    def __init__(self, world_size: int = 1, local_rank: int = 0):
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_distributed = world_size > 1
        self._initialized = False
        
    def setup(self, backend: str = 'nccl'):
        """Initialize distributed training environment."""
        if self.is_distributed and not self._initialized:
            try:
                import torch.distributed as dist
                if not dist.is_initialized():
                    dist.init_process_group(
                        backend=backend,
                        rank=self.local_rank,
                        world_size=self.world_size
                    )
                self._initialized = True
                logger.info(f"Distributed training initialized: rank {self.local_rank}/{self.world_size}")
            except Exception as e:
                logger.warning(f"Failed to initialize distributed training: {e}")
                self.is_distributed = False
            
    def all_reduce(self, tensor: torch.Tensor, op: str = 'sum') -> torch.Tensor:
        """All-reduce tensor across all processes."""
        if self.is_distributed and self._initialized:
            try:
                import torch.distributed as dist
                if op == 'sum':
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                elif op == 'mean':
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    tensor /= self.world_size
                elif op == 'max':
                    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            except Exception as e:
                logger.warning(f"all_reduce failed: {e}")
        return tensor
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_distributed and self._initialized:
            try:
                import torch.distributed as dist
                dist.barrier()
            except Exception as e:
                logger.warning(f"barrier failed: {e}")
    
    def is_main_process(self) -> bool:
        """Check if this is the main (rank 0) process."""
        return self.local_rank == 0
    
    def cleanup(self):
        """Clean up distributed training resources."""
        if self._initialized:
            try:
                import torch.distributed as dist
                dist.destroy_process_group()
                self._initialized = False
            except Exception:
                pass


# ============================================================================
# K6: Training Metrics Logger
# ============================================================================

class TrainingMetricsLogger:
    """Logs and aggregates training metrics."""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.metrics = {}
        self.step_count = 0
        self.start_time = time.time()
        
    def log(self, metrics: Dict):
        self.step_count += 1
        
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            
        if self.step_count % self.log_interval == 0:
            self._print_summary()
            
    def _print_summary(self):
        elapsed = time.time() - self.start_time
        steps_per_sec = self.step_count / elapsed
        
        summary = [f"Step {self.step_count} ({steps_per_sec:.1f} steps/s)"]
        for key, values in self.metrics.items():
            recent = values[-self.log_interval:]
            summary.append(f"{key}: {np.mean(recent):.4f}")
            
        logger.info(" | ".join(summary))
        
    def get_summary(self) -> Dict:
        return {key: np.mean(values[-100:]) for key, values in self.metrics.items()}


# ============================================================================
# Complete Training Pipeline
# ============================================================================

@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 1e-4
    num_epochs: int = 100
    checkpoint_interval: int = 1000
    device: str = 'cpu'

class TrainingPipeline:
    """Complete training infrastructure pipeline."""
    
    def __init__(self, config=None, monitor: Optional[TrainingMonitor] = None):
        self.config = config or TrainingConfig()
        self.replay_buffer = PrioritizedReplayBuffer(
            ReplayConfig(batch_size=self.config.batch_size)
        )
        self.curriculum = CurriculumScheduler()
        self.hp_scheduler = HyperparameterScheduler(initial_lr=self.config.learning_rate)
        self.monitor = monitor or TrainingMetricsLogger()
        self.step_count = 0
        
    def train_epoch(self, agent: nn.Module, optimizer: optim.Optimizer,
                   loss_fn, data_loader=None) -> Dict:
        """Train for one epoch."""
        agent.train()
        epoch_losses = []
        
        for _ in range(100):  # Batches per epoch
            batch = self.replay_buffer.sample()
            if batch is None:
                continue
                
            states, actions, rewards, next_states, dones, weights, indices = batch
            
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.hp_scheduler.get_lr()
            
            # Forward pass (placeholder)
            optimizer.zero_grad()
            loss = torch.tensor(0.0, requires_grad=True)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            self.step_count += 1
            
            # Update curriculum
            self.curriculum.step(1.0 - loss.item())
            
            # Log metrics
            metrics = {
                'loss': loss.item(),
                'lr': self.hp_scheduler.get_lr(),
                'exploration': self.hp_scheduler.get_exploration_rate(),
                'difficulty': self.curriculum.get_difficulty()
            }
            
            if isinstance(self.monitor, TrainingMonitor):
                self.monitor.log_metrics(metrics, step=self.step_count)
            else:
                self.monitor.log(metrics)
            
        return {
            'mean_loss': np.mean(epoch_losses) if epoch_losses else 0.0,
            'steps': self.step_count
        }
    
    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
