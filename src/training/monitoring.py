"""
HIMARI Layer 2 - Training Monitoring Infrastructure
Comprehensive monitoring with Weights & Biases integration for remote tracking.
"""

import os
import time
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# Optional W&B import - gracefully degrade if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Weights & Biases not installed. Install with: pip install wandb")


@dataclass
class MonitoringConfig:
    """Configuration for training monitoring."""
    # W&B settings
    use_wandb: bool = True
    wandb_project: str = "himari-layer2"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # Logging settings
    log_interval: int = 100  # Log every N steps
    checkpoint_interval: int = 3600  # Save checkpoint every N steps (every 6 hours ~= 360 steps if 1 min/step)
    
    # Storage settings
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Alert settings
    enable_alerts: bool = True
    alert_on_nan: bool = True
    alert_on_divergence: bool = True
    divergence_threshold: float = 10.0  # Loss > 10x initial = diverged
    
    # Performance tracking
    track_gpu: bool = True
    track_system: bool = True


class TrainingMonitor:
    """
    Comprehensive training monitoring with W&B integration.
    
    Features:
    - Real-time metric logging to W&B cloud dashboard
    - Automatic checkpoint management
    - GPU utilization tracking
    - Training progress estimation
    - Alert system for failures
    - Graceful degradation if W&B unavailable
    """
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.step_count = 0
        self.epoch_count = 0
        self.start_time = time.time()
        self.metrics_history = {}
        self.best_metric_value = float('-inf')
        self.initial_loss = None
        
        # Create directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B
        self.wandb_run = None
        if self.config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
        elif self.config.use_wandb and not WANDB_AVAILABLE:
            logger.warning("W&B requested but not available. Falling back to local logging.")
            self.config.use_wandb = False
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_run_name,
                tags=self.config.wandb_tags,
                config=vars(self.config),
                resume='allow'
            )
            logger.info(f"W&B initialized: {self.wandb_run.url}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.config.use_wandb = False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, commit: bool = True):
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Training step (uses internal counter if None)
            commit: Whether to commit to W&B immediately
        """
        if step is None:
            step = self.step_count
        else:
            self.step_count = step
        
        # Store in history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append((step, value))
        
        # Track initial loss for divergence detection
        if 'loss' in metrics and self.initial_loss is None:
            self.initial_loss = metrics['loss']
        
        # Check for anomalies
        self._check_anomalies(metrics)
        
        # Add system metrics
        if self.config.track_gpu:
            metrics.update(self._get_gpu_metrics())
        
        if self.config.track_system:
            metrics.update(self._get_system_metrics())
        
        # Log to W&B
        if self.config.use_wandb and self.wandb_run:
            try:
                wandb.log(metrics, step=step, commit=commit)
            except Exception as e:
                logger.error(f"Failed to log to W&B: {e}")
        
        # Console logging at intervals
        if step % self.config.log_interval == 0:
            self._print_metrics(metrics, step)
    
    def log_epoch_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log end-of-epoch metrics."""
        self.epoch_count = epoch
        metrics['epoch'] = epoch
        
        # Calculate epoch duration
        elapsed = time.time() - self.start_time
        metrics['epoch_duration'] = elapsed / (epoch + 1)
        
        self.log_metrics(metrics, commit=True)
    
    def _check_anomalies(self, metrics: Dict[str, float]):
        """Check for training anomalies and alert."""
        # Check for NaN
        if self.config.alert_on_nan:
            for key, value in metrics.items():
                if np.isnan(value) or np.isinf(value):
                    self._alert(f"NaN/Inf detected in {key}: {value}")
        
        # Check for divergence
        if self.config.alert_on_divergence and 'loss' in metrics and self.initial_loss:
            if metrics['loss'] > self.initial_loss * self.config.divergence_threshold:
                self._alert(f"Training divergence detected! Loss: {metrics['loss']:.4f} (initial: {self.initial_loss:.4f})")
    
    def _alert(self, message: str):
        """Send alert for critical issues."""
        logger.error(f"🚨 ALERT: {message}")
        
        if self.config.use_wandb and self.wandb_run:
            try:
                wandb.alert(
                    title="Training Alert",
                    text=message,
                    level=wandb.AlertLevel.ERROR
                )
            except Exception as e:
                logger.error(f"Failed to send W&B alert: {e}")
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU utilization metrics."""
        metrics = {}
        try:
            if torch.cuda.is_available():
                metrics['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
                metrics['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
                metrics['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
        except Exception as e:
            logger.debug(f"Failed to get GPU metrics: {e}")
        return metrics
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get system performance metrics."""
        metrics = {}
        elapsed = time.time() - self.start_time
        metrics['elapsed_time_hours'] = elapsed / 3600
        metrics['steps_per_second'] = self.step_count / elapsed if elapsed > 0 else 0
        return metrics
    
    def _print_metrics(self, metrics: Dict[str, float], step: int):
        """Print metrics to console."""
        elapsed = time.time() - self.start_time
        steps_per_sec = self.step_count / elapsed if elapsed > 0 else 0
        
        # Format key metrics
        metric_strs = []
        for key in ['loss', 'accuracy', 'sharpe_ratio', 'lr']:
            if key in metrics:
                metric_strs.append(f"{key}: {metrics[key]:.4f}")
        
        logger.info(
            f"Step {step} ({steps_per_sec:.1f} steps/s) | " + 
            " | ".join(metric_strs)
        )
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        is_best: bool = False,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            metrics: Current metrics
            is_best: Whether this is the best checkpoint so far
            metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'step': self.step_count,
            'epoch': self.epoch_count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        # Save regular checkpoint
        checkpoint_name = f"checkpoint_step_{self.step_count}.pt"
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save as best if applicable
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_checkpoint.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Best checkpoint updated: {best_path}")
            
            # Upload to W&B
            if self.config.use_wandb and self.wandb_run:
                try:
                    wandb.save(best_path)
                except Exception as e:
                    logger.error(f"Failed to upload checkpoint to W&B: {e}")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to restore state
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.step_count = checkpoint.get('step', 0)
        self.epoch_count = checkpoint.get('epoch', 0)
        
        logger.info(f"Checkpoint loaded from step {self.step_count}, epoch {self.epoch_count}")
        
        return checkpoint.get('metadata', {})
    
    def should_save_checkpoint(self) -> bool:
        """Check if it's time to save a checkpoint."""
        return self.step_count % self.config.checkpoint_interval == 0
    
    def is_best_model(self, metric_value: float, metric_name: str = 'sharpe_ratio') -> bool:
        """
        Check if current model is the best so far.
        
        Args:
            metric_value: Current metric value
            metric_name: Name of metric to track
            
        Returns:
            True if this is the best model
        """
        is_best = metric_value > self.best_metric_value
        if is_best:
            self.best_metric_value = metric_value
            logger.info(f"New best {metric_name}: {metric_value:.4f}")
        return is_best
    
    def estimate_time_remaining(self, total_steps: int) -> float:
        """
        Estimate remaining training time.
        
        Args:
            total_steps: Total training steps
            
        Returns:
            Estimated hours remaining
        """
        if self.step_count == 0:
            return float('inf')
        
        elapsed = time.time() - self.start_time
        steps_per_sec = self.step_count / elapsed
        remaining_steps = total_steps - self.step_count
        remaining_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
        
        return remaining_seconds / 3600  # Convert to hours
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        elapsed = time.time() - self.start_time
        
        summary = {
            'total_steps': self.step_count,
            'total_epochs': self.epoch_count,
            'elapsed_hours': elapsed / 3600,
            'steps_per_second': self.step_count / elapsed if elapsed > 0 else 0,
        }
        
        # Add recent metric averages
        for key, history in self.metrics_history.items():
            if len(history) > 0:
                recent_values = [v for _, v in history[-100:]]
                summary[f'{key}_recent_avg'] = np.mean(recent_values)
        
        return summary
    
    def finish(self):
        """Clean up and finalize monitoring."""
        summary = self.get_summary()
        
        logger.info("=" * 80)
        logger.info("Training Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
        
        # Save summary
        summary_path = os.path.join(self.config.log_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Finish W&B run
        if self.config.use_wandb and self.wandb_run:
            try:
                wandb.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.error(f"Failed to finish W&B run: {e}")
