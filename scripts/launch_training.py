"""
HIMARI Layer 2 - Training Launcher Script
Main entry point for training with comprehensive monitoring and checkpoint management.

FIXED: Now implements REAL training with actual model, data loading, and gradient updates.
"""

import argparse
import sys
import os
import signal
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Optional, Dict
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training.monitoring import TrainingMonitor, MonitoringConfig
from src.models.baseline_mlp import BaselineMLP, create_baseline_model
from src.data.dataset import load_training_data, compute_class_weights

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GracefulKiller:
    """Handle CTRL+C and other shutdown signals gracefully."""
    
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        logger.info("\n\n[STOP] Interrupt received. Saving checkpoint and exiting gracefully...")
        self.kill_now = True


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from: {config_path}")
    return config


def setup_device(gpu_id: Optional[int] = None, force_cpu: bool = False) -> torch.device:
    """Setup training device (GPU or CPU)."""
    if force_cpu:
        device = torch.device('cpu')
        logger.warning("Forced CPU training (will be SLOW!)")
    elif gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.warning("No GPU available. Training on CPU (will be SLOW!)")
    
    return device


def evaluate_model(model: nn.Module, data_loader, criterion, device) -> Dict:
    """Evaluate model on validation/test data."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    
    # Compute per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    class_names = ['SELL', 'HOLD', 'BUY']
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[f'acc_{name.lower()}'] = (all_preds[mask] == i).mean()
    
    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy,
        **per_class_acc
    }


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    monitor: Optional[TrainingMonitor] = None,
    grad_clip: float = 1.0
) -> Dict:
    """Train for one epoch with REAL gradient updates."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        # Move to device
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Log step metrics
        global_step = epoch * len(train_loader) + batch_idx
        step_metrics = {
            'train_loss': loss.item(),
            'train_accuracy': (predicted == labels).float().mean().item(),
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        if monitor:
            monitor.log_metrics(step_metrics, step=global_step)
        
        # Periodic logging
        if batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} Acc: {step_metrics['train_accuracy']:.4f}"
            )
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    
    return {
        'train_loss': avg_loss,
        'train_accuracy': accuracy
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict,
    checkpoint_dir: str,
    is_best: bool = False,
    keep_last_n: int = 5
):
    """Save model checkpoint with automatic cleanup of old checkpoints."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"[SAVE] Saved checkpoint: {checkpoint_path}")

    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        logger.info(f"[BEST] New best model saved: {best_path}")

    # Also save latest
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)

    # Cleanup old checkpoints (keep only last N)
    try:
        import glob
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')))
        if len(checkpoints) > keep_last_n:
            for old_checkpoint in checkpoints[:-keep_last_n]:
                os.remove(old_checkpoint)
                logger.debug(f"[CLEANUP] Removed old checkpoint: {old_checkpoint}")
    except Exception as e:
        logger.warning(f"Failed to cleanup old checkpoints: {e}")


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer) -> int:
    """Load checkpoint and return starting epoch."""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    logger.info(f"Resumed from checkpoint: epoch {checkpoint['epoch']}")
    
    return start_epoch


def main():
    """Main training launcher with REAL training loop."""
    parser = argparse.ArgumentParser(description='HIMARI Layer 2 Training')
    
    # Training config
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to training config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    
    # GPU settings
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID (e.g., 0, 1)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training (not recommended)')
    
    # Monitoring settings
    parser.add_argument('--wandb-project', type=str, default='himari-layer2',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default='charithliyanage52-himari',
                        help='Weights & Biases entity/username')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    
    # Checkpoint settings
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Epochs between checkpoints')
    parser.add_argument('--fresh-start', action='store_true',
                        help='Start fresh training (ignore existing checkpoints)')
    
    # Data settings
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to training data directory')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    
    # Debug settings
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (verbose logging)')
    parser.add_argument('--test-run', action='store_true',
                        help='Quick test run (2 epochs)')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Print banner
    logger.info("=" * 80)
    logger.info("HIMARI Layer 2 - Training Launcher")
    logger.info("=" * 80)
    
    # Load configuration
    config_dict = load_config(args.config)
    
    # Override config with command line arguments
    num_epochs = args.epochs or config_dict.get('training', {}).get('num_epochs', 50)
    if args.test_run:
        num_epochs = 2
        logger.info("[TEST] TEST RUN MODE: Only 2 epochs")
    
    batch_size = args.batch_size or config_dict.get('training', {}).get('batch_size', 256)
    
    # Setup device
    device = setup_device(args.gpu, args.cpu)
    
    # =========================================================================
    # LOAD TRAINING DATA (REAL DATA - Issue #3 FIXED)
    # =========================================================================
    logger.info("\n[DATA] Loading training data...")
    try:
        train_loader, val_loader, test_loader, data_metadata = load_training_data(
            data_dir=args.data_dir,
            batch_size=batch_size,
            train_split=config_dict.get('data', {}).get('train_split', 0.8),
            val_split=config_dict.get('data', {}).get('val_split', 0.1),
            test_split=config_dict.get('data', {}).get('test_split', 0.1)
        )
        logger.info(f"[OK] Data loaded: {data_metadata['train_size']} train, "
                   f"{data_metadata['val_size']} val, {data_metadata['test_size']} test")
    except FileNotFoundError as e:
        logger.error(f"[ERROR] Data loading failed: {e}")
        logger.error("Run: python scripts/preprocess_training_data.py first")
        sys.exit(1)
    
    # =========================================================================
    # CREATE MODEL (REAL MODEL - Issue #2 FIXED)
    # =========================================================================
    logger.info("\n[BUILD] Creating model...")
    model = create_baseline_model(
        model_type="mlp",
        input_dim=data_metadata['feature_dim'],
        hidden_dims=(128, 64, 32),
        num_classes=data_metadata['num_classes'],
        dropout=0.3
    )
    model = model.to(device)
    logger.info(f"[OK] Model created: {model.count_parameters():,} parameters")
    
    # =========================================================================
    # SETUP TRAINING COMPONENTS
    # =========================================================================
    # Optimizer
    lr = config_dict.get('training', {}).get('initial_lr', 1e-4)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config_dict.get('optimization', {}).get('weight_decay', 0.01)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Loss function with class weights for imbalanced data
    # Load labels to compute weights
    labels = np.load(os.path.join(args.data_dir, 'labels.npy'))
    class_weights = compute_class_weights(labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"[OK] Class weights: {class_weights.cpu().numpy()}")
    
    # =========================================================================
    # SETUP MONITORING (Issue #1 FIXED - Real metrics)
    # =========================================================================
    monitoring_config = MonitoringConfig(
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_name or f"baseline-{datetime.now().strftime('%Y%m%d-%H%M')}",
        wandb_tags=['baseline', 'mlp'] + (['test-run'] if args.test_run else []),
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        log_dir='./logs'
    )
    
    monitor = TrainingMonitor(monitoring_config)
    
    # Log hyperparameters
    if monitor.wandb_run:
        monitor.wandb_run.config.update({
            'model': 'BaselineMLP',
            'epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealing',
            'train_samples': data_metadata['train_size'],
            'val_samples': data_metadata['val_size'],
            'feature_dim': data_metadata['feature_dim']
        })
    
    # =========================================================================
    # RESUME FROM CHECKPOINT (Issue #4 FIXED)
    # =========================================================================
    start_epoch = 0
    best_val_accuracy = 0.0

    if args.resume_from:
        start_epoch = load_checkpoint(args.resume_from, model, optimizer)
    elif not args.fresh_start and os.path.exists(os.path.join(args.checkpoint_dir, 'latest_checkpoint.pt')):
        # Auto-resume from latest (unless --fresh-start is set)
        logger.info("[RESUME] Found existing checkpoint. Resuming training...")
        logger.info("Use --fresh-start to ignore existing checkpoints")
        start_epoch = load_checkpoint(
            os.path.join(args.checkpoint_dir, 'latest_checkpoint.pt'),
            model, optimizer
        )
    elif args.fresh_start:
        logger.info("[FRESH] Starting fresh training (--fresh-start enabled)")
    
    # Setup graceful shutdown
    killer = GracefulKiller()
    
    # =========================================================================
    # TRAINING LOOP (Issue #1 & #5 FIXED - Real training)
    # =========================================================================
    logger.info("\n[INFO] Training Configuration:")
    logger.info(f"  Epochs: {num_epochs} (starting from {start_epoch})")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Device: {device}")
    logger.info(f"  W&B: {'Enabled' if monitoring_config.use_wandb else 'Disabled'}")
    
    if monitor.wandb_run:
        logger.info(f"  W&B Dashboard: {monitor.wandb_run.url}")
    
    logger.info("\n[START] Starting training...")
    logger.info("Press CTRL+C to interrupt and save checkpoint\n")
    
    try:
        for epoch in range(start_epoch, num_epochs):
            if killer.kill_now:
                logger.info("Interrupt detected. Saving checkpoint...")
                save_checkpoint(model, optimizer, epoch, {}, args.checkpoint_dir)
                break
            
            # ===== TRAIN EPOCH (REAL GRADIENTS) =====
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device,
                epoch, monitor, grad_clip=config_dict.get('optimization', {}).get('gradient_clip', 1.0)
            )
            
            # ===== EVALUATE =====
            val_metrics = evaluate_model(model, val_loader, criterion, device)
            
            # Update learning rate
            scheduler.step()
            
            # Log epoch metrics
            epoch_metrics = {
                **train_metrics,
                **val_metrics,
                'epoch': epoch,
                'learning_rate': scheduler.get_last_lr()[0]
            }
            
            if monitor:
                monitor.log_epoch_metrics(epoch_metrics, epoch)
            
            # Check for best model
            is_best = val_metrics['val_accuracy'] > best_val_accuracy
            if is_best:
                best_val_accuracy = val_metrics['val_accuracy']
            
            # Log epoch summary
            logger.info(
                f"Epoch {epoch}/{num_epochs-1} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Train Acc: {train_metrics['train_accuracy']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_accuracy']:.4f} | "
                f"{'[BEST]' if is_best else ''}"
            )
            
            # Save checkpoint (Issue #4 FIXED)
            if (epoch + 1) % args.checkpoint_interval == 0 or is_best:
                keep_n = config_dict.get('checkpoints', {}).get('keep_last_n', 5)
                save_checkpoint(model, optimizer, epoch, epoch_metrics, args.checkpoint_dir, is_best, keep_last_n=keep_n)
        
        # ===== FINAL EVALUATION =====
        logger.info("\n[DATA] Final evaluation on test set...")
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        logger.info(f"Test Accuracy: {test_metrics['val_accuracy']:.4f}")
        logger.info(f"Test Loss: {test_metrics['val_loss']:.4f}")
        
        # Save final model
        save_checkpoint(model, optimizer, num_epochs-1, test_metrics, args.checkpoint_dir)
        
        logger.info("\n[OK] Training completed successfully!")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"\n[ERROR] Training failed with error: {e}", exc_info=True)
        # Try to save emergency checkpoint
        try:
            save_checkpoint(model, optimizer, epoch, {}, args.checkpoint_dir)
            logger.info("Emergency checkpoint saved.")
        except Exception as save_error:
            logger.error(f"Failed to save emergency checkpoint: {save_error}")
        sys.exit(1)
    
    finally:
        # Cleanup
        monitor.finish()
        logger.info("=" * 80)
        logger.info("Training session ended")
        logger.info("=" * 80)


if __name__ == '__main__':
    main()