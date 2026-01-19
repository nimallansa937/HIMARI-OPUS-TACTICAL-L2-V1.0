"""
HIMARI FLAG-TRADER Training Script for Vast.ai
Trains FLAG-TRADER model with balanced class weights to solve model collapse.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from typing import Tuple, Dict
import pickle

# Add model directory to path
models_path = Path(__file__).parent / "LAYER 2 TACTICAL HIMARI OPUS" / "src" / "models"
sys.path.insert(0, str(models_path))

from flag_trader import FLAGTRADERModel


class TradingDataset(Dataset):
    """Dataset for FLAG-TRADER training."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: (N, 60) feature array
            labels: (N,) label array (0=SELL, 1=HOLD, 2=BUY)
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess trading data.

    Returns:
        features: (N, 60) padded features
        labels: (N,) action labels
    """
    logger.info(f"Loading data from {data_path}...")

    # Check if preprocessed pickle exists
    pkl_path = data_path.replace('.csv', '_processed.pkl')
    if Path(pkl_path).exists():
        logger.info("Found preprocessed data, loading from pickle...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data['features'], data['labels']

    # Load CSV
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Extract OHLCV features (5 columns)
    ohlcv = df[['open', 'high', 'low', 'close', 'volume']].values

    # Compute basic technical indicators (44 features)
    features = compute_features(ohlcv)

    # Generate labels (simple momentum strategy)
    labels = generate_labels(df['close'].values)

    # Pad to 60D
    if features.shape[1] < 60:
        padding = np.zeros((features.shape[0], 60 - features.shape[1]))
        features = np.concatenate([features, padding], axis=1)

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Labels shape: {labels.shape}")

    # Calculate class distribution
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        logger.info(f"  Class {label}: {count} ({count/len(labels)*100:.1f}%)")

    # Save preprocessed data
    logger.info(f"Saving preprocessed data to {pkl_path}...")
    with open(pkl_path, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)

    return features, labels


def compute_features(ohlcv: np.ndarray) -> np.ndarray:
    """
    Compute technical indicators from OHLCV.

    Returns:
        features: (N, 49) feature array
    """
    close = ohlcv[:, 3]
    high = ohlcv[:, 1]
    low = ohlcv[:, 2]
    volume = ohlcv[:, 4]

    features = []

    # Price returns (5 features)
    for period in [1, 6, 24, 72, 168]:
        returns = np.zeros_like(close)
        returns[period:] = (close[period:] - close[:-period]) / close[:-period]
        features.append(returns)

    # Moving averages (4 features)
    for period in [12, 24, 72, 168]:
        ma = np.convolve(close, np.ones(period)/period, mode='same')
        ma_ratio = (close - ma) / ma
        features.append(ma_ratio)

    # Volatility (4 features)
    for period in [12, 24, 72, 168]:
        vol = np.zeros_like(close)
        for i in range(period, len(close)):
            vol[i] = np.std(close[i-period:i]) / close[i]
        features.append(vol)

    # RSI (14-period)
    rsi = compute_rsi(close, period=14)
    features.append(rsi)

    # Volume features (3 features)
    volume_ma = np.convolve(volume, np.ones(24)/24, mode='same')
    volume_ratio = volume / (volume_ma + 1e-8)
    features.append(volume_ratio)

    volume_std = np.zeros_like(volume)
    for i in range(24, len(volume)):
        volume_std[i] = np.std(volume[i-24:i]) / (volume[i] + 1e-8)
    features.append(volume_std)

    price_volume_corr = np.zeros_like(close)
    for i in range(24, len(close)):
        price_volume_corr[i] = np.corrcoef(close[i-24:i], volume[i-24:i])[0, 1]
    features.append(price_volume_corr)

    # High-low range (2 features)
    hl_range = (high - low) / close
    features.append(hl_range)

    hl_range_ma = np.convolve(hl_range, np.ones(24)/24, mode='same')
    features.append(hl_range_ma)

    # Momentum (3 features)
    for period in [12, 24, 72]:
        momentum = np.zeros_like(close)
        momentum[period:] = close[period:] - close[:-period]
        features.append(momentum / close)

    # Stack features (should be 49 total)
    features = np.column_stack(features)

    # Replace NaN/Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.convolve(gain, np.ones(period)/period, mode='same')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='same')

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    # Normalize to [-1, 1]
    rsi = (rsi - 50) / 50

    return rsi


def generate_labels(close: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Generate trading labels based on future returns.

    Args:
        close: Close prices
        threshold: Return threshold for BUY/SELL signals

    Returns:
        labels: 0=SELL, 1=HOLD, 2=BUY
    """
    # Look ahead 6 hours (reduced from 24 to increase signal frequency)
    lookahead = 6

    future_returns = np.zeros_like(close)
    future_returns[:-lookahead] = (close[lookahead:] - close[:-lookahead]) / close[:-lookahead]

    # Generate labels
    labels = np.ones(len(close), dtype=np.int64)  # Default: HOLD
    labels[future_returns > threshold] = 2  # BUY
    labels[future_returns < -threshold] = 0  # SELL

    return labels


def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: str,
                epoch: int) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.unsqueeze(1).to(device)  # (B, 1, 60)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward
        logits = model(features)  # (B, 1, 3)
        logits = logits.squeeze(1)  # (B, 3)

        # Loss
        loss = criterion(logits, labels)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 100 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                       f"Loss: {loss.item():.4f}, Acc: {correct/total*100:.2f}%")

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: str) -> Tuple[float, float, Dict]:
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Class-wise metrics
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.unsqueeze(1).to(device)
            labels = labels.to(device)

            logits = model(features).squeeze(1)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            # Class-wise accuracy
            for i in range(3):
                mask = (labels == i)
                class_correct[i] += ((pred == labels) & mask).sum().item()
                class_total[i] += mask.sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    class_acc = [class_correct[i] / (class_total[i] + 1e-8) for i in range(3)]

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'sell_acc': class_acc[0],
        'hold_acc': class_acc[1],
        'buy_acc': class_acc[2]
    }

    return avg_loss, accuracy, metrics


def main():
    parser = argparse.ArgumentParser(description="Train FLAG-TRADER model")
    parser.add_argument('--data', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--output', type=str, default='checkpoints/', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split')

    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    features, labels = load_data(args.data)

    # Train/val split
    n_samples = len(features)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val

    train_features = features[:n_train]
    train_labels = labels[:n_train]
    val_features = features[n_train:]
    val_labels = labels[n_train:]

    logger.info(f"Train samples: {n_train}, Val samples: {n_val}")

    # Calculate balanced class weights
    unique, counts = np.unique(train_labels, return_counts=True)
    total_samples = len(train_labels)
    class_weights = torch.tensor([total_samples / (len(unique) * count) for count in counts])
    class_weights = class_weights.to(device)

    logger.info(f"Class weights: SELL={class_weights[0]:.2f}, "
               f"HOLD={class_weights[1]:.2f}, BUY={class_weights[2]:.2f}")

    # Create datasets
    train_dataset = TradingDataset(train_features, train_labels)
    val_dataset = TradingDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    logger.info("Initializing FLAG-TRADER model...")
    model = FLAGTRADERModel(
        state_dim=60,
        action_dim=3,
        d_model=768,
        num_layers=12,
        num_heads=8,
        dim_feedforward=3072,
        lora_rank=args.lora_rank,
        dropout=0.1
    )
    model = model.to(device)

    # Loss with balanced class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer (only train LoRA parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*80}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")

        # Validate
        val_loss, val_acc, val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        logger.info(f"  SELL Acc: {val_metrics['sell_acc']*100:.2f}%")
        logger.info(f"  HOLD Acc: {val_metrics['hold_acc']*100:.2f}%")
        logger.info(f"  BUY Acc: {val_metrics['buy_acc']*100:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / 'flag_trader_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'config': {
                    'state_dim': 60,
                    'action_dim': 3,
                    'd_model': 768,
                    'num_layers': 12,
                    'num_heads': 8,
                    'dim_feedforward': 3072,
                    'lora_rank': args.lora_rank
                }
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")

        # Save latest
        latest_path = output_dir / f'flag_trader_epoch_{epoch}.pt'
        torch.save(model.state_dict(), latest_path)

        scheduler.step()

    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
