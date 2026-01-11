#!/bin/bash
# =============================================================================
# HIMARI Layer 2 - Transformer-A2C Training on Vast.ai
# =============================================================================
# Run this script on a Vast.ai GPU instance to train the model
#
# PREREQUISITES:
# - Vast.ai instance with GPU (RTX 3090/4090 recommended)
# - PyTorch CUDA template
#
# DATA: Will download from Google Drive automatically
# Google Drive File ID: 1_YMRsTCHjfsrqf63RI3xQ4jpehIsEaNW
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "HIMARI Layer 2 - Transformer-A2C Training"
echo "=========================================="

# Step 1: Clone the repository
echo "[1/6] Cloning repository..."
cd /root
if [ -d "HIMARI-OPUS-TACTICAL-L2-V1.0" ]; then
    echo "Repository already exists, pulling latest..."
    cd HIMARI-OPUS-TACTICAL-L2-V1.0
    git pull origin main
else
    git clone https://github.com/nimallansa937/HIMARI-OPUS-TACTICAL-L2-V1.0.git
    cd HIMARI-OPUS-TACTICAL-L2-V1.0
fi

# Step 2: Install dependencies
echo "[2/6] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn scipy gdown tqdm

# Step 3: Download BTC data from Google Drive
echo "[3/6] Downloading BTC data from Google Drive..."
mkdir -p "LAYER 2 TACTICAL HIMARI OPUS/data"
cd "LAYER 2 TACTICAL HIMARI OPUS/data"

# Google Drive file ID from the share link
FILE_ID="1_YMRsTCHjfsrqf63RI3xQ4jpehIsEaNW"
OUTPUT_FILE="btc_5min_2020_2024.pkl"

if [ -f "$OUTPUT_FILE" ]; then
    echo "Data file already exists, skipping download..."
else
    echo "Downloading from Google Drive (File ID: $FILE_ID)..."
    gdown --id "$FILE_ID" -O "$OUTPUT_FILE"
    echo "Download complete! File size: $(du -h $OUTPUT_FILE | cut -f1)"
fi

cd ../..

# Step 4: Verify data file
echo "[4/6] Verifying data file..."
python3 << 'EOF'
import pickle
import numpy as np

data_path = "LAYER 2 TACTICAL HIMARI OPUS/data/btc_5min_2020_2024.pkl"
try:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Type: {type(data)}")
    
    if isinstance(data, dict):
        for k, v in data.items():
            if hasattr(v, 'shape'):
                print(f"   {k}: shape={v.shape}, dtype={v.dtype}")
            elif hasattr(v, '__len__'):
                print(f"   {k}: len={len(v)}")
    elif hasattr(data, 'shape'):
        print(f"   Shape: {data.shape}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)
EOF

# Step 5: Set PYTHONPATH
echo "[5/6] Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:/root/HIMARI-OPUS-TACTICAL-L2-V1.0/LAYER 2 TACTICAL HIMARI OPUS"

# Step 6: Start training
echo "[6/6] Starting Transformer-A2C training..."
echo "=========================================="
echo "Training configuration:"
echo "  - Entropy coefficient: 0.07 (with decay)"
echo "  - Max steps: 100,000"
echo "  - Validation frequency: 25,000"
echo "  - Early stopping patience: 3"
echo "=========================================="

cd "LAYER 2 TACTICAL HIMARI OPUS"

python3 << 'EOF'
import os
import sys
import pickle
import numpy as np
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, '/root/HIMARI-OPUS-TACTICAL-L2-V1.0/LAYER 2 TACTICAL HIMARI OPUS')

from src.models.transformer_a2c import TransformerA2CConfig
from src.environment.transformer_a2c_env import TransformerA2CEnv, WalkForwardSplitter, TransformerEnvConfig
from src.training.transformer_a2c_trainer import train_transformer_a2c

# Load data
logger.info("Loading BTC data...")
data_path = "data/btc_5min_2020_2024.pkl"
with open(data_path, 'rb') as f:
    raw_data = pickle.load(f)

# Extract features and prices
if isinstance(raw_data, dict):
    if 'features' in raw_data and 'prices' in raw_data:
        features = raw_data['features']
        prices = raw_data['prices']
    elif 'data' in raw_data:
        features = raw_data['data']
        prices = raw_data.get('prices', raw_data.get('close', None))
    else:
        # Assume first array-like is features
        keys = list(raw_data.keys())
        features = raw_data[keys[0]]
        prices = raw_data[keys[1]] if len(keys) > 1 else None
else:
    features = raw_data
    prices = None

# Convert to numpy
features = np.array(features, dtype=np.float32)
if prices is not None:
    prices = np.array(prices, dtype=np.float32)
else:
    # Generate synthetic prices if not available
    logger.warning("No prices found in data, generating from first feature column")
    prices = features[:, 0] if features.ndim > 1 else features

logger.info(f"Data shape: features={features.shape}, prices={prices.shape}")

# Create train/val/test splits
splitter = WalkForwardSplitter(
    features, 
    prices,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Feature dimension from data
feature_dim = features.shape[1] if features.ndim > 1 else 1

# Create environments
env_config = TransformerEnvConfig(
    context_length=100,
    feature_dim=feature_dim,
)

train_env, val_env, test_env = splitter.create_envs(config=env_config)

# Training config (with fixes applied)
config = TransformerA2CConfig(
    input_dim=feature_dim,
    hidden_dim=256,
    num_heads=8,
    num_layers=4,
    context_length=100,
    entropy_coef=0.07,  # FIXED: increased from 0.01
    max_steps=100_000,
    val_frequency=25_000,
    patience=3,
)

logger.info("=" * 70)
logger.info("Starting training with FIXED configuration:")
logger.info(f"  - entropy_coef: {config.entropy_coef} (was 0.01)")
logger.info(f"  - Look-ahead bias: FIXED")
logger.info(f"  - Entropy decay: ENABLED (0.07 → 0.021)")
logger.info(f"  - Action distribution logging: ENABLED")
logger.info("=" * 70)

# Train model
result = train_transformer_a2c(
    train_env=train_env,
    val_env=val_env,
    config=config,
    device="cuda",
    output_dir="./output/transformer_a2c",
    use_wandb=False,
)

if result:
    logger.info("=" * 70)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info(f"Best checkpoint: {result['path']}")
    logger.info(f"Best validation Sharpe: {result['val_sharpe']:.4f}")
    logger.info("=" * 70)
else:
    logger.warning("Training completed but no best checkpoint found")

print("\n" + "=" * 70)
print("DONE! Checkpoints saved to: ./output/transformer_a2c/")
print("=" * 70)
EOF

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoints: /root/HIMARI-OPUS-TACTICAL-L2-V1.0/LAYER 2 TACTICAL HIMARI OPUS/output/transformer_a2c/"
echo "=========================================="
