#!/bin/bash
# HIMARI Layer 2 - Quick Start Training Script
# Run this on GPU instance after setup

set -e  # Exit on error

echo "=========================================="
echo "HIMARI Layer 2 - Training Launcher"
echo "=========================================="

# Check CUDA availability
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify data
echo ""
echo "Verifying training data..."
python scripts/verify_training_data.py --data-dir ./data

# Launch training
echo ""
echo "Starting training..."
echo "Press CTRL+C to stop (will save checkpoint)"
echo ""

python scripts/launch_training.py \
    --config configs/training_config.yaml \
    --fresh-start \
    --epochs 50 \
    --batch-size 256

echo ""
echo "=========================================="
echo "Training completed!"
echo "Check checkpoints/ for saved models"
echo "=========================================="
