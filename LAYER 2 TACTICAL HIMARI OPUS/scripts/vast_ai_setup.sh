#!/bin/bash
# HIMARI Layer 2 - Vast.ai GPU Deployment Script
# Run this on your rented Vast.ai A10 GPU instance

set -e  # Exit on error

echo "=============================================="
echo "HIMARI Layer 2 - Vast.ai Setup & Training"
echo "=============================================="

# Configuration
WANDB_ENTITY="charithliyanage52-himari"
WANDB_PROJECT="himari-layer2"
WORKSPACE="/workspace/himari-layer2"

# Step 1: System Update
echo ""
echo "[1/6] Updating system packages..."
apt-get update -qq
apt-get install -y -qq git screen htop

# Step 2: Install Python dependencies
echo ""
echo "[2/6] Installing Python dependencies..."
pip install -q torch numpy pandas scikit-learn
pip install -q wandb pyyaml tqdm
pip install -q stable-baselines3 transformers

# Step 3: Create workspace
echo ""
echo "[3/6] Setting up workspace..."
mkdir -p $WORKSPACE
cd $WORKSPACE

# Step 4: Copy code (user needs to do this manually or via git)
echo ""
echo "[4/6] Code setup..."
echo "Please copy your code to $WORKSPACE using one of:"
echo "  Option A: git clone <your-repo> ."
echo "  Option B: scp -r -P <PORT> local_path/* root@<IP>:$WORKSPACE/"
echo ""

# Step 5: Login to W&B
echo ""
echo "[5/6] Weights & Biases login..."
echo "Run: wandb login"
echo "Get API key from: https://wandb.ai/authorize"
echo ""

# Step 6: Launch training
echo ""
echo "[6/6] Ready to launch training!"
echo ""
echo "Commands:"
echo "=========="
echo ""
echo "# Quick test (5 min):"
echo "python scripts/launch_training.py --test-run --gpu 0"
echo ""
echo "# Full training (133 hours, ~\$22):"
echo "screen -S training"
echo "python scripts/launch_training.py \\"
echo "  --config configs/training_config.yaml \\"
echo "  --wandb-project $WANDB_PROJECT \\"
echo "  --wandb-entity $WANDB_ENTITY \\"
echo "  --gpu 0"
echo ""
echo "# Detach screen: Ctrl+A, D"
echo "# Reattach: screen -r training"
echo ""
echo "Monitor at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
echo "=============================================="
