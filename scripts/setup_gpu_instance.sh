#!/bin/bash
# HIMARI Layer 2 GPU Instance Setup Script
# Run this script in the Jupyter terminal to set up the training environment

set -e

echo "=================================================="
echo "  HIMARI Layer 2 Training Environment Setup"
echo "=================================================="

# Navigate to workspace
cd /workspace

# Create project directory
echo "[1/6] Creating project directory..."
mkdir -p himari-layer2
cd himari-layer2

# Install Python dependencies
echo "[2/6] Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install wandb pyyaml numpy pandas scipy scikit-learn tqdm

# Verify GPU is available
echo "[3/6] Verifying GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Create directory structure
echo "[4/6] Creating directory structure..."
mkdir -p src/training configs data scripts docs

echo "[5/6] Setup complete! Next steps:"
echo "  1. Upload your training code files"
echo "  2. Run: wandb login"
echo "  3. Run: python scripts/launch_training.py"

echo "=================================================="
echo "  GPU Environment Ready!"
echo "=================================================="
