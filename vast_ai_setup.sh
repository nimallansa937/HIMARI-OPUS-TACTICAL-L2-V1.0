#!/bin/bash
#
# HIMARI FLAG-TRADER - Vast.ai Training Setup Script
# Automatically installs dependencies, downloads data, and starts training
#

set -e  # Exit on error

echo "================================================================================"
echo "HIMARI FLAG-TRADER - Vast.ai Training Setup"
echo "================================================================================"
echo ""

# Check if running on GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. This script requires a GPU instance."
    echo "Please ensure you're running on a Vast.ai GPU instance."
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Update and install system dependencies
echo "Step 1: Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq wget git python3-pip > /dev/null 2>&1
echo "✓ System dependencies installed"
echo ""

# Install Python dependencies
echo "Step 2: Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers peft loguru numpy pandas scikit-learn gdown
echo "✓ Python dependencies installed"
echo ""

# Verify PyTorch CUDA
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Clone repository
echo "Step 3: Cloning HIMARI repository..."
if [ -d "HIMARI-OPUS-TACTICAL-L2-V1.0" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd HIMARI-OPUS-TACTICAL-L2-V1.0
    git pull
else
    git clone https://github.com/nimallansa937/HIMARI-OPUS-TACTICAL-L2-V1.0.git
    cd HIMARI-OPUS-TACTICAL-L2-V1.0
fi
echo "✓ Repository cloned"
echo ""

# Create data directory
mkdir -p data
mkdir -p checkpoints

# Download training data from Google Drive
echo "Step 4: Downloading training data from Google Drive..."
echo ""
echo "IMPORTANT: You need to provide your Google Drive file ID"
echo "If you haven't uploaded btc_1h_2020_2024.csv to Google Drive yet:"
echo "  1. Upload the file to Google Drive"
echo "  2. Right-click the file -> Get link -> Copy link"
echo "  3. Extract the file ID from the URL"
echo "  4. Edit this script and replace <YOUR_GOOGLE_DRIVE_FILE_ID>"
echo ""

# Replace with your Google Drive file ID
GDRIVE_FILE_ID="1da3Vv2o6pAtuLkvF8N3zPp8OK6Dm3RCB"

if [ "$GDRIVE_FILE_ID" = "<YOUR_GOOGLE_DRIVE_FILE_ID>" ]; then
    echo "ERROR: Please edit vast_ai_setup.sh and set your Google Drive file ID"
    echo ""
    echo "Alternatively, you can manually download the data:"
    echo "  1. Upload btc_1h_2020_2024.csv to Google Drive"
    echo "  2. Run: gdown --id YOUR_FILE_ID -O data/btc_1h_2020_2024.csv"
    echo "  3. Then run: python train_flagtrader.py --data data/btc_1h_2020_2024.csv --output checkpoints/ --epochs 10"
    echo ""
    exit 1
fi

# Download data
if [ ! -f "data/btc_1h_2020_2024.csv" ]; then
    gdown --id "$GDRIVE_FILE_ID" -O data/btc_1h_2020_2024.csv
    echo "✓ Training data downloaded"
else
    echo "✓ Training data already exists"
fi
echo ""

# Verify data
echo "Step 5: Verifying data..."
if [ ! -f "data/btc_1h_2020_2024.csv" ]; then
    echo "ERROR: Training data not found at data/btc_1h_2020_2024.csv"
    exit 1
fi

DATA_SIZE=$(wc -l < data/btc_1h_2020_2024.csv)
echo "Data file has $DATA_SIZE lines"

if [ "$DATA_SIZE" -lt 1000 ]; then
    echo "ERROR: Data file seems too small. Please check the download."
    exit 1
fi
echo "✓ Data verified"
echo ""

# Start training
echo "================================================================================"
echo "Step 6: Starting FLAG-TRADER training..."
echo "================================================================================"
echo ""
echo "Training configuration:"
echo "  - Data: data/btc_1h_2020_2024.csv"
echo "  - Output: checkpoints/"
echo "  - Epochs: 10"
echo "  - Batch size: 64"
echo "  - Learning rate: 1e-4"
echo "  - LoRA rank: 16"
echo ""
echo "Expected training time: 2-4 hours on RTX 3090"
echo ""

python train_flagtrader.py \
    --data data/btc_1h_2020_2024.csv \
    --output checkpoints/ \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-4 \
    --lora_rank 16

echo ""
echo "================================================================================"
echo "Training completed!"
echo "================================================================================"
echo ""
echo "Checkpoint saved to: checkpoints/flag_trader_best.pt"
echo ""
echo "To download the checkpoint from Vast.ai:"
echo "  1. Right-click on 'checkpoints/flag_trader_best.pt' in the file browser"
echo "  2. Click 'Download'"
echo "  3. Or use SCP: scp -P PORT root@HOST:~/HIMARI-OPUS-TACTICAL-L2-V1.0/checkpoints/flag_trader_best.pt ."
echo ""
echo "To run backtest on the trained model:"
echo "  python ensemble_backtest.py --confidence_threshold 0.7"
echo ""
