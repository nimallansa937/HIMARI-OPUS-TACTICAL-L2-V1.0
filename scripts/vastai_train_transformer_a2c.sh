#!/bin/bash
# HIMARI Layer 2 - Transformer-A2C Training on Vast.ai A10 GPU
# ================================================================
# This script runs the full Transformer-A2C training pipeline
# Expected runtime: ~1.5-2.5 hours for 500k steps on A10 GPU

set -e  # Exit on error

echo "========================================================"
echo "HIMARI Layer 2 - Transformer-A2C Training"
echo "GPU: NVIDIA A10 (24GB)"
echo "========================================================"

# Configuration
MAX_STEPS=${MAX_STEPS:-500000}
VAL_FREQUENCY=${VAL_FREQUENCY:-25000}
CHECKPOINT_FREQUENCY=${CHECKPOINT_FREQUENCY:-50000}
HIDDEN_DIM=${HIDDEN_DIM:-256}
NUM_LAYERS=${NUM_LAYERS:-4}
DROPOUT=${DROPOUT:-0.2}
OUTPUT_DIR=${OUTPUT_DIR:-./output/transformer_a2c_run}

# Install dependencies
echo "[1/5] Installing dependencies..."
pip install -q torch numpy wandb

# Check GPU availability
echo "[2/5] Checking GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Verify data exists
echo "[3/5] Checking training data..."
if [ -f "./data/training_data.pkl" ]; then
    echo "Training data found: ./data/training_data.pkl"
    DATA_PATH="./data/training_data.pkl"
elif [ -f "./data/btc_5min_2020_2024.pkl" ]; then
    echo "Training data found: ./data/btc_5min_2020_2024.pkl"
    DATA_PATH="./data/btc_5min_2020_2024.pkl"
else
    echo "No training data found. Using synthetic data for testing."
    DATA_PATH="SYNTHETIC"
fi

# Start training
echo "[4/5] Starting Transformer-A2C training..."
echo "  Max steps: $MAX_STEPS"
echo "  Validation frequency: $VAL_FREQUENCY"
echo "  Hidden dim: $HIDDEN_DIM"
echo "  Num layers: $NUM_LAYERS"
echo "  Output: $OUTPUT_DIR"
echo ""

if [ "$DATA_PATH" = "SYNTHETIC" ]; then
    python scripts/train_transformer_a2c.py \
        --synthetic \
        --synthetic_samples 100000 \
        --max_steps $MAX_STEPS \
        --val_frequency $VAL_FREQUENCY \
        --hidden_dim $HIDDEN_DIM \
        --num_layers $NUM_LAYERS \
        --dropout $DROPOUT \
        --device cuda \
        --output $OUTPUT_DIR
else
    python scripts/train_transformer_a2c.py \
        --data $DATA_PATH \
        --max_steps $MAX_STEPS \
        --val_frequency $VAL_FREQUENCY \
        --hidden_dim $HIDDEN_DIM \
        --num_layers $NUM_LAYERS \
        --dropout $DROPOUT \
        --device cuda \
        --output $OUTPUT_DIR
fi

# Package results
echo "[5/5] Packaging results..."
cd $OUTPUT_DIR
tar -czvf ../transformer_a2c_results.tar.gz .
cd ..

echo ""
echo "========================================================"
echo "Training Complete!"
echo "========================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "Packaged results: transformer_a2c_results.tar.gz"
echo ""
echo "To download results, use:"
echo "  scp user@instance:/workspace/transformer_a2c_results.tar.gz ."
echo "========================================================"
