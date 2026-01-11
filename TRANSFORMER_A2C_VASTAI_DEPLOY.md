# HIMARI Layer 2 - Transformer-A2C Vast.ai Deployment

## Quick Start on Vast.ai

### 1. Rent an A10 GPU Instance

- Go to [Vast.ai](https://vast.ai/console/create/)
- Filter: RTX A10, 24GB VRAM, PyTorch image
- Recommended: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`

### 2. Upload Files

```bash
# Upload the zip to your instance
scp transformer_a2c_deploy.zip root@[IP]:/workspace/

# SSH into instance
ssh root@[IP]

# Extract
cd /workspace
unzip transformer_a2c_deploy.zip

# Install requirements
pip install -r transformer_a2c_requirements.txt
```

### 3. Upload Your Training Data

```bash
# From local machine - upload your BTC data
scp btc_5min_2020_2024.pkl root@[IP]:/workspace/data/training_data.pkl
```

### 4. Run Training

```bash
# Option A: Use the deployment script
chmod +x scripts/vastai_train_transformer_a2c.sh
./scripts/vastai_train_transformer_a2c.sh

# Option B: Run directly with custom params
python scripts/train_transformer_a2c.py \
    --data ./data/training_data.pkl \
    --device cuda \
    --max_steps 500000 \
    --output ./output/transformer_a2c_v1
```

### 5. Download Results

```bash
# From local machine
scp root@[IP]:/workspace/output/transformer_a2c_v1/*.pt ./checkpoints/
```

## Expected Training Times on A10 GPU

| Steps | Time | Use Case |
|-------|------|----------|
| 100k | ~25 min | Quick test |
| 500k | ~2 hours | **Recommended** |
| 1M | ~4 hours | Extended training |

## Output Files

After training, you'll find:

- `checkpoint_XXXXX_best.pt` - Best model by validation Sharpe
- `checkpoint_XXXXX.pt` - Regular checkpoints
- `training_summary.json` - Training metrics and config

## Environment Variables (Optional)

```bash
export MAX_STEPS=500000
export VAL_FREQUENCY=25000
export HIDDEN_DIM=256
export NUM_LAYERS=4
./scripts/vastai_train_transformer_a2c.sh
```
