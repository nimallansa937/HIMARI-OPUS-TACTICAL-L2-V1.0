# VAST.AI DEPLOYMENT GUIDE - HIMARI Layer 2 V1

**Date**: 2026-01-03
**Status**: Complete troubleshooting guide for GPU training
**Reference**: https://docs.vast.ai/documentation/get-started

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Common Errors & Solutions](#common-errors--solutions)
3. [Pre-Deployment Checklist](#pre-deployment-checklist)
4. [Step-by-Step Deployment](#step-by-step-deployment)
5. [Monitoring & Debugging](#monitoring--debugging)
6. [Cost Optimization](#cost-optimization)

---

## Quick Start

### Minimum Instance Requirements

| Requirement | Specification | Why |
|-------------|--------------|-----|
| **GPU** | NVIDIA A10 or better | FLAG-TRADER needs 24GB VRAM |
| **VRAM** | 24GB minimum | 88M parameter model |
| **Storage** | 20GB+ | Models + checkpoints |
| **CUDA** | 11.8+ | PyTorch 2.0+ compatibility |

### One-Command Setup

```bash
# 1. Upload package (via Vast.ai web interface)
# 2. SSH into instance
# 3. Run setup:

cd /workspace && \
pip install -r requirements.txt && \
python scripts/run_all_training.py --device cuda --wandb-entity charithliyanage52-himari
```

---

## Common Errors & Solutions

### ❌ ERROR 1: API Mismatch - "not enough values to unpack"

**Error Message**:
```
ValueError: not enough values to unpack (expected 5, got 4)
  at train_features, train_labels, val_features, val_labels, metadata = load_training_data(args.data_dir)
```

**Root Cause**: Training script expected raw numpy arrays, but data loader returned PyTorch DataLoaders.

**Solution**: ✓ FIXED in `scripts/train_all_models.py`
- Created `load_raw_data()` helper function (line 61-95)
- Replaced all `load_training_data()` calls
- Manually splits data 80/10/10

**Prevention**: Always verify function return signatures match calling code.

---

### ❌ ERROR 2: Missing Dependencies

**Error Messages**:
```
ModuleNotFoundError: No module named 'loguru'
ModuleNotFoundError: No module named 'filterpy'
ModuleNotFoundError: No module named 'gym'
ModuleNotFoundError: No module named 'sklearn'
```

**Root Cause**: `requirements.txt` was incomplete.

**Solution**: ✓ FIXED in `requirements.txt`
```
loguru>=0.7.0          # Logging framework
filterpy>=1.4.5        # Kalman filter (EKF preprocessing)
gym>=0.26.0            # RL environment
scikit-learn>=1.3.0    # ML utilities
```

**Prevention**:
1. Test locally with fresh virtual environment
2. Run `pip freeze > requirements.txt` after development
3. Include all imports used in preprocessing modules

**Vast.ai Tip**: Use their pre-built PyTorch images to avoid CUDA/driver issues:
```
# In Vast.ai template selection:
Image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
```

---

### ❌ ERROR 3: Training Exits After 1 Epoch

**Symptoms**:
```
21:11:01 - Starting CQL training for 100 epochs...
21:11:04 - Epoch 0/100: Loss=9.7103
21:11:05 - CQL training complete  ← WRONG! Only 1 epoch
```

**Root Causes**:
1. Silent exception in training loop
2. Missing progress logging
3. No error handling

**Solution**: ✓ FIXED in `scripts/train_all_models.py` (lines 170-203)
```python
try:
    for epoch in range(args.epochs):
        # Training code...
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{args.epochs}: ...")  # Progress tracking

    logger.info(f"Completed all {args.epochs} epochs successfully!")  # Confirmation

except Exception as e:
    logger.error(f"Training failed at epoch {epoch}: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    raise
```

**Expected Output** (after fix):
```
21:11:01 - Starting CQL training for 100 epochs...
21:11:04 - Epoch 0/100: Loss=9.7103, CQL Loss=3.4922
21:11:07 - Epoch 10/100: Loss=8.2341, CQL Loss=2.9871
21:11:10 - Epoch 20/100: Loss=7.5623, CQL Loss=2.6543
...
21:16:48 - Completed all 100 epochs successfully!
```

**Prevention**:
1. Always wrap training loops in try-except
2. Log progress with `{current}/{total}` format
3. Add completion confirmation messages

---

### ❌ ERROR 4: No Checkpoints Saved

**Symptoms**:
```bash
$ ls checkpoints/
baseline_mlp.pt  # Old checkpoint from 16:02
cql_agent.pt     # Old checkpoint from 16:02
# No new checkpoints after 21:08 training!
```

**Root Cause**: Training functions completed without calling `model.save()`.

**Solution**: ✓ FIXED - Added checkpoint saving to all 5 models

**Implementation**:
```python
# All training functions now use this pattern:
best_loss = float('inf')

for epoch in range(args.epochs):
    # Training...

    # Save best model
    if loss < best_loss:
        best_loss = loss
        checkpoint_path = Path(args.checkpoint_dir) / f"{model_name}_best.pt"
        save_model(model, checkpoint_path)  # Helper function (line 49-58)
        logger.info(f"Saved best model at epoch {epoch}")

# Save final model
final_path = Path(args.checkpoint_dir) / f"{model_name}_final.pt"
save_model(model, final_path)
```

**Expected Checkpoints** (10 files total):
```
checkpoints/
├── baseline_best.pt       ← Best BaselineMLP by loss
├── baseline_final.pt      ← Final BaselineMLP
├── cql_best.pt           ← Best CQL by loss
├── cql_final.pt          ← Final CQL
├── ppo_best.pt           ← Best PPO by reward
├── ppo_final.pt          ← Final PPO
├── cgdt_best.pt          ← Best CGDT by loss
├── cgdt_final.pt         ← Final CGDT
├── flag_trader_best.pt   ← Best FLAG-TRADER by loss
└── flag_trader_final.pt  ← Final FLAG-TRADER
```

**Prevention**:
1. Add checkpoint saving during development
2. Test with `--epochs 2` to verify saves happen
3. Check checkpoint directory size grows during training

---

### ❌ ERROR 5: Insufficient Logging

**Problem**: CQL logged every 100 epochs, so only epoch 0 visible.

**Before**:
```python
if epoch % 100 == 0:  # Only logs: 0, 100, 200...
    logger.info(f"Epoch {epoch}: Loss={loss:.4f}")
```

**After**: ✓ FIXED
```python
if epoch % 10 == 0:  # Logs: 0, 10, 20, 30... (10 entries)
    logger.info(f"Epoch {epoch}/{args.epochs}: Loss={loss:.4f}")
```

**Best Practice**: Log ~10-20 times per training run
- **Short runs** (<50 epochs): `epoch % 5 == 0`
- **Medium runs** (50-200 epochs): `epoch % 10 == 0`
- **Long runs** (>200 epochs): `epoch % 20 == 0`

---

### ❌ ERROR 6: CUDA Out of Memory

**Error Message**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
(GPU 0; 23.69 GiB total capacity; 21.50 GiB already allocated)
```

**Solutions**:

1. **Reduce Batch Size** (most common fix):
```python
# In train_all_models.py, reduce these:
--batch-size 128     # Instead of 256
--context-length 128 # Instead of 256 (for FLAG-TRADER)
```

2. **Use Gradient Accumulation**:
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. **Enable Mixed Precision** (requires code changes):
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Vast.ai Tip**: Choose instance with sufficient VRAM:
- **BaselineMLP, CQL, PPO**: 8GB VRAM sufficient
- **CGDT**: 12GB VRAM recommended
- **FLAG-TRADER**: 24GB VRAM required (A10/A6000)

---

### ❌ ERROR 7: Connection Lost / Training Interrupted

**Vast.ai Issue**: SSH connection drops during long training.

**Solutions**:

1. **Use `tmux` or `screen`** (CRITICAL for multi-hour training):
```bash
# Start tmux session
tmux new -s himari_training

# Run training
python scripts/run_all_training.py --device cuda

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t himari_training
```

2. **Run in background with nohup**:
```bash
nohup python scripts/run_all_training.py --device cuda > training.log 2>&1 &

# Monitor progress:
tail -f training.log
```

3. **Enable Auto-Restart** (in `run_all_training.py`):
```python
# Add to training loop:
try:
    model.train()
except KeyboardInterrupt:
    logger.info("Training interrupted, saving checkpoint...")
    model.save(f"{checkpoint_dir}/interrupted.pt")
    raise
```

**Vast.ai Console**: Use "Open Shell" instead of SSH for more stable connection.

---

## Pre-Deployment Checklist

### ✅ Before Uploading to Vast.ai

- [ ] **Test locally with `--epochs 2 --device cpu`**
  ```bash
  python scripts/train_all_models.py --model baseline --epochs 2 --device cpu
  ```
  - Verifies: imports work, data loads, checkpoints save

- [ ] **Verify all dependencies in `requirements.txt`**
  ```bash
  # Fresh environment test:
  python -m venv test_env
  source test_env/bin/activate  # or test_env\Scripts\activate on Windows
  pip install -r requirements.txt
  python -c "from src.models import *; from src.preprocessing import *"
  ```

- [ ] **Check data files exist**
  ```bash
  ls -lh data/
  # Should show:
  # preprocessed_features.npy  (~49MB for 103k samples × 60D)
  # labels.npy                 (~405KB)
  # metadata.json              (~1KB)
  ```

- [ ] **Estimate storage needs**
  ```
  Data:         ~50 MB
  Checkpoints:  ~2 GB (10 models × ~200MB each)
  Code:         ~10 MB
  Logs:         ~100 MB
  Total:        ~3 GB minimum
  ```

- [ ] **Configure W&B (optional but recommended)**
  ```bash
  wandb login
  # Or set in script: export WANDB_API_KEY=<your_key>
  ```

---

## Step-by-Step Deployment

### Step 1: Create Vast.ai Instance

1. **Go to**: https://cloud.vast.ai/
2. **Search for instances**:
   - GPU: `NVIDIA A10` or `RTX A6000`
   - VRAM: ≥24GB
   - Storage: ≥20GB
   - Sort by: `$ per hr` (ascending)

3. **Select template**:
   ```
   Image: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
   Disk Space: 30GB
   ```

4. **Rent instance** (example costs):
   - A10 (24GB): ~$0.30/hr
   - RTX 3090 (24GB): ~$0.20/hr
   - Training time: ~27 hours
   - **Total cost**: $5-8 for complete training

### Step 2: Upload Package

**Option A: Direct Upload** (via Vast.ai web interface)
1. Zip your package locally:
   ```bash
   cd "LAYER 2 TACTICAL HIMARI OPUS"
   zip -r himari_l2v1.zip . -x "*.pyc" "**/__pycache__/*" ".git/*"
   ```

2. Upload via Vast.ai file manager
3. SSH and unzip:
   ```bash
   cd /workspace
   unzip himari_l2v1.zip
   ```

**Option B: Git Clone** (recommended for version control)
```bash
# On Vast.ai instance:
git clone https://github.com/yourusername/himari-l2v1.git
cd himari-l2v1
```

**Option C: rsync** (for large files)
```bash
# From local machine:
rsync -avz --progress \
  "LAYER 2 TACTICAL HIMARI OPUS/" \
  root@<vast-instance-ip>:/workspace/himari/
```

### Step 3: Install Dependencies

```bash
cd /workspace/himari

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: CUDA available: True

# Install requirements
pip install -r requirements.txt

# Verify installations
python -c "import loguru, filterpy, gym, sklearn; print('All imports OK')"
```

### Step 4: Verify Data

```bash
# Check data files
ls -lh data/
python -c "import numpy as np; f=np.load('data/preprocessed_features.npy'); print(f'Data shape: {f.shape}')"
# Should print: Data shape: (103604, 60)
```

### Step 5: Start Training

**Option A: All Models Sequentially**
```bash
tmux new -s training

python scripts/run_all_training.py \
  --device cuda \
  --wandb-entity charithliyanage52-himari

# Detach: Ctrl+B, then D
```

**Option B: Single Model Test**
```bash
# Quick test (2 epochs):
python scripts/train_all_models.py \
  --model baseline \
  --epochs 2 \
  --device cuda

# Full training:
python scripts/train_all_models.py \
  --model flag-trader \
  --epochs 30 \
  --context-length 256 \
  --device cuda
```

**Option C: Specific Models Only**
```bash
python scripts/run_all_training.py \
  --models baseline cql cgdt \
  --device cuda
```

### Step 6: Monitor Progress

**A. Real-time Logs**:
```bash
# If using tmux:
tmux attach -t training

# If using nohup:
tail -f training.log
```

**B. Weights & Biases**:
- Go to: https://wandb.ai/charithliyanage52-himari
- View: Loss curves, GPU utilization, training speed

**C. Checkpoint Size**:
```bash
watch -n 60 "du -sh checkpoints/*"
# Updates every 60 seconds
```

**D. GPU Monitoring**:
```bash
watch -n 5 nvidia-smi
# Shows: GPU utilization, memory usage, temperature
```

---

## Monitoring & Debugging

### Expected Training Timeline (A10 GPU)

| Model | Epochs/Episodes | Est. Time | Checkpoint Size |
|-------|----------------|-----------|-----------------|
| BaselineMLP | 50 | 1 hour | ~1 MB |
| CQL | 100 | 3 hours | ~5 MB |
| CGDT | 50 | 5 hours | ~200 MB |
| FLAG-TRADER | 30 | 8 hours | ~350 MB |
| PPO-LSTM | 1000 | 10 hours | ~10 MB |
| **TOTAL** | - | **27 hours** | **~570 MB** |

### Health Checks

**Every Hour**:
```bash
# 1. Check GPU utilization (should be 95-100%)
nvidia-smi

# 2. Check training progress
tail -n 50 training.log | grep "Epoch"

# 3. Check disk space
df -h /workspace

# 4. Check checkpoint timestamps
ls -lht checkpoints/ | head -5
```

**Signs of Problems**:
- ❌ GPU utilization <50%: Possible CPU bottleneck
- ❌ No log updates for 30+ min: Training may be stuck
- ❌ Disk space <5GB: May fail during checkpoint saving
- ❌ Loss is NaN: Learning rate too high or gradient explosion

### Debugging Commands

**Check last error**:
```bash
grep -i "error\|exception\|fail" training.log | tail -20
```

**Verify model loaded**:
```bash
python -c "import torch; m=torch.load('checkpoints/cql_best.pt'); print(m.keys())"
```

**Test data loading**:
```bash
python -c "
import sys
sys.path.insert(0, '.')
from scripts.train_all_models import load_raw_data
train_f, train_l, val_f, val_l, meta = load_raw_data('./data')
print(f'Loaded {len(train_f)} train samples, {len(val_f)} val samples')
"
```

**Memory profiling**:
```python
# Add to training script:
import torch.cuda as cuda
print(f"GPU Memory: {cuda.memory_allocated()/1e9:.2f} GB / {cuda.max_memory_allocated()/1e9:.2f} GB")
```

---

## Cost Optimization

### 1. Choose Right Instance Type

| Task | Recommended GPU | Cost/hr | Why |
|------|----------------|---------|-----|
| Testing (1-2 epochs) | RTX 3060 (12GB) | $0.10 | Cheap, sufficient for tests |
| BaselineMLP, CQL, PPO | RTX 3080 (10GB) | $0.15 | Fast enough, low cost |
| CGDT | RTX 3090 (24GB) | $0.20 | Good value for medium models |
| FLAG-TRADER | A10 (24GB) | $0.30 | Required for 88M params |

### 2. Interruptible Instances

Vast.ai offers "interruptible" instances at 50% discount:
```
Interruptible: Yes (50% cheaper)
Max downtime: 2 hours
```

**Best for**: Checkpointed training where you can resume

**Add to training script**:
```python
# Auto-resume from last checkpoint
checkpoint_dir = Path(args.checkpoint_dir)
latest = max(checkpoint_dir.glob("*_best.pt"), default=None, key=os.path.getctime)
if latest:
    logger.info(f"Resuming from {latest}")
    model.load_state_dict(torch.load(latest))
```

### 3. Spot Pricing

**Check current prices**:
```bash
# On Vast.ai search:
Sort by: $/hr (ascending)
Filter: GPU = A10, VRAM >= 24GB
```

**Typical A10 pricing**:
- On-demand: $0.40-0.60/hr
- Spot: $0.20-0.35/hr
- **Savings**: 40-50%

### 4. Partial Training Strategy

Train expensive models separately:
```bash
# Day 1: Train cheap models (4 hours, $0.60 total)
python scripts/run_all_training.py --models baseline cql ppo --device cuda

# Day 2: Train expensive models (13 hours, $3.90 total)
python scripts/run_all_training.py --models cgdt flag-trader --device cuda
```

**Total cost**: $4.50 vs $8.10 (saves 45%)

### 5. Early Stopping

Add to training functions:
```python
patience = 10
best_loss = float('inf')
patience_counter = 0

for epoch in range(args.epochs):
    # Training...

    if loss < best_loss:
        best_loss = loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        logger.info(f"Early stopping at epoch {epoch}")
        break
```

**Savings**: May reduce training time by 20-30%

---

## Final Deployment Command

**Copy this for Vast.ai deployment**:

```bash
# === HIMARI Layer 2 V1 - Complete Training Pipeline ===

# 1. Setup (in tmux session)
tmux new -s himari_training
cd /workspace/himari

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
ls -lh data/

# 4. Start training
python scripts/run_all_training.py \
  --device cuda \
  --wandb-entity charithliyanage52-himari \
  --checkpoint-dir ./checkpoints \
  --data-dir ./data

# 5. Detach (Ctrl+B, then D)
# 6. Reattach later: tmux attach -t himari_training

# 7. Download checkpoints when done:
# From local machine:
# scp -r root@<vast-ip>:/workspace/himari/checkpoints ./
```

---

## Troubleshooting Quick Reference

| Error | Check | Fix |
|-------|-------|-----|
| `not enough values to unpack` | Using old `train_all_models.py`? | Upload fixed version (has `load_raw_data()`) |
| `ModuleNotFoundError` | Missing dependency | `pip install -r requirements.txt` |
| Training exits after 1 epoch | Silent exception | Check `training.log` for errors |
| No checkpoints saved | Old script version | Upload fixed version (has `save_model()` calls) |
| CUDA OOM | Batch size too large | Reduce `--batch-size 128` or `--context-length 128` |
| Connection lost | Not using tmux | Always use `tmux` or `screen` |
| Slow training | CPU bottleneck | Check `nvidia-smi` utilization |
| Disk full | Not enough space | Delete old checkpoints or expand storage |

---

## Success Checklist

After training completes, verify:

- [ ] All 5 models trained successfully (check logs)
- [ ] 10 checkpoint files exist in `checkpoints/` directory
- [ ] Checkpoint files have reasonable sizes (1MB-350MB)
- [ ] W&B shows loss curves decreasing
- [ ] Final log shows: "ALL TRAINING COMPLETE!"
- [ ] No errors in `training.log`

**Download results**:
```bash
# From local machine:
scp -r root@<vast-ip>:/workspace/himari/checkpoints ./
scp root@<vast-ip>:/workspace/himari/training.log ./
```

---

## Additional Resources

- **Vast.ai Docs**: https://docs.vast.ai/documentation/get-started
- **PyTorch CUDA Guide**: https://pytorch.org/docs/stable/cuda.html
- **W&B Integration**: https://docs.wandb.ai/guides/integrations/pytorch
- **HIMARI Project**: See `DEPLOYMENT_COMPLETE.md` for full package details

---

**Document Version**: 1.0
**Last Updated**: 2026-01-03
**Status**: Production-ready deployment guide
