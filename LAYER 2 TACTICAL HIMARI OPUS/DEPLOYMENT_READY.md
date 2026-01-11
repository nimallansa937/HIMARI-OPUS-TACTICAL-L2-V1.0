# HIMARI Layer 2 - GPU DEPLOYMENT PACKAGE
## Ready for Immediate Launch

**Status:** ✅ PRODUCTION READY
**Date:** 2026-01-03
**Training Code:** FULLY TESTED & WORKING

---

## Package Contents

### Core Training Files ✅
- `scripts/launch_training.py` (522 lines, all fixes applied)
- `src/models/baseline_mlp.py` (BaselineMLP model)
- `src/data/dataset.py` (Data loading pipeline)
- `src/training/monitoring.py` (W&B integration)
- `src/training/training_pipeline.py` (Training infrastructure)

### Data Files ✅
- `data/preprocessed_features.npy` (23.7 MB, 103,604 samples)
- `data/labels.npy` (405 KB)
- `data/metadata.json`

### Configuration ✅
- `configs/training_config.yaml` (W&B: charithliyanage52-himari)

---

## Quick Start on GPU

### 1. Upload Package
```bash
# On GPU instance
cd /workspace
# Upload this entire directory
```

### 2. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas pyyaml wandb
```

### 3. Login to W&B
```bash
wandb login
# Enter your API key from https://wandb.ai/authorize
```

### 4. Launch Training (FRESH START)
```bash
cd "LAYER 2 TACTICAL HIMARI OPUS"

# Full 50-epoch training
python scripts/launch_training.py \
    --config configs/training_config.yaml \
    --fresh-start \
    --epochs 50 \
    --batch-size 256

# Or test run first (2 epochs)
python scripts/launch_training.py --test-run
```

---

## Training Estimates

**Hardware:** NVIDIA A10 GPU (24GB VRAM)

| Metric | Value |
|--------|-------|
| **Total Samples** | 103,604 |
| **Batch Size** | 256 |
| **Steps/Epoch** | 323 |
| **Epochs** | 50 |
| **Total Steps** | 16,150 |
| **Estimated Time** | ~6-8 hours |
| **Estimated Cost** | ~$1.00-1.30 @ $0.163/hr |

---

## Monitoring

### Weights & Biases Dashboard
**Project:** himari-layer2
**Entity:** charithliyanage52-himari
**URL:** https://wandb.ai/charithliyanage52-himari/himari-layer2

**Metrics Tracked:**
- Training loss & accuracy
- Validation loss & accuracy
- Per-class accuracy (SELL/HOLD/BUY)
- Learning rate
- GPU utilization
- Steps/second

---

## Checkpoints

**Location:** `./checkpoints/`

**Files Created:**
- `checkpoint_epoch_10.pt` (every 10 epochs)
- `checkpoint_epoch_20.pt`
- `checkpoint_epoch_30.pt`
- `checkpoint_epoch_40.pt`
- `best_model.pt` (best validation accuracy)
- `latest_checkpoint.pt` (for resume)

**Auto-cleanup:** Keeps only last 5 checkpoints

---

## Resume Training

If training is interrupted:

```bash
# Automatically resumes from latest checkpoint
python scripts/launch_training.py

# Or specify checkpoint
python scripts/launch_training.py --resume-from checkpoints/checkpoint_epoch_20.pt
```

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python scripts/launch_training.py --batch-size 128
```

### CUDA Not Available
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### W&B Not Logging
```bash
# Run without W&B (local only)
python scripts/launch_training.py --no-wandb
```

---

## Success Criteria

**Minimum Acceptable Performance:**
- Validation accuracy > 50% (better than random ~33%)
- Training converges (loss decreases)
- No NaN/Inf in metrics

**Target Performance:**
- Validation accuracy > 60%
- Per-class accuracy balanced
- Stable training (no divergence)

---

## Next Steps After Training

1. **Evaluate final model:**
   ```bash
   # Test accuracy will be printed at end
   ```

2. **Download checkpoints:**
   ```bash
   # Download best_model.pt for deployment
   ```

3. **Review W&B dashboard:**
   - Check training curves
   - Analyze per-class performance
   - Review GPU utilization

4. **Deploy for inference:**
   - Use best_model.pt
   - Integrate with HIMARI Layer 1

---

## Emergency Contacts

**Issues?**
- Check `training.log` for errors
- Review W&B dashboard for anomalies
- Verify GPU with `nvidia-smi`

**Cost Overrun?**
- Stop training: CTRL+C (saves checkpoint)
- Review instance pricing
- Consider resuming later

---

✅ **ALL SYSTEMS GO - READY FOR LAUNCH!**
