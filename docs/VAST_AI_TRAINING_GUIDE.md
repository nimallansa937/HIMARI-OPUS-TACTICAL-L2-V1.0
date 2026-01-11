# HIMARI Layer 2 - Vast.ai GPU Training Guide

> **Complete guide for training HIMARI models on Vast.ai GPU instances**
>
> This guide documents all errors encountered and their solutions, plus best practices for future training runs.

---

## Table of Contents

1. [Quick Start Checklist](#quick-start-checklist)
2. [Instance Setup](#instance-setup)
3. [Common Errors & Solutions](#common-errors--solutions)
4. [Training Commands](#training-commands)
5. [Monitoring Training](#monitoring-training)
6. [Checkpoints & Models](#checkpoints--models)
7. [Cost Optimization](#cost-optimization)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start Checklist

```
[ ] 1. Rent GPU instance (RTX 4090 recommended, ~$0.40-0.50/hr)
[ ] 2. Select PyTorch template (pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel)
[ ] 3. Upload project files to /workspace/
[ ] 4. Install dependencies: pip install -r requirements.txt
[ ] 5. Set PYTHONPATH: export PYTHONPATH=$PYTHONPATH:.
[ ] 6. Run training with nohup for background execution
[ ] 7. Monitor via log file or reconnect to terminal
```

---

## Instance Setup

### Recommended Instance Specs

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) |
| VRAM | 10 GB | 24 GB |
| RAM | 16 GB | 32 GB |
| Disk | 50 GB | 100 GB |
| Cost | ~$0.25/hr | ~$0.40-0.50/hr |

### Template Selection

Use the **PyTorch template** with CUDA support:

- `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel` (recommended)
- Or: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel`

### Initial Setup Commands

```bash
# Navigate to workspace
cd /workspace/

# Clone or upload your project
# Option 1: Git clone
git clone https://github.com/YOUR_REPO/himari-layer2.git

# Option 2: Upload via Jupyter (recommended for large files)
# Use the Jupyter file browser at https://YOUR_INSTANCE/tree

# Install dependencies
cd "LAYER 2 TACTICAL HIMARI OPUS"
pip install -r requirements.txt

# CRITICAL: Set Python path
export PYTHONPATH=$PYTHONPATH:.
```

---

## Common Errors & Solutions

### Error #1: `ModuleNotFoundError` for project imports

**Symptom:**

```
ModuleNotFoundError: No module named 'src.models.baseline_mlp'
```

**Cause:** Python can't find your project modules.

**Solution:**

```bash
# Add project root to Python path
export PYTHONPATH=$PYTHONPATH:.

# Or add to ~/.bashrc for persistence
echo 'export PYTHONPATH=$PYTHONPATH:.' >> ~/.bashrc
source ~/.bashrc
```

---

### Error #2: Missing Dependencies

**Symptom:**

```
ModuleNotFoundError: No module named 'loguru'
ModuleNotFoundError: No module named 'filterpy'
ModuleNotFoundError: No module named 'gym'
```

**Solution:**

```bash
# Install missing packages
pip install loguru filterpy gym scikit-learn wandb

# Or ensure requirements.txt includes:
# loguru>=0.7.0
# filterpy>=1.4.5
# gym>=0.26.0
# scikit-learn>=1.3.0
# wandb>=0.15.0
```

---

### Error #3: `AttributeError: object has no attribute 'save'`

**Symptom:**

```
AttributeError: 'BaselineMLP' object has no attribute 'save'
```

**Cause:** Training script calls `model.save()` but model class doesn't have this method.

**Solution:** Use a helper function:

```python
def save_model(model, path):
    """Save model checkpoint. Works for both nn.Module and agents."""
    if hasattr(model, 'save'):
        model.save(str(path))
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model.get_config() if hasattr(model, 'get_config') else {}
        }, str(path))
```

---

### Error #4: File Confusion (Two Scripts with Same Purpose)

**Symptom:**

```
train_all_models.py: error: unrecognized arguments: --model baseline --epochs 50
```

**Cause:** `run_all_training.py` (orchestrator) tries to call `train_all_models.py` (trainer), but both got confused/overwritten.

**Solution:**

- **`run_all_training.py`** = Orchestrator (loops through models, calls trainer)
- **`train_all_models.py`** = Trainer (accepts `--model` argument, trains one model)

Ensure correct file is uploaded. The trainer should accept:

```bash
python train_all_models.py --model baseline --epochs 50
python train_all_models.py --model cql --epochs 100
python train_all_models.py --model ppo --env-episodes 1000
```

---

### Error #5: CQL Only Trains 1 Epoch

**Symptom:**

```
CQL completed in 3 seconds (expected ~6 minutes for 100 epochs)
```

**Cause:** Logging interval was `epoch % 100 == 0`, so only epoch 0 was visible.

**Solution:** Change logging interval to `epoch % 10 == 0`:

```python
if epoch % 10 == 0:  # Was: epoch % 100
    logger.info(f"Epoch {epoch}/{args.epochs}: Loss={info['loss']:.4f}")
```

---

### Error #6: Terminal Freezes

**Symptom:** Jupyter terminal becomes unresponsive, commands don't execute.

**Solution:**

1. Open a **new terminal** from Jupyter Home: `https://YOUR_INSTANCE/tree` → New → Terminal
2. Or refresh the terminal page (F5)
3. Background processes continue running even if terminal freezes

---

### Error #7: `gdown` File Not Found

**Symptom:**

```
gdown: file not found or access denied
```

**Solution:**

1. Ensure Google Drive file is **shared** (Anyone with link can view)
2. Use `--fuzzy` flag: `gdown --fuzzy "YOUR_LINK" -O output.py`
3. Install gdown if missing: `pip install gdown`

---

### Error #8: Project Directory Path Issues

**Symptom:**

```
bash: cd: /workspace/himari-layer2: No such file or directory
```

**Cause:** Directory name has spaces or different than expected.

**Solution:**

```bash
# List directories to find correct name
ls -la /workspace/

# Use quotes for paths with spaces
cd "/workspace/LAYER 2 TACTICAL HIMARI OPUS"
```

---

## Training Commands

### Full Automated Training (All 5 Models)

```bash
cd "/workspace/LAYER 2 TACTICAL HIMARI OPUS"
export PYTHONPATH=$PYTHONPATH:.

# Run in background with nohup
nohup python scripts/run_all_training.py \
    --device cuda \
    --wandb-entity YOUR_WANDB_ENTITY \
    > training_all_models.log 2>&1 &

# Monitor progress
tail -f training_all_models.log
```

### Individual Model Training

```bash
# BaselineMLP
python scripts/train_all_models.py --model baseline --epochs 50 --device cuda

# CQL
python scripts/train_all_models.py --model cql --epochs 100 --device cuda

# CGDT
python scripts/train_all_models.py --model cgdt --epochs 50 --context-length 64 --device cuda

# FLAG-TRADER
python scripts/train_all_models.py --model flag-trader --epochs 30 --context-length 256 --lora-rank 16 --device cuda

# PPO-LSTM
python scripts/train_all_models.py --model ppo --env-episodes 1000 --device cuda
```

---

## Monitoring Training

### Check Training Status

```bash
# View last 50 lines of log
tail -50 training_all_models.log

# Follow log in real-time
tail -f training_all_models.log

# Check if training process is running
ps aux | grep python
```

### Check Checkpoints

```bash
# List saved checkpoints
ls -lh checkpoints/

# Expected files after training:
# baseline_best.pt, baseline_final.pt
# cql_best.pt, cql_final.pt
# cgdt_best.pt, cgdt_final.pt
# flag_trader_best.pt, flag_trader_final.pt
# ppo_best.pt, ppo_final.pt
```

### GPU Utilization

```bash
# Check GPU usage
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi
```

---

## Checkpoints & Models

### Expected Checkpoint Structure

```
checkpoints/
├── baseline_best.pt      # Best BaselineMLP (lowest loss)
├── baseline_final.pt     # Final BaselineMLP
├── cql_best.pt           # Best CQL agent
├── cql_final.pt          # Final CQL agent
├── cgdt_best.pt          # Best CGDT model
├── cgdt_final.pt         # Final CGDT model
├── flag_trader_best.pt   # Best FLAG-TRADER
├── flag_trader_final.pt  # Final FLAG-TRADER
├── ppo_best.pt           # Best PPO-LSTM
└── ppo_final.pt          # Final PPO-LSTM
```

### Download Checkpoints

```bash
# Compress checkpoints for download
cd /workspace/"LAYER 2 TACTICAL HIMARI OPUS"
tar -czvf himari_checkpoints.tar.gz checkpoints/

# Download via Jupyter file browser
# Navigate to: https://YOUR_INSTANCE/tree
# Right-click himari_checkpoints.tar.gz → Download
```

---

## Cost Optimization

### Tips to Reduce Costs

1. **Use interruptible instances** (up to 50% cheaper)
2. **Choose optimal GPU** - RTX 4090 has best price/performance
3. **Stop instance when not training** - you pay per hour!
4. **Use `nohup`** - allows training to continue if disconnected
5. **Monitor usage** - check nvidia-smi to ensure GPU is utilized

### Estimated Training Costs

| Model | Est. Time | Cost @ $0.45/hr |
|-------|-----------|-----------------|
| BaselineMLP | ~5 min | $0.04 |
| CQL | ~10 min | $0.08 |
| CGDT | ~3-4 hrs | $1.50 |
| FLAG-TRADER | ~3-5 hrs | $2.00 |
| PPO-LSTM | ~8-10 hrs | $4.50 |
| **Total** | **~15-20 hrs** | **~$8-10** |

---

## Troubleshooting

### Training Stuck / No Progress

```bash
# Check if process is running
ps aux | grep python

# Check GPU usage
nvidia-smi

# If GPU at 0%, training may have crashed - check log
tail -100 training_all_models.log
```

### Reconnecting After Disconnect

```bash
# Your training continues in background!
# Just reconnect and check log:
cd "/workspace/LAYER 2 TACTICAL HIMARI OPUS"
tail -50 training_all_models.log

# Check running processes
ps aux | grep python
```

### Instance Disconnected / Stopped

If the **Vast.ai instance** stops or is evicted:

- Training **is lost** (unless checkpoints were saved)
- Download checkpoints before stopping instance
- Consider using **reserved instances** for long training

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch-size 128  # Default is 256

# Reduce context length (for transformers)
--context-length 32  # Default is 64

# Use gradient checkpointing (if supported)
```

---

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════════════╗
║                 HIMARI VAST.AI QUICK REFERENCE                   ║
╠══════════════════════════════════════════════════════════════════╣
║ SETUP                                                            ║
║   cd "/workspace/LAYER 2 TACTICAL HIMARI OPUS"                  ║
║   export PYTHONPATH=$PYTHONPATH:.                               ║
║   pip install -r requirements.txt                               ║
╠══════════════════════════════════════════════════════════════════╣
║ TRAINING                                                         ║
║   nohup python scripts/run_all_training.py --device cuda \      ║
║     --wandb-entity YOUR_ENTITY > training.log 2>&1 &            ║
╠══════════════════════════════════════════════════════════════════╣
║ MONITORING                                                       ║
║   tail -f training_all_models.log       # Watch log             ║
║   ls -lh checkpoints/                   # Check saved models    ║
║   nvidia-smi                            # GPU usage             ║
╠══════════════════════════════════════════════════════════════════╣
║ DOWNLOAD MODELS                                                  ║
║   tar -czvf checkpoints.tar.gz checkpoints/                     ║
║   # Download via Jupyter file browser                           ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Resources

- **Vast.ai Documentation**: <https://docs.vast.ai/documentation/get-started>
- **Vast.ai Console**: <https://cloud.vast.ai/instances/>
- **Vast.ai Discord**: <https://discord.gg/vast>
- **W&B Dashboard**: <https://wandb.ai/YOUR_ENTITY/himari-layer2>

---

*Last Updated: January 2026*
*Guide Version: 1.0*
