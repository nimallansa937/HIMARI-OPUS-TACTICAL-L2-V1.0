# Quick Deploy Reference - Vast.ai

**1-page cheat sheet for rapid deployment**

---

## 🚀 Quick Start (5 Minutes)

```bash
# === On Vast.ai Instance ===

# 1. Start tmux
tmux new -s training

# 2. Setup
cd /workspace/himari
pip install -r requirements.txt

# 3. Train
python scripts/run_all_training.py --device cuda --wandb-entity charithliyanage52-himari

# 4. Detach: Ctrl+B, then D
```

---

## 🔧 Common Fixes

| Error | 1-Line Fix |
|-------|-----------|
| `not enough values to unpack` | ✓ Already fixed in `train_all_models.py` |
| `No module named 'loguru'` | `pip install -r requirements.txt` |
| Training stops after 1 epoch | ✓ Already fixed (error handling added) |
| No checkpoints | ✓ Already fixed (all models save now) |
| CUDA OOM | `--batch-size 128 --context-length 128` |
| Connection lost | Use `tmux` (see above) |

---

## 📊 Expected Results

| Model | Time | Checkpoints |
|-------|------|-------------|
| BaselineMLP | 1h | `baseline_best.pt`, `baseline_final.pt` |
| CQL | 3h | `cql_best.pt`, `cql_final.pt` |
| CGDT | 5h | `cgdt_best.pt`, `cgdt_final.pt` |
| FLAG-TRADER | 8h | `flag_trader_best.pt`, `flag_trader_final.pt` |
| PPO-LSTM | 10h | `ppo_best.pt`, `ppo_final.pt` |
| **TOTAL** | **27h** | **10 files (~570MB)** |

---

## 💰 Cost Estimate

- **Instance**: NVIDIA A10 (24GB VRAM)
- **Rate**: ~$0.30/hour
- **Time**: 27 hours
- **Total**: **~$8-9 for complete training**

---

## ✅ Pre-Flight Checklist

- [ ] Files uploaded: `scripts/`, `src/`, `data/`, `requirements.txt`
- [ ] Data exists: `data/preprocessed_features.npy` (103,604 samples)
- [ ] Instance: A10 with 24GB VRAM, 30GB storage
- [ ] Using tmux/screen for stability

---

## 🔍 Monitor Training

```bash
# GPU usage (should be 95-100%)
nvidia-smi

# Training progress
tail -f training.log | grep "Epoch"

# Checkpoints saved
ls -lht checkpoints/
```

---

## 📥 Download Results

```bash
# From your local machine:
scp -r root@<vast-ip>:/workspace/himari/checkpoints ./
```

---

## 🆘 Emergency Commands

```bash
# Check if training is stuck
ps aux | grep python

# Kill stuck process
pkill -f train_all_models.py

# Resume training from checkpoints
# (Add --resume flag if implemented)

# Free up space
rm checkpoints/*_final.pt  # Keep only best models
```

---

## 📞 Support Resources

- **Full Guide**: See `VAST_AI_DEPLOYMENT_GUIDE.md`
- **Fixes Applied**: See `ALL_FIXES_APPLIED.md`
- **Vast.ai Docs**: https://docs.vast.ai/documentation/get-started

---

**Last Updated**: 2026-01-03 | **Status**: Ready for deployment ✓
