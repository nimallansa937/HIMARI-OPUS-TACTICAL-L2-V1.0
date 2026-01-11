# HIMARI Layer 2 - Monitoring & Training Guide

This guide covers how to monitor your HIMARI Layer 2 training remotely using Weights & Biases.

## Quick Start

### 1. Setup Weights & Biases

Your W&B account is already configured: **charithliyanage52-himari**

1. **Login to W&B:**

   ```bash
   wandb login
   ```

   Enter your API key when prompted. Find it at: <https://wandb.ai/authorize>

2. **Verify login:**

   ```bash
   wandb verify
   ```

### 2. Verify Training Data

Before starting training, verify your data:

```bash
cd "c:/Users/chari/OneDrive/Documents/HIMARI OPUS 2/LAYER 2 V1/LAYER 2 TACTICAL HIMARI OPUS"
python scripts/verify_training_data.py --data-dir ./data
```

Expected output:

```
✅ Verification PASSED
Ready to start training!
```

### 3. Test Run (10 minutes, ~$0.027)

Test the setup with a short run:

```bash
python scripts/launch_training.py \
  --config configs/training_config.yaml \
  --test-run \
  --gpu 0
```

This runs 2 epochs to verify everything works.

### 4. Full Training (133 hours, $21.68)

Launch full training:

```bash
python scripts/launch_training.py \
  --config configs/training_config.yaml \
  --gpu 0
```

## Remote Monitoring

### Access Dashboard from Anywhere

1. **Open W&B Dashboard:**
   - Go to: <https://wandb.ai/charithliyanage52-himari/himari-layer2>
   - Or find your run at: <https://wandb.ai/charithliyanage52-himari>

2. **View Real-Time Metrics:**
   - Training loss (should decrease)
   - Validation Sharpe ratio (should increase)
   - GPU utilization (should be 80-95%)
   - Learning rate (should decay)
   - Estimated time remaining

3. **Mobile Access:**
   - Install W&B mobile app (iOS/Android)
   - Login with your account
   - View training progress on the go

### Key Metrics to Watch

| Metric | Expected Behavior | Alert If |
|--------|-------------------|----------|
| `loss` | Steadily decreasing | Spikes > 10x initial |
| `sharpe_ratio` | Increasing to 2.5-3.2 | Stays below 1.0 |
| `gpu_utilization` | 80-95% | Drops below 50% |
| `steps_per_second` | ~0.02 (1 min/step) | Drops below 0.01 |
| `epoch_duration` | ~100 minutes | Increases over time |

## Checkpoints

### Automatic Checkpoints

Checkpoints are saved automatically every **6 hours** (360 steps):

- Location: `./checkpoints/`
- Format: `checkpoint_step_<STEP>.pt`
- Best model: `best_checkpoint.pt` (tracked by Sharpe ratio)

### Download Checkpoints

From W&B dashboard:

1. Go to your run → Files → Artifacts
2. Download `best_checkpoint.pt`

From command line:

```bash
wandb artifact get charithliyanage52-himari/himari-layer2/best_checkpoint:latest
```

### Resume from checkpoint

If training interrupted:

```bash
python scripts/launch_training.py \
  --config configs/training_config.yaml \
  --resume-from ./checkpoints/checkpoint_step_3600.pt \
  --gpu 0
```

## Alerts & Notifications

### Setup Email Alerts

1. Go to W&B Settings: <https://wandb.ai/settings>
2. Navigate to "Notifications"
3. Enable email alerts for:
   - Run failures
   - Custom alerts (NaN, divergence)

### Setup Slack Alerts (Optional)

1. W&B Settings → Integrations → Slack
2. Connect your Slack workspace
3. Choose channels for alerts

## Troubleshooting

### Training Stuck / Not Progressing

**Symptoms:** `steps_per_second` = 0 or very low

**Solutions:**

1. Check GPU utilization - should be 80%+
2. Check if data loading is slow
3. Restart training with smaller batch size

### Loss Goes to NaN

**Symptoms:** `loss` = NaN in dashboard

**Solutions:**

- Training will auto-reload from previous checkpoint
- Check logs for gradient explosion
- May need to reduce learning rate

### W&B Not Logging

**Symptoms:** Dashboard not updating

**Solutions:**

1. Check internet connection on GPU instance
2. Verify W&B login: `wandb verify`
3. Check logs for W&B errors
4. Training continues with local logging as fallback

### Out of Memory (OOM)

**Symptoms:** CUDA out of memory error

**Solutions:**

1. Reduce batch size in config
2. Enable gradient checkpointing (already enabled)
3. Use smaller model variant

## Training Timeline

### Expected Progress

| Time | Epoch | Loss | Sharpe | GPU Cost |
|------|-------|------|--------|----------|
| 0h | 0 | ~1.2 | ~0.5 | $0.00 |
| 20h | 10 | ~0.6 | ~1.2 | $3.26 |
| 53h | 25 | ~0.3 | ~2.0 | $8.64 |
| 106h | 50 | ~0.15 | ~2.8 | $17.28 |
| 133h | Done | ~0.12 | ~3.0 | $21.68 |

*Progress may vary based on data and random initialization*

### What to Do During Training

**Daily checks (5 minutes):**

- Open W&B dashboard
- Verify metrics are updating
- Check no alerts/errors
- Note estimated time remaining

**Weekly checks (Not applicable for 5-6 day run):**

- Review metric trends
- Check checkpoint uploads
- Verify GPU instance still running

## Vast.ai GPU Instance Setup

### Rent GPU

1. Go to: <https://vast.ai/console/create/>
2. Search for: "A10" GPU
3. Select: Minnesota instance ($0.163/hr)
4. SSH into instance

### Sync Code to GPU

```bash
# On your local machine
cd "c:/Users/chari/OneDrive/Documents/HIMARI OPUS 2/LAYER 2 V1/LAYER 2 TACTICAL HIMARI OPUS"

# Copy to GPU (replace <GPU_IP> and <SSH_PORT>)
scp -r -P <SSH_PORT> . root@<GPU_IP>:/workspace/himari-layer2/
```

### Install Dependencies on GPU

```bash
# SSH into GPU instance
ssh -p <SSH_PORT> root@<GPU_IP>

# Install packages
cd /workspace/himari-layer2
pip install -r requirements.txt
pip install wandb

# Login to W&B
wandb login
# Paste your API key
```

### Launch Training

```bash
# Inside GPU instance
python scripts/launch_training.py \
  --config configs/training_config.yaml \
  --gpu 0
```

## Post-Training

### Download Results

1. **Best Model:**

   ```bash
   # From W&B
   wandb artifact get charithliyanage52-himari/himari-layer2/best_checkpoint:latest
   
   # Or via SCP from GPU
   scp -P <SSH_PORT> root@<GPU_IP>:/workspace/himari-layer2/checkpoints/best_checkpoint.pt ./
   ```

2. **Training Logs:**

   ```bash
   scp -P <SSH_PORT> root@<GPU_IP>:/workspace/himari-layer2/logs/training_summary.json ./
   ```

### Verify Performance

Check final metrics in W&B dashboard:

- Sharpe ratio: Should be 2.5-3.2
- Max drawdown: Should be -8% to -10%
- Win rate: Should be 62-68%

If metrics are significantly worse, may need to:

- Adjust hyperparameters
- Collect more training data
- Increase training duration

## Support & Resources

- **W&B Docs:** <https://docs.wandb.ai/>
- **HIMARI Docs:** See `LAYER 2 CLAUDE GUIDE/` directory
- **Training Config:** `configs/training_config.yaml`

---

**Created:** 2026-01-02  
**For:** HIMARI Layer 2 Initial Training  
**Estimated Cost:** $21.68 (133 hours on Vast.ai A10)
