# Instructions for Next Session

## Current Status ✅

1. **FLAG-TRADER model loading**: ✅ SOLVED
   - Found training script at: `LAYER 2 TACTICAL HIMARI OPUS/src/models/flag_trader.py`
   - Model loads successfully with logit bias correction

2. **Model collapse problem**: ✅ SOLVED with logit correction
   - Correction: `[3.0, -2.5, 1.0]` for [SELL, HOLD, BUY]
   - Results: Sharpe +0.063, +0.18% return, 1,001 trades

3. **Ensemble framework**: ✅ CREATED
   - File: `ensemble_backtest.py`
   - Ready for CGDT integration (not yet trained)

## What Needs to Be Done Next

### Task 1: Test Ensemble Locally (30 min)

Run the ensemble with confidence filtering:

```bash
cd "C:\Users\chari\OneDrive\Documents\HIMARI OPUS TESTING\LAYER 2 V1 - Copy"
python ensemble_backtest.py --confidence_threshold 0.7 --strategy voting
```

Expected: Sharpe 0.1-0.3 (better than 0.063)

---

### Task 2: Prepare for GitHub (1 hour)

**Create these files**:

1. **`README.md`** - Main project documentation
2. **`requirements.txt`** - Python dependencies
3. **`.gitignore`** - Exclude checkpoints/data
4. **`train_flagtrader.py`** - Training script for Vast.ai
5. **`vast_ai_setup.sh`** - Automated setup script

**Git commands**:
```bash
cd "C:\Users\chari\OneDrive\Documents\HIMARI OPUS TESTING\LAYER 2 V1 - Copy"

# Initialize git
git init
git add .
git commit -m "Initial commit: FLAG-TRADER with ensemble framework"

# Add remote
git remote add origin https://github.com/nimallansa937/HIMARI-TESTING-SUITE.git

# Push
git branch -M main
git push -u origin main
```

---

### Task 3: Locate Training Data (15 min)

**Find these files**:
1. `btc_1h_2020_2024.csv` - Training data (2020-2024)
   - Location: `L2 POSTION FINAL MODELS/orkspace/data/` (search for it)

2. Training labels (if separate file exists)

3. Preprocessed features (if cached)

**Check file sizes** to estimate upload time to Google Drive:
```bash
dir /s "btc_1h_*.csv" | findstr /C:"btc_1h"
```

---

### Task 4: Create Training Script (1 hour)

**File to create**: `train_flagtrader.py`

**Should include**:
- Model architecture (FLAG-TRADER with LoRA)
- Balanced class weights: `[2.5, 0.83, 2.5]` for [SELL, HOLD, BUY]
- Training loop with validation
- Checkpoint saving
- Google Drive download for data

**Key parameters**:
```python
# Use balanced loss
class_weights = torch.tensor([2.5, 0.83, 2.5])  # Inverse frequency
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Training config
batch_size = 64
learning_rate = 1e-4
num_epochs = 10
lora_rank = 16
```

---

### Task 5: Create Vast.ai Command (30 min)

**File to create**: `vast_ai_setup.sh`

```bash
#!/bin/bash

# Install dependencies
pip install torch transformers peft loguru numpy pandas scikit-learn

# Clone repository
git clone https://github.com/nimallansa937/HIMARI-TESTING-SUITE.git
cd HIMARI-TESTING-SUITE

# Download data from Google Drive (you'll provide link)
gdown --id <YOUR_GOOGLE_DRIVE_FILE_ID> -O data/btc_1h_2020_2024.csv

# Start training
python train_flagtrader.py --data data/btc_1h_2020_2024.csv --output checkpoints/ --epochs 10
```

**Single command for user**:
```bash
wget https://raw.githubusercontent.com/nimallansa937/HIMARI-TESTING-SUITE/main/vast_ai_setup.sh && bash vast_ai_setup.sh
```

---

## Files Already Created ✅

1. `end_to_end_backtest.py` - Main backtest with logit correction
2. `ensemble_backtest.py` - Ensemble framework
3. `test_model_load_flagtrader.py` - Model loading test
4. `diagnose_model_predictions.py` - Prediction analysis
5. `validate_flag_trader.py` - HIFA validation
6. `SOLUTION_SUMMARY.md` - Complete solution documentation
7. `FINDINGS_MODEL_COLLAPSE.md` - Problem analysis

---

## Key Information for Next Session

### Model Architecture
- **FLAG-TRADER**: 88M parameters, LoRA rank 16
- **Config**: d_model=768, layers=12, heads=8
- **Input**: 60D features (49D + 11 padding)
- **Output**: 3 actions (SELL, HOLD, BUY)

### Logit Correction (Current)
```python
logit_correction = torch.tensor([3.0, -2.5, 1.0])  # [SELL, HOLD, BUY]
```

### Training Data Requirements
- **Period**: 2020-2024 (5 years, ~43,800 hours)
- **Features**: 60D (49D market + 11D order flow)
- **Labels**: BUY/HOLD/SELL actions
- **Expected distribution**: ~60% HOLD, ~20% BUY, ~20% SELL

### Expected Performance (After Retraining)
- **Accuracy**: 50-55% (down from 61%, but diverse predictions)
- **Sharpe**: 0.8-2.0 (vs current 0.063)
- **Trades**: 20-35% of timesteps (vs current 11%)

---

## Quick Start for New Session

Tell Claude:
> "Continue from NEXT_SESSION_INSTRUCTIONS.md - we need to:
> 1. Finish ensemble testing
> 2. Create training script for Vast.ai
> 3. Locate training data files
> 4. Push everything to GitHub
> 5. Give me the Vast.ai command to run"

Claude will have all context from this document and can continue where we left off.

---

## Critical Files Locations

- **Checkpoints**: `checkpoints/flag_trader_best.pt`
- **Test Data**: `btc_1h_2025_2026_test_arrays.pkl`
- **Training Data**: Search for `btc_1h_2020_2024.csv`
- **Model Code**: `LAYER 2 TACTICAL HIMARI OPUS/src/models/flag_trader.py`

---

**Last Updated**: 2026-01-19
**Session Token Usage**: 94k/200k
