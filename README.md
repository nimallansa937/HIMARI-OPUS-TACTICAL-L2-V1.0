# HIMARI FLAG-TRADER

High-frequency trading system powered by large transformer models with LoRA fine-tuning.

## Overview

HIMARI FLAG-TRADER is a transformer-based trading agent designed for cryptocurrency markets. The system uses an 88M parameter model with Low-Rank Adaptation (LoRA) for efficient fine-tuning on trading data.

### Key Features

- **Large-scale transformer architecture**: 88M parameters with LoRA (only 2.9M trainable)
- **Balanced training**: Custom class weights to prevent model collapse
- **Ensemble framework**: Multiple model voting for robust predictions
- **Advanced backtesting**: Comprehensive performance metrics and risk analysis

## Quick Start

### 1. Local Testing

Test the ensemble backtest on your machine:

```bash
cd "LAYER 2 V1 - Copy"
python ensemble_backtest.py --confidence_threshold 0.7 --strategy voting
```

### 2. Training on Vast.ai

**Single command to run on your Vast.ai instance:**

```bash
wget https://raw.githubusercontent.com/nimallansa937/HIMARI-TESTING-SUITE/main/vast_ai_setup.sh && bash vast_ai_setup.sh
```

This will:
1. Install dependencies
2. Clone the repository
3. Download training data from Google Drive
4. Start training with balanced class weights

### 3. Manual Training

If you prefer manual setup:

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_flagtrader.py \
    --data data/btc_1h_2020_2024.csv \
    --output checkpoints/ \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-4
```

## Architecture

### FLAG-TRADER Model

- **Input**: 60D feature vectors (OHLCV + technical indicators)
- **Architecture**:
  - Embedding layer: 60 → 768D
  - 12 Transformer blocks with LoRA
  - 8 attention heads
  - 3072D feedforward dimension
- **Output**: 3 actions (SELL=0, HOLD=1, BUY=2)
- **LoRA rank**: 16 (configurable)

### Training Configuration

```python
# Balanced class weights (solves model collapse)
class_weights = [2.5, 0.83, 2.5]  # [SELL, HOLD, BUY]

# Training hyperparameters
batch_size = 64
learning_rate = 1e-4
num_epochs = 10
optimizer = AdamW (weight_decay=0.01)
scheduler = CosineAnnealingLR
```

## Performance

### Current Results (Pre-trained checkpoint)

- **Sharpe Ratio**: 0.066
- **Total Return**: 0.18%
- **Trades**: 1,001 (11% of timesteps)
- **Issue**: Model collapse due to class imbalance

### Expected Results (After retraining with balanced weights)

- **Sharpe Ratio**: 0.8 - 2.0
- **Accuracy**: 50-55% (with diverse predictions)
- **Trades**: 20-35% of timesteps
- **Max Drawdown**: < 10%

## Project Structure

```
LAYER 2 V1 - Copy/
├── train_flagtrader.py          # Training script for Vast.ai
├── ensemble_backtest.py         # Ensemble framework
├── end_to_end_backtest.py       # Main backtest pipeline
├── test_model_load_flagtrader.py # Model loading validation
├── vast_ai_setup.sh             # Automated Vast.ai setup
├── requirements.txt             # Python dependencies
│
├── checkpoints/
│   └── flag_trader_best.pt      # Trained model checkpoint
│
├── LAYER 2 TACTICAL HIMARI OPUS/
│   └── src/models/
│       ├── flag_trader.py       # FLAG-TRADER model code
│       └── cgdt.py              # CGDT model (future)
│
└── L2 POSTION FINAL MODELS/
    └── orkspace/data/
        ├── btc_1h_2020_2024.csv # Training data (2020-2024)
        └── btc_1h_2025_2026.csv # Test data (2025-2026)
```

## Model Collapse Solution

The original model suffered from **model collapse** - predicting HOLD 89% of the time due to class imbalance in the training data.

### Problem

```
Original distribution:
- HOLD: 60% of data
- BUY: 20% of data
- SELL: 20% of data

Result: Model learns to predict HOLD for nearly everything
```

### Solution

1. **Balanced class weights** in loss function:
   ```python
   class_weights = torch.tensor([2.5, 0.83, 2.5])  # Inverse frequency
   criterion = nn.CrossEntropyLoss(weight=class_weights)
   ```

2. **Logit bias correction** (temporary fix for pre-trained model):
   ```python
   logit_correction = torch.tensor([3.0, -2.5, 1.0])  # [SELL, HOLD, BUY]
   adjusted_logits = raw_logits + logit_correction
   ```

3. **Confidence filtering** in ensemble:
   ```python
   if confidence < threshold:
       return 'HOLD', confidence
   ```

## Training Data

### Format

CSV file with OHLCV data:
```
timestamp,open,high,low,close,volume
2020-01-01 00:00:00,7195.24,7196.25,7175.46,7177.02,511.814901
```

### Features (60D total)

**Computed features (49D)**:
- Price returns: 1h, 6h, 24h, 72h, 168h
- Moving averages: 12h, 24h, 72h, 168h
- Volatility: 12h, 24h, 72h, 168h
- RSI (14-period)
- Volume indicators
- High-low range
- Momentum indicators

**Padding**: 11D zeros (for future order flow features)

### Labels

Generated from future price movements:
- **BUY**: Next 6h return > 1%
- **HOLD**: Next 6h return between -1% and 1%
- **SELL**: Next 6h return < -1%

## Ensemble Strategies

The framework supports multiple ensemble strategies:

1. **Voting** (default): Both models vote, majority wins
2. **Confidence-weighted**: Weight predictions by model confidence
3. **Disagreement**: Only trade when both models agree

```bash
# Example: Run with different strategies
python ensemble_backtest.py --strategy voting
python ensemble_backtest.py --strategy confidence_weighted
python ensemble_backtest.py --strategy disagreement
```

## Vast.ai Training

### Prerequisites

1. Create Vast.ai account
2. Rent a GPU instance (recommended: RTX 3090 or better)
3. Upload `btc_1h_2020_2024.csv` to Google Drive
4. Get the Google Drive file ID

### Setup Script

Edit `vast_ai_setup.sh` and replace `<YOUR_GOOGLE_DRIVE_FILE_ID>` with your file ID:

```bash
# Download data from Google Drive
gdown --id <YOUR_GOOGLE_DRIVE_FILE_ID> -O data/btc_1h_2020_2024.csv
```

### Run Training

```bash
wget https://raw.githubusercontent.com/nimallansa937/HIMARI-TESTING-SUITE/main/vast_ai_setup.sh && bash vast_ai_setup.sh
```

Training takes approximately 2-4 hours on RTX 3090.

## Results & Metrics

The backtest generates comprehensive metrics:

- **Return metrics**: Total return, CAGR, Sharpe ratio, Sortino ratio
- **Risk metrics**: Max drawdown, volatility, VaR, CVaR
- **Trading metrics**: Win rate, profit factor, avg profit/trade
- **Market correlation**: Beta coefficient

Results are saved to:
- `backtest_results.json` - Full metrics
- `backtest_trades.csv` - Trade-by-trade log

## Development

### Running Tests

```bash
# Test model loading
python test_model_load_flagtrader.py

# Validate model predictions
python diagnose_model_predictions.py

# Run backtest with logit correction
python end_to_end_backtest.py
```

### Adding New Features

1. Edit `compute_features()` in `train_flagtrader.py`
2. Ensure output is 49D (will be padded to 60D)
3. Update feature documentation

### Training Custom Models

```python
from flag_trader import FLAGTRADERModel

model = FLAGTRADERModel(
    state_dim=60,
    action_dim=3,
    d_model=768,      # Model dimension
    num_layers=12,    # Transformer layers
    num_heads=8,      # Attention heads
    lora_rank=16      # LoRA rank (higher = more capacity)
)
```

## Troubleshooting

### Model predicts only HOLD

- **Issue**: Model collapse due to class imbalance
- **Solution**: Retrain with balanced class weights (see `train_flagtrader.py`)

### Low Sharpe ratio

- **Issue**: Model too conservative or not diverse enough
- **Solution**:
  - Increase confidence threshold
  - Retrain with balanced weights
  - Use ensemble with multiple models

### Out of memory during training

- **Issue**: Batch size too large
- **Solution**: Reduce `--batch_size` (try 32 or 16)

### Slow inference

- **Issue**: Model running on CPU
- **Solution**: Ensure CUDA is available and model is on GPU

## Citation

```
@software{himari_flagtrader_2026,
  title={HIMARI FLAG-TRADER: Large Language Model for Trading},
  author={Your Name},
  year={2026},
  url={https://github.com/nimallansa937/HIMARI-TESTING-SUITE}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- LoRA implementation inspired by Microsoft's LoRA paper
- Transformer architecture based on "Attention Is All You Need"
- Backtesting framework inspired by Backtrader and VectorBT

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Last Updated**: 2026-01-19
**Status**: Ready for training on Vast.ai
