# Part A: Data Preprocessing

8 methods for market data preprocessing and normalization.

## Methods

- A1: Extended Kalman Filter (EKF) - Nonlinear denoising
- A2: VecNormalize - Running normalization with clipping
- A3: TimeGAN+ - Synthetic minority augmentation
- A4: Conversational Autoencoder - Regime-specific compression
- A5: Fractional Differencing - Stationarity preservation
- A6: Feature Engineering - Technical indicators
- A7: Outlier Handling - Robust preprocessing
- A8: Missing Data - Forward-fill/interpolation

## Usage

```python
from src.preprocessing import TradingKalmanFilter, VecNormalize, DataAugmentor
```
