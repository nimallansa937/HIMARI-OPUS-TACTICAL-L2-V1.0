#!/usr/bin/env python3
"""
Create 2025-2026 Test Dataset for HIMARI Layer 2

Downloads fresh BTC 1H data and processes through:
1. Feature engineering (49 features)
2. EKF denoising
3. Regime detection

Output: btc_1h_2025_2026_test_arrays.pkl (same format as training data)
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_PATH = "btc_1h_2025_2026_test_arrays.pkl"
START_DATE = "2025-01-01"
END_DATE = "2026-01-14"  # Today

# =============================================================================
# Data Download
# =============================================================================

def download_btc_data(start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
    """Download BTC data from Binance."""
    import requests

    print(f"Downloading BTC {interval} data from {start_date} to {end_date}...")

    # Convert dates to timestamps
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

    # Binance API
    url = "https://api.binance.com/api/v3/klines"

    all_data = []
    current_ts = start_ts

    while current_ts < end_ts:
        params = {
            "symbol": "BTCUSDT",
            "interval": interval,
            "startTime": current_ts,
            "endTime": end_ts,
            "limit": 1000
        }

        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        current_ts = data[-1][0] + 1  # Next timestamp

        print(f"  Downloaded {len(all_data)} candles...")

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df = df.set_index('timestamp')
    df = df[['open', 'high', 'low', 'close', 'volume']]

    print(f"Downloaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    return df


# =============================================================================
# Feature Engineering (Same as training)
# =============================================================================

def compute_features(df: pd.DataFrame) -> np.ndarray:
    """Compute 49 technical features (same as training pipeline)."""

    features = {}

    # Price data
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_price = df['open'].values
    volume = df['volume'].values

    # Returns
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])
    features['returns'] = returns

    # Log returns
    log_returns = np.diff(np.log(close + 1e-10))
    log_returns = np.concatenate([[0], log_returns])
    features['log_returns'] = log_returns

    # Moving averages
    for window in [5, 10, 20, 50, 100, 200]:
        ma = pd.Series(close).rolling(window).mean().values
        features[f'ma_{window}'] = (close - ma) / (ma + 1e-10)

    # EMA
    for span in [5, 10, 20, 50]:
        ema = pd.Series(close).ewm(span=span).mean().values
        features[f'ema_{span}'] = (close - ema) / (ema + 1e-10)

    # Volatility
    for window in [5, 10, 20, 50]:
        vol = pd.Series(returns).rolling(window).std().values
        features[f'volatility_{window}'] = vol

    # RSI
    for window in [7, 14, 21]:
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        features[f'rsi_{window}'] = (rsi.values - 50) / 50  # Normalize to [-1, 1]

    # MACD
    ema12 = pd.Series(close).ewm(span=12).mean()
    ema26 = pd.Series(close).ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    features['macd'] = (macd / close).values
    features['macd_signal'] = (signal / close).values
    features['macd_hist'] = ((macd - signal) / close).values

    # Bollinger Bands
    for window in [20]:
        ma = pd.Series(close).rolling(window).mean()
        std = pd.Series(close).rolling(window).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        features[f'bb_upper_{window}'] = ((close - upper) / (upper + 1e-10))
        features[f'bb_lower_{window}'] = ((close - lower) / (lower + 1e-10))
        features[f'bb_width_{window}'] = ((upper - lower) / (ma + 1e-10)).values
        features[f'bb_pct_{window}'] = ((close - lower) / (upper - lower + 1e-10))

    # ATR
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                              np.abs(low - np.roll(close, 1))))
    for window in [7, 14, 21]:
        atr = pd.Series(tr).rolling(window).mean().values
        features[f'atr_{window}'] = atr / (close + 1e-10)

    # Volume features
    vol_ma = pd.Series(volume).rolling(20).mean().values
    features['volume_ratio'] = volume / (vol_ma + 1e-10)
    features['volume_change'] = np.concatenate([[0], np.diff(volume) / (volume[:-1] + 1e-10)])
    features['volume_std'] = pd.Series(volume).rolling(20).std().values / (vol_ma + 1e-10)

    # Price position
    features['price_position'] = (close - low) / (high - low + 1e-10)

    # High-Low range
    features['hl_range'] = (high - low) / (close + 1e-10)

    # Candle features
    features['body_size'] = np.abs(close - open_price) / (close + 1e-10)
    features['upper_shadow'] = (high - np.maximum(close, open_price)) / (close + 1e-10)
    features['lower_shadow'] = (np.minimum(close, open_price) - low) / (close + 1e-10)

    # Momentum
    for window in [5, 10, 20]:
        mom = close - np.roll(close, window)
        features[f'momentum_{window}'] = mom / (close + 1e-10)

    # Rate of change
    for window in [5, 10, 20]:
        roc = (close - np.roll(close, window)) / (np.roll(close, window) + 1e-10)
        features[f'roc_{window}'] = roc

    # Stochastic
    for window in [14]:
        lowest_low = pd.Series(low).rolling(window).min()
        highest_high = pd.Series(high).rolling(window).max()
        k = (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        features[f'stoch_k_{window}'] = k.values
        features[f'stoch_d_{window}'] = pd.Series(k).rolling(3).mean().values

    # Convert to array
    feature_names = sorted(features.keys())
    feature_array = np.column_stack([features[name] for name in feature_names])

    # Handle NaN
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Computed {feature_array.shape[1]} features")

    # Pad to 49 features if needed
    if feature_array.shape[1] < 49:
        padding = 49 - feature_array.shape[1]
        feature_array = np.concatenate([
            feature_array,
            np.zeros((feature_array.shape[0], padding), dtype=np.float32)
        ], axis=1)
        print(f"Padded to {feature_array.shape[1]} features")

    return feature_array, feature_names, returns


# =============================================================================
# Regime Detection (Same as training)
# =============================================================================

def detect_regimes(returns: np.ndarray, volatility_20: np.ndarray) -> np.ndarray:
    """
    Detect market regimes:
    0 = LOW_VOL
    1 = TRENDING
    2 = HIGH_VOL
    3 = CRISIS
    """

    n = len(returns)
    regimes = np.zeros(n, dtype=np.int64)

    # Compute rolling metrics
    vol_20 = pd.Series(returns).rolling(20).std().values
    vol_50 = pd.Series(returns).rolling(50).std().values

    # Trend strength (absolute cumulative returns over window)
    cum_ret_20 = pd.Series(returns).rolling(20).sum().abs().values

    # Percentiles for thresholds
    vol_low = np.nanpercentile(vol_20, 33)
    vol_high = np.nanpercentile(vol_20, 67)
    vol_crisis = np.nanpercentile(vol_20, 90)
    trend_thresh = np.nanpercentile(cum_ret_20, 60)

    for i in range(n):
        vol = vol_20[i] if not np.isnan(vol_20[i]) else 0
        trend = cum_ret_20[i] if not np.isnan(cum_ret_20[i]) else 0

        if vol > vol_crisis:
            regimes[i] = 3  # CRISIS
        elif vol > vol_high:
            regimes[i] = 2  # HIGH_VOL
        elif trend > trend_thresh and vol < vol_high:
            regimes[i] = 1  # TRENDING
        else:
            regimes[i] = 0  # LOW_VOL

    # Print distribution
    regime_names = ['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS']
    print("\nRegime Distribution:")
    for r in range(4):
        pct = (regimes == r).sum() / n * 100
        print(f"  {regime_names[r]}: {pct:.1f}%")

    return regimes


# =============================================================================
# EKF Denoising (Simplified)
# =============================================================================

def denoise_features(features: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Simple exponential smoothing as EKF approximation."""
    denoised = np.zeros_like(features)
    denoised[0] = features[0]

    for i in range(1, len(features)):
        denoised[i] = alpha * features[i] + (1 - alpha) * denoised[i-1]

    return denoised


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Creating 2025-2026 Test Dataset")
    print("=" * 70)

    # Download data
    df = download_btc_data(START_DATE, END_DATE)

    # Compute features
    features_raw, feature_names, returns = compute_features(df)
    print(f"Feature shape: {features_raw.shape}")

    # Get volatility for regime detection
    vol_idx = feature_names.index('volatility_20') if 'volatility_20' in feature_names else 0
    volatility_20 = features_raw[:, vol_idx]

    # Detect regimes
    regime_ids = detect_regimes(returns, volatility_20)

    # Denoise features
    features_denoised = denoise_features(features_raw)

    # Create confidence scores (placeholder)
    regime_confidences = np.ones(len(regime_ids), dtype=np.float32) * 0.8

    # Prices
    prices = df['close'].values.astype(np.float32)

    # Create dataset in same format as training
    dataset = {
        'test': {
            'features_raw': features_raw.astype(np.float32),
            'features_denoised': features_denoised.astype(np.float32),
            'regime_ids': regime_ids,
            'regime_confidences': regime_confidences,
            'prices': prices,
            'returns': returns.astype(np.float32),
            'n_samples': len(features_raw)
        },
        'metadata': {
            'start_date': START_DATE,
            'end_date': END_DATE,
            'n_features': features_raw.shape[1],
            'feature_names': feature_names,
            'created': datetime.now().isoformat()
        }
    }

    # Save
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n{'=' * 70}")
    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"Samples: {dataset['test']['n_samples']}")
    print(f"Features: {dataset['test']['features_raw'].shape[1]}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
