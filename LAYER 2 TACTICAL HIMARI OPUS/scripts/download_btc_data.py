"""
HIMARI Layer 2 - BTC Data Downloader using CCXT
Downloads 5-minute BTC/USDT candles from Binance (2020-2024)

Usage:
    pip install ccxt pandas numpy
    python download_btc_data.py --output ./data/btc_5min_2020_2024.pkl
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import argparse
import pickle
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def download_btc_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    exchange_id: str = "binance",
    rate_limit_delay: float = 0.5,
) -> pd.DataFrame:
    """
    Download OHLCV data using CCXT.
    
    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Candlestick timeframe (e.g., "5m", "15m", "1h")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        exchange_id: Exchange to use (e.g., "binance", "bybit")
        rate_limit_delay: Delay between requests
        
    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Initializing {exchange_id} exchange...")
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}  # Use futures for more data
    })
    
    # Convert dates to timestamps
    start_ts = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end_date}T23:59:59Z")
    
    logger.info(f"Downloading {symbol} {timeframe} from {start_date} to {end_date}")
    
    all_candles = []
    current_ts = start_ts
    
    # Calculate expected candles for progress
    tf_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30, 
        '1h': 60, '4h': 240, '1d': 1440
    }
    minutes = tf_minutes.get(timeframe, 5)
    total_expected = (end_ts - start_ts) / (minutes * 60 * 1000)
    
    batch_count = 0
    while current_ts < end_ts:
        try:
            # Fetch batch (max 1000 candles per request)
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_ts,
                limit=1000
            )
            
            if not candles:
                break
                
            all_candles.extend(candles)
            
            # Move to next batch
            current_ts = candles[-1][0] + 1
            batch_count += 1
            
            # Progress update
            if batch_count % 50 == 0:
                progress = len(all_candles) / total_expected * 100
                logger.info(f"  Downloaded {len(all_candles):,} candles ({progress:.1f}%)")
            
            # Rate limit
            time.sleep(rate_limit_delay)
            
        except Exception as e:
            logger.warning(f"Error fetching batch: {e}. Retrying in 5s...")
            time.sleep(5)
    
    logger.info(f"Downloaded {len(all_candles):,} total candles")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Convert to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    logger.info(f"Final DataFrame: {len(df):,} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def compute_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute 44D feature vector from OHLCV data.
    
    Features:
        0-9: Price features
        10-19: Volatility features
        20-29: Technical indicators
        30-39: Volume features
        40-43: Regime/state features
    """
    logger.info("Computing 44D features...")
    n = len(df)
    features = np.zeros((n, 44), dtype=np.float32)
    
    # Extract price data
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    returns = np.diff(close, prepend=close[0]) / (close + 1e-8)
    
    # 0-9: Price features
    features[:, 0] = returns  # 1-step return
    features[:, 1] = np.roll(returns, 1)  # Lagged return
    features[:, 2] = _rolling_mean(returns, 5)
    features[:, 3] = _rolling_mean(returns, 20)
    features[:, 4] = _rolling_mean(returns, 50)
    ma_5 = _rolling_mean(close, 5)
    ma_20 = _rolling_mean(close, 20)
    features[:, 5] = close / (ma_5 + 1e-8) - 1
    features[:, 6] = close / (ma_20 + 1e-8) - 1
    features[:, 7] = ma_5 / (ma_20 + 1e-8) - 1
    features[:, 8] = (close - close.min()) / (close.max() - close.min() + 1e-8)
    features[:, 9] = np.log(close / (np.roll(close, 1) + 1e-8) + 1e-8)
    
    # 10-19: Volatility features
    features[:, 10] = _rolling_std(returns, 5)
    features[:, 11] = _rolling_std(returns, 20)
    features[:, 12] = _rolling_std(returns, 50)
    features[:, 13] = features[:, 10] / (features[:, 11] + 1e-8)
    features[:, 14] = _atr(high, low, close, 14)
    features[:, 15] = returns / (features[:, 10] + 1e-8)  # Sharpe-like
    features[:, 16] = _rolling_max(returns, 20) - _rolling_min(returns, 20)
    features[:, 17] = _ewma(features[:, 10], 0.1)
    features[:, 18] = np.where(features[:, 10] > features[:, 11], 1, 0)
    features[:, 19] = _rolling_skew(returns, 20)
    
    # 20-29: Technical indicators
    features[:, 20] = _rsi(returns, 14)
    macd, signal = _macd(close)
    features[:, 21] = macd
    features[:, 22] = signal
    features[:, 23] = macd - signal
    bb_upper, bb_lower = _bollinger_bands(close, 20)
    features[:, 24] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
    features[:, 25] = (bb_upper - bb_lower) / (close + 1e-8)
    features[:, 26] = _stochastic(close, high, low, 14)
    features[:, 27] = _williams_r(close, high, low, 14)
    features[:, 28] = _cci(high, low, close, 20)
    features[:, 29] = close / (np.roll(close, 10) + 1e-8) - 1
    
    # 30-39: Volume features
    vol_ma = _rolling_mean(volume, 20)
    features[:, 30] = volume / (vol_ma + 1e-8)
    features[:, 31] = np.sign(returns) * volume  # OBV-like
    features[:, 32] = _ewma(features[:, 30], 0.2)
    features[:, 33] = np.zeros(n)  # Funding rate placeholder
    features[:, 34] = np.diff(volume, prepend=volume[0])  # Volume delta
    features[:, 35] = np.where(volume > _rolling_mean(volume, 5), 1, 0)
    features[:, 36] = _rolling_mean(features[:, 31], 10)
    features[:, 37] = returns * volume / 1e9
    features[:, 38] = np.where(volume > 2 * vol_ma, 1, 0)  # Large trade
    features[:, 39] = (high - low) / (close + 1e-8)  # Spread proxy
    
    # 40-43: Regime/state features
    features[:, 40] = _rolling_mean(features[:, 18], 20)  # Regime prob
    features[:, 41] = 0  # Position placeholder
    features[:, 42] = 0  # PnL placeholder
    features[:, 43] = 1  # Confidence placeholder
    
    # Handle NaNs and infinities
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Z-score normalize each feature
    for i in range(features.shape[1]):
        std = np.std(features[:, i])
        if std > 1e-8:
            features[:, i] = (features[:, i] - np.mean(features[:, i])) / std
    
    logger.info(f"Features computed: shape={features.shape}")
    return features


# Helper functions
def _rolling_mean(x, window):
    result = np.convolve(x, np.ones(window)/window, mode='same')
    return result

def _rolling_std(x, window):
    result = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        result[i] = np.std(x[start:i+1]) if i > 0 else 0
    return result

def _rolling_max(x, window):
    result = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        result[i] = np.max(x[start:i+1])
    return result

def _rolling_min(x, window):
    result = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        result[i] = np.min(x[start:i+1])
    return result

def _rolling_skew(x, window):
    result = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        if i - start >= 2:
            data = x[start:i+1]
            mean = np.mean(data)
            std = np.std(data)
            if std > 1e-8:
                result[i] = np.mean(((data - mean) / std) ** 3)
    return result

def _ewma(x, alpha):
    result = np.zeros_like(x)
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = alpha * x[i] + (1 - alpha) * result[i-1]
    return result

def _rsi(returns, period=14):
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)
    avg_gain = _ewma(gains, 1/period)
    avg_loss = _ewma(losses, 1/period)
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    return rsi / 100

def _macd(prices, fast=12, slow=26, signal=9):
    ema_fast = _ewma(prices, 2/(fast+1))
    ema_slow = _ewma(prices, 2/(slow+1))
    macd_line = (ema_fast - ema_slow) / (prices + 1e-8)
    signal_line = _ewma(macd_line, 2/(signal+1))
    return macd_line, signal_line

def _bollinger_bands(prices, period=20, std_mult=2.0):
    ma = _rolling_mean(prices, period)
    std = _rolling_std(prices, period)
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    return upper, lower

def _atr(high, low, close, period=14):
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    atr = _ewma(tr, 1/period)
    return atr / (close + 1e-8)

def _stochastic(close, high, low, period=14):
    highest = _rolling_max(high, period)
    lowest = _rolling_min(low, period)
    k = (close - lowest) / (highest - lowest + 1e-8)
    return k

def _williams_r(close, high, low, period=14):
    return _stochastic(close, high, low, period) - 0.5

def _cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    tp_ma = _rolling_mean(tp, period)
    tp_std = _rolling_std(tp, period)
    cci = (tp - tp_ma) / (0.015 * tp_std + 1e-8)
    return cci / 100


def save_training_data(
    df: pd.DataFrame,
    features: np.ndarray,
    output_path: str,
):
    """Save data in format expected by Transformer-A2C trainer."""
    data = {
        'features': features,
        'prices': df['close'].values.astype(np.float32),
        'timestamps': pd.to_datetime(df['timestamp']).values,
        'ohlcv': df[['open', 'high', 'low', 'close', 'volume']].values,
        'metadata': {
            'symbol': 'BTC/USDT',
            'timeframe': '5m',
            'start': str(df['timestamp'].min()),
            'end': str(df['timestamp'].max()),
            'n_samples': len(df),
            'n_features': features.shape[1],
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Saved training data to {output_path}")
    logger.info(f"  Samples: {len(df):,}")
    logger.info(f"  Features: {features.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BTC data for Transformer-A2C training")
    parser.add_argument("--output", type=str, default="./data/btc_5min_2020_2024.pkl")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--exchange", type=str, default="binance")
    
    args = parser.parse_args()
    
    # Download OHLCV
    df = download_btc_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        exchange_id=args.exchange,
    )
    
    # Compute features
    features = compute_features(df)
    
    # Save
    save_training_data(df, features, args.output)
    
    print(f"\n✅ Done! Data saved to {args.output}")
    print(f"   Run training with: python scripts/train_transformer_a2c.py --data {args.output}")
