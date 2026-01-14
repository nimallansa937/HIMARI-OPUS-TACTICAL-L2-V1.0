"""
Download BTC 1-hour OHLCV data from Binance (2024-01-01 to 2026-01-14)
For Experiment 10 - Anti-overtrading PPO training

Run locally: python scripts/download_btc_1h_2024_2026.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import requests
import time
import pickle
from pathlib import Path


def download_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_date: str = "2024-01-01",
    end_date: str = "2026-01-14",
) -> pd.DataFrame:
    """
    Download historical klines from Binance API.
    
    Args:
        symbol: Trading pair
        interval: Candle interval (1h, 5m, etc.)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with OHLCV data
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Convert dates to milliseconds
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_klines = []
    current_ts = start_ts
    
    print(f"Downloading {symbol} {interval} data from {start_date} to {end_date}...")
    
    while current_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ts,
            "endTime": end_ts,
            "limit": 1000  # Max per request
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
                
            all_klines.extend(klines)
            current_ts = klines[-1][0] + 1  # Next candle after last
            
            print(f"  Downloaded {len(all_klines)} candles...")
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    
    # Process columns
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df.set_index("timestamp")
    
    return df


def engineer_features(df: pd.DataFrame) -> np.ndarray:
    """
    Engineer 44 features matching Layer 2 specification.
    
    Returns numpy array of shape (N, 44)
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. Price returns (5 features)
    features["ret_1"] = df["close"].pct_change(1)
    features["ret_2"] = df["close"].pct_change(2)
    features["ret_4"] = df["close"].pct_change(4)
    features["ret_8"] = df["close"].pct_change(8)
    features["ret_24"] = df["close"].pct_change(24)  # 24 hours
    
    # 2. Moving averages (5 features)
    for window in [8, 21, 55, 89, 144]:
        features[f"sma_{window}"] = df["close"] / df["close"].rolling(window).mean() - 1
    
    # 3. Volatility (5 features)
    for window in [8, 21, 55, 89, 144]:
        features[f"vol_{window}"] = df["close"].pct_change().rolling(window).std()
    
    # 4. RSI (3 features)
    for window in [14, 21, 55]:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-10)
        features[f"rsi_{window}"] = (rs / (1 + rs)) - 0.5  # Centered at 0
    
    # 5. MACD (3 features)
    ema_12 = df["close"].ewm(span=12).mean()
    ema_26 = df["close"].ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    features["macd"] = macd / df["close"]
    features["macd_signal"] = signal / df["close"]
    features["macd_hist"] = (macd - signal) / df["close"]
    
    # 6. Bollinger Bands (3 features)
    bb_sma = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    features["bb_upper"] = (df["close"] - (bb_sma + 2 * bb_std)) / df["close"]
    features["bb_lower"] = (df["close"] - (bb_sma - 2 * bb_std)) / df["close"]
    features["bb_width"] = (4 * bb_std) / bb_sma
    
    # 7. Volume features (5 features)
    features["vol_ratio_8"] = df["volume"] / df["volume"].rolling(8).mean()
    features["vol_ratio_21"] = df["volume"] / df["volume"].rolling(21).mean()
    features["vol_ratio_55"] = df["volume"] / df["volume"].rolling(55).mean()
    features["vol_trend"] = df["volume"].rolling(8).mean() / df["volume"].rolling(21).mean()
    features["vol_volatility"] = df["volume"].pct_change().rolling(21).std()
    
    # 8. High/Low features (4 features)
    features["hl_range"] = (df["high"] - df["low"]) / df["close"]
    features["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)
    features["high_dist_8"] = df["close"] / df["high"].rolling(8).max() - 1
    features["low_dist_8"] = df["close"] / df["low"].rolling(8).min() - 1
    
    # 9. Trend indicators (5 features)
    features["trend_8"] = np.sign(df["close"] - df["close"].shift(8))
    features["trend_21"] = np.sign(df["close"] - df["close"].shift(21))
    features["trend_55"] = np.sign(df["close"] - df["close"].shift(55))
    features["higher_highs"] = (df["high"] > df["high"].shift(1)).astype(float).rolling(8).mean()
    features["lower_lows"] = (df["low"] < df["low"].shift(1)).astype(float).rolling(8).mean()
    
    # 10. Momentum (3 features)
    features["roc_8"] = df["close"].pct_change(8)
    features["roc_21"] = df["close"].pct_change(21)
    features["momentum_divergence"] = features["roc_8"] - features["roc_21"]
    
    # 11. ATR (3 features) - for 1h data, adjust windows
    tr = pd.DataFrame({
        "hl": df["high"] - df["low"],
        "hc": abs(df["high"] - df["close"].shift(1)),
        "lc": abs(df["low"] - df["close"].shift(1))
    }).max(axis=1)
    features["atr_14"] = tr.rolling(14).mean() / df["close"]
    features["atr_21"] = tr.rolling(21).mean() / df["close"]
    features["atr_ratio"] = features["atr_14"] / (features["atr_21"] + 1e-10)
    
    # Fill NaN and clip
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    features = features.clip(-10, 10)
    
    print(f"Engineered {len(features.columns)} features")
    
    return features.values.astype(np.float32)


def main():
    # Download data
    df = download_binance_klines(
        symbol="BTCUSDT",
        interval="1h",
        start_date="2024-01-01",
        end_date="2026-01-14"
    )
    
    print(f"\nDownloaded {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Engineer features
    features = engineer_features(df)
    prices = df["close"].values.astype(np.float32)
    
    # Trim to match (remove first 144 rows with NaN features)
    start_idx = 144
    features = features[start_idx:]
    prices = prices[start_idx:]
    
    print(f"\nFinal dataset:")
    print(f"  Features shape: {features.shape}")
    print(f"  Prices shape: {prices.shape}")
    
    # Save
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "btc_1h_2024_2026.pkl"
    with open(output_path, "wb") as f:
        pickle.dump({
            "features": features,
            "prices": prices,
            "info": {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "start_date": "2024-01-01",
                "end_date": "2026-01-14",
                "num_samples": len(prices),
                "feature_dim": features.shape[1]
            }
        }, f)
    
    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
