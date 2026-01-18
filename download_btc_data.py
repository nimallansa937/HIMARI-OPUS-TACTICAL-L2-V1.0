#!/usr/bin/env python3
"""
Download BTC 1H data from Binance and save to local storage.
"""

import os
import requests
import pandas as pd
from datetime import datetime

OUTPUT_DIR = r"C:\Users\chari\OneDrive\Documents\BTC DATA SETS"

def download_btc_data(start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
    """Download BTC data from Binance."""

    print(f"Downloading BTC {interval} data from {start_date} to {end_date}...")

    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

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
        current_ts = data[-1][0] + 1

        if len(all_data) % 5000 == 0:
            print(f"  Downloaded {len(all_data)} candles...")

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df = df.set_index('timestamp')
    df = df[['open', 'high', 'low', 'close', 'volume']]

    print(f"Downloaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    return df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download Training Data: 2020-2024
    print("\n" + "=" * 60)
    print("Downloading Training Data (2020-2024)")
    print("=" * 60)

    train_df = download_btc_data("2020-01-01", "2024-12-31", "1h")
    train_path = os.path.join(OUTPUT_DIR, "btc_1h_2020_2024.csv")
    train_df.to_csv(train_path)
    print(f"Saved to: {train_path}")

    # Download Test Data: 2025-2026
    print("\n" + "=" * 60)
    print("Downloading Test Data (2025-2026)")
    print("=" * 60)

    test_df = download_btc_data("2025-01-01", "2026-01-15", "1h")
    test_path = os.path.join(OUTPUT_DIR, "btc_1h_2025_2026.csv")
    test_df.to_csv(test_path)
    print(f"Saved to: {test_path}")

    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print(f"  - btc_1h_2020_2024.csv ({len(train_df)} candles)")
    print(f"  - btc_1h_2025_2026.csv ({len(test_df)} candles)")

if __name__ == "__main__":
    main()
