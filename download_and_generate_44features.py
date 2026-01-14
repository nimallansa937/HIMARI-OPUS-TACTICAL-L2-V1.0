"""
Download fresh BTC data via CCXT and generate 44-feature enriched dataset.

This script:
1. Downloads BTC/USDT 1h data from Binance (2020-2024)
2. Computes 44 technical features
3. Runs the enriched dataset pipeline (EKF denoising + regime detection)
4. Saves to BTC DATA SETS folder
"""

import sys
import os
from pathlib import Path

# Setup paths
THIS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(THIS_DIR))

import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import time

# Import our modules
from src.pipeline.feature_engineer import FeatureEngineer, download_btc_data_ccxt


def download_and_process():
    """Download BTC data and create 44-feature dataset."""

    output_dir = Path("C:/Users/chari/OneDrive/Documents/BTC DATA SETS")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HIMARI 44-Feature Dataset Generator")
    print("=" * 70)
    print()

    # Step 1: Download BTC data via CCXT
    print("STEP 1: Downloading BTC/USDT 1h data from Binance...")
    print("-" * 50)

    start_date = "2020-01-01"
    end_date = "2024-12-31"

    try:
        df_raw = download_btc_data_ccxt(
            start_date=start_date,
            end_date=end_date,
            timeframe='1h',
            exchange='binance'
        )
        print(f"Downloaded {len(df_raw)} candles")
        print(f"Date range: {df_raw.index[0]} to {df_raw.index[-1]}")
    except Exception as e:
        print(f"CCXT download failed: {e}")
        print("Attempting to use existing raw data...")

        # Fallback to existing data
        raw_path = output_dir / "btc_1h_2020_2024_raw.pkl"
        if raw_path.exists():
            df_raw = pd.read_pickle(raw_path)
            print(f"Loaded existing data: {len(df_raw)} candles")
        else:
            # Try the 5-min data and resample
            fivemin_path = output_dir / "btc_5min_2020_2024.pkl"
            if fivemin_path.exists():
                print("Resampling 5-min data to 1-hour...")
                df_5min = pd.read_pickle(fivemin_path)
                df_raw = df_5min.resample('1h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                print(f"Resampled to {len(df_raw)} hourly candles")
            else:
                raise ValueError("No data source available")

    # Save raw data
    raw_output = output_dir / "btc_1h_2020_2024_raw.pkl"
    df_raw.to_pickle(raw_output)
    print(f"Saved raw data to: {raw_output}")

    print()
    print("STEP 2: Computing 44 technical features...")
    print("-" * 50)

    # Step 2: Compute 44 features
    engineer = FeatureEngineer()
    features = engineer.compute_features(df_raw)

    print(f"Features shape: {features.shape}")
    print(f"Feature names ({len(engineer.feature_names)}):")
    for i, name in enumerate(engineer.feature_names):
        print(f"  {i:2d}: {name}")

    # Combine OHLCV with features
    df_with_features = pd.concat([df_raw, features], axis=1)

    # Drop warmup period (NaN rows)
    df_with_features = df_with_features.dropna()
    print(f"\nAfter dropping warmup NaNs: {len(df_with_features)} samples")

    # Save 44-feature dataset
    features_output = output_dir / "btc_1h_2020_2024_44features.pkl"
    df_with_features.to_pickle(features_output)
    print(f"Saved 44-feature data to: {features_output}")

    print()
    print("STEP 3: Running enriched dataset pipeline (balanced regimes)...")
    print("-" * 50)

    # Step 3: Run enriched pipeline (EKF + Balanced Regime detection)
    from src.pipeline.dataset_generator import DatasetGenerator, generate_enriched_dataset

    # Generate enriched dataset with balanced regime detector
    enriched_output = output_dir / "btc_1h_2020_2024_enriched_44f.pkl"

    result = generate_enriched_dataset(
        raw_data_path=str(features_output),
        output_path=str(enriched_output),
        train_ratio=0.6,
        val_ratio=0.2,
        use_hmm=False  # Use balanced detector instead of HMM
    )

    print()
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print()
    print("Output files:")
    print(f"  Raw OHLCV:      {raw_output}")
    print(f"  44 Features:    {features_output}")
    print(f"  Enriched:       {enriched_output}")

    # Print regime distribution
    if result:
        train_samples = result['train']
        regime_counts = {}
        for sample in train_samples:
            r = sample.regime_id
            regime_counts[r] = regime_counts.get(r, 0) + 1

        print()
        print("Regime Distribution (Train):")
        regime_names = {0: "LOW_VOL", 1: "TRENDING", 2: "HIGH_VOL", 3: "CRISIS"}
        total = len(train_samples)
        for rid in sorted(regime_counts.keys()):
            count = regime_counts[rid]
            pct = count / total * 100
            print(f"  {regime_names.get(rid, f'REGIME_{rid}'):12s}: {count:6d} ({pct:5.1f}%)")

    return result


if __name__ == "__main__":
    download_and_process()
