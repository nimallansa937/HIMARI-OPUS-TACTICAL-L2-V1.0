#!/usr/bin/env python3
"""
Convert enriched dataset from dataclass objects to plain numpy arrays.
This makes the pickle file portable without needing src.pipeline module.
"""

import pickle
import numpy as np
from pathlib import Path

INPUT_PATH = Path("C:/Users/chari/OneDrive/Documents/BTC DATA SETS/btc_1h_2020_2024_enriched_44f.pkl")
OUTPUT_PATH = Path("C:/Users/chari/OneDrive/Documents/BTC DATA SETS/btc_1h_2020_2024_enriched_44f_arrays.pkl")


def convert_samples_to_arrays(samples):
    """Convert list of EnrichedSample objects to dict of numpy arrays."""
    n = len(samples)

    # Get dimensions from first sample
    s0 = samples[0]
    feature_dim = len(s0.features_raw)

    # Pre-allocate arrays
    features_raw = np.zeros((n, feature_dim), dtype=np.float32)
    features_denoised = np.zeros((n, feature_dim), dtype=np.float32)
    regime_ids = np.zeros(n, dtype=np.int64)
    regime_confidences = np.zeros(n, dtype=np.float32)
    prices = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)

    # Fill arrays
    for i, s in enumerate(samples):
        features_raw[i] = s.features_raw
        features_denoised[i] = s.features_denoised
        regime_ids[i] = s.regime_id
        regime_confidences[i] = s.regime_confidence
        prices[i] = s.price
        returns[i] = s.returns

    return {
        'features_raw': features_raw,
        'features_denoised': features_denoised,
        'regime_ids': regime_ids,
        'regime_confidences': regime_confidences,
        'prices': prices,
        'returns': returns,
        'n_samples': n
    }


def main():
    print("Loading original dataset...")
    with open(INPUT_PATH, 'rb') as f:
        data = pickle.load(f)

    print(f"  Train samples: {len(data['train'])}")
    print(f"  Val samples: {len(data['val'])}")
    print(f"  Test samples: {len(data['test'])}")

    print("\nConverting to numpy arrays...")

    converted = {
        'train': convert_samples_to_arrays(data['train']),
        'val': convert_samples_to_arrays(data['val']),
        'test': convert_samples_to_arrays(data['test']),
        'metadata': data['metadata']
    }

    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Check file size
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")

    print("\nDone! Upload this file to Google Drive and update GDRIVE_FILE_ID in train_vast.py")


if __name__ == "__main__":
    main()
