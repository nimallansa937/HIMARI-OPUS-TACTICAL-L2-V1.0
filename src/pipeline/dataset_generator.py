"""
Enriched Dataset Generator

Orchestrates the full pipeline:
    Raw .pkl → EKF Denoise → Regime Detection → Save enriched .pkl

Creates training-ready datasets with:
    - Denoised features (EKF-filtered)
    - Regime labels (0-3) with confidence scores
    - Pre-computed metadata for fast training

Output Schema per Sample:
    - timestamp: pd.Timestamp
    - features_raw: np.ndarray (feature_dim,)
    - features_denoised: np.ndarray (feature_dim,)
    - regime_id: int (0-3)
    - regime_confidence: float (0-1)
    - price: float
    - returns: float

Usage:
    python -m src.pipeline.dataset_generator \
        --input data/btc_1h_raw.pkl \
        --output data/btc_1h_enriched.pkl

Author: HIMARI Development Team
Date: January 2026
"""

import argparse
import pickle
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
from .ekf_denoiser import EKFDenoiser
from .regime_detector import RegimeDetector, RegimeLabel


@dataclass
class EnrichedSample:
    """Single enriched sample for training."""
    timestamp: pd.Timestamp
    features_raw: np.ndarray        # Original features (feature_dim,)
    features_denoised: np.ndarray   # EKF-filtered features (feature_dim,)
    regime_id: int                  # 0=LOW_VOL, 1=TRENDING, 2=HIGH_VOL, 3=CRISIS
    regime_confidence: float        # 0-1 confidence score
    price: float                    # Close price
    returns: float                  # Log return


@dataclass
class DatasetMetadata:
    """Metadata about the generated dataset."""
    total_samples: int
    train_samples: int
    val_samples: int
    test_samples: int
    feature_dim: int
    regime_distribution: Dict[str, float]
    price_range: Tuple[float, float]
    date_range: Tuple[str, str]
    ekf_noise_reduction: float      # MSE reduction %
    generator_version: str = "1.0.0"


class DatasetGenerator:
    """
    Generates enriched training datasets from raw OHLCV data.

    Pipeline:
        1. Load raw data (DataFrame or numpy array)
        2. EKF denoise all features
        3. Detect regimes using StudentTAHHMM
        4. Package into EnrichedSample objects
        5. Split into train/val/test
        6. Save with metadata

    Parameters:
        ekf_process_noise: EKF Q parameter (default 0.001)
        ekf_obs_noise: EKF R parameter (default 0.01)
        regime_vol_window: Regime detector volatility window (default 24)
        price_col: Column name or index for price (default 'close' or 0)
        returns_col: Column name or index for returns (default 'returns_1h' or 1)
    """

    def __init__(
        self,
        ekf_process_noise: float = 0.001,
        ekf_obs_noise: float = 0.01,
        regime_vol_window: int = 24,
        price_col: str = 'close',
        returns_col: str = 'returns_1h',
        use_hmm: bool = False  # Default to balanced detector
    ):
        self.ekf_process_noise = ekf_process_noise
        self.ekf_obs_noise = ekf_obs_noise
        self.regime_vol_window = regime_vol_window
        self.price_col = price_col
        self.returns_col = returns_col
        self.use_hmm = use_hmm

        # Components (initialized lazily based on data dim)
        self.ekf: Optional[EKFDenoiser] = None
        self.regime_detector: Optional[RegimeDetector] = None

    def generate(
        self,
        input_path: str,
        output_path: str,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2
    ) -> Dict:
        """
        Generate enriched dataset from raw data file.

        Args:
            input_path: Path to raw data (.pkl, .csv, .parquet)
            output_path: Path to save enriched dataset
            train_ratio: Fraction for training (default 0.6)
            val_ratio: Fraction for validation (default 0.2)

        Returns:
            Dict with 'train', 'val', 'test' sample lists and 'metadata'
        """
        print(f"{'=' * 60}")
        print("HIMARI Enriched Dataset Generator")
        print(f"{'=' * 60}")

        # Step 1: Load data
        print(f"\n[1/4] Loading data from {input_path}...")
        df, prices, returns = self._load_data(input_path)
        feature_dim = df.shape[1]
        T = len(df)
        print(f"      Loaded {T} samples, {feature_dim} features")

        # Step 2: Initialize components
        print(f"\n[2/4] Initializing pipeline components...")
        self.ekf = EKFDenoiser(
            dim=feature_dim,
            process_noise=self.ekf_process_noise,
            obs_noise=self.ekf_obs_noise
        )
        self.regime_detector = RegimeDetector(
            vol_window=self.regime_vol_window,
            use_hmm=self.use_hmm
        )
        detector_type = "HMM" if self.regime_detector.use_hmm else "Balanced"
        print(f"      EKF: dim={feature_dim}, Q={self.ekf_process_noise}, R={self.ekf_obs_noise}")
        print(f"      Regime: vol_window={self.regime_vol_window}, detector={detector_type}")

        # Step 3: Process all samples
        print(f"\n[3/4] Processing samples through pipeline...")
        samples = self._process_samples(df, prices, returns)

        # Step 4: Split and save
        print(f"\n[4/4] Splitting and saving dataset...")
        dataset = self._split_and_package(
            samples, df, prices, train_ratio, val_ratio
        )

        # Save
        self._save_dataset(dataset, output_path)

        return dataset

    def _load_data(self, path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load data from various formats."""
        path = Path(path)

        if path.suffix == '.pkl':
            df = pd.read_pickle(path)
        elif path.suffix == '.csv':
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        elif path.suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Extract price and returns
        if isinstance(self.price_col, str) and self.price_col in df.columns:
            prices = df[self.price_col].values
        elif isinstance(self.price_col, int):
            prices = df.iloc[:, self.price_col].values
        else:
            # Try to find price column
            price_cols = [c for c in df.columns if 'close' in c.lower() or 'price' in c.lower()]
            if price_cols:
                prices = df[price_cols[0]].values
            else:
                prices = df.iloc[:, 0].values  # Default to first column
                print(f"      Warning: Using first column as price")

        if isinstance(self.returns_col, str) and self.returns_col in df.columns:
            returns = df[self.returns_col].values
        elif isinstance(self.returns_col, int):
            returns = df.iloc[:, self.returns_col].values
        else:
            # Compute returns from prices
            returns = np.zeros(len(prices))
            returns[1:] = np.diff(prices) / prices[:-1]
            print(f"      Warning: Computing returns from prices")

        return df, prices.astype(np.float64), returns.astype(np.float64)

    def _process_samples(
        self,
        df: pd.DataFrame,
        prices: np.ndarray,
        returns: np.ndarray
    ) -> List[EnrichedSample]:
        """Process all samples through EKF and regime detection."""
        T = len(df)
        samples = []

        # Track for noise reduction calculation
        raw_features_all = []
        denoised_features_all = []

        # Fit regime detector on initial portion (only if using HMM)
        if self.regime_detector.use_hmm:
            fit_size = min(1000, T // 3)
            if fit_size >= 100:
                print(f"      Fitting HMM on first {fit_size} samples...")
                self.regime_detector.fit(returns[:fit_size])

        # Process with progress bar
        for idx in tqdm(range(T), desc="      Processing"):
            row = df.iloc[idx]
            features_raw = row.values.astype(np.float32)

            # Get timestamp
            if isinstance(df.index, pd.DatetimeIndex):
                timestamp = df.index[idx]
            else:
                timestamp = pd.Timestamp.now()

            # EKF denoise
            ekf_result = self.ekf.update(features_raw)
            features_denoised = ekf_result.state.astype(np.float32)

            # Regime detection
            regime_result = self.regime_detector.update(
                price=float(prices[idx]),
                returns=float(returns[idx])
            )

            # Create sample
            sample = EnrichedSample(
                timestamp=timestamp,
                features_raw=features_raw,
                features_denoised=features_denoised,
                regime_id=regime_result.regime_id,
                regime_confidence=regime_result.confidence,
                price=float(prices[idx]),
                returns=float(returns[idx])
            )
            samples.append(sample)

            # Track for stats
            raw_features_all.append(features_raw)
            denoised_features_all.append(features_denoised)

        # Compute noise reduction
        raw_arr = np.array(raw_features_all)
        denoised_arr = np.array(denoised_features_all)

        # Estimate noise reduction by variance comparison
        raw_var = np.var(np.diff(raw_arr, axis=0))
        denoised_var = np.var(np.diff(denoised_arr, axis=0))
        noise_reduction = (raw_var - denoised_var) / raw_var * 100 if raw_var > 0 else 0

        print(f"\n      EKF noise reduction: {noise_reduction:.1f}%")

        return samples

    def _split_and_package(
        self,
        samples: List[EnrichedSample],
        df: pd.DataFrame,
        prices: np.ndarray,
        train_ratio: float,
        val_ratio: float
    ) -> Dict:
        """Split samples and create final dataset dict."""
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]

        # Compute regime distribution
        all_regimes = [s.regime_id for s in samples]
        regime_dist = {
            label.name: all_regimes.count(label.value) / len(all_regimes)
            for label in RegimeLabel
        }

        # Date range
        if isinstance(df.index, pd.DatetimeIndex):
            date_range = (str(df.index[0]), str(df.index[-1]))
        else:
            date_range = ("N/A", "N/A")

        # Compute EKF noise reduction for metadata
        raw_arr = np.array([s.features_raw for s in samples])
        denoised_arr = np.array([s.features_denoised for s in samples])
        raw_var = np.var(np.diff(raw_arr, axis=0))
        denoised_var = np.var(np.diff(denoised_arr, axis=0))
        noise_reduction = (raw_var - denoised_var) / raw_var * 100 if raw_var > 0 else 0

        metadata = DatasetMetadata(
            total_samples=n,
            train_samples=len(train_samples),
            val_samples=len(val_samples),
            test_samples=len(test_samples),
            feature_dim=df.shape[1],
            regime_distribution=regime_dist,
            price_range=(float(np.min(prices)), float(np.max(prices))),
            date_range=date_range,
            ekf_noise_reduction=float(noise_reduction)
        )

        print(f"\n      Train: {len(train_samples)} samples ({train_ratio*100:.0f}%)")
        print(f"      Val:   {len(val_samples)} samples ({val_ratio*100:.0f}%)")
        print(f"      Test:  {len(test_samples)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
        print(f"\n      Regime Distribution:")
        for regime, pct in regime_dist.items():
            print(f"        {regime:12s}: {pct*100:5.1f}%")

        return {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples,
            'metadata': asdict(metadata)
        }

    def _save_dataset(self, dataset: Dict, output_path: str) -> None:
        """Save dataset to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Compute file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n      Saved to: {output_path}")
        print(f"      File size: {size_mb:.1f} MB")


def generate_enriched_dataset(
    raw_data_path: str,
    output_path: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    **kwargs
) -> Dict:
    """
    Convenience function for generating enriched dataset.

    Args:
        raw_data_path: Path to raw data file
        output_path: Path to save enriched dataset
        train_ratio: Training set fraction
        val_ratio: Validation set fraction
        **kwargs: Additional arguments for DatasetGenerator

    Returns:
        Generated dataset dict
    """
    generator = DatasetGenerator(**kwargs)
    return generator.generate(
        input_path=raw_data_path,
        output_path=output_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate enriched training dataset from raw OHLCV data"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input raw data (.pkl, .csv, .parquet)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output enriched dataset (.pkl)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Fraction for training set (default: 0.6)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction for validation set (default: 0.2)"
    )
    parser.add_argument(
        "--ekf-q",
        type=float,
        default=0.001,
        help="EKF process noise (default: 0.001)"
    )
    parser.add_argument(
        "--ekf-r",
        type=float,
        default=0.01,
        help="EKF observation noise (default: 0.01)"
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=24,
        help="Regime volatility window (default: 24)"
    )

    args = parser.parse_args()

    dataset = generate_enriched_dataset(
        raw_data_path=args.input,
        output_path=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        ekf_process_noise=args.ekf_q,
        ekf_obs_noise=args.ekf_r,
        regime_vol_window=args.vol_window
    )

    print(f"\n{'=' * 60}")
    print("✅ Dataset generation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
