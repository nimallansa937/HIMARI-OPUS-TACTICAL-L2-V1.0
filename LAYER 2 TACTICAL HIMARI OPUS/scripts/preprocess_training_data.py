"""
HIMARI Layer 2 - Training Data Preprocessor
Converts raw OHLCV data to 60D feature vectors using HIMARI Layer 1 Signal Feed.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add Layer 1 to path
SIGNAL_LAYER_PATH = "c:/Users/chari/OneDrive/Documents/HIMARI OPUS 2/HIMARI SIGNAL LAYER"
sys.path.insert(0, SIGNAL_LAYER_PATH)

try:
    from feature_vector import FeatureVectorGenerator
    FEATURE_EXTRACTION_AVAILABLE = True
except ImportError:
    logger.warning("Could not import FeatureVectorGenerator from Layer 1")
    FEATURE_EXTRACTION_AVAILABLE = False


class TrainingDataPreprocessor:
    """
    Preprocesses raw OHLCV data into training-ready format for Layer 2.
    Uses HIMARI Layer 1 feature extraction to generate 60D feature vectors.
    """
    
    def __init__(self, input_dir: str = './data/raw', output_dir: str = './data'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if FEATURE_EXTRACTION_AVAILABLE:
            self.feature_gen = FeatureVectorGenerator()
            logger.info("✅ Using HIMARI Layer 1 feature extraction")
        else:
            self.feature_gen = None
            logger.warning("⚠️  Layer 1 feature extraction not available, using simplified features")
    
    def load_ohlcv_data(self, symbol: str) -> pd.DataFrame:
        """Load OHLCV data from CSV file."""
        safe_symbol = symbol.replace('/', '_')
        filepath = self.input_dir / f"{safe_symbol}_5m_6m.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"📥 Loading data from {filepath}...")
        df = pd.read_csv(filepath, index_col='datetime', parse_dates=True)
        
        logger.info(f"  Loaded {len(df):,} rows")
        logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def generate_simple_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate simplified 60D features using technical indicators.
        Fallback when Layer 1 feature extraction is not available.
        """
        logger.info("Generating simplified 60D feature vectors...")
        
        features_list = []
        
        # Calculate technical indicators
        for i in range(len(df)):
            if i < 50:  # Need history for indicators
                continue
            
            window = df.iloc[max(0, i-50):i+1]
            
            features = []
            
            # Tier 1: Price Features (10 features)
            features.append(window['close'].iloc[-1] / window['close'].iloc[0] - 1)  # Returns
            features.append(window['high'].max() / window['low'].min() - 1)  # Range
            features.append((window['close'].iloc[-1] - window['open'].iloc[0]) / window['open'].iloc[0])  # Net change
            features.extend([(window['close'].iloc[-1] - window['close'].iloc[-k]) / window['close'].iloc[-k] 
                           for k in [5, 10, 20]])  # Multi-period returns
            features.append(window['close'].std() / window['close'].mean())  # CoV
            features.append((window['high'] - window['low']).mean() / window['close'].mean())  # Avg range
            features.append(window['volume'].iloc[-1] / window['volume'].mean())  # Volume ratio
            features.append((window['close'] - window['close'].mean()) / window['close'].std())  # Z-score
            
            # Tier 2: Momentum Features (10 features)
            rsi = self._calculate_rsi(window['close'])
            macd, signal = self._calculate_macd(window['close'])
            features.extend([rsi, macd, signal, macd - signal])
            features.extend([window['close'].rolling(k).mean().iloc[-1] for k in [5, 10, 20]])
            features.append((window['close'].iloc[-1] - window['close'].rolling(20).mean().iloc[-1]) / window['close'].std())
            features.extend([window['close'].pct_change().mean(), window['close'].pct_change().std()])
            
            # Tier 3: Volatility Features (10 features)
            features.extend([window['close'].pct_change().std() * np.sqrt(k) for k in [5, 10, 20]])
            features.append((window['high'] - window['low']).mean())
            features.append((window['high'] - window['low']).std())
            features.extend([window['close'].rolling(k).std().iloc[-1] for k in [5, 10, 20]])
            features.append(window['high'].max() - window['low'].min())
            features.append((window['close'].iloc[-1] - window['close'].rolling(20).mean().iloc[-1]) / (2 * window['close'].rolling(20).std().iloc[-1]))
            
            # Tier 4: Volume Features (10 features)
            features.extend([window['volume'].rolling(k).mean().iloc[-1] for k in [5, 10, 20]])
            features.append(window['volume'].std() / window['volume'].mean())
            features.append(window['volume'].iloc[-1] / window['volume'].rolling(20).mean().iloc[-1])
            # Volume-weighted price momentum for different windows
            features.append((window['close'].pct_change() * window['volume']).rolling(5).mean().iloc[-1])
            features.append((window['close'].pct_change() * window['volume']).rolling(10).mean().iloc[-1])
            features.append((window['close'].pct_change() * window['volume']).rolling(20).mean().iloc[-1])
            features.append(window['volume'].max() / window['volume'].min())
            features.append(window['volume'].iloc[-1] - window['volume'].mean())
            
            # Tier 5: Pattern Features (10 features) - Convert booleans to float
            features.extend([float(window['close'].iloc[-1] > window['close'].rolling(k).mean().iloc[-1]) for k in [5, 10, 20]])
            features.append(float(window['close'].rolling(10).mean().iloc[-1] > window['close'].rolling(20).mean().iloc[-1]))
            features.extend([float(window['close'].iloc[-1] > window['high'].rolling(k).max().iloc[-1] * 0.99) for k in [10, 20]])
            features.extend([float(window['close'].iloc[-1] < window['low'].rolling(k).min().iloc[-1] * 1.01) for k in [10, 20]])
            bb_upper = window['close'].rolling(20).mean().iloc[-1] + 2 * window['close'].rolling(20).std().iloc[-1]
            bb_lower = window['close'].rolling(20).mean().iloc[-1] - 2 * window['close'].rolling(20).std().iloc[-1]
            features.append((window['close'].iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5)
            features.append(float((window['close'].iloc[-2] > window['open'].iloc[-2]) and (window['close'].iloc[-1] > window['open'].iloc[-1])))
            
            # Tier 6: Time Features (10 features)
            timestamp = window.index[-1]
            features.append(timestamp.hour / 24)
            features.append(timestamp.dayofweek / 7)
            features.append(timestamp.day / 31)
            features.append(timestamp.month / 12)
            features.append(np.sin(2 * np.pi * timestamp.hour / 24))
            features.append(np.cos(2 * np.pi * timestamp.hour / 24))
            features.append(np.sin(2 * np.pi * timestamp.dayofweek / 7))
            features.append(np.cos(2 * np.pi * timestamp.dayofweek / 7))
            features.append((timestamp - df.index[0]).total_seconds() / (365.25 * 24 * 3600))
            features.append(1.0 if (timestamp.dayofweek >= 5) else 0.0)  # Weekend
            
            # Ensure exactly 60 features
            if len(features) < 60:
                features.extend([0.0] * (60 - len(features)))
            elif len(features) > 60:
                features = features[:60]

            # Convert all features to scalar floats (debug nested arrays)
            features_clean = []
            for idx, f in enumerate(features):
                try:
                    # Convert to float, handling pandas Series, numpy arrays, etc.
                    if isinstance(f, (pd.Series, np.ndarray)):
                        f_val = float(f.iloc[0]) if hasattr(f, 'iloc') else float(f[0])
                    else:
                        f_val = float(f)

                    # Replace NaN/Inf with 0
                    if np.isnan(f_val) or np.isinf(f_val):
                        f_val = 0.0

                    features_clean.append(f_val)
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Feature {idx} conversion error at sample {i}: {type(f)} -> {e}")
                    features_clean.append(0.0)

            features_list.append(features_clean)
        
        features_array = np.array(features_list, dtype=np.float32)
        logger.info(f"✅ Generated {len(features_array):,} feature vectors of {features_array.shape[1]}D")
        
        return features_array
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        loss_val = loss.iloc[-1]
        gain_val = gain.iloc[-1]
        if loss_val == 0:
            return 100.0
        rs = gain_val / loss_val
        rsi = 100 - (100 / (1 + rs))
        return rsi if not np.isnan(rsi) else 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal_period=9) -> Tuple[float, float]:
        """Calculate MACD indicator."""
        if len(prices) < slow:
            return 0.0, 0.0
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period).mean()
        macd_val = macd.iloc[-1] if not np.isnan(macd.iloc[-1]) else 0.0
        signal_val = signal.iloc[-1] if not np.isnan(signal.iloc[-1]) else 0.0
        return macd_val, signal_val
    
    def generate_labels(self, df: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.002) -> np.ndarray:
        """
        Generate trading labels (BUY=2, HOLD=1, SELL=0) based on forward returns.
        
        Args:
            df: OHLCV dataframe
            forward_periods: Number of periods to look ahead
            threshold: Price change threshold for BUY/SELL signals
        """
        logger.info(f"Generating labels (forward_periods={forward_periods}, threshold={threshold})...")
        
        labels = []
        
        for i in range(len(df)):
            if i < 50:  # Skip warmup
                continue
            
            if i + forward_periods >= len(df):
                # Default to HOLD for last few samples
                labels.append(1)
            else:
                current_price = df['close'].iloc[i]
                future_price = df['close'].iloc[i + forward_periods]
                return_pct = (future_price - current_price) / current_price
                
                if return_pct > threshold:
                    labels.append(2)  # BUY
                elif return_pct < -threshold:
                    labels.append(0)  # SELL
                else:
                    labels.append(1)  # HOLD
        
        labels_array = np.array(labels, dtype=np.int32)
        logger.info(f"✅ Generated {len(labels_array):,} labels")
        logger.info(f"  Label distribution: SELL={np.sum(labels_array == 0)}, HOLD={np.sum(labels_array == 1)}, BUY={np.sum(labels_array == 2)}")
        
        return labels_array
    
    def process_symbol(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single symbol and return features and labels."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {symbol}...")
        logger.info(f"{'='*80}")
        
        # Load data
        df = self.load_ohlcv_data(symbol)
        
        # Generate features
        if self.feature_gen:
            try:
                # Use Layer 1 feature extraction
                features = self.feature_gen.extract_features(df)
                logger.info(f"✅ Generated features using Layer 1 (shape: {features.shape})")
            except Exception as e:
                logger.warning(f"Layer 1 feature extraction failed: {e}")
                logger.info("Falling back to simplified features...")
                features = self.generate_simple_features(df)
        else:
            features = self.generate_simple_features(df)
        
        # Generate labels
        labels = self.generate_labels(df)
        
        # Ensure matching lengths
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]
        
        logger.info(f"✅ Final dataset: {len(features):,} samples × {features.shape[1]} features")
        
        return features, labels
    
    def save_preprocessed_data(self, features: np.ndarray, labels: np.ndarray, metadata: dict):
        """Save preprocessed data to disk."""
        logger.info(f"\n💾 Saving preprocessed data...")
        
        # Save features
        features_path = self.output_dir / 'preprocessed_features.npy'
        np.save(features_path, features)
        logger.info(f"  Features saved: {features_path} ({features.nbytes / 1024**2:.1f} MB)")
        
        # Save labels  
        labels_path = self.output_dir / 'labels.npy'
        np.save(labels_path, labels)
        logger.info(f"  Labels saved: {labels_path} ({labels.nbytes / 1024:,.0f} KB)")
        
        # Save metadata
        import json
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"  Metadata saved: {metadata_path}")
        
        logger.info(f"✅ All files saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess OHLCV data for HIMARI Layer 2 training')
    parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT', 'ETH/USDT'],
                        help='Symbols to process')
    parser.add_argument('--input-dir', type=str, default='./data/raw',
                        help='Input directory with raw OHLCV data')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory for preprocessed data')
    parser.add_argument('--forward-periods', type=int, default=5,
                        help='Forward periods for label generation')
    parser.add_argument('--threshold', type=float, default=0.002,
                        help='Threshold for BUY/SELL labels (0.2%)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("HIMARI Layer 2 - Training Data Preprocessor")
    logger.info("=" * 80)
    logger.info(f"Symbols: {', '.join(args.symbols)}")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 80)
    
    # Initialize preprocessor
    preprocessor = TrainingDataPreprocessor(args.input_dir, args.output_dir)
    
    # Process each symbol
    all_features = []
    all_labels = []
    
    for symbol in args.symbols:
        try:
            features, labels = preprocessor.process_symbol(symbol)
            all_features.append(features)
            all_labels.append(labels)
        except Exception as e:
            logger.error(f"❌ Failed to process {symbol}: {e}", exc_info=True)
            continue
    
    if not all_features:
        logger.error("❌ No symbols processed successfully")
        sys.exit(1)
    
    # Combine all features and labels
    logger.info(f"\n{'='*80}")
    logger.info("Combining data from all symbols...")
    logger.info(f"{'='*80}")
    
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    
    logger.info(f"Combined dataset:")
    logger.info(f"  Features: {combined_features.shape}")
    logger.info(f"  Labels: {combined_labels.shape}")
    logger.info(f"  Memory: {combined_features.nbytes / 1024**2:.1f} MB")
    
    # Create metadata
    metadata = {
        'symbols': args.symbols,
        'num_samples': int(len(combined_features)),
        'feature_dim': int(combined_features.shape[1]),
        'label_distribution': {
            'SELL': int(np.sum(combined_labels == 0)),
            'HOLD': int(np.sum(combined_labels == 1)),
            'BUY': int(np.sum(combined_labels == 2))
        },
        'forward_periods': args.forward_periods,
        'threshold': args.threshold,
        'preprocessing_date': datetime.now().isoformat(),
        'feature_extraction_method': 'Layer1' if FEATURE_EXTRACTION_AVAILABLE else 'Simplified'
    }
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(combined_features, combined_labels, metadata)
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("✅ PREPROCESSING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total samples: {len(combined_features):,}")
    logger.info(f"Feature dimension: {combined_features.shape[1]}D")
    logger.info(f"Label distribution: SELL={metadata['label_distribution']['SELL']:,}, " +
                f"HOLD={metadata['label_distribution']['HOLD']:,}, " +
                f"BUY={metadata['label_distribution']['BUY']:,}")
    logger.info(f"\n🎯 Next steps:")
    logger.info(f"  1. Run data verification: python scripts/verify_training_data.py")
    logger.info(f"  2. Launch training: python scripts/launch_training.py --config configs/training_config.yaml")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
