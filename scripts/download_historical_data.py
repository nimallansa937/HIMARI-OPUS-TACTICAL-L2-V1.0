"""
HIMARI Layer 2 - Historical Data Downloader
Download 6 months of 5-minute OHLCV data from Binance using CCXT.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import logging
from typing import List, Dict, Optional
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class HistoricalDataDownloader:
    """Download historical OHLCV data from cryptocurrency exchanges."""
    
    def __init__(self, exchange_name: str = 'binance'):
        """
        Initialize data downloader.
        
        Args:
            exchange_name: Exchange to use (default: binance)
        """
        self.exchange_name = exchange_name
        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                'enableRateLimit': True,  # Respect rate limits
                'options': {
                    'defaultType': 'spot',  # Use spot market
                }
            })
            logger.info(f"✅ Connected to {exchange_name}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to {exchange_name}: {e}")
            raise
    
    def download_ohlcv(
        self,
        symbol: str,
        timeframe: str = '5m',
        months: int = 6,
        output_dir: str = './data/raw'
    ) -> pd.DataFrame:
        """
        Download OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            months: Number of months to download
            output_dir: Directory to save data
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"📥 Downloading {symbol} {timeframe} data for {months} months...")
        
        # Calculate date range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=months * 30)
        
        # Convert to milliseconds
        since = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        all_candles = []
        
        try:
            while since < end_ms:
                # Fetch candles
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000  # Max per request
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                since = candles[-1][0] + 1  # Next timestamp
                
                logger.info(f"  Downloaded {len(all_candles)} candles... (up to {datetime.fromtimestamp(since/1000)})")
                
                # Rate limit protection
                time.sleep(self.exchange.rateLimit / 1000)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by time
            df.sort_index(inplace=True)
            
            logger.info(f"✅ Downloaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
            
            # Save to file
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            safe_symbol = symbol.replace('/', '_')
            filename = f"{safe_symbol}_{timeframe}_{months}m.csv"
            filepath = output_path / filename
            
            df.to_csv(filepath)
            logger.info(f"💾 Saved to: {filepath}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            raise
    
    def download_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = '5m',
        months: int = 6,
        output_dir: str = './data/raw'
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple symbols.
        
        Args:
            symbols: List of trading pairs
            timeframe: Timeframe
            months: Number of months
            output_dir: Output directory
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
            try:
                df = self.download_ohlcv(symbol, timeframe, months, output_dir)
                results[symbol] = df
            except Exception as e:
                logger.error(f"⚠️  Failed to download {symbol}: {e}")
                continue
        
        return results
    
    def create_metadata(
        self,
        symbols: List[str],
        timeframe: str,
        months: int,
        output_dir: str = './data'
    ):
        """Create metadata file for downloaded data."""
        metadata = {
            'exchange': self.exchange_name,
            'symbols': symbols,
            'timeframe': timeframe,
            'months': months,
            'download_date': datetime.now().isoformat(),
            'start_date': (datetime.now() - timedelta(days=months * 30)).isoformat(),
            'end_date': datetime.now().isoformat(),
            'feature_dim': 60,  # Will be generated later
            'description': 'Raw OHLCV data for HIMARI Layer 2 training'
        }
        
        metadata_path = Path(output_dir) / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📝 Metadata saved to: {metadata_path}")


def main():
    """Main download script."""
    parser = argparse.ArgumentParser(description='Download historical crypto data for HIMARI')
    parser.add_argument('--symbols', type=str, nargs='+', 
                        default=['BTC/USDT', 'ETH/USDT'],
                        help='Trading pairs to download (default: BTC/USDT ETH/USDT)')
    parser.add_argument('--timeframe', type=str, default='5m',
                        help='Timeframe (default: 5m)')
    parser.add_argument('--months', type=int, default=6,
                        help='Number of months to download (default: 6)')
    parser.add_argument('--exchange', type=str, default='binance',
                        help='Exchange name (default: binance)')
    parser.add_argument('--output-dir', type=str, default='./data/raw',
                        help='Output directory (default: ./data/raw)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("HIMARI Layer 2 - Historical Data Downloader")
    logger.info("=" * 80)
    logger.info(f"Exchange: {args.exchange}")
    logger.info(f"Symbols: {', '.join(args.symbols)}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Duration: {args.months} months")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 80)
    
    # Initialize downloader
    downloader = HistoricalDataDownloader(args.exchange)
    
    # Download data
    results = downloader.download_multiple_symbols(
        symbols=args.symbols,
        timeframe=args.timeframe,
        months=args.months,
        output_dir=args.output_dir
    )
    
    # Create metadata
    downloader.create_metadata(
        symbols=args.symbols,
        timeframe=args.timeframe,
        months=args.months,
        output_dir=Path(args.output_dir).parent
    )
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 80)
    
    for symbol, df in results.items():
        logger.info(f"✅ {symbol}: {len(df):,} candles | {df.index[0]} to {df.index[-1]}")
    
    logger.info("\n📊 Data Statistics:")
    total_candles = sum(len(df) for df in results.values())
    logger.info(f"  Total candles: {total_candles:,}")
    logger.info(f"  Estimated size: ~{total_candles * len(results) * 6 * 8 / 1024**2:.1f} MB")
    
    logger.info("\n🎯 Next Steps:")
    logger.info("  1. Run preprocessing to generate 60D feature vectors")
    logger.info("  2. Run data verification: python scripts/verify_training_data.py")
    logger.info("  3. Launch training: python scripts/launch_training.py")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
