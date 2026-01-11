"""
HIMARI Layer 2 - Data Verification Script
Verify training data availability and integrity before launching training.
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DataVerifier:
    """Verify HIMARI Layer 2 training data requirements."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.errors = []
        self.warnings = []
        self.info = []
    
    def verify_all(self) -> bool:
        """Run all verification checks."""
        logger.info("=" * 80)
        logger.info("HIMARI Layer 2 - Data Verification")
        logger.info("=" * 80)
        
        checks = [
            ("Directory exists", self.check_directory_exists),
            ("Data files present", self.check_data_files),
            ("Feature dimensions", self.check_feature_dimensions),
            ("Data coverage", self.check_data_coverage),
            ("Memory requirements", self.estimate_memory),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            logger.info(f"\n🔍 Checking: {check_name}...")
            try:
                passed = check_func()
                if passed:
                    logger.info(f"  ✅ PASS")
                else:
                    logger.error(f"  ❌ FAIL")
                    all_passed = False
            except Exception as e:
                logger.error(f"  ❌ ERROR: {e}")
                self.errors.append(f"{check_name}: {e}")
                all_passed = False
        
        self.print_summary()
        return all_passed
    
    def check_directory_exists(self) -> bool:
        """Check if data directory exists."""
        if not self.data_dir.exists():
            self.errors.append(f"Data directory not found: {self.data_dir}")
            return False
        
        self.info.append(f"Data directory: {self.data_dir}")
        return True
    
    def check_data_files(self) -> bool:
        """Check if required data files exist."""
        # Expected data structure (customize based on your actual data format)
        expected_files = [
            "preprocessed_features.npy",  # Preprocessed feature vectors
            "labels.npy",  # Trading labels (BUY/HOLD/SELL)
            "metadata.json",  # Metadata about the data
        ]
        
        missing_files = []
        found_files = []
        
        for file in expected_files:
            file_path = self.data_dir / file
            if file_path.exists():
                found_files.append(file)
                size_mb = file_path.stat().st_size / (1024**2)
                self.info.append(f"  Found: {file} ({size_mb:.1f} MB)")
            else:
                missing_files.append(file)
        
        if missing_files:
            self.warnings.append(f"Missing files: {', '.join(missing_files)}")
            logger.warning(f"  ⚠️  Missing: {', '.join(missing_files)}")
            logger.warning(f"  Note: This is expected if data is in a different format")
        
        # Check for any .npy or .csv files as fallback
        data_files = list(self.data_dir.glob("*.npy")) + list(self.data_dir.glob("*.csv"))
        if not data_files:
            self.errors.append("No data files found (.npy or .csv)")
            return False
        
        self.info.append(f"Total data files found: {len(data_files)}")
        return True
    
    def check_feature_dimensions(self) -> bool:
        """Check if feature vectors have correct dimensions."""
        try:
            import numpy as np
            
            # Try to load feature file
            feature_file = self.data_dir / "preprocessed_features.npy"
            if not feature_file.exists():
                self.warnings.append("Cannot verify feature dims - preprocessed_features.npy not found")
                return True  # Don't fail if file not found
            
            features = np.load(feature_file, mmap_mode='r')
            
            expected_dim = 60  # HIMARI Layer 1 feature dimension
            actual_dim = features.shape[-1] if len(features.shape) >= 2 else features.shape[0]
            
            self.info.append(f"  Feature shape: {features.shape}")
            self.info.append(f"  Feature dimension: {actual_dim}")
            
            if actual_dim != expected_dim:
                self.warnings.append(
                    f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}"
                )
                logger.warning(f"  ⚠️  Dimension: {actual_dim} (expected {expected_dim})")
            
            return True
            
        except ImportError:
            self.warnings.append("NumPy not available - cannot verify feature dimensions")
            return True
        except Exception as e:
            self.warnings.append(f"Feature dimension check failed: {e}")
            return True
    
    def check_data_coverage(self) -> bool:
        """Check data coverage (6 months minimum)."""
        try:
            metadata_file = self.data_dir / "metadata.json"
            if not metadata_file.exists():
                self.warnings.append("No metadata.json - cannot verify data coverage")
                return True
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            start_date = metadata.get('start_date')
            end_date = metadata.get('end_date')
            
            if start_date and end_date:
                start = datetime.fromisoformat(start_date)
                end = datetime.fromisoformat(end_date)
                coverage_days = (end - start).days
                coverage_months = coverage_days / 30
                
                self.info.append(f"  Coverage: {coverage_months:.1f} months ({coverage_days} days)")
                self.info.append(f"  Start: {start_date}")
                self.info.append(f"  End: {end_date}")
                
                if coverage_months < 6:
                    self.warnings.append(
                        f"Data coverage ({coverage_months:.1f} months) less than recommended 6 months"
                    )
                    logger.warning(f"  ⚠️  Only {coverage_months:.1f} months of data")
            
            return True
            
        except Exception as e:
            self.warnings.append(f"Coverage check failed: {e}")
            return True
    
    def estimate_memory(self) -> bool:
        """Estimate memory requirements."""
        try:
            import numpy as np
            
            feature_file = self.data_dir / "preprocessed_features.npy"
            if not feature_file.exists():
                self.warnings.append("Cannot estimate memory - preprocessed_features.npy not found")
                return True
            
            features = np.load(feature_file, mmap_mode='r')
            
            # Estimate memory needed for training
            data_size_gb = features.nbytes / (1024**3)
            estimated_peak_gb = data_size_gb * 3  # 3x for batching, gradients, etc.
            
            self.info.append(f"  Data size: {data_size_gb:.2f} GB")
            self.info.append(f"  Estimated peak memory: {estimated_peak_gb:.2f} GB")
            
            if estimated_peak_gb > 16:
                self.warnings.append(
                    f"High memory usage expected ({estimated_peak_gb:.1f} GB). "
                    "Ensure GPU has sufficient memory."
                )
            
            return True
            
        except Exception as e:
            self.warnings.append(f"Memory estimation failed: {e}")
            return True
    
    def print_summary(self):
        """Print verification summary."""
        logger.info("\n" + "=" * 80)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 80)
        
        if self.info:
            logger.info("\n📊 Information:")
            for msg in self.info:
                logger.info(f"  {msg}")
        
        if self.warnings:
            logger.info("\n⚠️  Warnings:")
            for msg in self.warnings:
                logger.warning(f"  {msg}")
        
        if self.errors:
            logger.info("\n❌ Errors:")
            for msg in self.errors:
                logger.error(f"  {msg}")
        
        logger.info("\n" + "=" * 80)
        if not self.errors:
            logger.info("✅ Verification PASSED")
            logger.info("Ready to start training!")
        else:
            logger.error("❌ Verification FAILED")
            logger.error("Fix errors before starting training")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Verify HIMARI Layer 2 training data')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--strict', action='store_true',
                        help='Treat warnings as errors')
    
    args = parser.parse_args()
    
    verifier = DataVerifier(args.data_dir)
    passed = verifier.verify_all()
    
    if not passed:
        sys.exit(1)
    
    if args.strict and verifier.warnings:
        logger.error("\n❌ Strict mode: Warnings treated as errors")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
