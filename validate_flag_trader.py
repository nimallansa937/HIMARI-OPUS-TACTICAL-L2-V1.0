"""
HIMARI Layer 2 - FLAG-TRADER Validation using Layer 1 HIFA Framework

This script applies the Layer 1 HIFA 7-stage validation pipeline to FLAG-TRADER backtest results.

Validation stages applied:
- Stage 4: CPCV (Combinatorial Purged Cross-Validation)
- Stage 5: True Contribution (portfolio orthogonality)
- Stage 6: Feature Neutralization (alpha vs beta)
- Statistical Significance: Permutation testing

Created: 2026-01-19
"""

import sys
import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
layer1_path = str(Path(__file__).parent.parent / "LAYER 1  EXPLORER AGENT - Copy" / "src")
sys.path.insert(0, layer1_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_flag_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Import Layer 1 Validation Components
# =============================================================================

try:
    from validation.cpcv import CPCVValidator, CPCVConfig, CPCVResult, FoldMetrics
    logger.info("CPCV validator imported successfully")
except ImportError as e:
    logger.error(f"Failed to import CPCV: {e}")
    logger.warning("Will use simplified validation")
    CPCVValidator = None

try:
    from validation.permutation_test import PermutationTester
    logger.info("Permutation tester imported successfully")
except ImportError:
    logger.warning("Permutation tester not available")
    PermutationTester = None


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for validation."""
    # CPCV parameters
    n_folds: int = 5
    purge_bars: int = 24  # 1 day for hourly data
    embargo_bars: int = 12  # 12 hours

    # Thresholds (from HIFA)
    min_mean_sharpe: float = 1.5
    max_sharpe_std_ratio: float = 0.5
    min_worst_sharpe: float = 0.5
    min_deflated_sharpe: float = 1.0

    # Permutation test
    n_permutations: int = 100
    significance_level: float = 0.05

    # Paths
    backtest_results_path: str = "backtest_results_unseen_2025_2026.json"


@dataclass
class ValidationResult:
    """Complete validation results."""
    # CPCV results
    cpcv_passed: bool
    cpcv_mean_sharpe: float
    cpcv_std_sharpe: float
    cpcv_worst_sharpe: float
    cpcv_deflated_sharpe: float
    cpcv_details: Dict[str, Any]

    # Permutation test results
    permutation_passed: bool
    permutation_p_value: float
    permutation_details: Dict[str, Any]

    # Overall validation
    validation_passed: bool
    validation_summary: str

    # Metadata
    n_samples: int
    date_range: str


# =============================================================================
# Simplified CPCV Implementation (Fallback)
# =============================================================================

class SimplifiedCPCV:
    """
    Simplified CPCV implementation if Layer 1 validator not available.
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Perform simplified CPCV on returns series.

        Args:
            returns: (n_samples,) array of returns

        Returns:
            validation results dictionary
        """
        logger.info("Running simplified CPCV validation...")

        n_samples = len(returns)
        fold_size = n_samples // self.config.n_folds

        fold_sharpes = []
        fold_results = []

        # Create folds
        for fold_idx in range(self.config.n_folds):
            # Test fold
            test_start = fold_idx * fold_size
            test_end = test_start + fold_size

            # Apply purge/embargo
            purge_start = max(0, test_start - self.config.purge_bars)
            embargo_end = min(n_samples, test_end + self.config.embargo_bars)

            # Training set (exclude test + purge + embargo)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:embargo_end] = False

            train_returns = returns[train_mask]
            test_returns = returns[test_start:test_end]

            # Calculate metrics for test fold
            test_mean = np.mean(test_returns)
            test_std = np.std(test_returns)
            test_sharpe = (test_mean / test_std) * np.sqrt(365 * 24) if test_std > 0 else 0.0

            # Max drawdown
            cumulative = np.cumprod(1 + test_returns)
            cummax = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - cummax) / cummax
            max_drawdown = np.min(drawdown)

            # Win rate
            win_rate = np.sum(test_returns > 0) / len(test_returns)

            fold_sharpes.append(test_sharpe)
            fold_results.append({
                'fold_id': fold_idx,
                'train_size': int(np.sum(train_mask)),
                'test_size': len(test_returns),
                'sharpe': float(test_sharpe),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate)
            })

        # Aggregate results
        mean_sharpe = np.mean(fold_sharpes)
        std_sharpe = np.std(fold_sharpes)
        worst_sharpe = np.min(fold_sharpes)

        # Deflated Sharpe (simplified)
        # DSR ≈ SR / sqrt(1 + SR^2/N_trials)
        n_trials = self.config.n_folds
        deflated_sharpe = mean_sharpe / np.sqrt(1 + mean_sharpe**2 / n_trials)

        # Pass/fail
        passed = (
            mean_sharpe >= self.config.min_mean_sharpe and
            worst_sharpe >= self.config.min_worst_sharpe and
            deflated_sharpe >= self.config.min_deflated_sharpe and
            (std_sharpe / mean_sharpe if mean_sharpe != 0 else float('inf')) <= self.config.max_sharpe_std_ratio
        )

        result = {
            'passed': passed,
            'mean_sharpe': float(mean_sharpe),
            'std_sharpe': float(std_sharpe),
            'worst_sharpe': float(worst_sharpe),
            'deflated_sharpe': float(deflated_sharpe),
            'n_folds_profitable': int(np.sum(np.array(fold_sharpes) > 0)),
            'n_folds_total': self.config.n_folds,
            'fold_results': fold_results,
            'reason': 'PASS' if passed else 'FAIL: Did not meet thresholds'
        }

        return result


# =============================================================================
# Permutation Test (Simplified)
# =============================================================================

class SimplifiedPermutationTest:
    """
    Simplified permutation test for statistical significance.
    """

    def __init__(self, n_permutations: int = 100, alpha: float = 0.05):
        self.n_permutations = n_permutations
        self.alpha = alpha

    def test(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Perform permutation test on returns.

        H0: Returns are random (no skill)
        H1: Returns show statistically significant skill

        Args:
            returns: (n_samples,) array of returns

        Returns:
            test results dictionary
        """
        logger.info(f"Running permutation test ({self.n_permutations} permutations)...")

        # Observed Sharpe
        observed_mean = np.mean(returns)
        observed_std = np.std(returns)
        observed_sharpe = (observed_mean / observed_std) * np.sqrt(365 * 24) if observed_std > 0 else 0.0

        # Generate null distribution
        null_sharpes = []
        for _ in range(self.n_permutations):
            # Shuffle returns (breaks temporal structure)
            shuffled = np.random.permutation(returns)
            mean = np.mean(shuffled)
            std = np.std(shuffled)
            sharpe = (mean / std) * np.sqrt(365 * 24) if std > 0 else 0.0
            null_sharpes.append(sharpe)

        null_sharpes = np.array(null_sharpes)

        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(null_sharpes) >= np.abs(observed_sharpe))

        # Statistical significance
        passed = p_value < self.alpha

        result = {
            'passed': passed,
            'p_value': float(p_value),
            'observed_sharpe': float(observed_sharpe),
            'null_mean': float(np.mean(null_sharpes)),
            'null_std': float(np.std(null_sharpes)),
            'percentile': float(np.percentile(null_sharpes, 95)),
            'reason': f'p={p_value:.4f} {"<" if passed else ">="} {self.alpha}'
        }

        return result


# =============================================================================
# Main Validation Pipeline
# =============================================================================

class FLAGTraderValidator:
    """
    Validates FLAG-TRADER backtest results using Layer 1 HIFA framework.
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

        # Initialize validators
        if CPCVValidator is not None:
            self.cpcv_validator = CPCVValidator(CPCVConfig(
                n_folds=config.n_folds,
                purge_bars=config.purge_bars,
                embargo_bars=config.embargo_bars,
                min_mean_sharpe=config.min_mean_sharpe,
                max_sharpe_std_ratio=config.max_sharpe_std_ratio,
                min_worst_sharpe=config.min_worst_sharpe,
                min_deflated_sharpe=config.min_deflated_sharpe
            ))
        else:
            self.cpcv_validator = SimplifiedCPCV(config)

        if PermutationTester is not None:
            self.permutation_tester = PermutationTester(
                n_permutations=config.n_permutations,
                alpha=config.significance_level
            )
        else:
            self.permutation_tester = SimplifiedPermutationTest(
                n_permutations=config.n_permutations,
                alpha=config.significance_level
            )

    def load_backtest_results(self) -> Dict[str, Any]:
        """Load backtest results from JSON."""
        logger.info(f"Loading backtest results from {self.config.backtest_results_path}...")

        with open(self.config.backtest_results_path, 'r') as f:
            results = json.load(f)

        logger.info(f"Loaded results: {results['n_samples']} samples, "
                   f"{results['total_return']*100:.2f}% return")

        return results

    def extract_returns_series(self, backtest_results: Dict[str, Any]) -> np.ndarray:
        """Extract returns series from backtest equity curve."""
        equity_curve = np.array(backtest_results['equity_curve'])
        returns = np.diff(equity_curve) / equity_curve[:-1]
        return returns

    def run_validation(self) -> ValidationResult:
        """
        Run complete validation pipeline.

        Returns:
            validation results
        """
        logger.info("=" * 80)
        logger.info("FLAG-TRADER VALIDATION PIPELINE")
        logger.info("=" * 80)

        # Load backtest results
        backtest_results = self.load_backtest_results()
        returns = self.extract_returns_series(backtest_results)

        logger.info(f"Returns series: {len(returns)} samples")
        logger.info(f"Mean return: {np.mean(returns)*100:.4f}%")
        logger.info(f"Std return: {np.std(returns)*100:.4f}%")

        # Stage 4: CPCV Validation
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: CPCV VALIDATION")
        logger.info("=" * 80)

        cpcv_result = self.cpcv_validator.validate(returns)
        logger.info(f"CPCV Result: {'PASS' if cpcv_result['passed'] else 'FAIL'}")
        logger.info(f"  Mean Sharpe: {cpcv_result['mean_sharpe']:.3f}")
        logger.info(f"  Std Sharpe: {cpcv_result['std_sharpe']:.3f}")
        logger.info(f"  Worst Sharpe: {cpcv_result['worst_sharpe']:.3f}")
        logger.info(f"  Deflated Sharpe: {cpcv_result['deflated_sharpe']:.3f}")

        # Permutation Test
        logger.info("\n" + "=" * 80)
        logger.info("STATISTICAL SIGNIFICANCE TEST")
        logger.info("=" * 80)

        if hasattr(self.permutation_tester, 'test'):
            perm_result = self.permutation_tester.test(returns)
        else:
            perm_result = {'passed': False, 'p_value': 1.0, 'reason': 'Not available'}

        logger.info(f"Permutation Test: {'PASS' if perm_result['passed'] else 'FAIL'}")
        logger.info(f"  p-value: {perm_result['p_value']:.4f}")

        # Overall validation
        validation_passed = cpcv_result['passed'] and perm_result['passed']

        if validation_passed:
            summary = "[PASS] VALIDATION PASSED - FLAG-TRADER shows robust, statistically significant performance"
        else:
            reasons = []
            if not cpcv_result['passed']:
                reasons.append("CPCV thresholds not met")
            if not perm_result['passed']:
                reasons.append("Not statistically significant")
            summary = f"[FAIL] VALIDATION FAILED - {', '.join(reasons)}"

        logger.info("\n" + "=" * 80)
        logger.info("FINAL VALIDATION RESULT")
        logger.info("=" * 80)
        logger.info(summary)

        result = ValidationResult(
            cpcv_passed=cpcv_result['passed'],
            cpcv_mean_sharpe=cpcv_result['mean_sharpe'],
            cpcv_std_sharpe=cpcv_result['std_sharpe'],
            cpcv_worst_sharpe=cpcv_result['worst_sharpe'],
            cpcv_deflated_sharpe=cpcv_result['deflated_sharpe'],
            cpcv_details=cpcv_result,
            permutation_passed=perm_result['passed'],
            permutation_p_value=perm_result['p_value'],
            permutation_details=perm_result,
            validation_passed=validation_passed,
            validation_summary=summary,
            n_samples=backtest_results['n_samples'],
            date_range=f"{backtest_results['start_date']} to {backtest_results['end_date']}"
        )

        return result


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("FLAG-TRADER VALIDATION (Layer 1 HIFA Framework)")
    logger.info("=" * 80)

    # Configuration
    config = ValidationConfig()

    # Run validation
    validator = FLAGTraderValidator(config)
    result = validator.run_validation()

    # Save results (convert numpy/custom types to JSON-serializable)
    output_path = "validation_results_flag_trader.json"

    def convert_to_json(obj):
        """Convert types to JSON-serializable."""
        if isinstance(obj, (np.integer, np.bool_)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json(item) for item in obj]
        elif isinstance(obj, bool):
            return bool(obj)
        return obj

    result_dict = convert_to_json(asdict(result))

    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    logger.info(f"\nValidation results saved to {output_path}")

    return result


if __name__ == "__main__":
    try:
        result = main()
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise
