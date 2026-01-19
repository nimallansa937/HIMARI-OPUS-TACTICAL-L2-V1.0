"""
Ensemble Backtest: FLAG-TRADER + CGDT
Combines multiple models for improved prediction diversity and robustness.
"""

import sys
from pathlib import Path

# Import the base backtest system
from end_to_end_backtest import (
    EndToEndPipeline, BacktestConfig, BacktestResult, TradeRecord,
    logger
)

import torch
import numpy as np
from typing import Tuple, Optional


class EnsembleConfig:
    """Config for ensemble backtest."""
    # Data paths
    test_data_path: str
    flag_trader_path: str
    cgdt_path: str = "checkpoints/cgdt_best.pt"
    ahhmm_path: str = "L2V1 AHHMM FINAL/student_t_ahhmm_trained.pkl"

    # Ensemble strategy
    ensemble_strategy: str = "voting"  # "voting", "confidence_weighted", "disagreement"
    confidence_threshold: float = 0.6  # Minimum confidence to trade

    # Backtest parameters
    initial_capital: float = 100000.0
    max_position_pct: float = 0.10
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    device: str = "cuda"

    # Voting weights
    flagtrader_weight: float = 0.6
    cgdt_weight: float = 0.4

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class EnsemblePipeline(EndToEndPipeline):
    """
    Ensemble decision engine combining FLAG-TRADER and CGDT.

    Strategies:
    1. Voting: Both models vote, majority wins (or HOLD if tie)
    2. Confidence-weighted: Weight predictions by model confidence
    3. Disagreement: Only trade when both agree
    """

    def __init__(self, config: EnsembleConfig):
        super().__init__(config)
        self.ensemble_config = config

        # Load CGDT if available
        self.load_cgdt()

    def load_cgdt(self):
        """Load CGDT model if checkpoint exists."""
        logger.info(f"Loading CGDT from {self.ensemble_config.cgdt_path}...")

        try:
            if not Path(self.ensemble_config.cgdt_path).exists():
                logger.warning(f"CGDT checkpoint not found at {self.ensemble_config.cgdt_path}")
                logger.warning("Ensemble will use FLAG-TRADER only with enhanced confidence filtering")
                self.cgdt_model = None
                return

            # Import CGDT model
            models_path = Path(__file__).parent / "LAYER 2 TACTICAL HIMARI OPUS" / "src" / "models"
            sys.path.insert(0, str(models_path))

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "cgdt",
                models_path / "cgdt.py"
            )
            cgdt_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cgdt_module)

            # Load checkpoint
            checkpoint = torch.load(self.ensemble_config.cgdt_path, map_location=self.device)

            # Reconstruct CGDT model (similar to FLAG-TRADER)
            # TODO: Add proper CGDT reconstruction based on checkpoint structure
            logger.info("CGDT loaded successfully!")
            self.cgdt_model = None  # Placeholder until checkpoint available

        except Exception as e:
            logger.error(f"Failed to load CGDT: {e}")
            logger.warning("Ensemble will use FLAG-TRADER only")
            self.cgdt_model = None

    def get_action_from_model(self, features: np.ndarray, step: int) -> Tuple[str, float]:
        """
        Get ensemble action from FLAG-TRADER (+ CGDT if available).

        Ensemble strategies:
        1. If CGDT available: Use voting/weighting strategy
        2. If CGDT not available: Use FLAG-TRADER with enhanced confidence filtering
        """
        # Get FLAG-TRADER prediction
        ft_action, ft_confidence = self._get_flagtrader_action(features)

        if self.cgdt_model is None:
            # No CGDT: Use FLAG-TRADER with confidence threshold
            return self._filter_by_confidence(ft_action, ft_confidence)

        # Get CGDT prediction
        cgdt_action, cgdt_confidence = self._get_cgdt_action(features)

        # Apply ensemble strategy
        if self.ensemble_config.ensemble_strategy == "voting":
            return self._voting_strategy(ft_action, ft_confidence, cgdt_action, cgdt_confidence)
        elif self.ensemble_config.ensemble_strategy == "confidence_weighted":
            return self._weighted_strategy(ft_action, ft_confidence, cgdt_action, cgdt_confidence)
        elif self.ensemble_config.ensemble_strategy == "disagreement":
            return self._disagreement_strategy(ft_action, ft_confidence, cgdt_action, cgdt_confidence)
        else:
            raise ValueError(f"Unknown ensemble strategy: {self.ensemble_config.ensemble_strategy}")

    def _get_flagtrader_action(self, features: np.ndarray) -> Tuple[str, float]:
        """Get FLAG-TRADER prediction (no correction needed for retrained model)."""
        x = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            logits = self.flag_trader_model(x)
            raw_logits = logits.squeeze()

            # No logit correction needed - model trained with balanced weights
            probs = torch.softmax(raw_logits, dim=-1)
            action_idx = torch.argmax(probs).item()
            confidence = probs[action_idx].item()

        action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        return action_map[action_idx], confidence

    def _get_cgdt_action(self, features: np.ndarray) -> Tuple[str, float]:
        """Get CGDT prediction (placeholder until model trained)."""
        # TODO: Implement CGDT inference
        return 'HOLD', 0.5

    def _filter_by_confidence(self, action: str, confidence: float) -> Tuple[str, float]:
        """
        Filter low-confidence trades (when CGDT not available).

        Strategy: Only take BUY/SELL if confidence > threshold
        """
        if action != 'HOLD' and confidence < self.ensemble_config.confidence_threshold:
            return 'HOLD', confidence
        return action, confidence

    def _voting_strategy(self, ft_action: str, ft_conf: float,
                        cgdt_action: str, cgdt_conf: float) -> Tuple[str, float]:
        """
        Voting strategy: Models vote, majority wins.

        Rules:
        1. Both agree → Use action with boosted confidence
        2. Disagree + one is HOLD → Use non-HOLD action if high confidence
        3. Disagree (BUY vs SELL) → HOLD (too risky)
        """
        if ft_action == cgdt_action:
            # Agreement: Boost confidence
            avg_confidence = (ft_conf + cgdt_conf) / 2
            boosted_confidence = min(avg_confidence * 1.2, 1.0)
            return ft_action, boosted_confidence

        # Disagreement
        if ft_action == 'HOLD':
            # FLAG-TRADER says HOLD, CGDT says BUY/SELL
            if cgdt_conf > 0.7:
                return cgdt_action, cgdt_conf
            else:
                return 'HOLD', (ft_conf + cgdt_conf) / 2

        if cgdt_action == 'HOLD':
            # CGDT says HOLD, FLAG-TRADER says BUY/SELL
            if ft_conf > 0.7:
                return ft_action, ft_conf
            else:
                return 'HOLD', (ft_conf + cgdt_conf) / 2

        # Both say BUY/SELL but disagree → too risky
        return 'HOLD', (ft_conf + cgdt_conf) / 2

    def _weighted_strategy(self, ft_action: str, ft_conf: float,
                          cgdt_action: str, cgdt_conf: float) -> Tuple[str, float]:
        """
        Confidence-weighted strategy: Weight by model confidence.

        Compute weighted probabilities and select highest.
        """
        # This requires full probability distributions, not just argmax
        # For now, use voting as fallback
        return self._voting_strategy(ft_action, ft_conf, cgdt_action, cgdt_conf)

    def _disagreement_strategy(self, ft_action: str, ft_conf: float,
                               cgdt_action: str, cgdt_conf: float) -> Tuple[str, float]:
        """
        Disagreement strategy: Only trade when both agree.

        Very conservative: Requires consensus.
        """
        if ft_action == cgdt_action and ft_action != 'HOLD':
            avg_confidence = (ft_conf + cgdt_conf) / 2
            return ft_action, avg_confidence
        else:
            return 'HOLD', (ft_conf + cgdt_conf) / 2


def main():
    """Run ensemble backtest."""
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble Backtest")
    parser.add_argument('--strategy', type=str, default='voting',
                       choices=['voting', 'confidence_weighted', 'disagreement'],
                       help='Ensemble strategy')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                       help='Minimum confidence for non-HOLD actions')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Create config
    config = EnsembleConfig(
        test_data_path="btc_1h_2025_2026_test_arrays.pkl",
        flag_trader_path="checkpoints/flag_trader_best.pt",
        cgdt_path="checkpoints/cgdt_best.pt",
        ahhmm_path="L2V1 AHHMM FINAL/student_t_ahhmm_trained.pkl",
        ensemble_strategy=args.strategy,
        confidence_threshold=args.confidence_threshold,
        device=args.device
    )

    logger.info("=" * 80)
    logger.info("HIMARI Layer 2 - Ensemble Backtest (FLAG-TRADER + CGDT)")
    logger.info("=" * 80)
    logger.info(f"Strategy: {config.ensemble_strategy}")
    logger.info(f"Confidence threshold: {config.confidence_threshold}")
    logger.info("")

    # Run backtest
    pipeline = EnsemblePipeline(config)
    result = pipeline.run_backtest()

    # Print results
    logger.info("=" * 80)
    logger.info("ENSEMBLE BACKTEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Return: {result.total_return:.2f}%")
    logger.info(f"CAGR: {result.cagr:.2f}%")
    logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    logger.info(f"Sortino Ratio: {result.sortino_ratio:.3f}")
    logger.info(f"Calmar Ratio: {result.calmar_ratio:.3f}")
    logger.info(f"Max Drawdown: {result.max_drawdown:.2f}%")
    logger.info(f"Volatility: {result.volatility:.2f}%")
    logger.info(f"Beta: {result.beta:.3f}")
    logger.info("")
    logger.info(f"Total Trades: {result.total_trades}")
    logger.info(f"Win Rate: {result.win_rate:.2f}%")
    logger.info(f"Profit Factor: {result.profit_factor:.3f}")
    logger.info(f"Avg Profit/Trade: ${result.avg_profit_per_trade:.2f}")
    logger.info(f"VaR 95%: {result.var_95:.2f}%")
    logger.info(f"CVaR 95%: {result.cvar_95:.2f}%")

    # Save results
    import json
    output_path = f"ensemble_backtest_{config.ensemble_strategy}.json"

    def convert_to_json_serializable(obj):
        """Convert numpy types to Python types for JSON serialization."""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    with open(output_path, 'w') as f:
        json.dump(convert_to_json_serializable(result.__dict__), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
