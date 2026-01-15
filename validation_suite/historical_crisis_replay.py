#!/usr/bin/env python3
"""
HIMARI Layer 2 V1 - Historical Crisis Replay

Replays the model through known historical crisis events to test
how the system handles extreme market conditions.

Crisis Events:
1. COVID Crash (March 2020) - 50% drop in days
2. Luna/UST Collapse (May 2022) - Stablecoin death spiral
3. FTX Collapse (November 2022) - Exchange failure

Usage:
    python historical_crisis_replay.py
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

BASE_DIR = Path(r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1")

# =============================================================================
# Crisis Scenarios (Synthetic based on historical patterns)
# =============================================================================

@dataclass
class CrisisScenario:
    """Definition of a crisis scenario."""
    name: str
    description: str
    duration_hours: int
    initial_drop_pct: float
    recovery_pct: float
    volatility_multiplier: float
    correlation_breakdown: bool


CRISIS_SCENARIOS = [
    CrisisScenario(
        name="COVID_CRASH",
        description="March 2020 COVID crash - 50% drop in 2 days",
        duration_hours=72,
        initial_drop_pct=-50.0,
        recovery_pct=30.0,
        volatility_multiplier=5.0,
        correlation_breakdown=True
    ),
    CrisisScenario(
        name="LUNA_UST",
        description="May 2022 Luna/UST collapse - death spiral",
        duration_hours=168,  # 1 week
        initial_drop_pct=-35.0,
        recovery_pct=5.0,  # Almost no recovery
        volatility_multiplier=4.0,
        correlation_breakdown=True
    ),
    CrisisScenario(
        name="FTX_COLLAPSE",
        description="November 2022 FTX collapse",
        duration_hours=120,
        initial_drop_pct=-25.0,
        recovery_pct=10.0,
        volatility_multiplier=3.0,
        correlation_breakdown=False
    ),
    CrisisScenario(
        name="FLASH_CRASH",
        description="Hypothetical flash crash - 20% in 1 hour",
        duration_hours=24,
        initial_drop_pct=-20.0,
        recovery_pct=15.0,
        volatility_multiplier=10.0,
        correlation_breakdown=False
    ),
    CrisisScenario(
        name="SLOW_BLEED",
        description="Extended bear market - slow 40% decline",
        duration_hours=720,  # 30 days
        initial_drop_pct=-40.0,
        recovery_pct=5.0,
        volatility_multiplier=1.5,
        correlation_breakdown=False
    ),
]


# =============================================================================
# Model (same architecture)
# =============================================================================

class RegimeConditionedPolicy(nn.Module):
    def __init__(
        self,
        feature_dim: int = 49,
        context_len: int = 100,
        n_regimes: int = 4,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        n_actions: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.n_actions = n_actions

        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.regime_embed = nn.Embedding(n_regimes, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, context_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, regime_ids):
        batch_size, seq_len, _ = features.shape
        x = self.feature_proj(features)
        x = x + self.regime_embed(regime_ids)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        last_hidden = x[:, -1, :]
        mean_hidden = x.mean(dim=1)
        combined = torch.cat([last_hidden, mean_hidden], dim=-1)
        return self.actor(combined), self.critic(combined)

    def get_action(self, features, regime_ids):
        with torch.no_grad():
            logits, _ = self.forward(features, regime_ids)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item() - 1
            confidence = probs[0, action + 1].item()
        return action, confidence


# =============================================================================
# Crisis Generator
# =============================================================================

class CrisisDataGenerator:
    """Generate synthetic crisis price paths."""

    def __init__(self, base_features: np.ndarray, base_price: float = 50000.0):
        self.base_features = base_features
        self.base_price = base_price
        self.feature_dim = base_features.shape[1]

    def generate_crisis(self, scenario: CrisisScenario, seed: int = 42) -> Dict:
        """Generate synthetic crisis data."""
        np.random.seed(seed)

        n_samples = scenario.duration_hours
        prices = np.zeros(n_samples)
        returns = np.zeros(n_samples)
        features = np.zeros((n_samples, self.feature_dim))
        regime_ids = np.zeros(n_samples, dtype=np.int32)

        # Generate price path
        current_price = self.base_price

        # Phase 1: Initial crash
        crash_duration = n_samples // 3
        crash_per_hour = scenario.initial_drop_pct / crash_duration / 100

        # Phase 2: Volatile bottom
        bottom_duration = n_samples // 3

        # Phase 3: Partial recovery
        recovery_duration = n_samples - crash_duration - bottom_duration
        recovery_per_hour = scenario.recovery_pct / recovery_duration / 100 if recovery_duration > 0 else 0

        for i in range(n_samples):
            # Determine phase
            if i < crash_duration:
                # Crash phase
                base_return = crash_per_hour
                vol = scenario.volatility_multiplier * 0.02
                regime = 3  # CRISIS
            elif i < crash_duration + bottom_duration:
                # Volatile bottom
                base_return = 0
                vol = scenario.volatility_multiplier * 0.015
                regime = 3  # CRISIS
            else:
                # Recovery
                base_return = recovery_per_hour
                vol = scenario.volatility_multiplier * 0.01
                regime = 1  # BEAR (recovering)

            # Add noise
            noise = np.random.normal(0, vol)
            ret = base_return + noise
            returns[i] = ret
            current_price *= (1 + ret)
            prices[i] = current_price
            regime_ids[i] = regime

            # Generate features based on price action
            features[i] = self._generate_crisis_features(ret, vol, regime, i, n_samples)

        return {
            'prices': prices,
            'returns': returns,
            'features': features,
            'regime_ids': regime_ids,
            'scenario': scenario
        }

    def _generate_crisis_features(
        self,
        ret: float,
        vol: float,
        regime: int,
        idx: int,
        total: int
    ) -> np.ndarray:
        """Generate feature vector for crisis conditions."""
        # Start with random base
        features = np.random.randn(self.feature_dim) * 0.1

        # Adjust key features for crisis
        features[0] = ret * 10  # Return signal
        features[1] = vol * 20  # Volatility signal
        features[2] = -0.5 if regime == 3 else 0.2  # Momentum
        features[3] = 0.9 if regime == 3 else 0.3  # Volatility regime
        features[4] = -0.8 if regime == 3 else -0.2  # Trend

        # RSI-like feature (oversold in crash)
        features[5] = 0.1 if idx < total // 3 else 0.3

        return features


# =============================================================================
# Crisis Replay Engine
# =============================================================================

@dataclass
class CrisisResult:
    """Results from a crisis replay."""
    scenario_name: str
    total_return: float
    max_drawdown: float
    sharpe: float
    n_trades: int
    time_in_market: float
    worst_hour: float
    recovery_quality: float  # How well it recovered


class CrisisReplayEngine:
    """Replay model through crisis scenarios."""

    def __init__(self, device: torch.device, context_len: int = 100):
        self.device = device
        self.context_len = context_len

    def replay_crisis(
        self,
        model: nn.Module,
        crisis_data: Dict
    ) -> CrisisResult:
        """Replay model through a crisis scenario."""
        model.eval()

        features = torch.tensor(crisis_data['features'], dtype=torch.float32, device=self.device)
        regime_ids = torch.tensor(crisis_data['regime_ids'], dtype=torch.long, device=self.device)
        prices = crisis_data['prices']
        returns = crisis_data['returns']
        scenario = crisis_data['scenario']

        # Normalize features
        features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)

        # Need pre-crisis context
        # Pad with neutral data
        context_len = self.context_len
        n_samples = len(prices)

        # Pad features and regimes
        pad_features = torch.zeros(context_len, features.shape[1], device=self.device)
        pad_regimes = torch.ones(context_len, dtype=torch.long, device=self.device) * 2  # SIDEWAYS

        features = torch.cat([pad_features, features], dim=0)
        regime_ids = torch.cat([pad_regimes, regime_ids], dim=0)

        # Simulate
        capital = 10000.0
        position = 0
        entry_price = 0.0
        equity_curve = [capital]
        n_trades = 0
        hours_in_market = 0
        worst_hour = 0.0

        for i in range(context_len, context_len + n_samples):
            feat_window = features[i-context_len:i].unsqueeze(0)
            regime_window = regime_ids[i-context_len:i].unsqueeze(0)

            action, confidence = model.get_action(feat_window, regime_window)
            current_price = prices[i - context_len]
            current_return = returns[i - context_len]

            # Track worst hour
            if position != 0:
                hour_pnl = position * current_return * 0.25 * capital
                worst_hour = min(worst_hour, hour_pnl / capital * 100)

            # Position management
            if action != position:
                if position != 0:
                    # Close
                    if position == 1:
                        pnl = (current_price - entry_price) / entry_price * 0.25 * capital
                    else:
                        pnl = (entry_price - current_price) / entry_price * 0.25 * capital
                    capital += pnl * 0.999
                    n_trades += 1
                    position = 0

                if action != 0:
                    # Open
                    position = action
                    entry_price = current_price

            if position != 0:
                hours_in_market += 1

            equity_curve.append(capital)

        equity = np.array(equity_curve)
        period_returns = np.diff(equity) / equity[:-1]

        total_return = (capital - 10000) / 10000 * 100
        max_dd = np.max(1 - equity / np.maximum.accumulate(equity)) * 100

        if len(period_returns) > 1 and np.std(period_returns) > 0:
            sharpe = np.mean(period_returns) / np.std(period_returns) * np.sqrt(24 * 365)
        else:
            sharpe = 0.0

        time_in_market = hours_in_market / n_samples * 100

        # Recovery quality: how much of DD was recovered
        peak = np.max(equity)
        trough = np.min(equity)
        final = equity[-1]
        if peak > trough:
            recovery_quality = (final - trough) / (peak - trough) * 100
        else:
            recovery_quality = 100.0

        return CrisisResult(
            scenario_name=scenario.name,
            total_return=total_return,
            max_drawdown=max_dd,
            sharpe=sharpe,
            n_trades=n_trades,
            time_in_market=time_in_market,
            worst_hour=worst_hour,
            recovery_quality=recovery_quality
        )


# =============================================================================
# Main
# =============================================================================

def run_crisis_replay():
    """Run historical crisis replay."""
    print("=" * 70)
    print("HIMARI Layer 2 V1 - Historical Crisis Replay")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\n[1/3] Loading model...")
    checkpoint_path = BASE_DIR / "L2V1 PPO FINAL" / "himari_ppo_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = {
        'feature_dim': checkpoint.get('feature_dim', 49),
        'context_len': checkpoint.get('context_len', 100),
        'n_regimes': checkpoint.get('n_regimes', 4),
        'hidden_dim': checkpoint.get('hidden_dim', 256),
        'n_heads': checkpoint.get('n_heads', 4),
        'n_layers': checkpoint.get('n_layers', 3),
        'n_actions': checkpoint.get('n_actions', 3),
    }

    model = RegimeConditionedPolicy(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Generate crisis data
    print("\n[2/3] Generating crisis scenarios...")
    generator = CrisisDataGenerator(
        base_features=np.random.randn(100, config['feature_dim']),
        base_price=50000.0
    )

    # Run crisis replays
    print("\n[3/3] Replaying crisis scenarios...")
    engine = CrisisReplayEngine(device, context_len=config['context_len'])

    results: List[CrisisResult] = []

    for scenario in CRISIS_SCENARIOS:
        print(f"\n  Scenario: {scenario.name}")
        print(f"  Description: {scenario.description}")

        crisis_data = generator.generate_crisis(scenario)
        result = engine.replay_crisis(model, crisis_data)
        results.append(result)

        print(f"    Return:      {result.total_return:+.2f}%")
        print(f"    Max DD:      {result.max_drawdown:.2f}%")
        print(f"    Worst Hour:  {result.worst_hour:.2f}%")
        print(f"    Time in Mkt: {result.time_in_market:.1f}%")
        print(f"    Recovery:    {result.recovery_quality:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("CRISIS REPLAY SUMMARY")
    print("=" * 70)

    print("\n{:<15} {:>10} {:>10} {:>10} {:>10}".format(
        "Scenario", "Return", "Max DD", "Worst Hr", "Recovery"))
    print("-" * 55)

    for r in results:
        print("{:<15} {:>+9.1f}% {:>9.1f}% {:>9.1f}% {:>9.1f}%".format(
            r.scenario_name[:15], r.total_return, r.max_drawdown,
            r.worst_hour, r.recovery_quality))

    # Aggregate metrics
    avg_return = np.mean([r.total_return for r in results])
    avg_dd = np.mean([r.max_drawdown for r in results])
    worst_dd = np.max([r.max_drawdown for r in results])
    avg_recovery = np.mean([r.recovery_quality for r in results])

    print("-" * 55)
    print("{:<15} {:>+9.1f}% {:>9.1f}% {:>10} {:>9.1f}%".format(
        "AVERAGE", avg_return, avg_dd, "", avg_recovery))
    print("{:<15} {:>10} {:>9.1f}%".format("WORST", "", worst_dd))

    # Validation
    print("\n[VALIDATION]")
    survive_all = all(r.max_drawdown < 50 for r in results)
    avg_dd_pass = avg_dd < 30
    recovery_pass = avg_recovery > 30

    print(f"  Survive all (DD<50%):   {'PASS' if survive_all else 'FAIL'}")
    print(f"  Avg DD < 30%:           {'PASS' if avg_dd_pass else 'FAIL'} ({avg_dd:.1f}%)")
    print(f"  Avg Recovery > 30%:     {'PASS' if recovery_pass else 'FAIL'} ({avg_recovery:.1f}%)")

    print("=" * 70)

    return results


if __name__ == "__main__":
    try:
        results = run_crisis_replay()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
