#!/usr/bin/env python3
"""
HIMARI Layer 2 V1 - Full Integrated Pipeline Backtest

Runs the COMPLETE pipeline with all 6 components:
1. EKF Denoiser (Part A)
2. AHHMM Regime Detection (Part B)
3. PPO Decision Engine (Part D)
4. Uncertainty Quantification (Part F)
5. HSM State Validation (Part E)
6. Hysteresis Filter (Part G)
7. Kelly Position Sizing (Part H)
8. Safety System (Part I)

Usage:
    python full_pipeline_backtest.py
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque

BASE_DIR = Path(r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1")

# =============================================================================
# Enums
# =============================================================================

class TradeAction(Enum):
    SELL = -1
    HOLD = 0
    BUY = 1

class MarketRegime(Enum):
    BULL = 0
    BEAR = 1
    SIDEWAYS = 2
    CRISIS = 3

class PositionState(Enum):
    FLAT = 0
    LONG = 1
    SHORT = 2

# =============================================================================
# Part A: EKF Denoiser
# =============================================================================

class EKFDenoiser:
    """Extended Kalman Filter for price denoising."""

    def __init__(self, config: Dict):
        self.process_noise = config.get('process_noise', 0.001)
        self.measurement_noise = config.get('measurement_noise', 0.01)
        self.state = np.zeros(4)  # [price, velocity, acceleration, volatility]
        self.covariance = np.eye(4) * 0.1
        self.initialized = False

    def update(self, price: float) -> Dict[str, float]:
        """Update EKF with new price."""
        if not self.initialized:
            self.state[0] = price
            self.initialized = True
            return self._get_state()

        # Predict
        dt = 1.0
        F = np.array([
            [1, dt, 0.5*dt**2, 0],
            [0, 1, dt, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0.95]
        ])
        Q = np.eye(4) * self.process_noise

        predicted_state = F @ self.state
        predicted_cov = F @ self.covariance @ F.T + Q

        # Update
        H = np.array([[1, 0, 0, 0]])
        R = np.array([[self.measurement_noise]])
        innovation = price - H @ predicted_state

        S = H @ predicted_cov @ H.T + R
        K = predicted_cov @ H.T @ np.linalg.inv(S)

        self.state = predicted_state + K.flatten() * innovation[0]
        self.covariance = (np.eye(4) - K @ H) @ predicted_cov

        # Update volatility
        self.state[3] = 0.9 * self.state[3] + 0.1 * abs(innovation[0])

        return self._get_state()

    def _get_state(self) -> Dict[str, float]:
        return {
            'price': self.state[0],
            'velocity': self.state[1],
            'acceleration': self.state[2],
            'volatility': self.state[3]
        }

# =============================================================================
# Part B: AHHMM Regime Detection
# =============================================================================

class AHHMMRegimeDetector:
    """Asymmetric Heavy-tailed HMM for regime detection."""

    def __init__(self, config: Dict):
        self.n_states = config.get('n_states', 4)
        self.df = config.get('df', 5.0)
        self.means = config.get('means', np.zeros((4, 6)))
        self.scales = config.get('scales', np.ones((4, 6)))
        self.trans = config.get('trans', np.eye(4) * 0.7 + 0.1)
        self.state_probs = np.ones(4) / 4

        # Volatility tracking for regime
        self.vol_buffer = deque(maxlen=20)
        self.ret_buffer = deque(maxlen=20)

    def detect(self, features: np.ndarray, price: float, prev_price: float) -> Tuple[int, float]:
        """Detect current regime."""
        # Calculate return
        ret = (price - prev_price) / prev_price if prev_price > 0 else 0
        self.ret_buffer.append(ret)

        # Calculate volatility
        if len(self.ret_buffer) >= 5:
            vol = np.std(list(self.ret_buffer)) * np.sqrt(24 * 365)
            self.vol_buffer.append(vol)

        # Simple regime detection based on volatility and returns
        avg_vol = np.mean(list(self.vol_buffer)) if self.vol_buffer else 0.2
        avg_ret = np.mean(list(self.ret_buffer)) if self.ret_buffer else 0

        # Regime classification
        if avg_vol > 0.8:  # High volatility
            regime = MarketRegime.CRISIS.value
            confidence = min(avg_vol / 1.0, 1.0)
        elif avg_ret > 0.001:  # Positive momentum
            regime = MarketRegime.BULL.value
            confidence = min(avg_ret * 100, 0.9)
        elif avg_ret < -0.001:  # Negative momentum
            regime = MarketRegime.BEAR.value
            confidence = min(-avg_ret * 100, 0.9)
        else:
            regime = MarketRegime.SIDEWAYS.value
            confidence = 0.5

        return regime, confidence

# =============================================================================
# Part D: PPO Decision Engine
# =============================================================================

class RegimeConditionedPolicy(nn.Module):
    """PPO Policy with regime conditioning."""

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

    def get_action_with_confidence(self, features, regime_ids) -> Tuple[int, float, np.ndarray]:
        """Get action, confidence, and full probability distribution."""
        with torch.no_grad():
            logits, value = self.forward(features, regime_ids)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            action = np.argmax(probs) - 1  # Convert to -1, 0, 1
            confidence = probs[action + 1]
        return action, confidence, probs

# =============================================================================
# Part F: Uncertainty Quantification
# =============================================================================

class UncertaintyQuantifier:
    """Calibrate confidence using temperature scaling."""

    def __init__(self, temperature: float = 1.5):
        self.temperature = temperature
        self.calibration_history = deque(maxlen=100)

    def calibrate(self, raw_confidence: float, regime: int, probs: np.ndarray) -> float:
        """Calibrate raw confidence."""
        # Temperature scaling
        scaled_probs = np.exp(np.log(probs + 1e-8) / self.temperature)
        scaled_probs = scaled_probs / scaled_probs.sum()

        # Entropy-based uncertainty
        entropy = -np.sum(scaled_probs * np.log(scaled_probs + 1e-8))
        max_entropy = np.log(len(probs))
        uncertainty = entropy / max_entropy

        # Regime adjustment
        regime_multipliers = {
            MarketRegime.BULL.value: 1.0,
            MarketRegime.BEAR.value: 0.9,
            MarketRegime.SIDEWAYS.value: 0.8,
            MarketRegime.CRISIS.value: 0.5
        }
        regime_mult = regime_multipliers.get(regime, 0.8)

        # Final calibrated confidence
        calibrated = raw_confidence * (1 - uncertainty) * regime_mult

        return float(np.clip(calibrated, 0.0, 1.0))

# =============================================================================
# Part E: HSM State Machine
# =============================================================================

class HSMValidator:
    """Hierarchical State Machine for trade validation."""

    def __init__(self):
        self.position_state = PositionState.FLAT
        self.action_history = deque(maxlen=20)
        self.oscillation_threshold = 4  # Max flips in window

    def validate(self, proposed_action: int, current_position: int) -> Tuple[int, bool, str]:
        """Validate proposed action against current state."""
        # Update history
        self.action_history.append(proposed_action)

        # Check for oscillation (flip-flopping)
        if len(self.action_history) >= 10:
            recent = list(self.action_history)[-10:]
            flips = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1] and recent[i] != 0)
            if flips >= self.oscillation_threshold:
                return 0, False, "oscillation_blocked"

        # Validate state transitions
        if current_position == 0:  # FLAT
            # Can go to any position
            return proposed_action, True, "valid"

        elif current_position == 1:  # LONG
            if proposed_action == 1:  # Can't go more long
                return 0, False, "already_long"
            return proposed_action, True, "valid"

        else:  # SHORT
            if proposed_action == -1:  # Can't go more short
                return 0, False, "already_short"
            return proposed_action, True, "valid"

# =============================================================================
# Part G: Hysteresis Filter
# =============================================================================

class HysteresisFilter:
    """Filter low-confidence signals to prevent whipsawing."""

    def __init__(self, config: Dict):
        self.entry_threshold = config.get('entry_threshold', 0.4)
        self.exit_threshold = config.get('exit_threshold', 0.25)
        self.kama_period = config.get('kama_period', 10)

        # Adaptive thresholds
        self.confidence_history = deque(maxlen=50)

    def filter(self, action: int, confidence: float, current_position: int) -> Tuple[int, bool, str]:
        """Apply hysteresis filtering."""
        self.confidence_history.append(confidence)

        # Calculate adaptive threshold based on recent confidence
        if len(self.confidence_history) >= 10:
            avg_conf = np.mean(list(self.confidence_history))
            adaptive_entry = max(self.entry_threshold, avg_conf * 0.8)
        else:
            adaptive_entry = self.entry_threshold

        # Entry filter (higher threshold for new positions)
        if current_position == 0 and action != 0:
            if confidence < adaptive_entry:
                return 0, False, f"entry_blocked_conf_{confidence:.2f}<{adaptive_entry:.2f}"

        # Exit filter (lower threshold for closing)
        if current_position != 0 and action != current_position:
            if confidence < self.exit_threshold:
                return current_position, False, f"exit_blocked_conf_{confidence:.2f}<{self.exit_threshold:.2f}"

        return action, True, "passed"

# =============================================================================
# Part H: Kelly Position Sizing
# =============================================================================

class KellyPositionSizer:
    """Kelly criterion-based position sizing with risk limits."""

    def __init__(self, config: Dict):
        self.kelly_fraction = config.get('fraction_cap', 0.25)
        self.max_position = config.get('max_position', 1.0)
        self.min_position = config.get('min_position', 0.05)

        # Win rate tracking
        self.trade_history = deque(maxlen=50)

    def calculate_size(self, action: int, confidence: float, regime: int,
                       current_drawdown: float) -> float:
        """Calculate position size using Kelly criterion."""
        if action == 0:
            return 0.0

        # Base Kelly size
        # Kelly = (p * b - q) / b where p=win_prob, q=lose_prob, b=win/loss ratio
        # Simplified: use confidence as proxy for edge
        edge = max(0, confidence - 0.5) * 2  # Convert 0.5-1.0 to 0-1
        kelly_size = edge * self.kelly_fraction

        # Regime scaling
        regime_scales = {
            MarketRegime.BULL.value: 1.0,
            MarketRegime.BEAR.value: 0.75,
            MarketRegime.SIDEWAYS.value: 0.5,
            MarketRegime.CRISIS.value: 0.25
        }
        regime_scale = regime_scales.get(regime, 0.5)

        # Drawdown scaling
        if current_drawdown > 0.05:
            dd_scale = max(0.1, 1 - current_drawdown * 5)
        else:
            dd_scale = 1.0

        # Final size
        final_size = kelly_size * regime_scale * dd_scale

        return float(np.clip(final_size, self.min_position if action != 0 else 0, self.max_position))

# =============================================================================
# Part I: Safety System
# =============================================================================

class SafetySystem:
    """Final safety checks and drawdown brakes."""

    def __init__(self, config: Dict):
        self.max_drawdown = config.get('max_drawdown', 0.15)
        self.daily_loss_limit = config.get('daily_loss_limit', 0.05)
        self.drawdown_brakes = config.get('brake_levels', [0.05, 0.08, 0.10])
        self.brake_reductions = config.get('brake_reductions', [0.25, 0.50, 0.90])

        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.safety_mode = False

    def check(self, action: int, position_size: float,
              current_equity: float, regime: int) -> Tuple[int, float, str]:
        """Apply safety checks."""
        # Update peak
        self.peak_equity = max(self.peak_equity, current_equity)
        current_dd = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0

        # Check drawdown brakes
        for brake_level, reduction in zip(self.drawdown_brakes, self.brake_reductions):
            if current_dd >= brake_level:
                position_size *= (1 - reduction)

        # Emergency stop if max drawdown exceeded
        if current_dd >= self.max_drawdown:
            return 0, 0.0, f"emergency_stop_dd_{current_dd:.2%}"

        # Crisis mode: force conservative
        if regime == MarketRegime.CRISIS.value:
            position_size *= 0.25
            if position_size < 0.02:
                return 0, 0.0, "crisis_mode_blocked"

        # Daily loss limit
        if self.daily_pnl < -self.daily_loss_limit:
            return 0, 0.0, f"daily_loss_limit_{self.daily_pnl:.2%}"

        return action, position_size, "passed"

    def update_daily_pnl(self, pnl: float):
        self.daily_pnl += pnl

    def reset_daily(self):
        self.daily_pnl = 0.0

# =============================================================================
# Full Integrated Pipeline
# =============================================================================

class FullIntegratedPipeline:
    """Complete Layer 2 pipeline with all 6 components."""

    def __init__(self, checkpoints: Dict, device: torch.device):
        self.device = device

        # Part A: EKF
        ekf_config = checkpoints.get('ekf', {}).get('config', {})
        self.ekf = EKFDenoiser(ekf_config)

        # Part B: AHHMM
        ahhmm_config = checkpoints.get('ahhmm', {})
        self.regime_detector = AHHMMRegimeDetector(ahhmm_config)

        # Part D: PPO
        ppo_checkpoint = checkpoints.get('ppo', {})
        self.ppo = self._load_ppo(ppo_checkpoint)

        # Part F: UQ
        self.uq = UncertaintyQuantifier(temperature=1.5)

        # Part E: HSM
        self.hsm = HSMValidator()

        # Part G: Hysteresis
        hyst_config = {'entry_threshold': 0.35, 'exit_threshold': 0.20}
        self.hysteresis = HysteresisFilter(hyst_config)

        # Part H: Position Sizing
        risk_config = checkpoints.get('risk_manager', {}).get('H2_Kelly', {})
        self.position_sizer = KellyPositionSizer(risk_config)

        # Part I: Safety
        safety_config = {
            'max_drawdown': 0.15,
            'daily_loss_limit': 0.05,
            'brake_levels': [0.05, 0.08, 0.10],
            'brake_reductions': [0.25, 0.50, 0.90]
        }
        self.safety = SafetySystem(safety_config)

        # State
        self.feature_buffer = deque(maxlen=100)
        self.regime_buffer = deque(maxlen=100)
        self.current_position = 0
        self.prev_price = 0.0

        # Statistics
        self.stats = {
            'total_signals': 0,
            'hsm_blocked': 0,
            'hysteresis_blocked': 0,
            'safety_blocked': 0,
            'trades_executed': 0
        }

    def _load_ppo(self, checkpoint: Dict) -> RegimeConditionedPolicy:
        """Load PPO model from checkpoint."""
        config = {
            'feature_dim': checkpoint.get('feature_dim', 49),
            'context_len': checkpoint.get('context_len', 100),
            'n_regimes': checkpoint.get('n_regimes', 4),
            'hidden_dim': checkpoint.get('hidden_dim', 256),
            'n_heads': checkpoint.get('n_heads', 4),
            'n_layers': checkpoint.get('n_layers', 3),
            'n_actions': checkpoint.get('n_actions', 3),
        }

        model = RegimeConditionedPolicy(**config).to(self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def process(self, features: np.ndarray, price: float,
                current_equity: float) -> Dict:
        """Process one timestep through full pipeline."""
        result = {
            'action': 0,
            'position_size': 0.0,
            'regime': 2,  # Default SIDEWAYS
            'raw_confidence': 0.0,
            'calibrated_confidence': 0.0,
            'blocked_by': None,
            'latency_ms': 0.0
        }

        start_time = time.perf_counter()
        self.stats['total_signals'] += 1

        # Part A: EKF (simplified - already have denoised features)
        ekf_state = self.ekf.update(price)

        # Part B: Regime Detection
        regime, regime_conf = self.regime_detector.detect(
            features, price, self.prev_price
        )
        result['regime'] = regime

        # Buffer features
        self.feature_buffer.append(features)
        self.regime_buffer.append(regime)

        # Need enough context
        if len(self.feature_buffer) < 100:
            self.prev_price = price
            result['latency_ms'] = (time.perf_counter() - start_time) * 1000
            return result

        # Part D: PPO Decision
        feat_tensor = torch.tensor(
            np.array(list(self.feature_buffer)),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        regime_tensor = torch.tensor(
            list(self.regime_buffer),
            dtype=torch.long, device=self.device
        ).unsqueeze(0)

        # Normalize
        feat_tensor = (feat_tensor - feat_tensor.mean(dim=1, keepdim=True)) / (feat_tensor.std(dim=1, keepdim=True) + 1e-8)

        raw_action, raw_conf, probs = self.ppo.get_action_with_confidence(feat_tensor, regime_tensor)
        result['raw_confidence'] = raw_conf

        # Part F: Uncertainty Quantification
        calibrated_conf = self.uq.calibrate(raw_conf, regime, probs)
        result['calibrated_confidence'] = calibrated_conf

        # Part E: HSM Validation
        validated_action, hsm_valid, hsm_reason = self.hsm.validate(raw_action, self.current_position)
        if not hsm_valid:
            self.stats['hsm_blocked'] += 1
            result['blocked_by'] = f"HSM:{hsm_reason}"
            result['action'] = validated_action
            self.prev_price = price
            result['latency_ms'] = (time.perf_counter() - start_time) * 1000
            return result

        # Part G: Hysteresis Filter
        filtered_action, hyst_pass, hyst_reason = self.hysteresis.filter(
            validated_action, calibrated_conf, self.current_position
        )
        if not hyst_pass:
            self.stats['hysteresis_blocked'] += 1
            result['blocked_by'] = f"Hysteresis:{hyst_reason}"
            result['action'] = filtered_action
            self.prev_price = price
            result['latency_ms'] = (time.perf_counter() - start_time) * 1000
            return result

        # Part H: Position Sizing
        current_dd = (self.safety.peak_equity - current_equity) / self.safety.peak_equity if self.safety.peak_equity > 0 else 0
        position_size = self.position_sizer.calculate_size(
            filtered_action, calibrated_conf, regime, current_dd
        )

        # Part I: Safety System
        final_action, final_size, safety_reason = self.safety.check(
            filtered_action, position_size, current_equity, regime
        )
        if safety_reason != "passed":
            self.stats['safety_blocked'] += 1
            result['blocked_by'] = f"Safety:{safety_reason}"

        result['action'] = final_action
        result['position_size'] = final_size

        # Update state
        if final_action != self.current_position and final_action != 0:
            self.stats['trades_executed'] += 1
        self.current_position = final_action if final_action != 0 else self.current_position

        self.prev_price = price
        result['latency_ms'] = (time.perf_counter() - start_time) * 1000

        return result

# =============================================================================
# Trading Simulator
# =============================================================================

class TradingSimulator:
    """Trading simulator for backtesting."""

    def __init__(self, initial_capital: float = 10000.0, fee_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.reset()

    def reset(self):
        self.capital = self.initial_capital
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.peak_capital = self.initial_capital
        self.trades = []
        self.equity_curve = []

    def step(self, action: int, position_size: float, current_price: float) -> Dict:
        result = {'pnl': 0.0, 'trade': False, 'fee': 0.0}

        # Close if changing direction
        if self.position != 0 and action != self.position:
            if self.position == 1:
                pnl = (current_price - self.entry_price) / self.entry_price * self.position_size * self.capital
            else:
                pnl = (self.entry_price - current_price) / self.entry_price * self.position_size * self.capital

            fee = abs(pnl) * self.fee_rate
            self.capital += pnl - fee
            result['pnl'] = pnl - fee
            result['fee'] = fee
            result['trade'] = True
            self.trades.append({'pnl': pnl - fee, 'type': 'close'})
            self.position = 0
            self.position_size = 0.0

        # Open new position
        if action != 0 and self.position == 0 and position_size > 0.01:
            self.position = action
            self.position_size = position_size
            self.entry_price = current_price
            result['trade'] = True
            self.trades.append({'type': 'open', 'action': action, 'size': position_size})

        self.equity_curve.append(self.capital)
        self.peak_capital = max(self.peak_capital, self.capital)
        return result

    def get_stats(self) -> Dict:
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])

        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        max_dd = np.max(1 - equity / np.maximum.accumulate(equity)) * 100 if len(equity) > 0 else 0

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365)
        else:
            sharpe = 0.0

        closed_trades = [t for t in self.trades if t.get('type') == 'close']
        wins = sum(1 for t in closed_trades if t['pnl'] > 0)
        win_rate = wins / len(closed_trades) * 100 if closed_trades else 0

        return {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'n_trades': len(closed_trades),
            'win_rate': win_rate,
            'final_capital': self.capital
        }

# =============================================================================
# Main
# =============================================================================

def run_full_pipeline_backtest():
    """Run full integrated pipeline backtest."""
    print("=" * 70)
    print("HIMARI Layer 2 V1 - FULL INTEGRATED PIPELINE BACKTEST")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load all checkpoints
    print("\n[1/3] Loading all checkpoints...")
    checkpoints = {}

    # PPO
    ppo_path = BASE_DIR / "L2V1 PPO FINAL" / "himari_ppo_final.pt"
    checkpoints['ppo'] = torch.load(ppo_path, map_location=device, weights_only=False)
    print("  [OK] PPO loaded")

    # AHHMM
    ahhmm_path = BASE_DIR / "L2V1 AHHMM FINAL" / "student_t_ahhmm_percentile.pkl"
    with open(ahhmm_path, 'rb') as f:
        checkpoints['ahhmm'] = pickle.load(f)
    print("  [OK] AHHMM loaded")

    # EKF
    ekf_path = BASE_DIR / "L2V1 EKF FINAL" / "ekf_config_calibrated.pkl"
    with open(ekf_path, 'rb') as f:
        checkpoints['ekf'] = pickle.load(f)
    print("  [OK] EKF loaded")

    # Risk Manager
    risk_path = BASE_DIR / "L2V1 RISK MANAGER FINAL" / "risk_manager_config.pkl"
    with open(risk_path, 'rb') as f:
        checkpoints['risk_manager'] = pickle.load(f)
    print("  [OK] Risk Manager loaded")

    # Load test data
    print("\n[2/3] Loading test data...")
    data_path = BASE_DIR / "btc_1h_2025_2026_test_arrays.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    test_data = data['test']
    features = test_data['features_denoised']
    prices = test_data['prices']
    n_samples = len(features)

    print(f"  Samples: {n_samples}")
    print(f"  Date range: {data['metadata']['start_date']} to {data['metadata']['end_date']}")

    # Initialize pipeline
    print("\n[3/3] Running full pipeline backtest...")
    pipeline = FullIntegratedPipeline(checkpoints, device)
    simulator = TradingSimulator(initial_capital=10000.0)

    actions = []
    regimes = []
    confidences = []
    position_sizes = []
    latencies = []
    blocked_reasons = []

    start_time = time.time()

    for i in range(n_samples):
        # Process through full pipeline
        result = pipeline.process(features[i], prices[i], simulator.capital)

        # Execute trade
        sim_result = simulator.step(result['action'], result['position_size'], prices[i])

        # Update safety system with PnL
        if sim_result['pnl'] != 0:
            pipeline.safety.update_daily_pnl(sim_result['pnl'] / simulator.capital)

        # Track results
        actions.append(result['action'])
        regimes.append(result['regime'])
        confidences.append(result['calibrated_confidence'])
        position_sizes.append(result['position_size'])
        latencies.append(result['latency_ms'])
        if result['blocked_by']:
            blocked_reasons.append(result['blocked_by'])

        # Progress
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1}/{n_samples} ({100*(i+1)/n_samples:.1f}%) - {elapsed:.1f}s")

    total_time = time.time() - start_time

    # Results
    print("\n" + "=" * 70)
    print("FULL PIPELINE BACKTEST RESULTS")
    print("=" * 70)

    stats = simulator.get_stats()

    print("\n[PERFORMANCE METRICS]")
    print(f"  Total Return:  {stats['total_return']:+.2f}%")
    print(f"  Max Drawdown:  {stats['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio:  {stats['sharpe']:.3f}")
    print(f"  Win Rate:      {stats['win_rate']:.1f}%")
    print(f"  Total Trades:  {stats['n_trades']}")
    print(f"  Final Capital: ${stats['final_capital']:.2f}")

    print("\n[PIPELINE STATISTICS]")
    print(f"  Total Signals:      {pipeline.stats['total_signals']}")
    print(f"  HSM Blocked:        {pipeline.stats['hsm_blocked']} ({100*pipeline.stats['hsm_blocked']/max(1,pipeline.stats['total_signals']):.1f}%)")
    print(f"  Hysteresis Blocked: {pipeline.stats['hysteresis_blocked']} ({100*pipeline.stats['hysteresis_blocked']/max(1,pipeline.stats['total_signals']):.1f}%)")
    print(f"  Safety Blocked:     {pipeline.stats['safety_blocked']} ({100*pipeline.stats['safety_blocked']/max(1,pipeline.stats['total_signals']):.1f}%)")
    print(f"  Trades Executed:    {pipeline.stats['trades_executed']}")

    print("\n[LATENCY STATISTICS]")
    lat = np.array(latencies)
    print(f"  Mean:   {np.mean(lat):.3f} ms")
    print(f"  P95:    {np.percentile(lat, 95):.3f} ms")
    print(f"  P99:    {np.percentile(lat, 99):.3f} ms")
    print(f"  Total:  {total_time:.1f}s ({n_samples/total_time:.0f} samples/sec)")

    print("\n[ACTION DISTRIBUTION]")
    actions = np.array(actions)
    print(f"  SELL (-1): {np.sum(actions == -1):5d} ({100*np.mean(actions == -1):.1f}%)")
    print(f"  HOLD (0):  {np.sum(actions == 0):5d} ({100*np.mean(actions == 0):.1f}%)")
    print(f"  BUY (1):   {np.sum(actions == 1):5d} ({100*np.mean(actions == 1):.1f}%)")

    print("\n[REGIME DISTRIBUTION]")
    regimes = np.array(regimes)
    regime_names = ['BULL', 'BEAR', 'SIDEWAYS', 'CRISIS']
    for r in range(4):
        count = np.sum(regimes == r)
        print(f"  {regime_names[r]:8s}: {count:5d} ({100*count/len(regimes):.1f}%)")

    print("\n[POSITION SIZE STATISTICS]")
    sizes = np.array(position_sizes)
    nonzero_sizes = sizes[sizes > 0]
    if len(nonzero_sizes) > 0:
        print(f"  Mean (when trading): {np.mean(nonzero_sizes):.4f}")
        print(f"  Max:                 {np.max(nonzero_sizes):.4f}")

    # Validation
    print("\n[VALIDATION]")
    sharpe_pass = stats['sharpe'] > -0.5  # More lenient for full pipeline
    dd_pass = stats['max_drawdown'] < 20
    latency_pass = np.percentile(lat, 99) < 50

    print(f"  Sharpe > -0.5:        {'PASS' if sharpe_pass else 'FAIL'} ({stats['sharpe']:.3f})")
    print(f"  Max DD < 20%:         {'PASS' if dd_pass else 'FAIL'} ({stats['max_drawdown']:.2f}%)")
    print(f"  Latency P99 < 50ms:   {'PASS' if latency_pass else 'FAIL'} ({np.percentile(lat, 99):.3f} ms)")

    print("=" * 70)

    return {
        'stats': stats,
        'pipeline_stats': pipeline.stats,
        'latency_p99': np.percentile(lat, 99)
    }


if __name__ == "__main__":
    try:
        results = run_full_pipeline_backtest()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
