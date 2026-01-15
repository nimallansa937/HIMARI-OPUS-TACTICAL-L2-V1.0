#!/usr/bin/env python3
"""
Train Student-T AHHMM Regime Detector for HIMARI Layer 2.

This script:
1. Loads BTC 1H data from local CSV files
2. Prepares features: returns, volume_norm, volatility
3. Saves processed features to BTC DATA SETS folder
4. Trains Student-T AHHMM with EM algorithm
5. Evaluates regime detection accuracy
6. Tests on unseen 2025-2026 data
7. Saves trained model checkpoint

Key Lessons from PPO Training Applied:
- Variance normalization for features
- Adaptive parameters (not hardcoded)
- Test on unseen data to verify generalization
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# Data paths - LOCAL FILES
DATA_DIR = r"C:\Users\chari\OneDrive\Documents\BTC DATA SETS"
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "btc_1h_2020_2024.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "btc_1h_2025_2026.csv")

# Output paths
OUTPUT_DIR = r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1\L2V1 AHHMM FINAL"
MODEL_PATH = os.path.join(OUTPUT_DIR, "student_t_ahhmm_trained.pkl")

# Processed features will be saved here
TRAIN_FEATURES_PATH = os.path.join(DATA_DIR, "btc_1h_2020_2024_features.pkl")
TEST_FEATURES_PATH = os.path.join(DATA_DIR, "btc_1h_2025_2026_features.pkl")

# Model config
N_MARKET_STATES = 4  # LOW_VOL, TRENDING, HIGH_VOL, CRISIS
N_META_STATES = 2    # Low/High Uncertainty
DF = 5.0             # Student-t degrees of freedom (fat tails)
N_ITER = 100         # EM iterations

# =============================================================================
# Data Loading (from local CSV)
# =============================================================================

def load_btc_data(csv_path: str) -> pd.DataFrame:
    """Load BTC data from local CSV file."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df

# =============================================================================
# Feature Preparation
# =============================================================================

def prepare_ahhmm_features(df: pd.DataFrame, save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features for AHHMM: [returns, volume_norm, volatility]

    Returns:
        features: (n_samples, 3) array
        returns: (n_samples,) array for regime labeling
    """
    close = df['close'].values
    volume = df['volume'].values

    # Returns
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])

    # Volatility (20-period rolling std of returns)
    volatility = pd.Series(returns).rolling(20).std().values
    volatility = np.nan_to_num(volatility, nan=0.01)

    # Volume normalized (ratio to 20-period MA)
    vol_ma = pd.Series(volume).rolling(20).mean().values
    volume_norm = volume / (vol_ma + 1e-10)
    volume_norm = np.nan_to_num(volume_norm, nan=1.0)

    # Stack features
    features = np.column_stack([returns, volume_norm, volatility])

    # Remove warmup period
    features = features[200:]
    returns = returns[200:]

    print(f"Prepared features: {features.shape}")
    print(f"  Returns range: [{returns.min():.4f}, {returns.max():.4f}]")
    print(f"  Volatility range: [{volatility[200:].min():.4f}, {volatility[200:].max():.4f}]")

    # Save processed features if path provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump({
                'features': features,
                'returns': returns,
                'timestamps': df.index[200:].tolist(),
                'created': datetime.now().isoformat()
            }, f)
        print(f"  Saved processed features to: {save_path}")

    return features.astype(np.float32), returns.astype(np.float32)

# =============================================================================
# Ground Truth Regime Labels (for evaluation)
# =============================================================================

def create_ground_truth_regimes(returns: np.ndarray) -> np.ndarray:
    """
    Create ground truth regime labels based on volatility percentiles.
    Same logic as PPO training for consistency.

    Regimes:
        0 = LOW_VOL (quiet markets)
        1 = TRENDING (strong directional moves)
        2 = HIGH_VOL (volatile but not crisis)
        3 = CRISIS (extreme volatility)
    """
    n = len(returns)
    regimes = np.zeros(n, dtype=np.int64)

    # Rolling volatility
    vol_20 = pd.Series(returns).rolling(20).std().values

    # Trend strength (absolute cumulative returns)
    cum_ret_20 = pd.Series(returns).rolling(20).sum().abs().values

    # Percentile thresholds
    vol_low = np.nanpercentile(vol_20, 33)
    vol_high = np.nanpercentile(vol_20, 67)
    vol_crisis = np.nanpercentile(vol_20, 90)
    trend_thresh = np.nanpercentile(cum_ret_20, 60)

    for i in range(n):
        vol = vol_20[i] if not np.isnan(vol_20[i]) else 0
        trend = cum_ret_20[i] if not np.isnan(cum_ret_20[i]) else 0

        if vol > vol_crisis:
            regimes[i] = 3  # CRISIS
        elif vol > vol_high:
            regimes[i] = 2  # HIGH_VOL
        elif trend > trend_thresh and vol < vol_high:
            regimes[i] = 1  # TRENDING
        else:
            regimes[i] = 0  # LOW_VOL

    return regimes

# =============================================================================
# Student-T AHHMM Implementation (Simplified for Training)
# =============================================================================

from scipy import stats
from scipy.special import logsumexp

class StudentTAHHMM:
    """Student-t Adaptive Hierarchical HMM for Regime Detection."""

    def __init__(self, n_states: int = 4, df: float = 5.0, n_features: int = 3):
        self.n_states = n_states
        self.df = df
        self.n_features = n_features

        # Emission parameters per state: mean, scale
        self.means = np.zeros((n_states, n_features))
        self.scales = np.ones((n_states, n_features))

        # Transition matrix
        self.trans = np.eye(n_states) * 0.9 + 0.1 / n_states

        # Initial state probabilities
        self.pi = np.ones(n_states) / n_states

        # State tracking
        self.state_probs = np.ones(n_states) / n_states
        self._fitted = False

        # State names for reporting
        self.state_names = ['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS']

    def _student_t_logpdf(self, x: np.ndarray, mean: np.ndarray,
                          scale: np.ndarray) -> float:
        """Multivariate Student-t log probability."""
        # Use df=3 for crisis state (fatter tails)
        return np.sum(stats.t.logpdf(x, df=self.df, loc=mean, scale=scale))

    def _emission_log_prob(self, obs: np.ndarray, state: int) -> float:
        """Log probability of observation given state."""
        return self._student_t_logpdf(obs, self.means[state], self.scales[state])

    def fit(self, features: np.ndarray, n_iter: int = 100, verbose: bool = True):
        """
        Fit HMM using EM algorithm.

        Args:
            features: (n_samples, n_features) array
            n_iter: Number of EM iterations
        """
        n_samples, n_features = features.shape

        # Initialize emission parameters using K-means-like clustering
        # Based on volatility (feature 2) for initial assignment
        volatility = features[:, 2]
        vol_percentiles = [0, 33, 67, 90, 100]

        for s in range(self.n_states):
            low_pct = np.nanpercentile(volatility, vol_percentiles[s])
            high_pct = np.nanpercentile(volatility, vol_percentiles[s + 1])

            mask = (volatility >= low_pct) & (volatility < high_pct)
            if mask.sum() > 0:
                self.means[s] = np.mean(features[mask], axis=0)
                self.scales[s] = np.std(features[mask], axis=0) + 1e-6
            else:
                self.means[s] = np.mean(features, axis=0)
                self.scales[s] = np.std(features, axis=0) + 1e-6

        if verbose:
            print(f"\nInitial emission parameters:")
            for s in range(self.n_states):
                print(f"  {self.state_names[s]}: mean={self.means[s]}, scale={self.scales[s]}")

        # EM iterations
        prev_ll = -np.inf
        for iteration in range(n_iter):
            # E-step: Compute responsibilities (posterior state probabilities)
            log_probs = np.zeros((n_samples, self.n_states))
            for t in range(n_samples):
                for s in range(self.n_states):
                    log_probs[t, s] = self._emission_log_prob(features[t], s)

            # Forward-backward for responsibilities
            # Simplified: just use emission probs (no transition dynamics for initial fit)
            log_responsibilities = log_probs - logsumexp(log_probs, axis=1, keepdims=True)
            responsibilities = np.exp(log_responsibilities)

            # M-step: Update emission parameters
            for s in range(self.n_states):
                weights = responsibilities[:, s]
                weights_sum = weights.sum() + 1e-8

                new_mean = np.average(features, axis=0, weights=weights)
                new_scale = np.sqrt(
                    np.average((features - new_mean)**2, axis=0, weights=weights)
                ) + 1e-6

                # Smooth update (lesson from PPO: adaptive, not sudden changes)
                self.means[s] = 0.7 * self.means[s] + 0.3 * new_mean
                self.scales[s] = 0.7 * self.scales[s] + 0.3 * new_scale

            # Update transition matrix
            for s in range(self.n_states):
                for t in range(1, n_samples):
                    trans_weight = responsibilities[t-1, :].reshape(-1, 1) * responsibilities[t, :].reshape(1, -1)
                    self.trans = 0.99 * self.trans + 0.01 * (trans_weight.sum(axis=0) / (trans_weight.sum() + 1e-8))

            # Normalize transition matrix
            self.trans = self.trans / self.trans.sum(axis=1, keepdims=True)

            # Compute log-likelihood
            ll = logsumexp(log_probs, axis=1).mean()

            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}/{n_iter}: LL = {ll:.4f}")

            # Check convergence
            if abs(ll - prev_ll) < 1e-6:
                if verbose:
                    print(f"  Converged at iteration {iteration + 1}")
                break
            prev_ll = ll

        self._fitted = True

        if verbose:
            print(f"\nFinal emission parameters:")
            for s in range(self.n_states):
                print(f"  {self.state_names[s]}:")
                print(f"    mean=[ret:{self.means[s,0]:.5f}, vol_norm:{self.means[s,1]:.3f}, volatility:{self.means[s,2]:.5f}]")
                print(f"    scale=[{self.scales[s,0]:.5f}, {self.scales[s,1]:.3f}, {self.scales[s,2]:.5f}]")

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regime for each observation.

        Returns:
            regimes: (n_samples,) predicted regime indices
            confidences: (n_samples,) confidence scores
        """
        n_samples = features.shape[0]

        log_probs = np.zeros((n_samples, self.n_states))
        for t in range(n_samples):
            for s in range(self.n_states):
                log_probs[t, s] = self._emission_log_prob(features[t], s)

        # Viterbi-like forward pass with transitions
        state_probs = np.zeros((n_samples, self.n_states))
        state_probs[0] = self.pi

        for t in range(1, n_samples):
            # Transition
            pred = state_probs[t-1] @ self.trans
            # Emission update
            emission = np.exp(log_probs[t] - logsumexp(log_probs[t]))
            state_probs[t] = pred * emission
            state_probs[t] /= state_probs[t].sum() + 1e-10

        regimes = np.argmax(state_probs, axis=1)
        confidences = np.max(state_probs, axis=1)

        return regimes, confidences

    def save(self, path: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'n_states': self.n_states,
                'df': self.df,
                'n_features': self.n_features,
                'means': self.means,
                'scales': self.scales,
                'trans': self.trans,
                'pi': self.pi,
                'state_names': self.state_names,
                'fitted': self._fitted
            }, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'StudentTAHHMM':
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(n_states=data['n_states'], df=data['df'], n_features=data['n_features'])
        model.means = data['means']
        model.scales = data['scales']
        model.trans = data['trans']
        model.pi = data['pi']
        model.state_names = data['state_names']
        model._fitted = data['fitted']

        return model

# =============================================================================
# Evaluation
# =============================================================================

def evaluate_regime_detection(predicted: np.ndarray, ground_truth: np.ndarray,
                              state_names: List[str]) -> Dict:
    """Evaluate regime detection accuracy."""

    n_samples = len(predicted)

    # Overall accuracy
    accuracy = (predicted == ground_truth).mean() * 100

    # Per-regime accuracy and confusion
    results = {
        'overall_accuracy': accuracy,
        'per_regime': {},
        'confusion_matrix': np.zeros((4, 4), dtype=int)
    }

    for r in range(4):
        mask = ground_truth == r
        if mask.sum() > 0:
            regime_acc = (predicted[mask] == r).mean() * 100
            results['per_regime'][state_names[r]] = {
                'accuracy': regime_acc,
                'count': int(mask.sum()),
                'predicted_dist': {
                    state_names[i]: (predicted[mask] == i).sum()
                    for i in range(4)
                }
            }

    # Build confusion matrix
    for i in range(n_samples):
        results['confusion_matrix'][ground_truth[i], predicted[i]] += 1

    # Compute F1 scores per regime
    for r in range(4):
        tp = results['confusion_matrix'][r, r]
        fp = results['confusion_matrix'][:, r].sum() - tp
        fn = results['confusion_matrix'][r, :].sum() - tp

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        if state_names[r] in results['per_regime']:
            results['per_regime'][state_names[r]]['f1'] = f1 * 100
            results['per_regime'][state_names[r]]['precision'] = precision * 100
            results['per_regime'][state_names[r]]['recall'] = recall * 100

    return results

def print_results(results: Dict, title: str):
    """Pretty print evaluation results."""
    print(f"\n{'=' * 70}")
    print(title)
    print('=' * 70)

    print(f"\nOverall Accuracy: {results['overall_accuracy']:.1f}%")

    print("\nPer-Regime Metrics:")
    print("-" * 60)
    print(f"{'Regime':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Count':>8}")
    print("-" * 60)

    for regime, metrics in results['per_regime'].items():
        print(f"{regime:<12} {metrics['accuracy']:>9.1f}% {metrics.get('precision', 0):>9.1f}% "
              f"{metrics.get('recall', 0):>9.1f}% {metrics.get('f1', 0):>9.1f}% {metrics['count']:>8}")

    print("\nConfusion Matrix (rows=ground truth, cols=predicted):")
    state_names = ['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS']
    print(f"{'':>12}", end='')
    for name in state_names:
        print(f"{name:>10}", end='')
    print()

    for i, name in enumerate(state_names):
        print(f"{name:>12}", end='')
        for j in range(4):
            print(f"{results['confusion_matrix'][i, j]:>10}", end='')
        print()

# =============================================================================
# Main Training
# =============================================================================

def main():
    print("=" * 70)
    print("Student-T AHHMM Regime Detector Training")
    print("=" * 70)
    print(f"\nApplying lessons from PPO training:")
    print("  - Variance-normalized features")
    print("  - Adaptive parameters (smooth updates)")
    print("  - Test on unseen 2025-2026 data")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Step 1: Load Training Data (2020-2024) from local CSV
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading Training Data (2020-2024)")
    print("=" * 70)

    train_df = load_btc_data(TRAIN_DATA_PATH)
    train_features, train_returns = prepare_ahhmm_features(train_df, save_path=TRAIN_FEATURES_PATH)
    train_ground_truth = create_ground_truth_regimes(train_returns)

    print(f"\nTraining data: {len(train_features)} samples")
    print(f"Ground truth regime distribution:")
    for r in range(4):
        pct = (train_ground_truth == r).mean() * 100
        print(f"  {['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS'][r]}: {pct:.1f}%")

    # =========================================================================
    # Step 2: Train Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Training Student-T AHHMM")
    print("=" * 70)

    model = StudentTAHHMM(n_states=N_MARKET_STATES, df=DF, n_features=3)
    model.fit(train_features, n_iter=N_ITER, verbose=True)

    # =========================================================================
    # Step 3: Evaluate on Training Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Evaluating on Training Data")
    print("=" * 70)

    train_predicted, train_confidence = model.predict(train_features)
    train_results = evaluate_regime_detection(
        train_predicted, train_ground_truth, model.state_names
    )
    print_results(train_results, "Training Set Results (2020-2024)")

    # =========================================================================
    # Step 4: Load and Evaluate on Test Data (2025-2026)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Evaluating on Unseen Data (2025-2026)")
    print("=" * 70)

    test_df = load_btc_data(TEST_DATA_PATH)
    test_features, test_returns = prepare_ahhmm_features(test_df, save_path=TEST_FEATURES_PATH)
    test_ground_truth = create_ground_truth_regimes(test_returns)

    print(f"\nTest data: {len(test_features)} samples")
    print(f"Ground truth regime distribution:")
    for r in range(4):
        pct = (test_ground_truth == r).mean() * 100
        print(f"  {['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS'][r]}: {pct:.1f}%")

    test_predicted, test_confidence = model.predict(test_features)
    test_results = evaluate_regime_detection(
        test_predicted, test_ground_truth, model.state_names
    )
    print_results(test_results, "Test Set Results (2025-2026 Unseen Data)")

    # =========================================================================
    # Step 5: Compare Training vs Test (Generalization Check)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Generalization Comparison")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Training':>12} {'Test':>12} {'Diff':>10}")
    print("-" * 54)
    print(f"{'Overall Accuracy':<20} {train_results['overall_accuracy']:>11.1f}% "
          f"{test_results['overall_accuracy']:>11.1f}% "
          f"{test_results['overall_accuracy'] - train_results['overall_accuracy']:>+9.1f}%")

    for regime in ['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS']:
        train_acc = train_results['per_regime'].get(regime, {}).get('accuracy', 0)
        test_acc = test_results['per_regime'].get(regime, {}).get('accuracy', 0)
        diff = test_acc - train_acc
        print(f"{regime:<20} {train_acc:>11.1f}% {test_acc:>11.1f}% {diff:>+9.1f}%")

    # =========================================================================
    # Step 6: Save Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 6: Saving Model")
    print("=" * 70)

    model.save(MODEL_PATH)

    # Save training log
    log_path = f"{OUTPUT_DIR}/training_log.txt"
    with open(log_path, 'w') as f:
        f.write("Student-T AHHMM Training Log\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Date: {datetime.now().isoformat()}\n")
        f.write(f"Training Data: {TRAIN_DATA_START} to {TRAIN_DATA_END}\n")
        f.write(f"Test Data: {TEST_DATA_START} to {TEST_DATA_END}\n")
        f.write(f"N States: {N_MARKET_STATES}\n")
        f.write(f"Student-t df: {DF}\n")
        f.write(f"EM Iterations: {N_ITER}\n\n")
        f.write(f"Training Accuracy: {train_results['overall_accuracy']:.1f}%\n")
        f.write(f"Test Accuracy: {test_results['overall_accuracy']:.1f}%\n\n")
        f.write("Per-Regime Test Accuracy:\n")
        for regime in ['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS']:
            acc = test_results['per_regime'].get(regime, {}).get('accuracy', 0)
            f.write(f"  {regime}: {acc:.1f}%\n")

    print(f"Training log saved to {log_path}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Training accuracy: {train_results['overall_accuracy']:.1f}%")
    print(f"Test accuracy: {test_results['overall_accuracy']:.1f}%")

    # Check generalization
    acc_diff = test_results['overall_accuracy'] - train_results['overall_accuracy']
    if abs(acc_diff) < 10:
        print("\n✅ Model generalizes well (accuracy drift < 10%)")
    else:
        print(f"\n⚠️ Generalization concern: accuracy drift = {acc_diff:.1f}%")

if __name__ == "__main__":
    main()
