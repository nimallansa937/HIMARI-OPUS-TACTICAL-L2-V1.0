"""
HIMARI Layer 2 - Comprehensive Model Evaluation (Best Practices 2026)
Evaluates all trained models with multiple metrics and statistical rigor.

Usage:
    python scripts/evaluate_trained_models_fixed.py --checkpoints-dir ./checkpoints --data-dir ./data
"""

import argparse
import sys
import os
import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
from scipy import stats

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.baseline_mlp import create_baseline_model
from src.models.cql import create_cql_agent
from src.models.ppo_lstm import create_ppo_lstm_agent
from src.models.cgdt import create_cgdt_agent
from src.models.flag_trader import create_flag_trader_agent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load test data with proper time-series split.

    Best practice: For time-series, use chronological split (not random).
    Train: 0-80%, Val: 80-90%, Test: 90-100%
    """
    data_path = Path(data_dir)
    features = np.load(data_path / "preprocessed_features.npy")
    labels = np.load(data_path / "labels.npy")

    # Time-series split (last 10% as test set - NEVER seen during training/validation)
    total_samples = len(features)
    test_start = int(0.9 * total_samples)

    test_features = features[test_start:]
    test_labels = labels[test_start:]

    # Check class distribution
    unique, counts = np.unique(test_labels, return_counts=True)
    class_dist = dict(zip(unique.astype(int), counts))

    logger.info(f"Test set: {len(test_features)} samples (last 10% chronologically)")
    logger.info(f"Class distribution: {class_dist}")

    metadata = {
        'total_samples': total_samples,
        'test_samples': len(test_features),
        'test_start_idx': test_start,
        'class_distribution': class_dist
    }

    return test_features, test_labels, metadata


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    class_names: List[str] = ['SELL', 'HOLD', 'BUY']
) -> Dict:
    """
    Calculate comprehensive metrics following best practices.

    Metrics:
    - Accuracy, Precision, Recall, F1 (macro & weighted)
    - Per-class metrics
    - Confusion matrix
    - ROC-AUC (if probabilities available)
    - Confidence intervals (bootstrap)
    """

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1 (macro and weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Bootstrap confidence interval for accuracy (1000 samples)
    bootstrap_accuracies = []
    n_bootstrap = 1000
    n_samples = len(y_true)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        acc = accuracy_score(y_true[indices], y_pred[indices])
        bootstrap_accuracies.append(acc)

    acc_ci_lower = np.percentile(bootstrap_accuracies, 2.5)
    acc_ci_upper = np.percentile(bootstrap_accuracies, 97.5)
    acc_std = np.std(bootstrap_accuracies)

    # ROC-AUC (if probabilities provided)
    roc_auc = None
    if y_proba is not None and y_proba.ndim == 2:
        try:
            # One-vs-rest ROC-AUC
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
            roc_auc = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
        except:
            pass

    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(f1_per_class[i]),
            'support': int(support[i]),
            'accuracy': float((y_pred[y_true == i] == i).sum() / max(support[i], 1))
        }

    return {
        # Overall metrics
        'accuracy': float(accuracy),
        'accuracy_ci_lower': float(acc_ci_lower),
        'accuracy_ci_upper': float(acc_ci_upper),
        'accuracy_std': float(acc_std),

        # Macro metrics (treat all classes equally)
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),

        # Weighted metrics (account for class imbalance)
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),

        # Per-class metrics
        'per_class': per_class_metrics,

        # Confusion matrix
        'confusion_matrix': cm.tolist(),

        # ROC-AUC
        'roc_auc_macro': float(roc_auc) if roc_auc else None
    }


def evaluate_baselinemlp(
    checkpoint_path: str,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    device: str
) -> Dict:
    """Evaluate BaselineMLP with comprehensive metrics."""
    logger.info("Evaluating BaselineMLP...")

    try:
        # Create model
        model = create_baseline_model('mlp', input_dim=60, hidden_dims=[128, 64, 32], num_classes=3)
        model.to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        # Evaluate
        start_time = time.time()
        with torch.no_grad():
            inputs = torch.FloatTensor(test_features).to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)

        inference_time = time.time() - start_time

        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(test_labels, predictions, probabilities)

        result = {
            'model': 'BaselineMLP',
            'status': 'success',
            'params': 18691,
            'inference_time': inference_time,
            'samples_per_sec': len(test_features) / inference_time,
            **metrics
        }

        logger.info(f"  Accuracy: {metrics['accuracy']:.4f} (95% CI: [{metrics['accuracy_ci_lower']:.4f}, {metrics['accuracy_ci_upper']:.4f}])")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")

        return result

    except Exception as e:
        logger.error(f"  Failed: {e}")
        return {'model': 'BaselineMLP', 'status': 'failed', 'error': str(e)}


def evaluate_cql(
    checkpoint_path: str,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    device: str
) -> Dict:
    """Evaluate CQL with comprehensive metrics."""
    logger.info("Evaluating CQL...")

    try:
        # Create agent
        agent = create_cql_agent(65, 3, 256).to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'q_network1' in checkpoint:
            agent.q_network1.load_state_dict(checkpoint['q_network1'])
            agent.q_network2.load_state_dict(checkpoint['q_network2'])
        else:
            agent.load_state_dict(checkpoint.get('model', checkpoint))

        agent.eval()

        # Add dummy env features
        test_states = np.concatenate([test_features, np.zeros((len(test_features), 5))], axis=1)

        # Evaluate
        start_time = time.time()
        with torch.no_grad():
            states = torch.FloatTensor(test_states).to(device)
            q_values = agent(states).cpu().numpy()
            predictions = np.argmax(q_values, axis=1)
            # CQL outputs Q-values, convert to probabilities with softmax
            probabilities = np.exp(q_values) / np.exp(q_values).sum(axis=1, keepdims=True)

        inference_time = time.time() - start_time

        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(test_labels, predictions, probabilities)

        result = {
            'model': 'CQL',
            'status': 'success',
            'params': 116995,
            'inference_time': inference_time,
            'samples_per_sec': len(test_features) / inference_time,
            **metrics
        }

        logger.info(f"  Accuracy: {metrics['accuracy']:.4f} (95% CI: [{metrics['accuracy_ci_lower']:.4f}, {metrics['accuracy_ci_upper']:.4f}])")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")

        return result

    except Exception as e:
        logger.error(f"  Failed: {e}")
        return {'model': 'CQL', 'status': 'failed', 'error': str(e)}


def evaluate_cgdt(
    checkpoint_path: str,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    device: str
) -> Dict:
    """Evaluate CGDT with comprehensive metrics."""
    logger.info("Evaluating CGDT...")

    try:
        # Create agent
        agent = create_cgdt_agent(60, 3, 256, 6).to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'dt' in checkpoint:
            agent.dt.load_state_dict(checkpoint['dt'])
            agent.critic.load_state_dict(checkpoint['critic'])
        else:
            agent.load_state_dict(checkpoint.get('model', checkpoint))

        agent.eval()

        # CGDT needs sequences
        context_length = 64
        num_sequences = len(test_features) // context_length

        predictions_list = []
        probabilities_list = []

        start_time = time.time()
        with torch.no_grad():
            for i in range(num_sequences):
                start_idx = i * context_length
                end_idx = start_idx + context_length

                states = torch.FloatTensor(test_features[start_idx:end_idx]).unsqueeze(0).to(device)
                actions = torch.zeros(1, context_length).long().to(device)
                returns_to_go = torch.zeros(1, context_length).to(device)
                timesteps = torch.arange(context_length).unsqueeze(0).to(device)

                action_preds = agent(states, actions, returns_to_go, timesteps)
                probs = torch.softmax(action_preds, dim=-1).cpu().numpy()[0]
                preds = np.argmax(probs, axis=-1)

                predictions_list.extend(preds)
                probabilities_list.extend(probs)

        inference_time = time.time() - start_time

        predictions = np.array(predictions_list)
        probabilities = np.array(probabilities_list)
        labels_truncated = test_labels[:len(predictions)]

        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(labels_truncated, predictions, probabilities)

        result = {
            'model': 'CGDT',
            'status': 'success',
            'params': 4822276,
            'inference_time': inference_time,
            'samples_per_sec': len(predictions) / inference_time,
            'samples_evaluated': len(predictions),
            **metrics
        }

        logger.info(f"  Accuracy: {metrics['accuracy']:.4f} (95% CI: [{metrics['accuracy_ci_lower']:.4f}, {metrics['accuracy_ci_upper']:.4f}])")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")

        return result

    except Exception as e:
        logger.error(f"  Failed: {e}")
        return {'model': 'CGDT', 'status': 'failed', 'error': str(e)}


def evaluate_flag_trader(
    checkpoint_path: str,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    device: str
) -> Dict:
    """Evaluate FLAG-TRADER with comprehensive metrics."""
    logger.info("Evaluating FLAG-TRADER...")

    try:
        # Create agent
        agent = create_flag_trader_agent(60, 3, "135M", 16).to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.model.load_state_dict(checkpoint['model'])
        agent.eval()

        # FLAG-TRADER needs sequences
        context_length = 128
        num_sequences = len(test_features) // context_length

        predictions_list = []
        probabilities_list = []

        start_time = time.time()
        with torch.no_grad():
            for i in range(num_sequences):
                start_idx = i * context_length
                end_idx = start_idx + context_length

                states = torch.FloatTensor(test_features[start_idx:end_idx]).unsqueeze(0).to(device)
                action_preds = agent(states)
                probs = torch.softmax(action_preds, dim=-1).cpu().numpy()[0]
                preds = np.argmax(probs, axis=-1)

                predictions_list.extend(preds)
                probabilities_list.extend(probs)

        inference_time = time.time() - start_time

        predictions = np.array(predictions_list)
        probabilities = np.array(probabilities_list)
        labels_truncated = test_labels[:len(predictions)]

        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(labels_truncated, predictions, probabilities)

        result = {
            'model': 'FLAG-TRADER',
            'status': 'success',
            'params': 87955971,
            'trainable_params': 2938371,
            'inference_time': inference_time,
            'samples_per_sec': len(predictions) / inference_time,
            'samples_evaluated': len(predictions),
            **metrics
        }

        logger.info(f"  Accuracy: {metrics['accuracy']:.4f} (95% CI: [{metrics['accuracy_ci_lower']:.4f}, {metrics['accuracy_ci_upper']:.4f}])")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")

        return result

    except Exception as e:
        logger.error(f"  Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'model': 'FLAG-TRADER', 'status': 'failed', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation (Best Practices 2026)")
    parser.add_argument('--checkpoints-dir', type=str, default='../checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing data')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run evaluation on')
    parser.add_argument('--output', type=str, default='../evaluation_results_comprehensive.json',
                       help='Output file for results')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("HIMARI Layer 2 - Comprehensive Model Evaluation (Best Practices 2026)")
    logger.info("=" * 80)
    logger.info(f"Reproducibility: Random seed set to {SEED}")
    logger.info("")

    # Load test data
    test_features, test_labels, metadata = load_data(args.data_dir)
    logger.info("")

    # Evaluate each model
    results = []
    checkpoint_dir = Path(args.checkpoints_dir)

    # BaselineMLP
    if (checkpoint_dir / "baseline_best.pt").exists():
        result = evaluate_baselinemlp(
            str(checkpoint_dir / "baseline_best.pt"),
            test_features, test_labels, args.device
        )
        if result['status'] == 'success':
            results.append(result)
        logger.info("")

    # CQL
    if (checkpoint_dir / "cql_best.pt").exists():
        result = evaluate_cql(
            str(checkpoint_dir / "cql_best.pt"),
            test_features, test_labels, args.device
        )
        if result['status'] == 'success':
            results.append(result)
        logger.info("")

    # CGDT
    if (checkpoint_dir / "cgdt_best.pt").exists():
        result = evaluate_cgdt(
            str(checkpoint_dir / "cgdt_best.pt"),
            test_features, test_labels, args.device
        )
        if result['status'] == 'success':
            results.append(result)
        logger.info("")

    # FLAG-TRADER
    if (checkpoint_dir / "flag_trader_best.pt").exists():
        result = evaluate_flag_trader(
            str(checkpoint_dir / "flag_trader_best.pt"),
            test_features, test_labels, args.device
        )
        if result['status'] == 'success':
            results.append(result)
        logger.info("")

    # Print comprehensive results
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info("")

    # Sort by F1 macro (better metric than accuracy for imbalanced data)
    results = sorted(results, key=lambda x: x.get('f1_macro', 0), reverse=True)

    for i, result in enumerate(results, 1):
        logger.info(f"{i}. {result['model']}")
        logger.info(f"   Accuracy: {result['accuracy']:.4f} ± {result['accuracy_std']:.4f}")
        logger.info(f"   95% CI: [{result['accuracy_ci_lower']:.4f}, {result['accuracy_ci_upper']:.4f}]")
        logger.info(f"   F1 (macro): {result['f1_macro']:.4f}")
        logger.info(f"   F1 (weighted): {result['f1_weighted']:.4f}")
        logger.info(f"   Precision (macro): {result['precision_macro']:.4f}")
        logger.info(f"   Recall (macro): {result['recall_macro']:.4f}")
        logger.info(f"   Parameters: {result['params']:,}")
        logger.info("")

    # Save comprehensive results
    output_data = {
        'evaluation_date': '2026-01-04',
        'methodology': 'Best Practices 2026 - Comprehensive Metrics',
        'random_seed': SEED,
        'metadata': metadata,
        'results': results,
        'best_model_by_f1': results[0]['model'] if results else None,
        'best_f1_macro': results[0]['f1_macro'] if results else None
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Comprehensive results saved to: {args.output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
