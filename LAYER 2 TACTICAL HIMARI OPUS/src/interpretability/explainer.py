"""
HIMARI Layer 2 - Interpretability
Subsystem N: Explainability (Methods N1-N4)

Purpose:
    Provide interpretable explanations for trading decisions.

Methods:
    N1: LIME/SHAP Attribution - Feature importance for individual predictions
    N2: Attention Visualization - Timeframe importance heatmaps
    N3: Causal Graph Queries - Causal reasoning for decisions
    N4: Decision Tree Distillation - Extract interpretable rules from neural nets

Expected Use Cases:
    - Regulatory compliance (MiFID II)
    - Model debugging
    - Trust building with stakeholders
    - Feature engineering insights

Reference:
    - Lundberg & Lee "A Unified Approach to Interpreting Model Predictions" (2017)
    - Ribeiro et al. "Why Should I Trust You?" (2016)
"""

from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available - install with: pip install shap")

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")


@dataclass
class ExplanationConfig:
    """Configuration for interpretability"""
    # N1: SHAP config
    shap_samples: int = 100  # Background samples for SHAP
    shap_method: str = "kernel"  # or "deep", "gradient"

    # N2: Attention viz
    attention_threshold: float = 0.1  # Min attention to display

    # N3: Causal graph
    causal_threshold: float = 0.3  # Min edge weight

    # N4: Distillation
    tree_max_depth: int = 5  # Max depth for distilled tree
    tree_min_samples_leaf: int = 50  # Min samples per leaf


@dataclass
class Explanation:
    """Explanation for a single prediction"""
    predicted_action: int
    confidence: float
    feature_importances: Dict[str, float]  # N1: SHAP values
    attention_weights: Optional[Dict[str, float]] = None  # N2: Attention
    causal_path: Optional[List[str]] = None  # N3: Causal chain
    rule_path: Optional[str] = None  # N4: Decision tree path


class SHAPExplainer:
    """
    SHAP-based feature attribution (Method N1).

    Example:
        >>> explainer = SHAPExplainer(model, background_data)
        >>> shap_values = explainer.explain(state)
    """

    def __init__(
        self,
        model: Callable,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        config: Optional[ExplanationConfig] = None
    ):
        self.model = model
        self.config = config or ExplanationConfig()
        self.feature_names = feature_names or [f"feature_{i}" for i in range(background_data.shape[1])]

        if not SHAP_AVAILABLE:
            logger.error("SHAP not installed - explanations unavailable")
            self.explainer = None
            return

        # Create SHAP explainer
        if self.config.shap_method == "kernel":
            # Model-agnostic
            self.explainer = shap.KernelExplainer(
                model,
                background_data[:self.config.shap_samples]
            )
        elif self.config.shap_method == "deep":
            # For PyTorch models
            self.explainer = shap.DeepExplainer(
                model,
                torch.tensor(background_data[:self.config.shap_samples], dtype=torch.float32)
            )
        else:
            # Gradient-based
            self.explainer = shap.GradientExplainer(
                model,
                torch.tensor(background_data[:self.config.shap_samples], dtype=torch.float32)
            )

        logger.info(f"SHAP Explainer initialized with {self.config.shap_method} method")

    def explain(self, state: np.ndarray) -> Dict[str, float]:
        """
        Compute SHAP values for a prediction (Method N1).

        Args:
            state: Input state to explain

        Returns:
            feature_importances: Dict mapping feature names to SHAP values
        """
        if self.explainer is None:
            return {}

        # Compute SHAP values
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

        if state.ndim == 1:
            state = state.reshape(1, -1)

        shap_values = self.explainer.shap_values(state)

        # For multi-class, take values for predicted class
        if isinstance(shap_values, list):
            # Multi-class output
            shap_values = shap_values[0]  # Take first class for simplicity

        if shap_values.ndim > 1:
            shap_values = shap_values[0]  # Take first sample

        # Create feature importance dict
        importances = {
            name: float(value)
            for name, value in zip(self.feature_names, shap_values)
        }

        # Sort by absolute importance
        importances = dict(sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True))

        return importances

    def plot_explanation(self, state: np.ndarray, save_path: Optional[str] = None):
        """Generate SHAP waterfall plot"""
        if self.explainer is None or not SHAP_AVAILABLE:
            logger.warning("Cannot plot - SHAP not available")
            return

        shap_values = self.explainer.shap_values(state)
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0] if isinstance(shap_values, list) else shap_values,
            base_values=self.explainer.expected_value,
            data=state[0] if state.ndim > 1 else state,
            feature_names=self.feature_names
        ))

        if save_path:
            import matplotlib.pyplot as plt
            plt.savefig(save_path)
            logger.info(f"SHAP plot saved to {save_path}")


class AttentionVisualizer:
    """
    Attention weight visualization (Method N2).

    Extracts and visualizes attention weights from multi-timeframe fusion.
    """

    def __init__(self, config: Optional[ExplanationConfig] = None):
        self.config = config or ExplanationConfig()
        self.attention_history = []

    def extract_attention(self, attention_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Extract attention weights from model (Method N2).

        Args:
            attention_tensor: Attention weights from model (timeframes,)

        Returns:
            timeframe_importance: Dict mapping timeframes to weights
        """
        if isinstance(attention_tensor, torch.Tensor):
            attention_tensor = attention_tensor.detach().cpu().numpy()

        # Assuming order: 1m, 5m, 1h, 4h
        timeframes = ["1m", "5m", "1h", "4h"]

        # Normalize if needed
        if not np.isclose(np.sum(attention_tensor), 1.0):
            attention_tensor = attention_tensor / (np.sum(attention_tensor) + 1e-8)

        importance = {
            tf: float(weight)
            for tf, weight in zip(timeframes, attention_tensor)
            if weight >= self.config.attention_threshold
        }

        # Store for trend analysis
        self.attention_history.append(importance)
        if len(self.attention_history) > 1000:
            self.attention_history.pop(0)

        return importance

    def get_attention_trends(self) -> Dict[str, float]:
        """Get average attention over recent history"""
        if not self.attention_history:
            return {}

        # Aggregate recent attention
        timeframes = list(self.attention_history[0].keys())
        trends = {}

        for tf in timeframes:
            weights = [h.get(tf, 0.0) for h in self.attention_history[-100:]]
            trends[tf] = float(np.mean(weights))

        return trends


class CausalGraphExplainer:
    """
    Causal graph-based explanations (Method N3).

    Uses learned causal relationships to explain decisions.
    """

    def __init__(
        self,
        causal_graph: Optional[Dict[str, List[str]]] = None,
        config: Optional[ExplanationConfig] = None
    ):
        self.config = config or ExplanationConfig()
        # Causal graph: {cause: [effects]}
        self.causal_graph = causal_graph or self._default_causal_graph()

        logger.info(f"Causal Graph with {len(self.causal_graph)} nodes")

    def _default_causal_graph(self) -> Dict[str, List[str]]:
        """
        Default causal graph for crypto trading.

        In production, this would be learned via NOTEARS or PC algorithm.
        """
        return {
            "btc_volume": ["btc_price", "market_sentiment"],
            "btc_price": ["eth_price", "market_sentiment"],
            "market_sentiment": ["trading_action"],
            "volatility": ["trading_action", "position_size"],
            "regime": ["trading_action", "position_size"],
        }

    def explain_via_causality(
        self,
        decision: str,
        observed_features: Dict[str, float]
    ) -> List[str]:
        """
        Explain decision via causal path (Method N3).

        Args:
            decision: The decision to explain (e.g., "trading_action")
            observed_features: Current feature values

        Returns:
            causal_path: List of causal explanations
        """
        # Find all causes leading to decision
        causes = self._find_causes(decision)

        # Build explanation
        explanations = []
        for cause in causes:
            if cause in observed_features:
                value = observed_features[cause]
                explanations.append(
                    f"{cause} = {value:.3f} → {decision}"
                )

        return explanations

    def _find_causes(self, effect: str, visited: Optional[set] = None) -> List[str]:
        """Recursively find all causes of an effect"""
        if visited is None:
            visited = set()

        if effect in visited:
            return []

        visited.add(effect)
        causes = []

        # Direct causes
        for cause, effects in self.causal_graph.items():
            if effect in effects:
                causes.append(cause)
                # Recursive causes
                causes.extend(self._find_causes(cause, visited))

        return causes


class DecisionTreeDistiller:
    """
    Distill neural network to decision tree (Method N4).

    Creates interpretable rule-based approximation of neural net.
    """

    def __init__(self, config: Optional[ExplanationConfig] = None):
        self.config = config or ExplanationConfig()
        self.tree = None
        self.feature_names = None

        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not installed - distillation unavailable")

    def distill(
        self,
        model: Callable,
        train_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Train decision tree to mimic neural network (Method N4).

        Args:
            model: Neural network to distill
            train_data: Data to generate soft labels
            feature_names: Feature names for rules
        """
        if not SKLEARN_AVAILABLE:
            logger.error("Cannot distill - sklearn not available")
            return

        self.feature_names = feature_names or [f"f{i}" for i in range(train_data.shape[1])]

        # Get soft labels from neural network
        logger.info("Generating soft labels from neural network...")
        soft_labels = []
        for i in range(0, len(train_data), 32):
            batch = train_data[i:i+32]
            if isinstance(model, nn.Module):
                with torch.no_grad():
                    preds = model(torch.tensor(batch, dtype=torch.float32))
                    preds = preds.argmax(dim=-1).cpu().numpy()
            else:
                preds = model(batch)
                if isinstance(preds, torch.Tensor):
                    preds = preds.argmax(dim=-1).cpu().numpy()

            soft_labels.extend(preds)

        soft_labels = np.array(soft_labels)

        # Train decision tree
        logger.info("Training decision tree surrogate...")
        self.tree = DecisionTreeClassifier(
            max_depth=self.config.tree_max_depth,
            min_samples_leaf=self.config.tree_min_samples_leaf,
            random_state=42
        )
        self.tree.fit(train_data, soft_labels)

        # Compute fidelity
        tree_preds = self.tree.predict(train_data)
        fidelity = np.mean(tree_preds == soft_labels)

        logger.info(f"Decision tree fidelity: {fidelity:.1%} (accuracy matching neural net)")

    def explain_prediction(self, state: np.ndarray) -> str:
        """
        Get rule-based explanation for prediction (Method N4).

        Args:
            state: Input state

        Returns:
            rule_path: Human-readable decision path
        """
        if self.tree is None:
            return "Tree not trained"

        if state.ndim == 1:
            state = state.reshape(1, -1)

        # Get decision path
        path = self.tree.decision_path(state)
        node_indicator = path.toarray()[0]

        # Extract rules
        feature = self.tree.tree_.feature
        threshold = self.tree.tree_.threshold

        rules = []
        for node_id in np.where(node_indicator)[0]:
            if feature[node_id] != -2:  # Not a leaf
                feature_name = self.feature_names[feature[node_id]]
                threshold_val = threshold[node_id]

                # Determine direction
                if state[0, feature[node_id]] <= threshold_val:
                    rules.append(f"{feature_name} <= {threshold_val:.3f}")
                else:
                    rules.append(f"{feature_name} > {threshold_val:.3f}")

        return " AND ".join(rules)

    def export_rules(self) -> str:
        """Export all decision rules as text"""
        if self.tree is None:
            return "Tree not trained"

        return export_text(self.tree, feature_names=self.feature_names)


class IntegratedExplainer:
    """
    Combines all explanation methods (N1-N4) into unified interface.
    """

    def __init__(
        self,
        model: Callable,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        config: Optional[ExplanationConfig] = None
    ):
        self.config = config or ExplanationConfig()

        # N1: SHAP
        self.shap_explainer = SHAPExplainer(model, background_data, feature_names, self.config)

        # N2: Attention
        self.attention_viz = AttentionVisualizer(self.config)

        # N3: Causal
        self.causal_explainer = CausalGraphExplainer(config=self.config)

        # N4: Distillation
        self.tree_distiller = DecisionTreeDistiller(self.config)
        self.tree_distiller.distill(model, background_data, feature_names)

        logger.info("Integrated Explainer initialized (N1-N4)")

    def explain_decision(
        self,
        state: np.ndarray,
        predicted_action: int,
        confidence: float,
        attention_weights: Optional[torch.Tensor] = None
    ) -> Explanation:
        """
        Generate complete explanation for a trading decision.

        Returns:
            Explanation object with all attribution methods
        """
        # N1: Feature importances
        feature_importances = self.shap_explainer.explain(state)

        # N2: Attention weights
        attention_dict = None
        if attention_weights is not None:
            attention_dict = self.attention_viz.extract_attention(attention_weights)

        # N3: Causal path
        top_features = list(feature_importances.keys())[:3]
        observed_features = {name: state[0, i] for i, name in enumerate(self.shap_explainer.feature_names)}
        causal_path = self.causal_explainer.explain_via_causality("trading_action", observed_features)

        # N4: Decision tree path
        rule_path = self.tree_distiller.explain_prediction(state)

        return Explanation(
            predicted_action=predicted_action,
            confidence=confidence,
            feature_importances=feature_importances,
            attention_weights=attention_dict,
            causal_path=causal_path,
            rule_path=rule_path
        )


__all__ = [
    'SHAPExplainer',
    'AttentionVisualizer',
    'CausalGraphExplainer',
    'DecisionTreeDistiller',
    'IntegratedExplainer',
    'Explanation',
    'ExplanationConfig'
]
