"""
HIMARI Layer 2 - Part J: LLM Integration
LLM-based market analysis and decision support.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================================
# J1: Market Narrative Generator
# ============================================================================

@dataclass
class NarrativeConfig:
    max_length: int = 256
    temperature: float = 0.7
    use_cache: bool = True

class MarketNarrativeGenerator:
    """Generates market narrative from structured data."""
    
    def __init__(self, config=None):
        self.config = config or NarrativeConfig()
        self.cache = {}
        
    def generate(self, market_data: Dict) -> str:
        """Generate narrative from market conditions."""
        regime = market_data.get('regime', 'unknown')
        trend = market_data.get('trend_direction', 0)
        volatility = market_data.get('volatility', 'normal')
        confidence = market_data.get('confidence', 0.5)
        
        regime_desc = {
            0: "bullish trending",
            1: "bearish trending", 
            2: "ranging/consolidating",
            3: "high volatility",
            4: "crisis mode"
        }.get(regime, "uncertain")
        
        trend_desc = "upward" if trend > 0 else "downward" if trend < 0 else "sideways"
        
        narrative = f"Market is currently {regime_desc} with {trend_desc} bias. "
        narrative += f"Volatility is {volatility}. "
        narrative += f"Model confidence: {confidence:.1%}."
        
        return narrative


# ============================================================================
# J2: Signal Explainer
# ============================================================================

class SignalExplainer:
    """Explains trading signals in natural language."""
    
    def __init__(self):
        self.factor_descriptions = {
            'momentum': "Price momentum indicates {direction} pressure",
            'trend': "The primary trend is {direction}",
            'volatility': "Volatility is {level}, suggesting {implication}",
            'sentiment': "Market sentiment appears {sentiment}",
            'regime': "Current regime is {regime_name}"
        }
        
    def explain(self, signal: int, factors: Dict) -> str:
        """Generate explanation for a trading signal."""
        action = {1: "BUY", 0: "HOLD", -1: "SELL"}.get(signal, "UNKNOWN")
        
        explanations = []
        if 'momentum' in factors:
            direction = "bullish" if factors['momentum'] > 0 else "bearish"
            explanations.append(f"Momentum is {direction}")
            
        if 'trend_strength' in factors:
            strength = "strong" if factors['trend_strength'] > 0.6 else "weak"
            explanations.append(f"Trend strength is {strength}")
            
        if 'confidence' in factors:
            conf = factors['confidence']
            conf_level = "high" if conf > 0.7 else "moderate" if conf > 0.4 else "low"
            explanations.append(f"Signal confidence is {conf_level} ({conf:.1%})")
        
        explanation = f"Signal: {action}. " + ". ".join(explanations) + "."
        return explanation


# ============================================================================
# J3: Risk Narrator
# ============================================================================

class RiskNarrator:
    """Narrates risk conditions in natural language."""
    
    def narrate(self, risk_metrics: Dict) -> str:
        """Generate risk narrative."""
        parts = []
        
        if 'drawdown' in risk_metrics:
            dd = risk_metrics['drawdown']
            if dd > 0.1:
                parts.append(f"WARNING: Drawdown at {dd:.1%}")
            elif dd > 0.05:
                parts.append(f"Drawdown elevated at {dd:.1%}")
                
        if 'volatility' in risk_metrics:
            vol = risk_metrics['volatility']
            if vol > 0.3:
                parts.append("Volatility is very high")
            elif vol > 0.2:
                parts.append("Volatility is elevated")
                
        if 'position_size' in risk_metrics:
            pos = risk_metrics['position_size']
            parts.append(f"Position size: {pos:.1%}")
            
        return ". ".join(parts) if parts else "Risk metrics normal."


# ============================================================================
# J4: Embedding Generator
# ============================================================================

class TextEmbedder:
    """Generates embeddings for market text with transformer fallback."""
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self._model = None
        self._use_transformer = False
        self._init_model()
        
    def _init_model(self):
        """Initialize embedding model (transformer or fallback)."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            self._use_transformer = True
            self._transformer_dim = 384  # MiniLM output dim
            logger.info("TextEmbedder: Using SentenceTransformer")
        except ImportError:
            logger.warning("TextEmbedder: sentence-transformers not available, using TF-IDF fallback")
            self._use_transformer = False
        except Exception as e:
            logger.warning(f"TextEmbedder: Error loading transformer: {e}, using fallback")
            self._use_transformer = False
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self._use_transformer and self._model is not None:
            return self._transformer_embed(text)
        return self._tfidf_embed(text)
    
    def _transformer_embed(self, text: str) -> np.ndarray:
        """Generate embedding using transformer model."""
        full_emb = self._model.encode(text, convert_to_numpy=True)
        # Project to desired dimension if needed
        if len(full_emb) > self.dim:
            # Simple dimensionality reduction via slicing (could use PCA)
            return full_emb[:self.dim]
        elif len(full_emb) < self.dim:
            # Pad with zeros
            result = np.zeros(self.dim)
            result[:len(full_emb)] = full_emb
            return result
        return full_emb
    
    def _tfidf_embed(self, text: str) -> np.ndarray:
        """TF-IDF inspired fallback embedding with semantic hashing."""
        words = text.lower().split()
        embedding = np.zeros(self.dim)
        
        # Financial keyword weighting
        financial_keywords = {
            'bullish': 2.0, 'bearish': 2.0, 'buy': 1.5, 'sell': 1.5,
            'rally': 1.5, 'crash': 2.0, 'volatile': 1.5, 'stable': 1.2,
            'growth': 1.3, 'decline': 1.3, 'trend': 1.2, 'momentum': 1.2,
            'risk': 1.5, 'opportunity': 1.3, 'resistance': 1.2, 'support': 1.2
        }
        
        for word in words:
            word_hash = hash(word) % self.dim
            weight = financial_keywords.get(word, 1.0)
            embedding[word_hash] += weight
        
        # Log-normalize for TF-IDF-like behavior
        embedding = np.log1p(embedding)
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
            
        return embedding


# ============================================================================
# J5: Context Aggregator
# ============================================================================

class ContextAggregator:
    """Aggregates context from multiple sources."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = []
        
    def add(self, context_type: str, content: str, importance: float = 1.0):
        self.history.append({
            'type': context_type,
            'content': content,
            'importance': importance
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def get_context(self) -> str:
        """Get aggregated context string."""
        if not self.history:
            return ""
        
        # Weight by importance, most recent first
        sorted_ctx = sorted(self.history, key=lambda x: x['importance'], reverse=True)
        return " | ".join(c['content'] for c in sorted_ctx[:5])


# ============================================================================
# J6: LLM Decision Advisor
# ============================================================================

class LLMDecisionAdvisor:
    """LLM-based decision advisor (simplified without actual LLM)."""
    
    def __init__(self):
        self.narrative_gen = MarketNarrativeGenerator()
        self.signal_explainer = SignalExplainer()
        self.risk_narrator = RiskNarrator()
        self.embedder = TextEmbedder()
        self.context = ContextAggregator()
        
    def advise(self, market_data: Dict, signal: int, 
               confidence: float, risk_metrics: Dict) -> Dict:
        """Generate comprehensive trading advice."""
        
        # Generate narratives
        market_narrative = self.narrative_gen.generate(market_data)
        signal_explanation = self.signal_explainer.explain(signal, {
            'momentum': market_data.get('momentum', 0),
            'trend_strength': market_data.get('trend_strength', 0.5),
            'confidence': confidence
        })
        risk_narrative = self.risk_narrator.narrate(risk_metrics)
        
        # Aggregate context
        self.context.add('market', market_narrative)
        self.context.add('signal', signal_explanation)
        self.context.add('risk', risk_narrative)
        
        # Generate embedding for downstream use
        full_context = f"{market_narrative} {signal_explanation} {risk_narrative}"
        embedding = self.embedder.embed(full_context)
        
        return {
            'action': signal,
            'confidence': confidence,
            'market_narrative': market_narrative,
            'signal_explanation': signal_explanation,
            'risk_narrative': risk_narrative,
            'context_embedding': embedding,
            'recommendation': self._format_recommendation(signal, confidence, risk_metrics)
        }
    
    def _format_recommendation(self, signal: int, confidence: float, 
                              risk_metrics: Dict) -> str:
        action = {1: "BUY", 0: "HOLD", -1: "SELL"}.get(signal, "HOLD")
        dd = risk_metrics.get('drawdown', 0)
        
        if dd > 0.1:
            return f"CAUTION: {action} with reduced size due to drawdown"
        elif confidence > 0.7:
            return f"STRONG {action} signal"
        elif confidence > 0.5:
            return f"MODERATE {action} signal"
        else:
            return f"WEAK {action} signal - consider waiting"


# ============================================================================
# Complete LLM Integration Pipeline
# ============================================================================

@dataclass
class LLMConfig:
    embedding_dim: int = 64
    max_context_length: int = 512

class LLMIntegrationPipeline:
    """Complete LLM integration for trading decisions."""
    
    def __init__(self, config=None):
        self.config = config or LLMConfig()
        self.advisor = LLMDecisionAdvisor()
        
    def process(self, market_data: Dict, signal: int,
               confidence: float, risk_metrics: Dict) -> Dict:
        """Full LLM processing pipeline."""
        return self.advisor.advise(market_data, signal, confidence, risk_metrics)
