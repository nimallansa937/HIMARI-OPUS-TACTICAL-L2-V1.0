"""HIMARI Layer 2 - Part J: LLM Integration Components"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class MarketStateEncoder:
    """Encode market state for LLM context."""
    def __init__(self):
        self.feature_names = []
        
    def encode(self, features: np.ndarray, regime: int = 2) -> str:
        regime_names = {0: "Crisis", 1: "Bearish", 2: "Ranging", 3: "Bullish"}
        return f"Market Regime: {regime_names.get(regime, 'Unknown')}. Features: {len(features)} dimensions."


class ThoughtChainDecoder:
    """Decode LLM thought chain into structured actions."""
    def __init__(self):
        pass
        
    def decode(self, thought_chain: str) -> Dict:
        actions = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for action in actions:
            if action.lower() in thought_chain.lower():
                actions[action] += 1
        best = max(actions, key=actions.get)
        return {"action": best, "reasoning": thought_chain[:100]}


class ActionSpaceConverter:
    """Convert between LLM outputs and trading actions."""
    def __init__(self):
        self.mapping = {"buy": "BUY", "sell": "SELL", "hold": "HOLD", "long": "BUY", "short": "SELL"}
        
    def convert(self, llm_output: str) -> str:
        llm_lower = llm_output.lower()
        for key, action in self.mapping.items():
            if key in llm_lower:
                return action
        return "HOLD"
