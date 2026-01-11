"""HIMARI Layer 2 - Part J: LLM Integration Tests"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestLLMIntegration(unittest.TestCase):
    """Tests for Part J: LLM Integration."""
    
    def test_market_state_encoder(self):
        from llm_integration.market_state_encoder import MarketStateEncoder
        encoder = MarketStateEncoder()
        encoded = encoder.encode(np.random.randn(60), regime=3)
        self.assertIn("Bullish", encoded)
        
    def test_action_space_converter(self):
        from llm_integration.market_state_encoder import ActionSpaceConverter
        converter = ActionSpaceConverter()
        action = converter.convert("I think we should buy")
        self.assertEqual(action, "BUY")
        
    def test_contextual_bandit(self):
        from llm_integration.alignment_loss import ContextualBandit
        bandit = ContextualBandit()
        action = bandit.select_action(np.random.randn(10))
        self.assertIn(action, [0, 1, 2])
        bandit.update(action, 1.0)
        
    def test_fewshot_adaptation(self):
        from llm_integration.fewshot_adaptation import FewShotAdaptation
        fsa = FewShotAdaptation()
        fsa.add_example("price up", "BUY")
        prompt = fsa.format_prompt("price down")
        self.assertIn("Examples:", prompt)


if __name__ == '__main__':
    unittest.main()
