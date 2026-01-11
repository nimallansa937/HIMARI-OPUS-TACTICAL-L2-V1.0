"""HIMARI Layer 2 - Part M: Adaptation Tests"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestAdaptation(unittest.TestCase):
    """Tests for Part M: Adaptation Framework."""
    
    def test_adaptive_memory(self):
        from adaptation.adaptive_memory import AdaptiveMemory
        mem = AdaptiveMemory()
        mem.add({"state": 1}, priority=1.0)
        mem.add({"state": 2}, priority=2.0)
        sample = mem.sample(1)
        self.assertEqual(len(sample), 1)
        
    def test_thompson_sampling(self):
        from adaptation.adaptive_memory import ThompsonSampling
        ts = ThompsonSampling()
        action = ts.select()
        self.assertIn(action, [0, 1, 2])
        ts.update(action, 1.0)
        
    def test_page_hinkley(self):
        from adaptation.adaptive_memory import PageHinkley
        ph = PageHinkley()
        for i in range(100):
            ph.update(0.01)
        result = ph.update(1.0)
        self.assertIsInstance(result, bool)
        
    def test_counterfactual_regret(self):
        from adaptation.adaptive_memory import CounterfactualRegret
        cfr = CounterfactualRegret()
        cfr.update_regret(0, np.array([0.5, 0.3, 0.2]))
        self.assertEqual(len(cfr.strategy), 3)


if __name__ == '__main__':
    unittest.main()
