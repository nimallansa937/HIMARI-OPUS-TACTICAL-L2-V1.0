"""HIMARI Layer 2 - Part I: Simplex Safety Tests"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSimplexSafety(unittest.TestCase):
    """Tests for Part I: Simplex Safety."""
    
    def test_fallback_cascade(self):
        from simplex_safety.fallback_cascade import FallbackCascade
        fc = FallbackCascade()
        self.assertEqual(fc.get_action(), "NORMAL")
        fc.escalate()
        self.assertEqual(fc.get_action(), "REDUCE")
        
    def test_predictive_safety(self):
        from simplex_safety.predictive_safety import PredictiveSafety
        ps = PredictiveSafety()
        for i in range(10):
            ps.update(0.5 + i * 0.05)
        violation, _ = ps.predict_violation()
        self.assertTrue(violation)
        
    def test_reachability(self):
        from simplex_safety.reachability import ReachabilityAnalyzer
        ra = ReachabilityAnalyzer()
        transitions = {"FLAT": ["LONG"], "LONG": ["OVERLEVERAGED"]}
        self.assertTrue(ra.can_reach_unsafe("FLAT", transitions))
        
    def test_stop_loss(self):
        from simplex_safety.safety_invariants import StopLossEnforcer
        sle = StopLossEnforcer(stop_loss_pct=0.02)
        self.assertTrue(sle.should_stop(100, 97, is_long=True))
        self.assertFalse(sle.should_stop(100, 99, is_long=True))


if __name__ == '__main__':
    unittest.main()
