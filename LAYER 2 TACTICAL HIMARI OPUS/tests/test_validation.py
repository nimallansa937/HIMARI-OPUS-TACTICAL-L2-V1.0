"""HIMARI Layer 2 - Part L: Validation Tests"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestValidation(unittest.TestCase):
    """Tests for Part L: Validation Framework."""
    
    def test_walk_forward(self):
        from validation.walk_forward import WalkForwardValidator
        wfv = WalkForwardValidator()
        splits = wfv.generate_splits(500)
        self.assertGreater(len(splits), 0)
        
    def test_regime_aware_cv(self):
        from validation.regime_aware_cv import RegimeAwareCV
        cv = RegimeAwareCV()
        X = np.random.randn(100, 10)
        regimes = np.random.randint(0, 4, 100)
        splits = cv.split(X, regimes)
        self.assertEqual(len(splits), 5)
        
    def test_deflated_sharpe(self):
        from validation.deflated_sharpe import DeflatedSharpe
        ds = DeflatedSharpe()
        psr = ds.compute(sharpe=1.5, n_trials=100, n_obs=252)
        self.assertGreaterEqual(psr, 0)
        self.assertLessEqual(psr, 1)
        
    def test_sequential_testing(self):
        from validation.sequential_testing import SequentialTesting
        st = SequentialTesting()
        result = st.update(0.02)
        self.assertIn(result, ["ACCEPT_ALT", "ACCEPT_NULL", "CONTINUE"])


if __name__ == '__main__':
    unittest.main()
