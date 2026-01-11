"""
HIMARI Layer 2 - Part H: Risk Management Tests
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestRiskManagement(unittest.TestCase):
    """Tests for Part H: Risk Management."""
    
    def test_evt_tail_risk(self):
        from risk_management.evt_tail_risk import EVTTailRisk
        evt = EVTTailRisk()
        returns = np.random.randn(100) * 0.01
        evt.fit(returns)
        var = evt.var(0.99)
        self.assertGreater(var, 0)
        
    def test_ddpg_kelly(self):
        from risk_management.ddpg_kelly import DDPGKelly
        kelly = DDPGKelly()
        for i in range(50):
            kelly.update(i % 2 == 0, 0.01 if i % 2 == 0 else -0.01)
        frac = kelly.kelly_fraction()
        self.assertGreaterEqual(frac, 0)
        
    def test_dcc_garch(self):
        from risk_management.dcc_garch import DCCGARCH
        garch = DCCGARCH()
        for i in range(50):
            garch.update(np.random.randn() * 0.01)
        vol = garch.get_volatility()
        self.assertGreater(vol, 0)
        
    def test_drawdown_brake(self):
        from risk_management.drawdown_brake import DrawdownBrake
        brake = DrawdownBrake()
        brake.update(100)
        brake.update(90)  # 10% DD
        scale = brake.get_scale()
        self.assertLess(scale, 1.0)
        
    def test_portfolio_var(self):
        from risk_management.portfolio_var import PortfolioVaR
        pvar = PortfolioVaR()
        for r in np.random.randn(100) * 0.01:
            pvar.update(r)
        var = pvar.calculate_var()
        self.assertGreater(var, 0)


if __name__ == '__main__':
    unittest.main()
