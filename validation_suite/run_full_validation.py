#!/usr/bin/env python3
"""
HIMARI Layer 2 V1 - Full Validation Suite

Runs all validation tests:
1. Walk-Forward Backtest
2. Monte Carlo Stress Test
3. Historical Crisis Replay

Usage:
    python run_full_validation.py
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add validation suite to path
BASE_DIR = Path(r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 2 V1")
sys.path.insert(0, str(BASE_DIR / "validation_suite"))

from walk_forward_backtest import run_walk_forward
from monte_carlo_stress_test import run_monte_carlo
from historical_crisis_replay import run_crisis_replay


def run_full_validation():
    """Run complete validation suite."""
    print("=" * 70)
    print("HIMARI Layer 2 V1 - FULL VALIDATION SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}
    total_start = time.time()

    # ==========================================================================
    # Test 1: Walk-Forward Backtest
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 1/3: WALK-FORWARD BACKTEST")
    print("=" * 70)

    try:
        wf_start = time.time()
        wf_results = run_walk_forward()
        wf_time = time.time() - wf_start
        all_results['walk_forward'] = {
            'status': 'PASS',
            'runtime': wf_time,
            'avg_test_sharpe': wf_results['avg_test_sharpe'],
            'is_oos_ratio': wf_results['is_oos_ratio'],
            'pct_profitable': wf_results['pct_profitable_folds'],
            'worst_dd': wf_results['worst_dd']
        }
        print(f"\n[Walk-Forward] COMPLETED in {wf_time:.1f}s")
    except Exception as e:
        print(f"\n[Walk-Forward] FAILED: {e}")
        all_results['walk_forward'] = {'status': 'FAIL', 'error': str(e)}

    # ==========================================================================
    # Test 2: Monte Carlo Stress Test
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 2/3: MONTE CARLO STRESS TEST")
    print("=" * 70)

    try:
        mc_start = time.time()
        mc_results = run_monte_carlo()
        mc_time = time.time() - mc_start
        all_results['monte_carlo'] = {
            'status': 'PASS',
            'runtime': mc_time,
            'prob_profit': mc_results['prob_profit'],
            'prob_positive_sharpe': mc_results['prob_positive_sharpe'],
            'sharpe_5th': mc_results['mc_sharpe_5th'],
            'sharpe_95th': mc_results['mc_sharpe_95th'],
            'dd_95th': mc_results['mc_dd_95th']
        }
        print(f"\n[Monte Carlo] COMPLETED in {mc_time:.1f}s")
    except Exception as e:
        print(f"\n[Monte Carlo] FAILED: {e}")
        all_results['monte_carlo'] = {'status': 'FAIL', 'error': str(e)}

    # ==========================================================================
    # Test 3: Historical Crisis Replay
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 3/3: HISTORICAL CRISIS REPLAY")
    print("=" * 70)

    try:
        cr_start = time.time()
        cr_results = run_crisis_replay()
        cr_time = time.time() - cr_start

        avg_return = sum(r.total_return for r in cr_results) / len(cr_results)
        avg_dd = sum(r.max_drawdown for r in cr_results) / len(cr_results)
        worst_dd = max(r.max_drawdown for r in cr_results)
        survived = all(r.max_drawdown < 50 for r in cr_results)

        all_results['crisis_replay'] = {
            'status': 'PASS',
            'runtime': cr_time,
            'n_scenarios': len(cr_results),
            'avg_return': avg_return,
            'avg_dd': avg_dd,
            'worst_dd': worst_dd,
            'survived_all': survived
        }
        print(f"\n[Crisis Replay] COMPLETED in {cr_time:.1f}s")
    except Exception as e:
        print(f"\n[Crisis Replay] FAILED: {e}")
        all_results['crisis_replay'] = {'status': 'FAIL', 'error': str(e)}

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("VALIDATION SUITE SUMMARY")
    print("=" * 70)

    print(f"\nTotal Runtime: {total_time:.1f}s")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n{:<25} {:>10} {:>15}".format("Test", "Status", "Key Metric"))
    print("-" * 50)

    # Walk-Forward
    wf = all_results.get('walk_forward', {})
    if wf.get('status') == 'PASS':
        print("{:<25} {:>10} {:>15}".format(
            "Walk-Forward",
            "PASS",
            f"IS/OOS: {wf.get('is_oos_ratio', 0):.3f}"
        ))
    else:
        print("{:<25} {:>10} {:>15}".format("Walk-Forward", "FAIL", "-"))

    # Monte Carlo
    mc = all_results.get('monte_carlo', {})
    if mc.get('status') == 'PASS':
        print("{:<25} {:>10} {:>15}".format(
            "Monte Carlo",
            "PASS",
            f"P(profit): {mc.get('prob_profit', 0):.1f}%"
        ))
    else:
        print("{:<25} {:>10} {:>15}".format("Monte Carlo", "FAIL", "-"))

    # Crisis Replay
    cr = all_results.get('crisis_replay', {})
    if cr.get('status') == 'PASS':
        print("{:<25} {:>10} {:>15}".format(
            "Crisis Replay",
            "PASS",
            f"Worst DD: {cr.get('worst_dd', 0):.1f}%"
        ))
    else:
        print("{:<25} {:>10} {:>15}".format("Crisis Replay", "FAIL", "-"))

    # Overall assessment
    print("\n" + "-" * 50)

    all_passed = all(
        all_results.get(t, {}).get('status') == 'PASS'
        for t in ['walk_forward', 'monte_carlo', 'crisis_replay']
    )

    if all_passed:
        print("\n[OVERALL] ALL TESTS PASSED")

        # Detailed pass/fail criteria
        criteria = []

        # Walk-forward criteria
        if wf.get('is_oos_ratio', 0) > 0.5:
            criteria.append(("IS/OOS Ratio > 0.5", "PASS"))
        else:
            criteria.append(("IS/OOS Ratio > 0.5", "FAIL"))

        if wf.get('worst_dd', 100) < 25:
            criteria.append(("WF Worst DD < 25%", "PASS"))
        else:
            criteria.append(("WF Worst DD < 25%", "FAIL"))

        # Monte Carlo criteria
        if mc.get('prob_profit', 0) > 50:
            criteria.append(("MC P(profit) > 50%", "PASS"))
        else:
            criteria.append(("MC P(profit) > 50%", "FAIL"))

        if mc.get('sharpe_5th', -10) > -1.0:
            criteria.append(("MC 5th Sharpe > -1.0", "PASS"))
        else:
            criteria.append(("MC 5th Sharpe > -1.0", "FAIL"))

        # Crisis criteria
        if cr.get('survived_all', False):
            criteria.append(("Crisis Survival", "PASS"))
        else:
            criteria.append(("Crisis Survival", "FAIL"))

        if cr.get('worst_dd', 100) < 50:
            criteria.append(("Crisis DD < 50%", "PASS"))
        else:
            criteria.append(("Crisis DD < 50%", "FAIL"))

        print("\n[DETAILED CRITERIA]")
        for name, result in criteria:
            print(f"  {name}: {result}")

        n_pass = sum(1 for _, r in criteria if r == "PASS")
        print(f"\n  Score: {n_pass}/{len(criteria)} criteria passed")

    else:
        print("\n[OVERALL] SOME TESTS FAILED")
        for test, result in all_results.items():
            if result.get('status') != 'PASS':
                print(f"  - {test}: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 70)

    # Save results
    results_path = BASE_DIR / "validation_results.json"
    with open(results_path, 'w') as f:
        # Convert non-serializable items
        save_results = {}
        for k, v in all_results.items():
            save_results[k] = {
                kk: vv for kk, vv in v.items()
                if not isinstance(vv, (list,)) or len(vv) < 100
            }
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    return all_results


if __name__ == "__main__":
    try:
        results = run_full_validation()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Validation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
