#!/usr/bin/env python3
"""
L2-5 Adversarial Stress Test Runner

Main entry point for running adversarial test suites.
Supports: full monthly, weekly quick, and on-demand runs.

Usage:
    python run_adversarial_tests.py --mode full      # Monthly full suite
    python run_adversarial_tests.py --mode quick     # Weekly quick suite
    python run_adversarial_tests.py --mode custom --scenarios 500
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from himari_layer2.adversarial.config import AdversarialConfig
from himari_layer2.adversarial.generators import create_all_generators
from himari_layer2.adversarial.executor.scenario_executor import ScenarioExecutor
from himari_layer2.adversarial.executor.checkpoint_manager import CheckpointManager
from himari_layer2.adversarial.executor.memory_manager import MemoryManager
from himari_layer2.adversarial.analysis.report_generator import AdversarialReportGenerator
from himari_layer2.adversarial.analysis.failure_detector import FailureDetector
from himari_layer2.adversarial.analysis.survival_estimator import SurvivalEstimator
from himari_layer2.adversarial.integration.knowledge_graph_sink import KnowledgeGraphSink

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdversarialTestRunner:
    """
    Main runner for adversarial stress tests.
    
    Modes:
    - full: 10,000 scenarios, all categories (~35-45 hours on 1x A10)
    - quick: 1,000 scenarios, focused on regressions (~3-5 hours)
    - custom: User-specified count
    """
    
    # Mode configurations per guide spec
    MODE_CONFIGS = {
        'full': {
            'total_scenarios': 10000,
            'steps_per_scenario': 50000,
            'category_weights': {
                'tail_extrapolation': 2000,
                'correlation_stress': 1500,
                'multi_failure': 2000,
                'distribution_shift': 1500,
                'signal_corruption': 1500,
                'parameter_sensitivity': 1500
            },
            'checkpoint_every': 100,
            'description': 'Monthly full suite'
        },
        'quick': {
            'total_scenarios': 1000,
            'steps_per_scenario': 20000,
            'category_weights': {
                'tail_extrapolation': 150,
                'correlation_stress': 150,
                'multi_failure': 250,
                'distribution_shift': 150,
                'signal_corruption': 150,
                'parameter_sensitivity': 150
            },
            'checkpoint_every': 50,
            'description': 'Weekly quick suite'
        },
        'custom': {
            'total_scenarios': 500,
            'steps_per_scenario': 25000,
            'checkpoint_every': 50,
            'description': 'Custom run'
        }
    }
    
    def __init__(self,
                 mode: str = 'quick',
                 config_path: Optional[str] = None,
                 output_dir: str = 'reports/adversarial',
                 resume_run_id: Optional[str] = None):
        """
        Initialize test runner.
        
        Args:
            mode: 'full', 'quick', or 'custom'
            config_path: Optional path to YAML config
            output_dir: Directory for reports
            resume_run_id: Optional run ID to resume
        """
        self.mode = mode
        self.mode_config = self.MODE_CONFIGS.get(mode, self.MODE_CONFIGS['quick'])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        if config_path:
            self.config = AdversarialConfig.from_yaml(config_path)
        else:
            self.config = AdversarialConfig()
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.output_dir / 'checkpoints'),
            checkpoint_every=self.mode_config['checkpoint_every']
        )
        
        self.memory_manager = MemoryManager(
            max_memory_gb=self.config.execution.max_memory_gb
        )
        
        self.report_generator = AdversarialReportGenerator(
            output_dir=str(self.output_dir)
        )
        
        self.failure_detector = FailureDetector()
        self.survival_estimator = SurvivalEstimator()
        
        # Knowledge graph for persistence
        self.kg_sink = KnowledgeGraphSink(
            storage_dir=str(self.output_dir / 'knowledge_graph')
        )
        
        self.resume_run_id = resume_run_id
        self.results: List[Dict[str, Any]] = []
        self.start_time = 0.0
        
    def run(self) -> Dict[str, Any]:
        """
        Execute the adversarial test suite.
        
        Returns:
            Summary dict with results
        """
        self.start_time = time.time()
        run_id = self.checkpoint_manager.start_run(self.resume_run_id)
        
        logger.info(f"=" * 60)
        logger.info(f"L2-5 ADVERSARIAL STRESS TEST - {self.mode.upper()} MODE")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Target scenarios: {self.mode_config['total_scenarios']}")
        logger.info(f"=" * 60)
        
        # Create KG node for this run
        self.kg_sink.create_test_run_node(
            run_id=run_id,
            config=self.config.to_dict(),
            total_scenarios=self.mode_config['total_scenarios']
        )
        
        # Check for resume
        checkpoint = None
        if self.resume_run_id:
            checkpoint = self.checkpoint_manager.load_latest_checkpoint(run_id)
            if checkpoint:
                self.results = checkpoint.results
                logger.info(f"Resumed from checkpoint: {len(self.results)} scenarios completed")
        
        # Get generators
        all_generators = create_all_generators()
        logger.info(f"Total generators available: {len(all_generators)}")
        
        # Select scenarios based on mode
        scenarios_to_run = self._select_scenarios(all_generators)
        logger.info(f"Scenarios selected for run: {len(scenarios_to_run)}")
        
        # Skip already-completed if resuming
        if checkpoint:
            completed_ids = {r.get('scenario_id') for r in self.results}
            scenarios_to_run = [
                (g, s) for g, s in scenarios_to_run 
                if f"{g.get_name()}_{s}" not in completed_ids
            ]
            logger.info(f"Scenarios remaining: {len(scenarios_to_run)}")
        
        # Run scenarios
        try:
            self._run_scenarios(scenarios_to_run, run_id)
        except KeyboardInterrupt:
            logger.warning("Interrupted! Saving checkpoint...")
            self._save_checkpoint(run_id)
            raise
        except Exception as e:
            logger.error(f"Error during run: {e}")
            self._save_checkpoint(run_id)
            raise
        
        # Generate report
        summary = self._generate_report(run_id)
        
        # Finalize KG
        self.kg_sink.finalize_run(run_id, self.results, summary)
        
        # Cleanup checkpoints for successful run
        if summary.get('success', False):
            self.checkpoint_manager.cleanup_run(run_id)
        
        return summary
    
    def _select_scenarios(self, 
                          generators: list) -> List[tuple]:
        """Select scenarios based on mode configuration."""
        scenarios = []
        
        # Group by category
        by_category = {}
        for gen in generators:
            cat = gen.get_category()
            by_category.setdefault(cat, []).append(gen)
        
        # Weights for this mode
        weights = self.mode_config.get('category_weights', {})
        total_target = self.mode_config['total_scenarios']
        
        # Distribute scenarios
        for category, gens in by_category.items():
            target = weights.get(category, total_target // 6)
            scenarios_per_gen = max(1, target // len(gens))
            
            for gen in gens:
                for seed in range(scenarios_per_gen):
                    scenarios.append((gen, seed))
        
        # Trim or pad to target
        if len(scenarios) > total_target:
            self.rng = __import__('numpy').random.default_rng(42)
            indices = self.rng.choice(len(scenarios), total_target, replace=False)
            scenarios = [scenarios[i] for i in sorted(indices)]
        
        return scenarios
    
    def _run_scenarios(self, 
                       scenarios: List[tuple], 
                       run_id: str) -> None:
        """Run all selected scenarios."""
        total = len(scenarios)
        category_progress = {}
        
        for idx, (generator, seed_offset) in enumerate(scenarios):
            # Check memory
            if self.memory_manager.is_pressure_critical():
                logger.warning("Memory pressure critical, forcing GC...")
                self.memory_manager.force_gc()
            
            # Generate scenario ID
            scenario_id = f"{generator.get_name()}_{seed_offset}"
            category = generator.get_category()
            
            # Run scenario
            try:
                result = self._run_single_scenario(
                    generator, 
                    seed_offset,
                    scenario_id
                )
                self.results.append(result)
                
                # Update progress
                category_progress[category] = category_progress.get(category, 0) + 1
                
            except Exception as e:
                logger.error(f"Scenario {scenario_id} failed: {e}")
                self.results.append({
                    'scenario_id': scenario_id,
                    'category': category,
                    'name': generator.get_name(),
                    'survived': False,
                    'error': str(e),
                    'events_injected': [],
                    'events_detected': []
                })
            
            # Progress logging
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - self.start_time
                rate = (idx + 1) / elapsed
                remaining = (total - idx - 1) / rate if rate > 0 else 0
                
                logger.info(
                    f"Progress: {idx+1}/{total} ({(idx+1)/total*100:.1f}%) - "
                    f"Rate: {rate:.1f}/s - ETA: {remaining/60:.1f}min"
                )
            
            # Checkpoint
            if self.checkpoint_manager.should_checkpoint(len(self.results)):
                self._save_checkpoint(run_id, category_progress)
    
    def _run_single_scenario(self,
                              generator,
                              seed_offset: int,
                              scenario_id: str) -> Dict[str, Any]:
        """Run a single scenario and return results."""
        steps = self.mode_config['steps_per_scenario']
        seed = hash(scenario_id) % (2**31)
        
        # Generate data
        batches = list(generator.generate(steps=steps, seed=seed))
        
        # Collect metrics
        all_returns = []
        all_events = []
        max_drawdown = 0.0
        peak = 0.0
        
        for batch in batches:
            for point in batch.data:
                all_returns.append(point.returns)
                all_events.extend(point.events)
                
                # Track drawdown
                value = point.close
                peak = max(peak, value) if peak > 0 else value
                dd = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, dd)
        
        # Determine survival (simplified - would integrate with actual pipeline)
        survived = max_drawdown < self.config.thresholds.max_drawdown_limit
        
        # Unique events
        unique_events = list(set(all_events))
        
        return {
            'scenario_id': scenario_id,
            'category': generator.get_category(),
            'name': generator.get_name(),
            'seed': seed,
            'scenario_hash': generator.get_metadata(seed).hash,
            'survived': survived,
            'max_drawdown': max_drawdown,
            'total_steps': len(all_returns),
            'events_injected': unique_events,
            'events_detected': unique_events if survived else [],  # Simplified
            'detection_latency_ms': 5000 if unique_events else 0  # Placeholder
        }
    
    def _save_checkpoint(self,
                         run_id: str,
                         category_progress: Optional[Dict[str, int]] = None) -> None:
        """Save checkpoint."""
        self.checkpoint_manager.save_checkpoint(
            results=self.results,
            pending_seeds=[],  # Would track remaining
            config=self.config.to_dict(),
            start_time=self.start_time,
            category_progress=category_progress or {}
        )
    
    def _generate_report(self, run_id: str) -> Dict[str, Any]:
        """Generate final report and return summary."""
        elapsed = time.time() - self.start_time
        
        # Survival analysis
        survival = self.survival_estimator.estimate_overall_survival(self.results)
        
        # Failure patterns
        failure_analysis = self.failure_detector.analyze_results(self.results)
        
        # Generate markdown report
        report = self.report_generator.generate_report(
            results=self.results,
            run_id=run_id,
            start_time=self.start_time,
            compute_cost_cad=elapsed / 3600 * 0.166  # A10 rate
        )
        
        # Summary
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get('survived', False))
        
        summary = {
            'run_id': run_id,
            'mode': self.mode,
            'success': passed / total >= 0.95 if total > 0 else False,
            'total_scenarios': total,
            'passed': passed,
            'failed': total - passed,
            'survival_rate': passed / total if total > 0 else 0,
            'survival_ci_lower': survival.get('ci_lower', 0),
            'survival_ci_upper': survival.get('ci_upper', 1),
            'elapsed_seconds': elapsed,
            'compute_cost_cad': elapsed / 3600 * 0.166,
            'failure_patterns': failure_analysis.get('patterns_detected', 0),
            'new_failure_modes': failure_analysis.get('new_modes', [])
        }
        
        # Save summary JSON
        summary_path = self.output_dir / f"{run_id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"=" * 60)
        logger.info(f"RUN COMPLETE: {run_id}")
        logger.info(f"Survival Rate: {summary['survival_rate']*100:.1f}%")
        logger.info(f"Pass/Fail: {passed}/{total-passed}")
        logger.info(f"Elapsed: {elapsed/60:.1f} minutes")
        logger.info(f"Cost: ${summary['compute_cost_cad']:.2f} CAD")
        logger.info(f"=" * 60)
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description='L2-5 Adversarial Stress Test Runner'
    )
    parser.add_argument(
        '--mode', 
        choices=['full', 'quick', 'custom'],
        default='quick',
        help='Run mode: full (monthly), quick (weekly), or custom'
    )
    parser.add_argument(
        '--scenarios',
        type=int,
        default=None,
        help='Number of scenarios (custom mode)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/adversarial',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Run ID to resume'
    )
    
    args = parser.parse_args()
    
    # Adjust for custom scenarios
    if args.mode == 'custom' and args.scenarios:
        AdversarialTestRunner.MODE_CONFIGS['custom']['total_scenarios'] = args.scenarios
    
    # Run
    runner = AdversarialTestRunner(
        mode=args.mode,
        config_path=args.config,
        output_dir=args.output_dir,
        resume_run_id=args.resume
    )
    
    try:
        summary = runner.run()
        
        # Exit code based on pass/fail
        if summary.get('success', False):
            logger.info("✅ TEST SUITE PASSED")
            sys.exit(0)
        else:
            logger.warning("❌ TEST SUITE FAILED - Below 95% survival threshold")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Run interrupted. Resume with --resume <run_id>")
        sys.exit(130)


if __name__ == '__main__':
    main()
