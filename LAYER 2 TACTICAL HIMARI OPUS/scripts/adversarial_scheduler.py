#!/usr/bin/env python3
"""
L2-5 Adversarial Test Scheduler

Cron-like scheduler for automated adversarial testing.
Supports monthly full runs, weekly quick runs, and on-demand triggers.

Can be run as:
1. Standalone script with schedule
2. Called from CI/CD (GitHub Actions, Jenkins)
3. Cloud functions (AWS Lambda, GCP Cloud Functions)
"""

import argparse
import logging
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdversarialScheduler:
    """
    Scheduler for automated adversarial tests.
    
    Schedule (per guide):
    - Monthly: Full 10,000 scenario suite (1st of month)
    - Weekly: Quick 1,000 scenario suite (Sunday)
    - On-demand: Before parameter changes, after incidents
    """
    
    # State file tracks last runs
    STATE_FILE = 'adversarial_schedule_state.json'
    
    def __init__(self, 
                 base_dir: str = '.',
                 output_dir: str = 'reports/adversarial'):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.state_file = self.base_dir / self.STATE_FILE
        self.state = self._load_state()
        
    def _load_state(self) -> Dict[str, Any]:
        """Load scheduler state."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            'last_full_run': None,
            'last_quick_run': None,
            'run_history': []
        }
    
    def _save_state(self) -> None:
        """Save scheduler state."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def should_run_full(self) -> bool:
        """Check if full monthly run is due."""
        last = self.state.get('last_full_run')
        if not last:
            return True
        
        last_date = datetime.fromisoformat(last)
        # Run on 1st of each month
        now = datetime.now()
        if now.day == 1 and (now - last_date).days >= 28:
            return True
        # Or if more than 35 days since last run
        if (now - last_date).days >= 35:
            return True
        return False
    
    def should_run_quick(self) -> bool:
        """Check if weekly quick run is due."""
        last = self.state.get('last_quick_run')
        if not last:
            return True
        
        last_date = datetime.fromisoformat(last)
        now = datetime.now()
        # Run on Sunday (weekday 6) or if > 7 days since last
        if now.weekday() == 6 and (now - last_date).days >= 5:
            return True
        if (now - last_date).days >= 8:
            return True
        return False
    
    def run_tests(self, mode: str) -> Dict[str, Any]:
        """Execute adversarial tests in specified mode."""
        logger.info(f"Starting {mode} adversarial test run...")
        
        # Build command
        script_path = self.base_dir / 'scripts' / 'run_adversarial_tests.py'
        cmd = [
            sys.executable,
            str(script_path),
            '--mode', mode,
            '--output-dir', str(self.output_dir)
        ]
        
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.base_dir)
            )
            
            success = result.returncode == 0
            
            # Parse output for summary
            summary = {
                'mode': mode,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                'success': success,
                'return_code': result.returncode,
                'stdout': result.stdout[-5000:] if result.stdout else '',  # Last 5k chars
                'stderr': result.stderr[-2000:] if result.stderr else ''
            }
            
            # Update state
            if mode == 'full':
                self.state['last_full_run'] = start_time.isoformat()
            elif mode == 'quick':
                self.state['last_quick_run'] = start_time.isoformat()
            
            self.state['run_history'].append({
                'mode': mode,
                'timestamp': start_time.isoformat(),
                'success': success
            })
            # Keep last 100 runs
            self.state['run_history'] = self.state['run_history'][-100:]
            
            self._save_state()
            
            return summary
            
        except Exception as e:
            logger.error(f"Test run failed: {e}")
            return {
                'mode': mode,
                'start_time': start_time.isoformat(),
                'success': False,
                'error': str(e)
            }
    
    def run_scheduled(self) -> Optional[Dict[str, Any]]:
        """Run scheduled tests based on current date."""
        # Check full (monthly) first
        if self.should_run_full():
            logger.info("Monthly full run is due")
            return self.run_tests('full')
        
        # Check quick (weekly)
        if self.should_run_quick():
            logger.info("Weekly quick run is due")
            return self.run_tests('quick')
        
        logger.info("No scheduled runs due")
        return None
    
    def run_on_demand(self, 
                      mode: str = 'quick',
                      reason: str = 'manual') -> Dict[str, Any]:
        """Run on-demand test with reason tracking."""
        logger.info(f"On-demand {mode} run triggered: {reason}")
        result = self.run_tests(mode)
        result['trigger_reason'] = reason
        return result
    
    def get_next_scheduled(self) -> Dict[str, Any]:
        """Get info about next scheduled runs."""
        now = datetime.now()
        
        # Next full run (1st of next month)
        if now.day == 1:
            next_full = now
        else:
            if now.month == 12:
                next_full = now.replace(year=now.year+1, month=1, day=1)
            else:
                next_full = now.replace(month=now.month+1, day=1)
        
        # Next quick run (next Sunday)
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0:
            next_quick = now
        else:
            next_quick = now + timedelta(days=days_until_sunday)
        
        return {
            'next_full': next_full.isoformat(),
            'next_quick': next_quick.isoformat(),
            'days_until_full': (next_full - now).days,
            'days_until_quick': (next_quick.date() - now.date()).days,
            'last_full': self.state.get('last_full_run'),
            'last_quick': self.state.get('last_quick_run')
        }


def notify_on_failure(summary: Dict[str, Any]) -> None:
    """
    Send notifications on test failure.
    Integrate with your preferred notification system.
    """
    if summary.get('success', True):
        return
    
    logger.warning("=" * 50)
    logger.warning("ADVERSARIAL TEST FAILURE ALERT")
    logger.warning(f"Mode: {summary.get('mode')}")
    logger.warning(f"Time: {summary.get('start_time')}")
    logger.warning("=" * 50)
    
    # TODO: Integrate with:
    # - Slack webhook
    # - PagerDuty
    # - Email
    # - Discord
    
    # Example Slack webhook (uncomment to use):
    # import requests
    # webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    # if webhook_url:
    #     requests.post(webhook_url, json={
    #         'text': f"⚠️ L2-5 Adversarial Test Failed\nMode: {summary['mode']}\nSee logs for details"
    #     })


def main():
    parser = argparse.ArgumentParser(
        description='L2-5 Adversarial Test Scheduler'
    )
    parser.add_argument(
        'action',
        choices=['schedule', 'full', 'quick', 'status', 'on-demand'],
        help='Action to perform'
    )
    parser.add_argument(
        '--reason',
        type=str,
        default='manual',
        help='Reason for on-demand run'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='.',
        help='Base directory'
    )
    
    args = parser.parse_args()
    
    scheduler = AdversarialScheduler(base_dir=args.base_dir)
    
    if args.action == 'schedule':
        # Check and run if due
        result = scheduler.run_scheduled()
        if result:
            notify_on_failure(result)
            sys.exit(0 if result.get('success') else 1)
        else:
            sys.exit(0)
    
    elif args.action == 'full':
        result = scheduler.run_tests('full')
        notify_on_failure(result)
        sys.exit(0 if result.get('success') else 1)
    
    elif args.action == 'quick':
        result = scheduler.run_tests('quick')
        notify_on_failure(result)
        sys.exit(0 if result.get('success') else 1)
    
    elif args.action == 'on-demand':
        result = scheduler.run_on_demand(reason=args.reason)
        notify_on_failure(result)
        sys.exit(0 if result.get('success') else 1)
    
    elif args.action == 'status':
        schedule = scheduler.get_next_scheduled()
        print(json.dumps(schedule, indent=2))


if __name__ == '__main__':
    main()
