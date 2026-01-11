"""HIMARI Layer 2 - Part K: Training Tests"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestTraining(unittest.TestCase):
    """Tests for Part K: Training Infrastructure."""
    
    def test_training_pipeline_import(self):
        from training import CurriculumTrainer, MAMLTrainer
        self.assertIsNotNone(CurriculumTrainer)
        
    def test_curriculum_learning(self):
        from training.curriculum_learning import CurriculumLearning
        cl = CurriculumLearning()
        difficulty = cl.get_current_difficulty()
        self.assertGreaterEqual(difficulty, 0)
        
    def test_replay_buffer(self):
        from training.training_pipeline import ReplayBuffer
        buffer = ReplayBuffer(capacity=100)
        buffer.add({"state": np.random.randn(10), "action": 1, "reward": 0.5})
        batch = buffer.sample(1)
        self.assertEqual(len(batch), 1)
        
    def test_reward_shaping(self):
        from training.reward_shaping import RewardShaper
        shaper = RewardShaper()
        reward = shaper.shape(base_reward=1.0, action="BUY", regime=3)
        self.assertIsInstance(reward, float)


if __name__ == '__main__':
    unittest.main()
