"""
HIMARI Layer 2 - Training Module
Subsystem K: Training Infrastructure (8 Methods)
Subsystem L: Validation Framework (6 Methods)

Components (K):
    - K1: 3-Stage Curriculum Learning
    - K2: MAML Meta-Learning
    - K3: Causal Data Augmentation
    - K4: Multi-Task Learning
    - K5: Adversarial Training
    - K6: FGSM/PGD Robustness
    - K7: Reward Shaping
    - K8: Rare Event Synthesis

Components (L):
    - L1: Backtesting Engine
    - L2: Walk-Forward Analysis
    - L3: Statistical Tests
    - L4: Cross-Validation
    - L5: Performance Attribution
    - L6: Robustness Checks
"""

# K1: Curriculum Learning (Separate Module)
from .curriculum_learning import (
    CurriculumDataset,
    CurriculumTrainer,
    CurriculumConfig,
    DifficultyScorer,
    DifficultyScore
)

# K2: MAML Meta-Learning (Separate Module)
from .maml_meta_learning import (
    MAMLTrainer,
    MAMLConfig,
    MAMLTask,
    RegimeTaskGenerator,
    create_maml_trainer
)

# K3: Causal Data Augmentation (Separate Module)
from .causal_augmentation import (
    CausalDataAugmenter,
    CausalAugmentConfig,
    GaussianNoiseAugmenter,
    TimeWarpAugmenter,
    MagnitudeScaleAugmenter,
    create_causal_augmenter
)

# K4: Multi-Task Learning (Separate Module)
from .multi_task_learning import (
    MultiTaskModel,
    MultiTaskTrainer,
    MultiTaskConfig,
    SharedEncoder,
    TaskHead,
    create_multi_task_model
)

# K5-K6: Adversarial Training
from .adversarial import (
    AdversarialTrainer,
    AdversarialConfig
)
# Create aliases for K5/K6 methods
FGSM = lambda trainer, state, target: trainer.fgsm_attack(state, target)
PGD = lambda trainer, state, target: trainer.pgd_attack(state, target)
create_adversarial_trainer = lambda model, **kwargs: AdversarialTrainer(model, AdversarialConfig(**kwargs))

# K7: Reward Shaping
from .reward_shaping import (
    RewardShaper,
    RewardShapingConfig,
    CurriculumLearning
)
# Alias for consistency
RewardConfig = RewardShapingConfig
create_reward_shaper = lambda **kwargs: RewardShaper(RewardShapingConfig(**kwargs))

# K8: Rare Event Synthesis (Separate Module)
from .rare_event_synthesis import (
    RareEventSynthesizer,
    RareEventConfig,
    CrashScenarioGenerator,
    CascadeEventGenerator,
    create_rare_event_synthesizer
)

# Training Pipeline (utility classes)
from .training_pipeline import (
    TrainingPipeline,
    TrainingConfig,
    PrioritizedReplayBuffer,
    ReplayConfig,
    CurriculumScheduler,
    MultiAgentCoordinator,
    HyperparameterScheduler,
    DistributedTrainingManager,
    TrainingMetricsLogger
)

# Part L: Validation Framework
from .validation_pipeline import (
    ValidationPipeline,
    ValidationConfig,
    BacktestEngine,
    BacktestConfig,
    WalkForwardAnalyzer,
    StatisticalValidator,
    TimeSeriesCrossValidator,
    PerformanceAttributor,
    RobustnessChecker
)

# Sortino Reward (for Transformer-A2C)
from .sortino_reward import (
    SimpleSortinoReward,
    SortinoWithDrawdownPenalty,
    create_reward_function
)

# Transformer-A2C Trainer
from .transformer_a2c_trainer import (
    TransformerA2CTrainer,
    train_transformer_a2c
)

# Walk-Forward Optimization
from .walk_forward_optimizer import (
    WFOConfig,
    WFOWindow,
    WalkForwardOptimizer,
    run_wfo_training
)

# Synthetic Crash Generator
from .synthetic_crash_generator import (
    CrashScenarioConfig,
    SyntheticCrashGenerator
)

__all__ = [
    # K1: Curriculum
    'CurriculumDataset',
    'CurriculumTrainer',
    'CurriculumConfig',
    'DifficultyScorer',
    'DifficultyScore',
    # K2: MAML
    'MAMLTrainer',
    'MAMLConfig',
    'MAMLTask',
    'RegimeTaskGenerator',
    'create_maml_trainer',
    # K3: Augmentation
    'CausalDataAugmenter',
    'CausalAugmentConfig',
    'GaussianNoiseAugmenter',
    'TimeWarpAugmenter',
    'MagnitudeScaleAugmenter',
    'create_causal_augmenter',
    # K4: Multi-Task
    'MultiTaskModel',
    'MultiTaskTrainer',
    'MultiTaskConfig',
    'SharedEncoder',
    'TaskHead',
    'create_multi_task_model',
    # K5-K6: Adversarial
    'AdversarialTrainer',
    'AdversarialConfig',
    'FGSM',
    'PGD',
    'create_adversarial_trainer',
    # K7: Reward
    'RewardShaper',
    'RewardConfig',
    'create_reward_shaper',
    # K8: Rare Events
    'RareEventSynthesizer',
    'RareEventConfig',
    'CrashScenarioGenerator',
    'CascadeEventGenerator',
    'create_rare_event_synthesizer',
    # Training Pipeline
    'TrainingPipeline',
    'TrainingConfig',
    'PrioritizedReplayBuffer',
    'ReplayConfig',
    'CurriculumScheduler',
    'MultiAgentCoordinator',
    'HyperparameterScheduler',
    'DistributedTrainingManager',
    'TrainingMetricsLogger',
    # Part L: Validation
    'ValidationPipeline',
    'ValidationConfig',
    'BacktestEngine',
    'BacktestConfig',
    'WalkForwardAnalyzer',
    'StatisticalValidator',
    'TimeSeriesCrossValidator',
    'PerformanceAttributor',
    'RobustnessChecker',
    # Sortino Reward
    'SimpleSortinoReward',
    'SortinoWithDrawdownPenalty',
    'create_reward_function',
    # Transformer-A2C Trainer
    'TransformerA2CTrainer',
    'train_transformer_a2c',
    # Walk-Forward Optimization
    'WFOConfig',
    'WFOWindow',
    'WalkForwardOptimizer',
    'run_wfo_training',
    # Synthetic Crash Generator
    'CrashScenarioConfig',
    'SyntheticCrashGenerator',
]
