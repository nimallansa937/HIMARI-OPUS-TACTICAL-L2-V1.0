"""
HIMARI Layer 2 - Decision Engine Module (v5.0)
Subsystem D: Decision Engine Ensemble

Components (v5.0 - 10 methods architecture):
    - FLAG-TRADER (D1): 135M LLM as policy network with rsLoRA - NEW
    - Critic-Guided DT (D2): Advantage-weighted Decision Transformer - UPGRADED
    - CQL Agent (D3): Conservative Q-Learning for offline RL fallback - NEW
    - rsLoRA (D4): Rank-stabilized LoRA fine-tuning - NEW
    - PPO-LSTM (D5): 25M parameter online RL workhorse - KEPT
    - SAC Agent (D6): Entropy-regularized exploration agent - KEPT
    - Sharpe-Weighted Voting (D7): Ensemble aggregation - KEPT
    - Disagreement Scaling (D8): Confidence from variance - KEPT
    - Return Conditioning (D9): Target Sharpe input - KEPT
    - FinRL-DT Pipeline (D10): Training infrastructure - NEW

Legacy components (v4.0 - backward compatibility):
    - Decision Transformer: Basic offline RL via sequence modeling
"""

# v5.0 Main Orchestrator
from .decision_engine import (
    DecisionEngine,
    DecisionEngineConfig,
    DecisionOutput,
    TradeAction,
    create_decision_engine
)

# D7: Sharpe-Weighted Ensemble Voting
from .ensemble_voting import (
    SharpeWeightedEnsemble,
    EnsembleConfig,
    create_sharpe_ensemble
)

# D8: Disagreement Scaling
from .disagreement_scaling import (
    DisagreementScaler,
    DisagreementConfig,
    create_disagreement_scaler
)

# D4: rsLoRA
from .rslora import (
    apply_rslora,
    get_lora_params,
    count_trainable_params,
    merge_lora_weights
)

# D9: Return Conditioning
from .return_conditioning import (
    ReturnConditioner,
    ReturnConditionerConfig,
    create_return_conditioner
)

# D10: FinRL-DT Pipeline
from .finrl_dt_pipeline import (
    TrajectoryDataset,
    TrajectoryDatasetConfig,
    FinRLDTPipeline,
    FinRLDTPipelineConfig,
    Trajectory,
    create_dt_training_pipeline
)

# v5.0 Agent Components (wrapped in try/except for optional dependencies)
try:
    from .flag_trader import (
        FLAGTrader,
        FLAGTraderConfig,
        TradeAction as FLAGTradeAction,
        create_flag_trader
    )
except (ImportError, SyntaxError) as e:
    FLAGTrader = None
    FLAGTraderConfig = None
    FLAGTradeAction = None
    create_flag_trader = None

try:
    from .cgdt import (
        CriticGuidedDecisionTransformer,
        CGDTConfig,
        CGDTTrainer
    )
except (ImportError, SyntaxError):
    CriticGuidedDecisionTransformer = None
    CGDTConfig = None
    CGDTTrainer = None

try:
    from .cql_agent import (
        CQLAgent,
        CQLConfig,
        create_cql_agent
    )
except (ImportError, SyntaxError):
    CQLAgent = None
    CQLConfig = None
    create_cql_agent = None

# v4.0 Legacy (backward compatibility)
try:
    from .decision_transformer import (
        DecisionTransformer,
        DecisionTransformerConfig,
        DecisionTransformerTrainer
    )
except (ImportError, SyntaxError):
    DecisionTransformer = None
    DecisionTransformerConfig = None
    DecisionTransformerTrainer = None

try:
    from .ppo_lstm import (
        PPOLSTMAgent,
        create_ppo_lstm_agent
    )
except (ImportError, SyntaxError):
    PPOLSTMAgent = None
    create_ppo_lstm_agent = None

try:
    from .sac_agent import (
        SACAgent,
        create_sac_agent
    )
except (ImportError, SyntaxError):
    SACAgent = None
    create_sac_agent = None

try:
    from .ensemble import (
        DecisionEngineEnsemble,
        EnsembleVote,
        AgentPrediction,
        create_ensemble
    )
except (ImportError, SyntaxError):
    DecisionEngineEnsemble = None
    EnsembleVote = None
    AgentPrediction = None
    create_ensemble = None

__all__ = [
    # v5.0 Main Orchestrator
    'DecisionEngine',
    'DecisionEngineConfig',
    'DecisionOutput',
    'TradeAction',
    'DisagreementScaler',
    'SharpeWeightedEnsemble',
    'create_decision_engine',
    
    # D4: rsLoRA
    'apply_rslora',
    'get_lora_params',
    'count_trainable_params',
    'merge_lora_weights',
    
    # D9: Return Conditioning
    'ReturnConditioner',
    'ReturnConditionerConfig',
    'create_return_conditioner',
    
    # D10: FinRL-DT Pipeline
    'TrajectoryDataset',
    'TrajectoryDatasetConfig',
    'FinRLDTPipeline',
    'FinRLDTPipelineConfig',
    'Trajectory',
    'create_dt_training_pipeline',
    
    # v5.0 Agents
    'FLAGTrader',
    'FLAGTraderConfig',
    'FLAGTradeAction',
    'create_flag_trader',
    'CriticGuidedDecisionTransformer',
    'CGDTConfig',
    'CGDTTrainer',
    'CQLAgent',
    'CQLConfig',
    'create_cql_agent',
    
    # v4.0 Legacy
    'DecisionTransformer',
    'DecisionTransformerConfig',
    'DecisionTransformerTrainer',
    'PPOLSTMAgent',
    'create_ppo_lstm_agent',
    'SACAgent',
    'create_sac_agent',
    'DecisionEngineEnsemble',
    'EnsembleVote',
    'AgentPrediction',
    'create_ensemble'
]
