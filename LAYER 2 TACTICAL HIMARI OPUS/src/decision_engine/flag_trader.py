"""
HIMARI Layer 2 - FLAG-TRADER: Fusion LLM-Agent with Gradient-based RL
Subsystem D: Decision Engine (Method D1)

Purpose:
    135M parameter LLM as policy network using rsLoRA for parameter-efficient
    fine-tuning with PPO-based gradient reinforcement learning.

Why FLAG-TRADER?
    - LLM as policy network leverages pre-trained "world knowledge"
    - 135M params beats GPT-4 (zero-shot) on trading tasks (ACL 2025)
    - Sharpe 3.344 on JNJ, 1.373 on MSFT vs 1.039 Buy&Hold
    - 10x cheaper inference than 70B models

Architecture:
    - Frozen LLM backbone (SmolLM2-135M)
    - LoRA adapters in attention layers (rank=16, rsLoRA scaling)
    - Policy head: LLM hidden → action logits
    - Value head: LLM hidden → state value

Training: PPO with textual state representation

Reference:
    - FLAG-TRADER (ACL 2025): Fusion LLM-Agent with Gradient-based RL
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
from enum import Enum
from loguru import logger

# Optional imports - gracefully handle if not installed
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not installed - FLAG-TRADER will use fallback mode")

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    logger.warning("peft not installed - LoRA fine-tuning unavailable")


class TradeAction(Enum):
    """Trading action enumeration"""
    STRONG_SELL = -2
    SELL = -1
    HOLD = 0
    BUY = 1
    STRONG_BUY = 2


@dataclass
class FLAGTraderConfig:
    """FLAG-TRADER configuration"""
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"  # 135M params
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    use_rslora: bool = True      # Rank-Stabilized LoRA (+2-5% Sharpe)
    learning_rate: float = 1e-4
    gamma: float = 0.99          # Discount factor
    gae_lambda: float = 0.95     # GAE lambda
    clip_epsilon: float = 0.2    # PPO clip
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 576        # SmolLM2-135M hidden size
    num_actions: int = 3         # BUY, HOLD, SELL (or 5 for STRONG variants)
    use_fallback: bool = False   # Use simple MLP if LLM unavailable


class FallbackPolicyNetwork(nn.Module):
    """Fallback MLP policy when LLM not available"""
    
    def __init__(self, state_dim: int = 60, hidden_dim: int = 256, num_actions: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        h = self.encoder(state)
        return self.policy_head(h), self.value_head(h)


class FLAGTrader(nn.Module):
    """
    FLAG-TRADER: Fusion LLM-Agent with Gradient-based Reinforcement Learning.
    
    Converts market state to text prompt, processes through LLM backbone,
    and outputs action probabilities via learned policy/value heads.
    
    Example:
        >>> config = FLAGTraderConfig(use_fallback=True)  # Use MLP fallback
        >>> trader = FLAGTrader(config)
        >>> action, confidence, value = trader.get_action(market_state)
    """
    
    def __init__(self, config: FLAGTraderConfig, device: str = 'cuda'):
        super().__init__()
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Determine if we can use full LLM or need fallback
        self.use_llm = HAS_TRANSFORMERS and HAS_PEFT and not config.use_fallback
        
        if self.use_llm:
            self._init_llm_backbone()
        else:
            self._init_fallback()
        
        logger.info(
            f"FLAGTrader initialized: mode={'LLM' if self.use_llm else 'Fallback'}, "
            f"device={self.device}"
        )
    
    def _init_fallback(self):
        """Initialize fallback MLP policy"""
        self.fallback = FallbackPolicyNetwork(
            state_dim=60,
            hidden_dim=self.config.hidden_dim,
            num_actions=self.config.num_actions
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.fallback.parameters(), 
            lr=self.config.learning_rate
        )
    
    def _init_llm_backbone(self):
        """Initialize LLM with LoRA adapters"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base LLM
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        
        # Apply LoRA with rsLoRA (Rank-Stabilized LoRA)
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=self.config.use_rslora,  # lora_alpha/sqrt(r) scaling
        )
        
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        
        # Get hidden dimension from model
        hidden_dim = self.model.config.hidden_size
        
        # Policy and Value heads
        self.policy_head = nn.Linear(hidden_dim, self.config.num_actions).to(self.device)
        self.value_head = nn.Linear(hidden_dim, 1).to(self.device)
        
        # Optimizer (only train LoRA params + heads)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_params += list(self.policy_head.parameters())
        trainable_params += list(self.value_head.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.config.learning_rate)
    
    def _create_prompt(self, market_state: Dict) -> str:
        """
        Serialize market state to text prompt.
        
        This is the key innovation: multimodal data → text → LLM reasoning.
        """
        prompt = f"""You are an expert trading agent. Analyze the current market state and choose an action.

MARKET STATE:
- Price Change (1h): {market_state.get('price_change_1h', 0):.2%}
- Price Change (4h): {market_state.get('price_change_4h', 0):.2%}
- Volume (vs avg): {market_state.get('volume_ratio', 1.0):.1f}x
- RSI (14): {market_state.get('rsi', 50):.1f}
- MACD Signal: {market_state.get('macd_signal', 'neutral')}
- Volatility: {market_state.get('volatility', 0.02):.2%}
- Regime: {market_state.get('regime', 'unknown')}
- Regime Confidence: {market_state.get('regime_confidence', 0.5):.1%}

SENTIMENT:
- News Sentiment: {market_state.get('news_sentiment', 'neutral')}
- Social Sentiment: {market_state.get('social_sentiment', 'neutral')}

PORTFOLIO:
- Current Position: {market_state.get('position', 'flat')}
- Cash Available: ${market_state.get('cash', 10000):.2f}
- Unrealized PnL: {market_state.get('unrealized_pnl', 0):.2%}

RISK:
- Confidence Interval Width: {market_state.get('ci_width', 0.02):.2%}
- Uncertainty Score: {market_state.get('uncertainty', 0.3):.2f}

Based on this analysis, choose one action: BUY, HOLD, or SELL.

Action:"""
        return prompt
    
    def _state_dict_to_tensor(self, market_state: Dict) -> torch.Tensor:
        """Convert market state dict to tensor for fallback mode"""
        # Extract numeric features
        features = [
            market_state.get('price_change_1h', 0),
            market_state.get('price_change_4h', 0),
            market_state.get('volume_ratio', 1.0),
            market_state.get('rsi', 50) / 100.0,
            market_state.get('volatility', 0.02),
            market_state.get('regime_confidence', 0.5),
            market_state.get('uncertainty', 0.3),
            market_state.get('cash', 10000) / 100000.0,
            market_state.get('unrealized_pnl', 0),
        ]
        # Pad to expected dimension
        while len(features) < 60:
            features.append(0.0)
        
        return torch.FloatTensor(features[:60]).to(self.device)
    
    def forward(self, market_states: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through FLAG-TRADER.
        
        Args:
            market_states: List of market state dicts
            
        Returns:
            action_logits: (batch, num_actions) logits
            state_values: (batch, 1) estimated values
        """
        if not self.use_llm:
            # Fallback mode: direct tensor input
            states = torch.stack([self._state_dict_to_tensor(s) for s in market_states])
            return self.fallback(states)
        
        # LLM mode: text prompts
        prompts = [self._create_prompt(state) for state in market_states]
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Forward through LLM
        with torch.cuda.amp.autocast():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get last hidden state (last token)
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
        last_hidden = hidden_states[:, -1, :]      # (batch, hidden)
        
        # Policy and value heads
        action_logits = self.policy_head(last_hidden.float())
        state_values = self.value_head(last_hidden.float())
        
        return action_logits, state_values
    
    def get_action(self, market_state: Dict) -> Tuple[TradeAction, float, float]:
        """
        Get action for single market state.
        
        Args:
            market_state: Dict with market features
            
        Returns:
            action: TradeAction enum
            confidence: Action probability
            value: State value estimate
        """
        self.eval()
        with torch.no_grad():
            logits, value = self.forward([market_state])
            probs = torch.softmax(logits, dim=-1)[0]
            action_idx = torch.argmax(probs).item()
            confidence = probs[action_idx].item()
        
        # Map index to action
        if self.config.num_actions == 3:
            action_map = {0: TradeAction.SELL, 1: TradeAction.HOLD, 2: TradeAction.BUY}
        else:
            action_map = {
                0: TradeAction.STRONG_SELL, 
                1: TradeAction.SELL, 
                2: TradeAction.HOLD, 
                3: TradeAction.BUY, 
                4: TradeAction.STRONG_BUY
            }
        
        return action_map.get(action_idx, TradeAction.HOLD), confidence, value.item()
    
    def compute_ppo_loss(self, 
                         states: List[Dict],
                         actions: torch.Tensor,
                         old_log_probs: torch.Tensor,
                         advantages: torch.Tensor,
                         returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss for training.
        
        Args:
            states: List of market state dicts
            actions: (batch,) action indices
            old_log_probs: (batch,) log probabilities from old policy
            advantages: (batch,) advantage estimates
            returns: (batch,) return targets
            
        Returns:
            Dict with loss components
        """
        self.train()
        logits, values = self.forward(states)
        
        # Policy loss (PPO clipped objective)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ratio = torch.exp(action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                           1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.functional.mse_loss(values.squeeze(), returns)
        
        # Entropy bonus (encourages exploration)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.config.value_coef * value_loss - 
                     self.config.entropy_coef * entropy)
        
        return {
            'total': total_loss,
            'policy': policy_loss,
            'value': value_loss,
            'entropy': entropy
        }
    
    def update(self, loss: torch.Tensor):
        """Perform optimization step"""
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.use_llm:
            params = list(self.model.parameters()) + \
                     list(self.policy_head.parameters()) + \
                     list(self.value_head.parameters())
        else:
            params = self.fallback.parameters()
        
        torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
        self.optimizer.step()
    
    def save(self, path: str):
        """Save model to path"""
        if self.use_llm:
            self.model.save_pretrained(f"{path}/lora")
            torch.save({
                'policy_head': self.policy_head.state_dict(),
                'value_head': self.value_head.state_dict(),
                'config': self.config
            }, f"{path}/heads.pt")
        else:
            torch.save({
                'fallback': self.fallback.state_dict(),
                'config': self.config
            }, f"{path}/model.pt")
        logger.info(f"FLAGTrader saved to {path}")
    
    def load(self, path: str):
        """Load model from path"""
        if self.use_llm:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model.base_model, f"{path}/lora")
            heads = torch.load(f"{path}/heads.pt", map_location=self.device)
            self.policy_head.load_state_dict(heads['policy_head'])
            self.value_head.load_state_dict(heads['value_head'])
        else:
            checkpoint = torch.load(f"{path}/model.pt", map_location=self.device)
            self.fallback.load_state_dict(checkpoint['fallback'])
        logger.info(f"FLAGTrader loaded from {path}")


# Factory function
def create_flag_trader(use_llm: bool = False, device: str = 'cuda') -> FLAGTrader:
    """
    Create FLAG-TRADER instance.
    
    Args:
        use_llm: Whether to use full LLM (requires transformers + peft)
        device: Compute device
        
    Returns:
        Configured FLAGTrader
    """
    config = FLAGTraderConfig(use_fallback=not use_llm)
    return FLAGTrader(config, device=device)
