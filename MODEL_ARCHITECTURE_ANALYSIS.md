# FLAG-TRADER Model Architecture Analysis

## Summary

The trained FLAG-TRADER checkpoint (`flag_trader_best.pt`) contains a **custom Decision Transformer with LoRA adapters** that doesn't match any of the existing model architectures in the codebase.

## Checkpoint Structure

**Total Parameters:** 343 weight tensors

### Layer Structure

```
pos_embedding                                    # Learned positional encoding
state_embedding.weight                            # Input projection (60D → 256D)
state_embedding.bias

transformer_blocks.0-11.attention.q_proj.base.weight    # Query projection base
transformer_blocks.0-11.attention.q_proj.base.bias
transformer_blocks.0-11.attention.q_proj.lora.lora_A    # LoRA adapter (rank 16)
transformer_blocks.0-11.attention.q_proj.lora.lora_B

transformer_blocks.0-11.attention.k_proj.base.*         # Key projection + LoRA
transformer_blocks.0-11.attention.v_proj.base.*         # Value projection + LoRA
transformer_blocks.0-11.attention.out_proj.base.*       # Output projection + LoRA

transformer_blocks.0-11.norm1.weight                     # Layer norm 1
transformer_blocks.0-11.norm1.bias
transformer_blocks.0-11.norm2.weight                     # Layer norm 2
transformer_blocks.0-11.norm2.bias

transformer_blocks.0-11.ffn.0.base.*                    # FFN layer 1 + LoRA
transformer_blocks.0-11.ffn.3.base.*                    # FFN layer 2 + LoRA

output_norm.weight                                       # Final normalization
output_norm.bias

action_head.weight                                       # Action logits (256D → 3D)
action_head.bias
```

### Architecture Specifications

- **12 Transformer Blocks**
- **256 Hidden Dimensions** (inferred from action_head input)
- **LoRA Rank:** 16 (inferred from lora_A/lora_B dimensions)
- **Input:** 60D feature vectors
- **Output:** 3D action logits (SELL/HOLD/BUY)
- **LoRA Applied To:**
  - All attention projections (q, k, v, out)
  - All FFN layers

## Architecture Mismatch

### Existing Models in Codebase

**Transformer-A2C** (`LAYER 2 TACTICAL HIMARI OPUS/src/models/transformer_a2c.py`):
- Uses `in_proj_weight` (combined Q/K/V projection)
- No LoRA adapters
- Different layer naming (`encoder.transformer_blocks...`)
- **Result:** ❌ Incompatible

**FLAG-TRADER** (`LAYER 2 TACTICAL HIMARI OPUS/src/decision_engine/flag_trader.py`):
- Uses HuggingFace transformers + PEFT LoRA
- Different architecture (SmolLM2-135M)
- **Result:** ❌ Incompatible

**CGDT** (`LAYER 2 TACTICAL HIMARI OPUS/src/decision_engine/cgdt.py`):
- Conditional Decision Transformer
- Different layer structure
- **Result:** ❌ Incompatible

## Solution Options

### Option 1: Find Original Training Script (Recommended)

The model was trained with a specific architecture. Find the training script that created this checkpoint:

```bash
# Search for training scripts
find . -name "*train*transformer*.py" -o -name "*train*lora*.py"
```

Look for files that create a model with:
- 12 transformer blocks
- LoRA adapters on attention + FFN
- `pos_embedding`, `state_embedding`, `action_head`

### Option 2: Reconstruct Architecture from Checkpoint

Create a custom model class that matches the checkpoint structure exactly:

**File to create:** `LAYER 2 V1 - Copy/models/custom_decision_transformer_lora.py`

```python
import torch
import torch.nn as nn
from typing import Tuple

class LoRALayer(nn.Module):
    """LoRA adapter for efficient fine-tuning."""
    def __init__(self, base_layer: nn.Linear, rank: int = 16):
        super().__init__()
        self.base = base_layer
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, out_features) * 0.01)

    def forward(self, x):
        # base output + LoRA adaptation
        return self.base(x) + x @ self.lora_A @ self.lora_B

class TransformerBlockWithLoRA(nn.Module):
    """Transformer block with LoRA adapters."""
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, lora_rank: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Attention with LoRA
        self.q_proj = LoRALayer(nn.Linear(hidden_dim, hidden_dim), lora_rank)
        self.k_proj = LoRALayer(nn.Linear(hidden_dim, hidden_dim), lora_rank)
        self.v_proj = LoRALayer(nn.Linear(hidden_dim, hidden_dim), lora_rank)
        self.out_proj = LoRALayer(nn.Linear(hidden_dim, hidden_dim), lora_rank)

        # FFN with LoRA
        self.ffn = nn.Sequential(
            LoRALayer(nn.Linear(hidden_dim, hidden_dim * 4), lora_rank),
            nn.GELU(),
            nn.Dropout(0.1),
            LoRALayer(nn.Linear(hidden_dim * 4, hidden_dim), lora_rank)
        )

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

    def forward(self, x):
        # Multi-head attention
        residual = x
        x = self.norm1(x)

        # ... (implement multi-head attention)

        x = residual + x

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x

class CustomDecisionTransformerLoRA(nn.Module):
    """Decision Transformer with LoRA adapters matching checkpoint."""
    def __init__(self, input_dim: int = 60, hidden_dim: int = 256,
                 num_layers: int = 12, lora_rank: int = 16):
        super().__init__()

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, hidden_dim) * 0.02)

        # State embedding (input projection)
        self.state_embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlockWithLoRA(hidden_dim, lora_rank=lora_rank)
            for _ in range(num_layers)
        ])

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 3)  # BUY/HOLD/SELL

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        # Embed
        x = self.state_embedding(x)  # (batch, seq_len, hidden_dim)

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Output
        x = self.output_norm(x)
        x = x[:, -1, :]  # Last timestep
        logits = self.action_head(x)

        return logits
```

**Note:** This is a simplified version. The full implementation requires proper multi-head attention.

### Option 3: Use Untrained Fallback (Current Status)

Continue using the untrained MLP fallback. This provides:
- ✅ Complete workflow validation
- ✅ Framework testing
- ❌ Poor performance (expected)

## Recommendations

1. **Search for Training Script:**
   ```bash
   cd "C:\Users\chari\OneDrive\Documents\HIMARI OPUS TESTING\LAYER 2 V1 - Copy"
   grep -r "lora_A" --include="*.py"
   grep -r "pos_embedding" --include="*.py"
   ```

2. **Check Training Logs:**
   Look for logs that show model architecture or training configuration.

3. **Contact Original Developer:**
   If this was trained by someone else, ask for the model definition file.

4. **Alternative Approach:**
   Since the workflow is validated, you could:
   - Retrain with an architecture you have (Transformer-A2C)
   - Use existing evaluation results (61.42% accuracy documented)
   - Focus on Layer 1→2→3 integration testing

## Current Workflow Status

**✅ Complete and Functional:**
- Data loading and verification
- Layer 1→2→3 pipeline
- Position sizing and execution
- HIFA validation framework
- Results export and reporting

**⚠️ Pending:**
- Trained model loading (architecture mismatch)

**Workaround:**
The workflow runs successfully with untrained fallback MLP, demonstrating that all components work correctly. With the correct trained model, performance should improve to match the documented 61.42% accuracy.

## Files to Check

Search these locations for the training script:

1. `LAYER 2 V1 - Copy/scripts/train_*.py`
2. `LAYER 2 V1 - Copy/training/`
3. `LAYER 2 TACTICAL HIMARI OPUS/scripts/`
4. `LAYER 2 TACTICAL HIMARI OPUS/training/`

Look for files that:
- Import LoRA or PEFT libraries
- Create 12 transformer blocks
- Have `pos_embedding` parameter
- Save checkpoints to `checkpoints/flag_trader_best.pt`

---

**Status:** Architecture identified, needs matching model class
**Date:** 2026-01-19
**Checkpoint:** `checkpoints/flag_trader_best.pt` (343 parameters)
