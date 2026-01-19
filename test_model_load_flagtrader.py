"""
Test script to verify FLAG-TRADER model loading from checkpoint.
"""
import torch
import sys
from pathlib import Path

# Add model path
models_path = Path(__file__).parent / "LAYER 2 TACTICAL HIMARI OPUS" / "src" / "models"
sys.path.insert(0, str(models_path))

# Import directly from the file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "flag_trader",
    models_path / "flag_trader.py"
)
flag_trader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flag_trader_module)

FLAGTRADERAgent = flag_trader_module.FLAGTRADERAgent
FLAGTRADERModel = flag_trader_module.FLAGTRADERModel

print("=" * 80)
print("FLAG-TRADER Model Loading Test")
print("=" * 80)

# Load checkpoint
print("\n[1/5] Loading checkpoint...")
try:
    checkpoint = torch.load('checkpoints/flag_trader_best.pt', map_location='cpu')
    print("[OK] Checkpoint loaded successfully")
    print(f"  Keys: {list(checkpoint.keys())}")
    print(f"  Config: {checkpoint['config']}")
    print(f"  Model state dict keys (first 10):")
    for i, key in enumerate(list(checkpoint['model'].keys())[:10]):
        print(f"    {i+1}. {key}")
except Exception as e:
    print(f"[FAIL] Failed to load checkpoint: {e}")
    sys.exit(1)

# Analyze checkpoint structure to determine model dimensions
print("\n[2/5] Analyzing checkpoint structure...")
try:
    state_dict = checkpoint['model']

    # Check action_head to determine d_model
    if 'action_head.weight' in state_dict:
        action_head_shape = state_dict['action_head.weight'].shape
        d_model = action_head_shape[1]  # input dimension
        action_dim = action_head_shape[0]  # output dimension
        print(f"[OK] Detected dimensions from action_head.weight: {action_head_shape}")
        print(f"  d_model (hidden_dim): {d_model}")
        print(f"  action_dim: {action_dim}")

    # Count transformer blocks
    num_layers = 0
    for key in state_dict.keys():
        if key.startswith('transformer_blocks.'):
            layer_num = int(key.split('.')[1])
            num_layers = max(num_layers, layer_num + 1)
    print(f"  num_layers: {num_layers}")

    # Check for num_heads by analyzing attention weights
    if 'transformer_blocks.0.attention.q_proj.base.weight' in state_dict:
        q_proj_shape = state_dict['transformer_blocks.0.attention.q_proj.base.weight'].shape
        # q_proj: (d_model, d_model)
        num_heads = 8  # Common default, can't detect from weights
        print(f"  q_proj shape: {q_proj_shape}")
        print(f"  num_heads: {num_heads} (default)")

    # Check LoRA rank
    if 'transformer_blocks.0.attention.q_proj.lora.lora_A' in state_dict:
        lora_A_shape = state_dict['transformer_blocks.0.attention.q_proj.lora.lora_A'].shape
        lora_rank = lora_A_shape[1]  # (in_features, rank)
        print(f"  lora_A shape: {lora_A_shape}")
        print(f"  lora_rank: {lora_rank}")

    # Check max_length from pos_embedding
    if 'pos_embedding' in state_dict:
        pos_emb_shape = state_dict['pos_embedding'].shape
        max_length = pos_emb_shape[1]  # (1, max_length, d_model)
        print(f"  pos_embedding shape: {pos_emb_shape}")
        print(f"  max_length: {max_length}")

except Exception as e:
    print(f"[FAIL] Failed to analyze checkpoint: {e}")
    sys.exit(1)

# Create model with detected configuration
print("\n[3/5] Creating FLAG-TRADER model with detected config...")
try:
    state_dim = checkpoint['config']['state_dim']

    # Create model
    model = FLAGTRADERModel(
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=d_model * 4,  # Standard transformer ratio
        max_length=max_length,
        lora_rank=lora_rank,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("[OK] Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Configuration:")
    print(f"    state_dim={state_dim}, action_dim={action_dim}")
    print(f"    d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}")
    print(f"    max_length={max_length}, lora_rank={lora_rank}")
except Exception as e:
    print(f"[FAIL] Failed to create model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load weights
print("\n[4/5] Loading trained weights...")
try:
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("[OK] Weights loaded successfully")
    print("  Model is in eval mode")
except Exception as e:
    print(f"[FAIL] Failed to load weights: {e}")
    print(f"\nError details: {type(e).__name__}: {str(e)}")

    # Print mismatched keys for debugging
    print("\nAttempting to identify mismatched keys...")
    try:
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint['model'].keys())

        missing = checkpoint_keys - model_keys
        unexpected = model_keys - checkpoint_keys

        if missing:
            print(f"\nMissing keys in model ({len(missing)}):")
            for key in list(missing)[:5]:
                print(f"  - {key}")
            if len(missing) > 5:
                print(f"  ... and {len(missing)-5} more")

        if unexpected:
            print(f"\nUnexpected keys in model ({len(unexpected)}):")
            for key in list(unexpected)[:5]:
                print(f"  - {key}")
            if len(unexpected) > 5:
                print(f"  ... and {len(unexpected)-5} more")
    except:
        pass

    sys.exit(1)

# Test forward pass
print("\n[5/5] Testing forward pass...")
try:
    import numpy as np

    # Create test input (batch=1, seq=1, features=60)
    test_input = torch.randn(1, 1, state_dim)

    with torch.no_grad():
        output = model(test_input)

    action_logits = output[0, 0].numpy()
    probabilities = torch.softmax(output[0, 0], dim=-1).numpy()
    predicted_action = np.argmax(probabilities)
    action_names = ['SELL', 'HOLD', 'BUY']

    print("[OK] Forward pass successful")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Action logits: {action_logits}")
    print(f"  Probabilities: {probabilities}")
    print(f"  Predicted action: {action_names[predicted_action]} ({probabilities[predicted_action]:.1%} confidence)")

except Exception as e:
    print(f"[FAIL] Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("[SUCCESS] FLAG-TRADER model loaded and ready!")
print("=" * 80)
print("\nNext step: Run 'python end_to_end_backtest.py'")
