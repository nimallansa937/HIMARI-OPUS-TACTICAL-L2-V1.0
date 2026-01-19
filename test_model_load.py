"""
Test script to verify Transformer-A2C model loading from checkpoint.
"""
import torch
import sys
from pathlib import Path

# Add model path (direct to models directory to avoid package import issues)
models_path = Path(__file__).parent / "LAYER 2 TACTICAL HIMARI OPUS" / "src" / "models"
sys.path.insert(0, str(models_path))

# Import directly from the file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "transformer_a2c",
    models_path / "transformer_a2c.py"
)
transformer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformer_module)

TransformerA2C = transformer_module.TransformerA2C
TransformerA2CConfig = transformer_module.TransformerA2CConfig

print("=" * 80)
print("FLAG-TRADER Model Loading Test")
print("=" * 80)

# Load checkpoint
print("\n[1/4] Loading checkpoint...")
try:
    checkpoint = torch.load('checkpoints/flag_trader_best.pt', map_location='cpu')
    print("[OK] Checkpoint loaded successfully")
    print(f"  Keys: {list(checkpoint.keys())}")
    print(f"  Config: {checkpoint['config']}")
    print(f"  Model state dict keys (first 5): {list(checkpoint['model'].keys())[:5]}")
except Exception as e:
    print(f"[FAIL] Failed to load checkpoint: {e}")
    sys.exit(1)

# Create model
print("\n[2/4] Creating Transformer-A2C model...")
try:
    config = TransformerA2CConfig(
        input_dim=60,
        hidden_dim=256,
        num_heads=8,
        num_layers=12,
        context_length=256
    )

    model = TransformerA2C(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("[OK] Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
except Exception as e:
    print(f"[FAIL] Failed to create model: {e}")
    sys.exit(1)

# Load weights
print("\n[3/4] Loading trained weights...")
try:
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("[OK] Weights loaded successfully")
    print("  Model is in eval mode")
except Exception as e:
    print(f"[FAIL] Failed to load weights: {e}")
    print(f"\nError details: {type(e).__name__}: {str(e)}")
    sys.exit(1)

# Test forward pass
print("\n[4/4] Testing forward pass...")
try:
    import numpy as np

    # Create test input (batch=1, seq=1, features=60)
    test_input = torch.randn(1, 1, 60)

    with torch.no_grad():
        output = model(test_input)

    action_logits = output[0].numpy()
    probabilities = torch.softmax(output[0], dim=-1).numpy()
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
    sys.exit(1)

print("\n" + "=" * 80)
print("[SUCCESS] Model is ready for backtesting!")
print("=" * 80)
print("\nNext step: Run 'python end_to_end_backtest.py'")
