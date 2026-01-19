"""
Test logit bias correction to verify it produces diverse predictions.
"""
import torch
import numpy as np
import pickle
from pathlib import Path
import sys
from collections import Counter

# Import FLAG-TRADER model
models_path = Path(__file__).parent / "LAYER 2 TACTICAL HIMARI OPUS" / "src" / "models"
sys.path.insert(0, str(models_path))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "flag_trader",
    models_path / "flag_trader.py"
)
flag_trader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flag_trader_module)

FLAGTRADERModel = flag_trader_module.FLAGTRADERModel

print("=" * 80)
print("Logit Bias Correction Test")
print("=" * 80)

# Load test data (first 1000 samples for quick test)
print("\n[1/3] Loading test data...")
with open('btc_1h_2025_2026_test_arrays.pkl', 'rb') as f:
    data = pickle.load(f)

features = data['test']['features_denoised'][:1000]  # First 1000 samples
features_60d = np.pad(features, ((0, 0), (0, 11)), mode='constant', constant_values=0.0)
print(f"Loaded {len(features_60d)} samples")

# Load model
print("\n[2/3] Loading FLAG-TRADER model...")
checkpoint = torch.load('checkpoints/flag_trader_best.pt', map_location='cpu')
state_dict = checkpoint['model']

d_model = state_dict['action_head.weight'].shape[1]
num_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('transformer_blocks.')]) + 1
lora_rank = state_dict['transformer_blocks.0.attention.q_proj.lora.lora_A'].shape[1]
max_length = state_dict['pos_embedding'].shape[1]

model = FLAGTRADERModel(
    state_dim=60, action_dim=3, d_model=d_model, num_layers=num_layers,
    num_heads=8, dim_feedforward=d_model * 4, max_length=max_length,
    lora_rank=lora_rank, dropout=0.1
)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded")

# Test with and without correction
print("\n[3/3] Comparing predictions with/without correction...")
print("=" * 80)

# Without correction
predictions_no_corr = []
with torch.no_grad():
    batch_tensor = torch.from_numpy(features_60d).float().unsqueeze(1)
    logits = model(batch_tensor)
    probs = torch.softmax(logits.squeeze(1), dim=-1)
    predictions_no_corr = torch.argmax(probs, dim=-1).numpy()

counter_no_corr = Counter(predictions_no_corr)
print("\nWITHOUT CORRECTION:")
action_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
for action_idx in [0, 1, 2]:
    count = counter_no_corr[action_idx]
    pct = (count / len(predictions_no_corr)) * 100
    print(f"  {action_names[action_idx]}: {count:4d} ({pct:5.2f}%)")

# With correction
predictions_with_corr = []
logit_correction = torch.tensor([4.5, -4.0, 2.5])

with torch.no_grad():
    batch_tensor = torch.from_numpy(features_60d).float().unsqueeze(1)
    logits = model(batch_tensor)
    raw_logits = logits.squeeze(1)

    # Apply correction
    adjusted_logits = raw_logits + logit_correction.unsqueeze(0)
    probs = torch.softmax(adjusted_logits, dim=-1)
    predictions_with_corr = torch.argmax(probs, dim=-1).numpy()

counter_with_corr = Counter(predictions_with_corr)
print("\nWITH CORRECTION:")
for action_idx in [0, 1, 2]:
    count = counter_with_corr[action_idx]
    pct = (count / len(predictions_with_corr)) * 100
    print(f"  {action_names[action_idx]}: {count:4d} ({pct:5.2f}%)")

# Summary
print("\n" + "=" * 80)
print("SUMMARY:")
print(f"  Before: {counter_no_corr[1]/len(predictions_no_corr)*100:.1f}% HOLD")
print(f"  After:  {counter_with_corr[1]/len(predictions_with_corr)*100:.1f}% HOLD")

non_hold_before = (counter_no_corr[0] + counter_no_corr[2]) / len(predictions_no_corr) * 100
non_hold_after = (counter_with_corr[0] + counter_with_corr[2]) / len(predictions_with_corr) * 100
print(f"  BUY+SELL: {non_hold_before:.1f}% -> {non_hold_after:.1f}%")

if counter_with_corr[0] > 0 and counter_with_corr[2] > 0:
    print("\n[SUCCESS] Correction produces diverse predictions!")
else:
    print("\n[WARNING] Correction may need tuning")

print("=" * 80)
