"""
Diagnostic script to analyze FLAG-TRADER predictions across all test data.
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
print("FLAG-TRADER Prediction Diagnosis")
print("=" * 80)

# Load test data
print("\n[1/4] Loading test data...")
with open('btc_1h_2025_2026_test_arrays.pkl', 'rb') as f:
    data = pickle.load(f)

features = data['test']['features_denoised']
n_samples = len(features)
print(f"Loaded {n_samples} samples")
print(f"Feature shape: {features.shape}")

# Pad 49D -> 60D
features_60d = np.pad(
    features,
    ((0, 0), (0, 11)),
    mode='constant',
    constant_values=0.0
)
print(f"Padded to 60D: {features_60d.shape}")

# Load model
print("\n[2/4] Loading trained FLAG-TRADER model...")
checkpoint = torch.load('checkpoints/flag_trader_best.pt', map_location='cpu')
state_dict = checkpoint['model']

# Detect model config
d_model = state_dict['action_head.weight'].shape[1]
num_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('transformer_blocks.')]) + 1
lora_rank = state_dict['transformer_blocks.0.attention.q_proj.lora.lora_A'].shape[1]
max_length = state_dict['pos_embedding'].shape[1]

model = FLAGTRADERModel(
    state_dim=60,
    action_dim=3,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=8,
    dim_feedforward=d_model * 4,
    max_length=max_length,
    lora_rank=lora_rank,
    dropout=0.1
)
model.load_state_dict(state_dict)
model.eval()

print(f"Model loaded: d_model={d_model}, layers={num_layers}, lora_rank={lora_rank}")

# Run predictions on all samples
print("\n[3/4] Running predictions on all samples...")
all_predictions = []
all_confidences = []
all_logits = []

batch_size = 256
num_batches = (n_samples + batch_size - 1) // batch_size

with torch.no_grad():
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        batch_features = features_60d[start_idx:end_idx]
        batch_tensor = torch.from_numpy(batch_features).float().unsqueeze(1)  # (batch, 1, 60)

        # Forward pass
        logits = model(batch_tensor)  # (batch, 1, 3)
        probs = torch.softmax(logits.squeeze(1), dim=-1)  # (batch, 3)

        # Store results
        predictions = torch.argmax(probs, dim=-1).numpy()
        confidences = torch.max(probs, dim=-1).values.numpy()

        all_predictions.extend(predictions)
        all_confidences.extend(confidences)
        all_logits.append(logits.squeeze(1).numpy())  # (batch, 3)

        if (i + 1) % 10 == 0:
            print(f"  Processed {end_idx}/{n_samples} samples...")

all_predictions = np.array(all_predictions)
all_confidences = np.array(all_confidences)
all_logits = np.vstack(all_logits)

# Analyze prediction distribution
print("\n[4/4] Analyzing predictions...")
print("=" * 80)

action_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
counter = Counter(all_predictions)

print("\nACTION DISTRIBUTION:")
for action_idx in [0, 1, 2]:
    count = counter[action_idx]
    pct = (count / n_samples) * 100
    print(f"  {action_names[action_idx]}: {count:5d} ({pct:5.2f}%)")

print("\nCONFIDENCE STATISTICS:")
print(f"  Mean confidence: {np.mean(all_confidences):.4f}")
print(f"  Std confidence:  {np.std(all_confidences):.4f}")
print(f"  Min confidence:  {np.min(all_confidences):.4f}")
print(f"  Max confidence:  {np.max(all_confidences):.4f}")

print("\nCONFIDENCE BY ACTION:")
for action_idx in [0, 1, 2]:
    mask = all_predictions == action_idx
    if mask.sum() > 0:
        action_confidences = all_confidences[mask]
        print(f"  {action_names[action_idx]}: mean={np.mean(action_confidences):.4f}, "
              f"std={np.std(action_confidences):.4f}")

print("\nLOGITS STATISTICS:")
print(f"  SELL logits: mean={np.mean(all_logits[:, 0]):.4f}, std={np.std(all_logits[:, 0]):.4f}")
print(f"  HOLD logits: mean={np.mean(all_logits[:, 1]):.4f}, std={np.std(all_logits[:, 1]):.4f}")
print(f"  BUY logits:  mean={np.mean(all_logits[:, 2]):.4f}, std={np.std(all_logits[:, 2]):.4f}")

# Check if model is stuck
unique_predictions = len(set(all_predictions))
if unique_predictions == 1:
    print("\n[WARNING] Model predicts only ONE action for ALL samples!")
    print("          This indicates the model has collapsed to a single prediction.")
    print("          Possible causes:")
    print("          1. Class imbalance during training (HOLD dominates)")
    print("          2. Model hasn't learned meaningful patterns")
    print("          3. Input features are not representative")
elif unique_predictions == 2:
    print("\n[WARNING] Model uses only 2 out of 3 actions")
    print(f"          Missing action: {[action_names[i] for i in range(3) if counter[i] == 0]}")
else:
    print("\n[OK] Model uses all 3 actions")

# Sample some predictions
print("\nSAMPLE PREDICTIONS (first 20):")
print("  Step | Action | Confidence | SELL   | HOLD   | BUY")
print("  " + "-" * 60)
for i in range(min(20, n_samples)):
    pred = all_predictions[i]
    conf = all_confidences[i]
    logits = all_logits[i]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    print(f"  {i:4d} | {action_names[pred]:4s} | {conf:6.4f} | "
          f"{probs[0]:6.4f} | {probs[1]:6.4f} | {probs[2]:6.4f}")

print("\n" + "=" * 80)
print("Diagnosis complete!")
print("=" * 80)
