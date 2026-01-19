"""
Diagnose predictions from the newly trained model
"""
import torch
import numpy as np
import pickle
from pathlib import Path
import sys

# Add model path
models_path = Path(__file__).parent / "LAYER 2 TACTICAL HIMARI OPUS" / "src" / "models"
sys.path.insert(0, str(models_path))

from flag_trader import FLAGTRADERModel

# Load checkpoint
checkpoint_path = "checkpoints/flag_trader_best.pt"
checkpoint = torch.load(checkpoint_path, map_location='cuda')

# Load model
model = FLAGTRADERModel(
    state_dim=60,
    action_dim=3,
    d_model=768,
    num_layers=12,
    num_heads=8,
    dim_feedforward=3072,
    lora_rank=16
)
model.load_state_dict(checkpoint['model_state_dict'])
model =model.cuda()
model.eval()

# Load test data
with open('btc_1h_2025_2026_test_arrays.pkl', 'rb') as f:
    data = pickle.load(f)

features = data['test']['features_denoised']  # (N, 49)

# Pad to 60D
padding = np.zeros((features.shape[0], 11))
features = np.concatenate([features, padding], axis=1)

# Sample 100 random predictions
np.random.seed(42)
indices = np.random.choice(len(features), 100, replace=False)

predictions = {'SELL': 0, 'HOLD': 0, 'BUY': 0}
confidences = []

print("Sampling 100 predictions...")
print("="*80)

for i, idx in enumerate(indices[:20]):  # Show first 20
    x = torch.from_numpy(features[idx]).float().unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        logits = model(x).squeeze()
        probs = torch.softmax(logits, dim=-1)
        action_idx = torch.argmax(probs).item()
        confidence = probs[action_idx].item()

    action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    action = action_map[action_idx]

    print(f"{i+1}. Logits: [{logits[0]:.3f}, {logits[1]:.3f}, {logits[2]:.3f}] "
          f"Probs: [{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}] "
          f"Action: {action} (conf: {confidence:.3f})")

    predictions[action] += 1
    confidences.append(confidence)

# Count all 100
for idx in indices:
    x = torch.from_numpy(features[idx]).float().unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        logits = model(x).squeeze()
        probs = torch.softmax(logits, dim=-1)
        action_idx = torch.argmax(probs).item()

    action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    predictions[action_map[action_idx]] += 1

print("="*80)
print("\nPrediction Distribution (100 samples):")
print(f"  SELL: {predictions['SELL']}/100 ({predictions['SELL']}%)")
print(f"  HOLD: {predictions['HOLD']}/100 ({predictions['HOLD']}%)")
print(f"  BUY:  {predictions['BUY']}/100 ({predictions['BUY']}%)")
print(f"\nAvg Confidence: {np.mean(confidences):.3f}")
