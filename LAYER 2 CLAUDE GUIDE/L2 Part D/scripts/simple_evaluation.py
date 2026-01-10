"""
Simple evaluation script that directly loads and evaluates checkpoints.
"""
import torch
import numpy as np
import json
from pathlib import Path

# Load data
print("Loading validation data...")
data_dir = Path("LAYER 2 TACTICAL HIMARI OPUS/data")
features = np.load(data_dir / "preprocessed_features.npy")
labels = np.load(data_dir / "labels.npy")

# Use last 10% as validation
val_start = int(0.9 * len(features))
val_features = features[val_start:]
val_labels = labels[val_start:]

print(f"Validation samples: {len(val_features)}")
print()

results = []

# ============================================================================
# BaselineMLP
# ============================================================================
print("=" * 70)
print("Evaluating BaselineMLP...")
try:
    ckpt = torch.load("checkpoints/baseline_best.pt", map_location='cpu')

    from LAYER_2_TACTICAL_HIMARI_OPUS.src.models.baseline_mlp import create_baseline_model
    model = create_baseline_model(60, [128, 64, 32], 3)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with torch.no_grad():
        inputs = torch.FloatTensor(val_features)
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1).numpy()

    accuracy = (predictions == val_labels).mean()

    results.append({
        'model': 'BaselineMLP',
        'accuracy': float(accuracy),
        'params': 18691
    })

    print(f"✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

except Exception as e:
    print(f"✗ Failed: {e}")

print()

# ============================================================================
# CQL
# ============================================================================
print("=" * 70)
print("Evaluating CQL...")
try:
    ckpt = torch.load("checkpoints/cql_best.pt", map_location='cpu')

    from LAYER_2_TACTICAL_HIMARI_OPUS.src.models.cql import create_cql_agent
    agent = create_cql_agent(65, 3, 256)
    agent.q_network1.load_state_dict(ckpt['q_network1'])
    agent.eval()

    # Add dummy env features
    val_states = np.concatenate([val_features, np.zeros((len(val_features), 5))], axis=1)

    with torch.no_grad():
        states = torch.FloatTensor(val_states)
        q_values = agent(states)
        predictions = torch.argmax(q_values, dim=1).numpy()

    accuracy = (predictions == val_labels).mean()

    results.append({
        'model': 'CQL',
        'accuracy': float(accuracy),
        'params': 116995
    })

    print(f"✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

except Exception as e:
    print(f"✗ Failed: {e}")

print()

# ============================================================================
# CGDT
# ============================================================================
print("=" * 70)
print("Evaluating CGDT...")
try:
    ckpt = torch.load("checkpoints/cgdt_best.pt", map_location='cpu')

    from LAYER_2_TACTICAL_HIMARI_OPUS.src.models.cgdt import create_cgdt_agent
    agent = create_cgdt_agent(60, 3, 256, 6)

    # CGDT saves nested dict
    if 'dt' in ckpt:
        agent.dt.load_state_dict(ckpt['dt'])
        agent.critic.load_state_dict(ckpt['critic'])
    else:
        agent.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)

    agent.eval()

    # CGDT needs sequences
    context_length = 64
    num_sequences = len(val_features) // context_length

    predictions_list = []
    with torch.no_grad():
        for i in range(num_sequences):
            start_idx = i * context_length
            end_idx = start_idx + context_length

            states = torch.FloatTensor(val_features[start_idx:end_idx]).unsqueeze(0)
            actions = torch.zeros(1, context_length).long()
            returns_to_go = torch.zeros(1, context_length)
            timesteps = torch.arange(context_length).unsqueeze(0)

            action_preds = agent(states, actions, returns_to_go, timesteps)
            preds = torch.argmax(action_preds, dim=-1).numpy()[0]
            predictions_list.extend(preds)

    predictions = np.array(predictions_list)
    labels_truncated = val_labels[:len(predictions)]
    accuracy = (predictions == labels_truncated).mean()

    results.append({
        'model': 'CGDT',
        'accuracy': float(accuracy),
        'params': 4822276
    })

    print(f"✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

except Exception as e:
    print(f"✗ Failed: {e}")

print()

# ============================================================================
# FLAG-TRADER
# ============================================================================
print("=" * 70)
print("Evaluating FLAG-TRADER...")
try:
    ckpt = torch.load("checkpoints/flag_trader_best.pt", map_location='cpu')

    from LAYER_2_TACTICAL_HIMARI_OPUS.src.models.flag_trader import create_flag_trader_agent
    agent = create_flag_trader_agent(60, 3, "135M", 16)

    # FLAG-TRADER model attribute stores the actual model
    agent.model.load_state_dict(ckpt['model'])
    agent.eval()

    # FLAG-TRADER needs sequences
    context_length = 128  # Use smaller context for faster evaluation
    num_sequences = len(val_features) // context_length

    predictions_list = []
    with torch.no_grad():
        for i in range(num_sequences):
            start_idx = i * context_length
            end_idx = start_idx + context_length

            states = torch.FloatTensor(val_features[start_idx:end_idx]).unsqueeze(0)
            action_preds = agent(states)
            preds = torch.argmax(action_preds, dim=-1).numpy()[0]
            predictions_list.extend(preds)

    predictions = np.array(predictions_list)
    labels_truncated = val_labels[:len(predictions)]
    accuracy = (predictions == labels_truncated).mean()

    results.append({
        'model': 'FLAG-TRADER',
        'accuracy': float(accuracy),
        'params': 87955971
    })

    print(f"✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

except Exception as e:
    print(f"✗ Failed: {e}")

print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)
print()

# Sort by accuracy
results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

for i, result in enumerate(results, 1):
    print(f"{i}. {result['model']:15s} - {result['accuracy']*100:6.2f}% accuracy ({result['params']:,} params)")

print()

# Save results
with open("evaluation_results_simple.json", 'w') as f:
    json.dump({
        'results': results,
        'best_model': results[0]['model'] if results else None,
        'best_accuracy': results[0]['accuracy'] if results else None
    }, f, indent=2)

print("Results saved to: evaluation_results_simple.json")
print("=" * 70)
