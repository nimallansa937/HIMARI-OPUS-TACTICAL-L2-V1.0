
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def check_leakage(data_path='data/btc_5min_2020_2024.pkl'):
    print(f"Loading {data_path}...")
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    features = np.array(raw_data.get('features', list(raw_data.values())[0]), dtype=np.float32)
    prices = np.array(raw_data.get('prices', features[:, 0]), dtype=np.float32)
    
    print(f"Features: {features.shape}, Prices: {prices.shape}")
    
    # Calculate returns (Target)
    # Return[t] = (Price[t] - Price[t-1]) / Price[t-1]
    returns = np.diff(prices) / prices[:-1]
    returns = np.insert(returns, 0, 0) # Pad to match length
    
    # Check correlation of Features with Returns
    # Note: TransformerEnv gives features[t-1] to the model to predict return at t
    # So we want to check if features[t-1] contains info about return[t]
    
    # Shift features by +1 to align feature[t-1] with return[t]
    # No, wait. 
    # At step t, we see features[t-1]. We trade. We get return[t] = (P[t]-P[t-1])/P[t-1].
    # So we check correlation(features[t-1], returns[t])
    
    threshold = 0.95
    suspicious_cols = []
    
    print("Checking correlations between Features[t-1] and Returns[t]...")
    
    targets = returns[1:] # Returns at t=1..N
    feats = features[:-1] # Features at t=0..N-1
    
    for i in range(feats.shape[1]):
        corr, _ = spearmanr(feats[:, i], targets)
        if abs(corr) > 0.1:
            print(f"Feature {i}: Corr = {corr:.4f}")
        
        if abs(corr) > threshold:
            suspicious_cols.append((i, corr))
            
    if suspicious_cols:
        print("\n🚨 LEAKAGE DETECTED! Use of future data likely.")
        for idx, val in suspicious_cols:
            print(f"Feature {idx} correlates {val:.4f} with NEXT return.")
    else:
        print("\n✅ No direct linear leakage found (max corr < 0.95)")
        
    # Also check if Prices are in features
    # If Feature k == Price t, then Feature[t-1] == Price[t-1] != Price[t] (Safe)
    # If Feature k == Price t+1, then Feature[t-1] == Price[t] (Leakage!)

if __name__ == "__main__":
    check_leakage()
