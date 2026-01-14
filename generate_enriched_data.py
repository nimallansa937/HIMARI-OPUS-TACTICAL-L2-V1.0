"""
Script to generate enriched BTC dataset from raw OHLCV data.
"""
import sys
import os
from pathlib import Path

# Setup paths - must be done before imports
THIS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TACTICAL_DIR = THIS_DIR / 'LAYER 2 TACTICAL HIMARI OPUS'

# Clear any cached src module that might conflict
for mod in list(sys.modules.keys()):
    if mod.startswith('src'):
        del sys.modules[mod]

# Add paths - order matters!
# Insert TACTICAL first, then THIS_DIR
# Since insert(0, x) puts x at front, the LAST insert wins
sys.path.insert(0, str(TACTICAL_DIR))  # Second priority
sys.path.insert(0, str(THIS_DIR))       # First priority (our src.pipeline)

print(f'THIS_DIR: {THIS_DIR}')
print(f'sys.path[0]: {sys.path[0]}')
print(f'sys.path[1]: {sys.path[1]}')

# Now import
from src.pipeline.dataset_generator import generate_enriched_dataset

if __name__ == "__main__":
    print()
    print('=' * 60)
    print('HIMARI Enriched Dataset Generator')
    print('=' * 60)
    print()

    input_path = 'C:/Users/chari/OneDrive/Documents/BTC DATA SETS/btc_1h_2020_2024_raw.pkl'
    output_path = 'C:/Users/chari/OneDrive/Documents/BTC DATA SETS/btc_1h_2020_2024_enriched.pkl'

    print(f'Input:  {input_path}')
    print(f'Output: {output_path}')
    print()

    dataset = generate_enriched_dataset(
        raw_data_path=input_path,
        output_path=output_path,
        train_ratio=0.6,
        val_ratio=0.2
    )

    print()
    print('=' * 60)
    print('COMPLETE!')
    print('=' * 60)
