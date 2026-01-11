"""
HIMARI Layer 2 - Data Module
Handles data loading, preprocessing, and dataset management.
"""

from src.data.dataset import (
    HIMARIDataset,
    CurriculumDataset,
    load_training_data,
    compute_class_weights,
    verify_data_files
)

__all__ = [
    'HIMARIDataset',
    'CurriculumDataset',
    'load_training_data',
    'compute_class_weights',
    'verify_data_files'
]
