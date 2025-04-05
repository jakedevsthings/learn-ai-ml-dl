# /learn-ai-ml-dl/ml_core_torch/utils/data.py

"""
Data utilities for machine learning core.
"""
import torch

def generate_xor_dataset():
    """Generate an XOR dataset."""
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    return X, y
