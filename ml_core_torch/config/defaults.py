# /learn-ai-ml-dl/ml_core_torch/config/defaults.py

"""
Default configuration for machine learning core.
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULTS = {
    "epochs": 5000,
    "learning_rate": 0.2,
    "batch_size": 4,
}
