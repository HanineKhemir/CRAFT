"""
seed.py — reproducibility helpers.
Call set_seed(cfg.data.seed) once at the top of train.py.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all RNG seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic CUDNN — slight speed cost, worth it for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False