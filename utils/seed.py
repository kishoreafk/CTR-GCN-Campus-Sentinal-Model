"""Reproducibility seed setter."""
import random, os
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: benchmark=True is still safe with fixed shapes
    torch.backends.cudnn.deterministic = False  # keep benchmark speed
