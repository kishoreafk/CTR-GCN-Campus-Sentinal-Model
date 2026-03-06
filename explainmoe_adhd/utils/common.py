"""
Common utilities for ExplainMoE-ADHD v2.13.
"""

import os
import random
import torch
import numpy as np
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_device(prefer_cuda: bool = True) -> str:
    """Get available device."""
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def move_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Move all tensors in a batch to the specified device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
