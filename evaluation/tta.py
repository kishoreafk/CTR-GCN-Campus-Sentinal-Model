"""
Test-Time Augmentation for skeleton-based action recognition.

Augmentations applied at inference:
  1. Original                 (weight: 1.0)
  2. Horizontal flip          (weight: 1.0)
  3. Temporal reverse         (weight: 0.8)
  4. Horizontal flip + reverse (weight: 0.8)

Predictions are weighted-averaged across all augmentations.
"""

import torch
import torch.nn as nn
from typing import List


# OpenPose-18 flip pairs (left/right body symmetry)
FLIP_PAIRS = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (14, 15), (16, 17)]


def apply_horizontal_flip(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N, C=3, T=64, V=18, M=2)
    Returns horizontally flipped skeleton.
    """
    x = x.clone()
    # Flip x coordinate (channel 0)
    x[:, 0, :, :, :] = 1.0 - x[:, 0, :, :, :]
    # Swap left/right joint pairs
    for l, r in FLIP_PAIRS:
        x[:, :, :, [l, r], :] = x[:, :, :, [r, l], :]
    return x


def apply_temporal_reverse(x: torch.Tensor) -> torch.Tensor:
    """Reverse the temporal dimension."""
    return x.flip(dims=[2])


class TTAEvaluator:

    def __init__(self, model: nn.Module, device: str,
                 augmentations: List[str] = None):
        self.model  = model
        self.device = device
        self.augmentations = augmentations or [
            "original", "flip", "reverse", "flip_reverse"
        ]
        self.weights = {
            "original":     1.0,
            "flip":         1.0,
            "reverse":      0.8,
            "flip_reverse": 0.8,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, T, V, M)
        Returns averaged sigmoid probabilities: (N, num_classes)
        """
        x = x.to(self.device)
        total_weight = 0.0
        total_probs  = None

        for aug_name in self.augmentations:
            aug_x = self._apply(x, aug_name)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = self.model(aug_x)
            probs = torch.sigmoid(logits)   # (N, num_classes)

            w = self.weights[aug_name]
            if total_probs is None:
                total_probs = probs * w
            else:
                total_probs = total_probs + probs * w
            total_weight += w

        return total_probs / total_weight

    def _apply(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if name == "original":
            return x
        elif name == "flip":
            return apply_horizontal_flip(x)
        elif name == "reverse":
            return apply_temporal_reverse(x)
        elif name == "flip_reverse":
            return apply_temporal_reverse(apply_horizontal_flip(x))
        else:
            raise ValueError(f"Unknown augmentation: {name}")
