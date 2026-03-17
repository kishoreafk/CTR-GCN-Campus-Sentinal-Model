"""
Asymmetric Loss is the default for both phases — better than standard BCE
for AVA's severe positive/negative imbalance in multi-label setting.
Reference: 'Asymmetric Loss For Multi-Label Classification' (Ben-Baruch 2021)
"""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional

class AsymmetricLoss(nn.Module):
    """
    gamma_neg=4, gamma_pos=0, clip=0.05 are strong defaults for AVA.
    clip: probability margin to shift down hard negatives.
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Asymmetric clip: shift hard negatives
        if self.clip > 0:
            probs = (probs + self.clip).clamp(max=1.0)

        p_m = probs     * targets     + (1 - probs) * (1 - targets)
        log_p = torch.log(p_m.clamp(min=1e-8))

        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        weight = (1 - p_m) ** gamma

        loss = -(weight * log_p)
        return loss.mean()


class WeightedBCELoss(nn.Module):
    """Standard BCE with per-class pos_weight (fallback option)."""
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight)


def build_loss(config, train_dataset=None, device="cuda"):
    if config.loss_type == "asymmetric":
        return AsymmetricLoss(
            gamma_neg = config.asymmetric_gamma_neg,
            gamma_pos = config.asymmetric_gamma_pos,
            clip      = config.asymmetric_clip,
        )
    elif config.loss_type == "bce":
        pw = None
        if train_dataset is not None:
            # CRITICAL: move pos_weight to same device as model
            pw = train_dataset.get_class_pos_weights().to(device)
        return WeightedBCELoss(pos_weight=pw)
    else:
        raise ValueError(f"Unknown loss_type: {config.loss_type}")
