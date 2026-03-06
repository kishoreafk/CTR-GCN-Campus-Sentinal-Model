"""
Phase 3: MMD Latent Space Alignment for ExplainMoE-ADHD v2.13.

Cross-modal alignment of projection heads via class-conditional MMD.
Only projection heads are trainable; all encoders are frozen.

4 default MMD pairs (Section 6, Phase 3):
  1. child_eeg_19ch ↔ child_eeg_10ch
  2. child_eeg_19ch ↔ clinical
  3. child_eeg_10ch ↔ clinical
  4. adult_eeg_5ch  ↔ actigraphy

LR: 1e-4
Max steps: 1000, eval every 50, patience 10 eval cycles
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List

from explainmoe_adhd.training.phases.base_phase import BasePhase
from explainmoe_adhd.training.losses import MMDLoss


class Phase3MMDAlignment(BasePhase):
    """MMD latent space alignment phase.

    Trains projection heads to align latent spaces across modalities.
    Encoders are frozen; only projection heads g_m are trainable.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/phase3",
    ):
        super().__init__(model, config, device, checkpoint_dir)
        self.mmd_loss = MMDLoss(
            kernel=config.get("kernel_type", "multibandwidth_rbf"),
            min_batch_per_class=config.get("min_batch_per_class", 16),
        )

    def is_maximize(self) -> bool:
        return False  # minimize MMD loss

    def get_monitor_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get("mean_mmd_loss", float("inf"))

    def forward_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step: compute MMD on one cross-modal pair."""
        # batch should contain data from two modalities, split by class
        mod_a = batch.get("modality_a", "child_eeg_19ch")
        mod_b = batch.get("modality_b", "child_eeg_10ch")

        if isinstance(mod_a, list):
            mod_a = mod_a[0]
        if isinstance(mod_b, list):
            mod_b = mod_b[0]

        # Encode modality A
        with torch.no_grad():
            h_a = self.model.encode(batch, mod_a)
        z_a = self.model.projection_heads[mod_a](h_a)

        # Encode modality B
        with torch.no_grad():
            h_b = self.model.encode(batch, mod_b)
        z_b = self.model.projection_heads[mod_b](h_b)

        # Split by class
        labels = batch["label"]
        adhd_mask = labels == 1
        ctrl_mask = labels == 0

        # Separate per-modality labels if provided
        labels_a = batch.get("labels_a", labels)
        labels_b = batch.get("labels_b", labels)

        if "z_a_adhd" in batch:
            # Pre-split data provided
            z_a_adhd = batch["z_a_adhd"]
            z_b_adhd = batch["z_b_adhd"]
            z_a_ctrl = batch["z_a_ctrl"]
            z_b_ctrl = batch["z_b_ctrl"]
        else:
            # Split by actual class labels
            adhd_mask_a = (labels_a == 1)
            ctrl_mask_a = (labels_a == 0)
            adhd_mask_b = (labels_b == 1)
            ctrl_mask_b = (labels_b == 0)
            z_a_adhd = z_a[adhd_mask_a]
            z_a_ctrl = z_a[ctrl_mask_a]
            z_b_adhd = z_b[adhd_mask_b]
            z_b_ctrl = z_b[ctrl_mask_b]

        loss = self.mmd_loss(z_a_adhd, z_b_adhd, z_a_ctrl, z_b_ctrl)

        return loss, {"mmd_loss": loss.item()}

    def evaluate_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate: compute MMD on validation data."""
        with torch.no_grad():
            _, metrics = self.forward_step(batch)
        return {"mean_mmd_loss": metrics["mmd_loss"]}
