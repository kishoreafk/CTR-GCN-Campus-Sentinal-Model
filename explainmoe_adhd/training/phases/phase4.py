"""
Phase 4: FuseMoE Training for ExplainMoE-ADHD v2.13.

Joint training of FuseMoE + task heads across all modalities.
Encoders are frozen. Projection heads trainable at 0.1x main LR.

LR: 3e-4 (main), 3e-5 (projection heads)
Max steps: 3000, eval every 100, patience 10 eval cycles
Batch: 80 (16 per modality x 5)

Active losses: L_CE(diag) w=1.0, L_CE(sub) w=0.25, L_balance w=0.01
Inactive: severity, MMD
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_auc_score

from explainmoe_adhd.training.phases.base_phase import BasePhase
from explainmoe_adhd.training.losses import CombinedLoss


class Phase4FuseMoE(BasePhase):
    """FuseMoE training phase.

    Trains FuseMoE module + task heads with modality-balanced batches.
    Encoders are frozen; projection heads at 0.1x LR.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/phase4",
    ):
        super().__init__(model, config, device, checkpoint_dir)

        self.loss_fn = CombinedLoss(
            diagnosis_weight=config.get("diagnosis_loss_weight", 1.0),
            subtype_weight=config.get("subtype_loss_weight", 0.25),
            severity_weight=0.0,  # INACTIVE in Phase 4
            mmd_weight=0.0,      # INACTIVE in Phase 4
            load_balance_weight=config.get("load_balance_weight", 0.01),
            use_subtype=True,
            use_severity=False,
            use_mmd=False,
        )

    def is_maximize(self) -> bool:
        return True  # maximize val_auroc_macro

    def get_monitor_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get("val_auroc_macro", 0.0)

    def forward_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step on a modality-balanced batch."""
        modality = batch.get("modality", "clinical")
        if isinstance(modality, list):
            modality = modality[0]

        outputs = self.model(batch, modality)

        diagnosis_logits = outputs["diagnosis_logits"]
        subtype_logits = outputs["subtype_logits"]
        expert_weights = outputs.get("expert_weights", None)

        targets = batch["label"].float()
        subtype_targets = batch.get("subtype_label", None)
        subtype_mask = batch.get("subtype_mask", None)

        total_loss, loss_dict = self.loss_fn(
            diagnosis_logits=diagnosis_logits,
            diagnosis_targets=targets,
            subtype_logits=subtype_logits,
            subtype_targets=subtype_targets,
            subtype_mask=subtype_mask,
            expert_weights=expert_weights,
        )

        return total_loss, loss_dict

    def evaluate_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate: compute AUROC on validation batch."""
        with torch.no_grad():
            modality = batch.get("modality", "clinical")
            if isinstance(modality, list):
                modality = modality[0]

            outputs = self.model(batch, modality)
            diagnosis_logits = outputs["diagnosis_logits"]
            targets = batch["label"].float()

            probs = torch.sigmoid(diagnosis_logits).squeeze(-1).cpu().numpy()
            labels = targets.squeeze(-1).cpu().numpy()

            try:
                auroc = roc_auc_score(labels, probs)
            except ValueError:
                auroc = 0.5

        return {
            "val_auroc": auroc,
            "val_auroc_macro": auroc,  # Single modality; macro computed externally
        }
