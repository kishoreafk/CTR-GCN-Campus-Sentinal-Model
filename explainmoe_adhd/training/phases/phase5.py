"""
Phase 5: Joint Fine-Tuning for ExplainMoE-ADHD v2.13.

End-to-end fine-tuning of all components at low LR (1e-5).
Eye-tracking encoder remains permanently frozen.

All losses active:
  L_CE(diag) w=1.0, L_CE(sub) w=0.25, L_MSE(sev) w=0.1,
  L_MMD w=0.3, L_balance w=0.01

Max steps: 1000, eval every 50, patience 10 eval cycles
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_auc_score

from explainmoe_adhd.training.phases.base_phase import BasePhase
from explainmoe_adhd.training.losses import CombinedLoss


class Phase5FineTuning(BasePhase):
    """Joint fine-tuning phase.

    All components trainable at 1e-5 (except eye-tracking encoder).
    All losses active including severity and MMD regularization.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/phase5",
    ):
        super().__init__(model, config, device, checkpoint_dir)

        self.loss_fn = CombinedLoss(
            diagnosis_weight=config.get("diagnosis_loss_weight", 1.0),
            subtype_weight=config.get("subtype_loss_weight", 0.25),
            severity_weight=config.get("severity_loss_weight", 0.1),
            mmd_weight=config.get("mmd_loss_weight", 0.3),
            load_balance_weight=config.get("load_balance_weight", 0.01),
            use_subtype=True,
            use_severity=True,
            use_mmd=True,
        )

    def is_maximize(self) -> bool:
        return True  # maximize val_auroc_macro

    def get_monitor_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get("val_auroc_macro", 0.0)

    def forward_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step with all losses active."""
        modality = batch.get("modality", "clinical")
        if isinstance(modality, list):
            modality = modality[0]

        outputs = self.model(batch, modality)

        diagnosis_logits = outputs["diagnosis_logits"]
        subtype_logits = outputs["subtype_logits"]
        severity_preds = outputs["severity_preds"]
        expert_weights = outputs.get("expert_weights", None)

        targets = batch["label"].float()
        subtype_targets = batch.get("subtype_label", None)
        subtype_mask = batch.get("subtype_mask", None)
        severity_targets = batch.get("severity_targets", None)
        severity_mask = batch.get("severity_mask", None)
        mmd_pairs = batch.get("mmd_pairs", None)

        total_loss, loss_dict = self.loss_fn(
            diagnosis_logits=diagnosis_logits,
            diagnosis_targets=targets,
            subtype_logits=subtype_logits,
            subtype_targets=subtype_targets,
            subtype_mask=subtype_mask,
            severity_preds=severity_preds,
            severity_targets=severity_targets,
            severity_mask=severity_mask,
            mmd_pairs=mmd_pairs,
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
            "val_auroc_macro": auroc,
        }
