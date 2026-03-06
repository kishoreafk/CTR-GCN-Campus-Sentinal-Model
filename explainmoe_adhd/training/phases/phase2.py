"""
Phase 2: Supervised Encoder Training for ExplainMoE-ADHD v2.13.

Per-encoder supervised training on Group A (ADHD-labeled) data.
Uses temporary MLP diagnosis heads that are discarded after this phase.
Early stopping on per-encoder val_auroc with patience=10 epochs.

LR: 5e-4, AdamW, weight_decay=1e-3
Max epochs: 50
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_auc_score

from explainmoe_adhd.training.phases.base_phase import BasePhase
from explainmoe_adhd.training.losses import DiagnosisLoss


class Phase2SupervisedEncoder(BasePhase):
    """Supervised encoder training with temporary diagnosis heads.

    Encoder backbones are trainable; all other components frozen.
    Eye-tracking encoder remains permanently frozen.
    Uses a temporary MLP head (discarded after this phase).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/phase2",
    ):
        super().__init__(model, config, device, checkpoint_dir)
        self.diagnosis_loss = DiagnosisLoss()

        # Temporary MLP heads per modality (discarded after Phase 2)
        latent_dim = getattr(model, 'latent_dim', 256)
        self.temp_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
            )
            for name in getattr(model, 'MODALITY_NAMES', [
                'child_eeg_19ch', 'child_eeg_10ch', 'adult_eeg_5ch',
                'clinical', 'actigraphy',
            ])
        }).to(device)

    def is_maximize(self) -> bool:
        return True  # maximize val_auroc

    def get_monitor_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get("val_auroc", 0.0)

    def forward_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step: encoder → temporary MLP head."""
        modality = batch.get("modality", "clinical")
        if isinstance(modality, list):
            modality = modality[0]

        # Encoder only — do NOT go through projection/FuseMoE/task heads
        h_m = self.model.encode(batch, modality)
        diagnosis_logits = self.temp_heads[modality](h_m)
        targets = batch["label"].float()

        loss = self.diagnosis_loss(diagnosis_logits, targets)

        probs = torch.sigmoid(diagnosis_logits.detach()).squeeze(-1)
        preds = (probs > 0.5).float()
        acc = (preds == targets.squeeze(-1)).float().mean().item()

        return loss, {
            "diagnosis_loss": loss.item(),
            "accuracy": acc,
        }

    def evaluate_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate: compute AUROC on validation batch."""
        with torch.no_grad():
            modality = batch.get("modality", "clinical")
            if isinstance(modality, list):
                modality = modality[0]

            h_m = self.model.encode(batch, modality)
            diagnosis_logits = self.temp_heads[modality](h_m)
            targets = batch["label"].float()

            loss = self.diagnosis_loss(diagnosis_logits, targets)
            probs = torch.sigmoid(diagnosis_logits).squeeze(-1).cpu().numpy()
            labels = targets.squeeze(-1).cpu().numpy()

            try:
                auroc = roc_auc_score(labels, probs)
            except ValueError:
                auroc = 0.5

        return {
            "val_loss": loss.item(),
            "val_auroc": auroc,
        }
