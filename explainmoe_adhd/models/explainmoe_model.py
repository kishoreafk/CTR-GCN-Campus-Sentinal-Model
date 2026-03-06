"""
ExplainMoE Model — Main Model Class for ExplainMoE-ADHD v2.13.

Assembles all components:
  Encoders → Projection Heads → FuseMoE → Task Heads

Supports all 5 training phases with appropriate freezing/unfreezing.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any

from explainmoe_adhd.models.components.eeg_encoders import (
    ChildEEG19chEncoder,
    ChildEEG10chEncoder,
    AdultEEG5chEncoder,
)
from explainmoe_adhd.models.components.clinical_encoder import ClinicalEncoder
from explainmoe_adhd.models.components.actigraphy_encoder import ActigraphyEncoder
from explainmoe_adhd.models.components.eye_tracking_encoder import EyeTrackingEncoder
from explainmoe_adhd.models.components.projection_heads import ProjectionHead
from explainmoe_adhd.models.components.fusemoe import FuseMoE
from explainmoe_adhd.models.components.task_heads import (
    DiagnosisHead,
    SubtypeHead,
    SeverityHead,
)
from explainmoe_adhd.config.dataset_configs import MODALITY_ROUTER_INDEX


class ExplainMoEModel(nn.Module):
    """
    Full ExplainMoE-ADHD model.

    Data flow:
        raw input → encoder f_m → h_m ∈ ℝ^256
        h_m → projection head g_m → z_m ∈ ℝ^256
        z_m → FuseMoE(router_m) → y_moe ∈ ℝ^256
        y_moe → task heads → predictions

    The model can operate in different modes depending on the training phase:
        Phase 1: Only encoders (self-supervised pretraining, handled externally)
        Phase 2: encoder + temporary diagnosis head (per-encoder)
        Phase 3: projection heads only (MMD alignment, encoders frozen)
        Phase 4: FuseMoE + task heads (encoders frozen, projection 0.1× LR)
        Phase 5: All components at 1e-5 (except eye-tracking, permanently frozen)
    """

    MODALITY_NAMES = [
        "child_eeg_19ch",
        "child_eeg_10ch",
        "adult_eeg_5ch",
        "clinical",
        "actigraphy",
    ]

    def __init__(
        self,
        latent_dim: int = 256,
        num_experts: int = 4,
        top_k: int = 2,
        projection_depth: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # ---- Encoders ----
        self.encoders = nn.ModuleDict({
            "child_eeg_19ch": ChildEEG19chEncoder(),
            "child_eeg_10ch": ChildEEG10chEncoder(),
            "adult_eeg_5ch": AdultEEG5chEncoder(),
            "clinical": ClinicalEncoder(),
            "actigraphy": ActigraphyEncoder(),
        })
        # Eye-tracking encoder (proof-of-concept, permanently frozen)
        self.eye_tracking_encoder = EyeTrackingEncoder()

        # ---- Projection Heads (one per modality, 5 total) ----
        self.projection_heads = nn.ModuleDict({
            name: ProjectionHead(
                input_dim=latent_dim,
                output_dim=latent_dim,
                depth=projection_depth,
            )
            for name in self.MODALITY_NAMES
        })

        # ---- FuseMoE ----
        self.fusemoe = FuseMoE(
            input_dim=latent_dim,
            num_experts=num_experts,
            top_k=top_k,
            num_routers=len(self.MODALITY_NAMES),
        )

        # ---- Task Heads ----
        self.diagnosis_head = DiagnosisHead(input_dim=latent_dim)
        self.subtype_head = SubtypeHead(input_dim=latent_dim)
        self.severity_head = SeverityHead(input_dim=latent_dim)

        # Router index mapping
        self.router_index = dict(MODALITY_ROUTER_INDEX)

    # ------------------------------------------------------------------
    # Encoder forward pass (modality → h_m)
    # ------------------------------------------------------------------
    def encode(self, batch: Dict[str, Any], modality: str) -> torch.Tensor:
        """Run the appropriate encoder for a given modality.

        Args:
            batch: dict with modality-specific data tensors
            modality: one of MODALITY_NAMES

        Returns:
            h_m: (B, latent_dim)
        """
        if modality in ("child_eeg_19ch", "child_eeg_10ch", "adult_eeg_5ch"):
            eeg_data = batch["eeg"]
            hw_id = batch.get("hardware_token_id", None)
            if hw_id is not None and isinstance(hw_id, torch.Tensor):
                hw_id = int(hw_id[0].item())
            return self.encoders[modality](eeg_data, hw_id=hw_id)

        elif modality == "clinical":
            tabular = batch.get("tabular", None)
            fmri = batch.get("fmri", None)
            mean_fd = batch.get("mean_fd", None)
            # ClinicalEncoder expects flat tensor (B, 6): [age, sex, handedness, IQ, site_id, source]
            if isinstance(tabular, dict):
                cat = tabular["categorical"].float()   # (B, 4): sex, handedness, site, source
                cont = tabular["continuous"]            # (B, 2): age, IQ
                # Reorder to [age, sex, handedness, IQ, site_id, dataset_source]
                tabular = torch.cat([
                    cont[:, 0:1],   # age
                    cat[:, 0:1],    # sex
                    cat[:, 1:2],    # handedness
                    cont[:, 1:2],   # IQ
                    cat[:, 2:3],    # site_id
                    cat[:, 3:4],    # dataset_source
                ], dim=-1)
            return self.encoders["clinical"](tabular, fmri, mean_fd)

        elif modality == "actigraphy":
            timeseries = batch["timeseries"]
            age = batch.get("age", torch.zeros(timeseries.size(0)))
            sex = batch.get("sex", torch.zeros(timeseries.size(0))).float()
            return self.encoders["actigraphy"](timeseries, age, sex)

        else:
            raise ValueError(f"Unknown modality: {modality}")

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        batch: Dict[str, Any],
        modality: str,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encoder → projection → FuseMoE → task heads.

        Args:
            batch: data dict for one modality
            modality: modality name
            return_intermediates: if True, also return h_m and z_m

        Returns:
            Dict with keys: diagnosis_logits, subtype_logits, severity_preds,
                            expert_weights, balance_loss, and optionally h_m, z_m, y_moe
        """
        # Encoder
        h_m = self.encode(batch, modality)

        # Projection head
        z_m = self.projection_heads[modality](h_m)

        # FuseMoE
        router_idx = self.router_index[modality]
        y_moe, expert_weights = self.fusemoe(z_m, router_idx, return_expert_weights=True)

        # Task heads
        diagnosis_logits = self.diagnosis_head(y_moe)
        subtype_logits = self.subtype_head(y_moe)
        severity_preds = self.severity_head(y_moe)

        # Balance loss
        balance_loss = self.fusemoe.load_balance_loss(expert_weights) if expert_weights is not None else torch.tensor(0.0)

        outputs = {
            "diagnosis_logits": diagnosis_logits,
            "subtype_logits": subtype_logits,
            "severity_preds": severity_preds,
            "expert_weights": expert_weights,
            "balance_loss": balance_loss,
        }

        if return_intermediates:
            outputs["h_m"] = h_m
            outputs["z_m"] = z_m
            outputs["y_moe"] = y_moe

        return outputs

    # ------------------------------------------------------------------
    # Phase-specific freeze/unfreeze
    # ------------------------------------------------------------------
    def freeze_encoders(self):
        """Freeze all encoder backbones."""
        for encoder in self.encoders.values():
            for p in encoder.parameters():
                p.requires_grad = False
        for p in self.eye_tracking_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self, exclude_eye_tracking: bool = True):
        """Unfreeze encoder backbones (except eye-tracking by default)."""
        for encoder in self.encoders.values():
            for p in encoder.parameters():
                p.requires_grad = True
        if not exclude_eye_tracking:
            for p in self.eye_tracking_encoder.parameters():
                p.requires_grad = True

    def freeze_projection_heads(self):
        for head in self.projection_heads.values():
            for p in head.parameters():
                p.requires_grad = False

    def unfreeze_projection_heads(self):
        for head in self.projection_heads.values():
            for p in head.parameters():
                p.requires_grad = True

    def freeze_fusemoe(self):
        for p in self.fusemoe.parameters():
            p.requires_grad = False

    def unfreeze_fusemoe(self):
        for p in self.fusemoe.parameters():
            p.requires_grad = True

    def freeze_task_heads(self):
        for p in self.diagnosis_head.parameters():
            p.requires_grad = False
        for p in self.subtype_head.parameters():
            p.requires_grad = False
        for p in self.severity_head.parameters():
            p.requires_grad = False

    def unfreeze_task_heads(self):
        for p in self.diagnosis_head.parameters():
            p.requires_grad = True
        for p in self.subtype_head.parameters():
            p.requires_grad = True
        for p in self.severity_head.parameters():
            p.requires_grad = True

    def configure_for_phase(self, phase: int):
        """Configure trainable parameters for a specific phase.

        Phase 2: encoders trainable, everything else frozen
        Phase 3: only projection heads trainable
        Phase 4: FuseMoE + task heads + projection (0.1× LR), encoders frozen
        Phase 5: everything trainable at 1e-5 (except eye-tracking)
        """
        if phase == 2:
            self.unfreeze_encoders(exclude_eye_tracking=True)
            # Explicitly freeze eye-tracking (always frozen)
            for p in self.eye_tracking_encoder.parameters():
                p.requires_grad = False
            self.freeze_projection_heads()
            self.freeze_fusemoe()
            self.freeze_task_heads()

        elif phase == 3:
            self.freeze_encoders()
            self.unfreeze_projection_heads()
            self.freeze_fusemoe()
            self.freeze_task_heads()

        elif phase == 4:
            self.freeze_encoders()
            self.unfreeze_projection_heads()  # At 0.1× LR
            self.unfreeze_fusemoe()
            self.unfreeze_task_heads()

        elif phase == 5:
            self.unfreeze_encoders(exclude_eye_tracking=True)
            # Explicitly freeze eye-tracking (always frozen)
            for p in self.eye_tracking_encoder.parameters():
                p.requires_grad = False
            self.unfreeze_projection_heads()
            self.unfreeze_fusemoe()
            self.unfreeze_task_heads()

    def get_parameter_groups(self, phase: int) -> List[Dict]:
        """Get parameter groups with phase-appropriate learning rates.

        Returns list suitable for torch.optim optimizer.
        """
        if phase == 2:
            return [
                {"params": [p for enc in self.encoders.values() for p in enc.parameters() if p.requires_grad]},
                {"params": [p for p in self.diagnosis_head.parameters() if p.requires_grad]},
            ]

        elif phase == 3:
            return [
                {"params": [p for head in self.projection_heads.values() for p in head.parameters() if p.requires_grad]},
            ]

        elif phase == 4:
            return [
                {"params": [p for p in self.fusemoe.parameters() if p.requires_grad], "lr": 3e-4},
                {"params": [p for p in self.diagnosis_head.parameters() if p.requires_grad], "lr": 3e-4},
                {"params": [p for p in self.subtype_head.parameters() if p.requires_grad], "lr": 3e-4},
                {"params": [p for p in self.severity_head.parameters() if p.requires_grad], "lr": 3e-4},
                {"params": [p for head in self.projection_heads.values() for p in head.parameters() if p.requires_grad], "lr": 3e-5},
            ]

        elif phase == 5:
            all_params = [p for p in self.parameters() if p.requires_grad]
            return [{"params": all_params, "lr": 1e-5}]

        return []

    def reinitialize_projection_heads(self):
        """Re-initialize projection heads with Xavier uniform (Phase 3 start)."""
        for head in self.projection_heads.values():
            head._init_weights()

    def reinitialize_task_heads(self):
        """Re-initialize all task heads (Phase 4 start — fresh diagnosis head)."""
        for module in [self.diagnosis_head, self.subtype_head, self.severity_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
