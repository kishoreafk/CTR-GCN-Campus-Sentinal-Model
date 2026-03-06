"""
Ablation Baselines for ExplainMoE-ADHD v2.13 (Section 10, Table 9).

Condition A — Baseline_M:
    Single-modality encoder → diagnosis MLP (no FuseMoE, no alignment).
    Trained with Phase 1 + Phase 2 only.

Condition B — MoE_M_only:
    Single-modality through FuseMoE WITHOUT Phase 3 alignment.
    Skips Phase 3; projection heads use full LR in Phase 4;
    FuseMoE centroids initialized randomly (no K-means).

Condition D — Simple_M:
    Classical ML baseline using sklearn. Not a torch model.
    Train logistic regression / gradient boosting on hand-crafted features.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from explainmoe_adhd.models.explainmoe_model import ExplainMoEModel


# =============================================================================
# Condition A: Baseline_M (encoder + MLP, no FuseMoE)
# =============================================================================

class BaselineModel(nn.Module):
    """Condition A: Single-modality encoder → MLP diagnosis head.

    No projection heads, no FuseMoE, no alignment.
    Serves as the lower-bound baseline.
    """

    MODALITY_NAMES = ExplainMoEModel.MODALITY_NAMES

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        from explainmoe_adhd.models.components.eeg_encoders import (
            ChildEEG19chEncoder, ChildEEG10chEncoder, AdultEEG5chEncoder,
        )
        from explainmoe_adhd.models.components.clinical_encoder import ClinicalEncoder
        from explainmoe_adhd.models.components.actigraphy_encoder import ActigraphyEncoder

        self.latent_dim = latent_dim
        self.encoders = nn.ModuleDict({
            "child_eeg_19ch": ChildEEG19chEncoder(),
            "child_eeg_10ch": ChildEEG10chEncoder(),
            "adult_eeg_5ch": AdultEEG5chEncoder(),
            "clinical": ClinicalEncoder(),
            "actigraphy": ActigraphyEncoder(),
        })

        # Per-modality diagnosis MLP
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
            )
            for name in self.MODALITY_NAMES
        })

    def encode(self, batch: Dict[str, Any], modality: str) -> torch.Tensor:
        """Delegate to the same encode logic as ExplainMoEModel."""
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
            if isinstance(tabular, dict):
                cat = tabular["categorical"].float()
                cont = tabular["continuous"]
                tabular = torch.cat([
                    cont[:, 0:1], cat[:, 0:1], cat[:, 1:2],
                    cont[:, 1:2], cat[:, 2:3], cat[:, 3:4],
                ], dim=-1)
            return self.encoders["clinical"](tabular, fmri, mean_fd)
        elif modality == "actigraphy":
            ts = batch["timeseries"]
            age = batch.get("age", torch.zeros(ts.size(0)))
            sex = batch.get("sex", torch.zeros(ts.size(0))).float()
            return self.encoders["actigraphy"](ts, age, sex)
        raise ValueError(f"Unknown modality: {modality}")

    def forward(self, batch: Dict[str, Any], modality: str) -> Dict[str, torch.Tensor]:
        h_m = self.encode(batch, modality)
        logits = self.heads[modality](h_m)
        return {"diagnosis_logits": logits}


# =============================================================================
# Condition B: MoE_M_only (FuseMoE without Phase 3 alignment)
# =============================================================================

class MoEWithoutAlignmentModel(ExplainMoEModel):
    """Condition B: Full FuseMoE pipeline WITHOUT Phase 3 alignment.

    Differences from the main ExplainMoE pipeline:
    - Phase 3 is entirely skipped.
    - Projection heads trained at full LR in Phase 4 (not 0.1×).
    - FuseMoE centroids stay random (no K-means init).
    - MMD loss disabled in Phases 4 and 5.
    """

    def configure_for_phase(self, phase: int):
        """Condition B skips Phase 3 entirely."""
        if phase == 3:
            raise ValueError("Condition B does not use Phase 3")
        super().configure_for_phase(phase)

    def get_parameter_groups(self, phase: int) -> List[Dict]:
        """In Phase 4, projection heads use the same LR as FuseMoE (full LR)."""
        if phase == 4:
            return [
                {"params": [p for p in self.fusemoe.parameters() if p.requires_grad], "lr": 3e-4},
                {"params": [p for p in self.diagnosis_head.parameters() if p.requires_grad], "lr": 3e-4},
                {"params": [p for p in self.subtype_head.parameters() if p.requires_grad], "lr": 3e-4},
                {"params": [p for p in self.severity_head.parameters() if p.requires_grad], "lr": 3e-4},
                {"params": [p for head in self.projection_heads.values()
                            for p in head.parameters() if p.requires_grad], "lr": 3e-4},
            ]
        return super().get_parameter_groups(phase)


# =============================================================================
# Condition D: Simple_M (classical ML)
# =============================================================================

class SimpleMLBaseline:
    """Condition D: Classical ML baseline (not a torch model).

    Uses sklearn LogisticRegression or GradientBoosting on hand-crafted
    summary features. Call fit() / predict() like a normal sklearn API.
    """

    def __init__(self, model_type: str = "logistic_regression"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier

        if model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000, solver="lbfgs")
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
