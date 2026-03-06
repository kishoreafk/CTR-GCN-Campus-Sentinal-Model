"""
Inference script for ExplainMoE-ADHD v2.13.

Loads a trained model and runs inference on new data.
Produces predictions with confidence scores.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
) -> nn.Module:
    """Load a trained ExplainMoE model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pt file
        device: Device to load model onto

    Returns:
        Loaded model in eval mode
    """
    from explainmoe_adhd.models.explainmoe_model import ExplainMoEModel

    model = ExplainMoEModel()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def predict(
    model: nn.Module,
    batch: Dict[str, Any],
    modality: str,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Run inference on a single batch.

    Args:
        model: Trained ExplainMoE model
        batch: Input data dict
        modality: Modality name
        device: Device string

    Returns:
        Dict with predictions:
          - diagnosis_prob: P(ADHD) per subject
          - subtype_probs: P(Combined, HI, Inattentive) per subject
          - severity_scores: [inattentive, hyperactive] per subject
    """
    model.eval()
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    with torch.no_grad():
        outputs = model(batch, modality, return_intermediates=True)

    diagnosis_prob = torch.sigmoid(outputs["diagnosis_logits"]).squeeze(-1).cpu().numpy()
    subtype_probs = torch.softmax(outputs["subtype_logits"], dim=-1).cpu().numpy()
    severity_scores = outputs["severity_preds"].cpu().numpy()

    result = {
        "diagnosis_prob": diagnosis_prob,
        "subtype_probs": subtype_probs,
        "severity_scores": severity_scores,
    }

    if "y_moe" in outputs:
        result["embeddings"] = outputs["y_moe"].cpu().numpy()

    return result


def predict_single(
    model: nn.Module,
    data: Dict[str, Any],
    modality: str,
    device: str = "cuda",
) -> Dict[str, float]:
    """Run inference on a single subject.

    Args:
        model: Trained ExplainMoE model
        data: Single subject data dict (no batch dim)
        modality: Modality name
        device: Device string

    Returns:
        Dict with scalar predictions
    """
    # Add batch dimension
    batch = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)
        else:
            batch[k] = v

    result = predict(model, batch, modality, device)

    return {
        "diagnosis_prob": float(result["diagnosis_prob"][0]),
        "adhd_prediction": bool(result["diagnosis_prob"][0] > 0.5),
        "subtype_probs": {
            "combined": float(result["subtype_probs"][0, 0]),
            "hyperactive_impulsive": float(result["subtype_probs"][0, 1]),
            "inattentive": float(result["subtype_probs"][0, 2]),
        },
        "severity_scores": {
            "inattentive": float(result["severity_scores"][0, 0]),
            "hyperactive": float(result["severity_scores"][0, 1]),
        },
    }
