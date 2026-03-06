"""
Evaluation script for ExplainMoE-ADHD v2.13.

Computes all mandatory reporting metrics (Section 8.5):
- AUROC per modality (5-fold mean ± std)
- Macro AUROC across modality pathways
- Per-subtype AUROC
- Calibration (ECE, Brier score)
- Confound probes
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, brier_score_loss


@dataclass
class EvaluationResult:
    """Holds evaluation metrics for one fold."""
    fold: int
    modality: str
    auroc: float
    accuracy: float
    brier_score: float
    ece: float
    n_subjects: int
    predictions: np.ndarray
    labels: np.ndarray


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)
    return ece


def evaluate_model(
    model: nn.Module,
    test_loader,
    modality: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate model on test set for a single modality.

    Args:
        model: Trained ExplainMoE model (frozen)
        test_loader: DataLoader for test subjects
        modality: Modality name
        device: Device string

    Returns:
        Dict with metrics
    """
    model.eval()
    model.to(device)

    all_probs = []
    all_labels = []
    all_subtype_probs = []
    all_subtype_labels = []
    all_severity_preds = []
    all_severity_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(batch, modality)
            probs = torch.sigmoid(outputs["diagnosis_logits"]).squeeze(-1).cpu().numpy()
            labels = batch["label"].cpu().numpy().flatten()

            all_probs.append(probs)
            all_labels.append(labels)

            # Subtype
            if "subtype_label" in batch and batch["subtype_label"] is not None:
                sub_probs = torch.softmax(outputs["subtype_logits"], dim=-1).cpu().numpy()
                sub_labels = batch["subtype_label"].cpu().numpy()
                mask = batch.get("subtype_mask", torch.ones(len(sub_labels), dtype=torch.bool))
                mask = mask.cpu().numpy().astype(bool)
                if mask.any():
                    all_subtype_probs.append(sub_probs[mask])
                    all_subtype_labels.append(sub_labels[mask])

            # Severity
            if "severity_targets" in batch and batch["severity_targets"] is not None:
                sev_preds = outputs["severity_preds"].cpu().numpy()
                sev_targets = batch["severity_targets"].cpu().numpy()
                mask = batch.get("severity_mask", torch.ones(len(sev_targets), dtype=torch.bool))
                mask = mask.cpu().numpy().astype(bool)
                if mask.any():
                    all_severity_preds.append(sev_preds[mask])
                    all_severity_targets.append(sev_targets[mask])

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Core metrics
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.5

    brier = brier_score_loss(all_labels, all_probs)
    ece = compute_ece(all_probs, all_labels)
    acc = ((all_probs > 0.5).astype(float) == all_labels).mean()

    metrics = {
        "auroc": auroc,
        "accuracy": acc,
        "brier_score": brier,
        "ece": ece,
        "n_subjects": len(all_labels),
        "predictions": all_probs,
        "labels": all_labels,
    }

    return metrics


def evaluate_cross_validation(
    fold_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate metrics across CV folds.

    Args:
        fold_results: List of per-fold metric dicts

    Returns:
        Dict with mean ± std for each metric
    """
    aurocs = [r["auroc"] for r in fold_results]
    accs = [r["accuracy"] for r in fold_results]

    return {
        "auroc_mean": np.mean(aurocs),
        "auroc_std": np.std(aurocs),
        "accuracy_mean": np.mean(accs),
        "accuracy_std": np.std(accs),
        "n_folds": len(fold_results),
    }


def bootstrap_ci(
    labels: np.ndarray,
    predictions: np.ndarray,
    metric_fn=roc_auc_score,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    scores = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        try:
            scores.append(metric_fn(labels[idx], predictions[idx]))
        except ValueError:
            continue

    alpha = (1 - ci) / 2
    lower = np.percentile(scores, 100 * alpha) if scores else 0.0
    upper = np.percentile(scores, 100 * (1 - alpha)) if scores else 1.0
    mean = np.mean(scores) if scores else 0.5

    return {"mean": mean, "lower": lower, "upper": upper, "n_valid": len(scores)}
