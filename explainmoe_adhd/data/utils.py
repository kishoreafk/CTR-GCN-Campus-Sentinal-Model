"""
Data utility functions for ExplainMoE-ADHD v2.13.

Collation, batching helpers, and data integrity checks.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional


def collate_modality_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for mixed-modality batches.

    Groups items by modality and stacks tensors where shapes match.
    Non-tensor fields (subject_id, modality, dataset_id) are kept as lists.
    """
    collated: Dict[str, Any] = {
        "subject_id": [item["subject_id"] for item in batch],
        "modality": [item["modality"] for item in batch],
        "dataset_id": [item["dataset_id"] for item in batch],
    }

    # Stack scalar tensors
    scalar_keys = [
        "label", "has_subtype", "subtype", "has_severity",
        "age", "sex", "site_id", "dataset_source", "hardware_token_id", "mean_fd",
    ]
    for key in scalar_keys:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])

    # Stack severity (fixed size 2)
    if "severity" in batch[0]:
        collated["severity"] = torch.stack([item["severity"] for item in batch])

    # Modality-specific data: group by modality name
    data_keys = set()
    for item in batch:
        for k in item:
            if k not in scalar_keys and k not in ("subject_id", "modality", "dataset_id", "severity"):
                data_keys.add(k)

    for key in data_keys:
        tensors = [item[key] for item in batch if key in item and isinstance(item[key], torch.Tensor)]
        if tensors and all(t.shape == tensors[0].shape for t in tensors):
            collated[key] = torch.stack(tensors)
        elif tensors:
            # Variable shapes — pad to max
            collated[key] = tensors  # Keep as list; specific handling in model

    return collated


def verify_no_tier3_leakage(feature_names: List[str]) -> bool:
    """
    Verify that no Tier 3 (label-proximate) features are used as input.

    Raises ValueError if DX, ADHD_index, inattentive_score, or hyperactive_score
    appear in the feature list.
    """
    TIER_3 = {"DX", "ADHD_index", "inattentive_score", "hyperactive_score",
              "dx", "adhd_index"}
    violations = set(feature_names) & TIER_3
    if violations:
        raise ValueError(
            f"Tier 3 (FORBIDDEN) features found in input: {violations}. "
            f"These are label-proximate and must NEVER be used as model input."
        )
    return True


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for BCE loss."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor([1.0])
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)
