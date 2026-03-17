"""
Converts RTMPose COCO-17 keypoints to OpenPose-18.
The pretrained CTR-GCN graph has 18 vertices; we must match this layout.

Neck (joint 1) = midpoint of left_shoulder (COCO 5) and right_shoulder (COCO 6).
Neck confidence = min(l_shoulder_conf, r_shoulder_conf).
All other joints are direct index remaps (see configs/joint_mappings.yaml).
"""
import numpy as np
from typing import Tuple

# Direct remaps: openpose_idx → coco_idx  (joint 1 handled separately)
DIRECT = {0:0, 2:6, 3:8, 4:10, 5:5, 6:7, 7:9,
          8:12, 9:14, 10:16, 11:11, 12:13, 13:15,
          14:2, 15:1, 16:4, 17:3}

def coco17_to_openpose18(
    kpts: np.ndarray,     # (17, 2) or (17, 3)  — pixel coords + optional conf
    scores: np.ndarray    # (17,)
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns kpts_18 (18, C), scores_18 (18,)."""
    C = kpts.shape[-1]
    out_kpts   = np.zeros((18, C), dtype=np.float32)
    out_scores = np.zeros(18,      dtype=np.float32)

    for op_idx, co_idx in DIRECT.items():
        out_kpts[op_idx]   = kpts[co_idx]
        out_scores[op_idx] = scores[co_idx]

    # Joint 1: neck = midpoint of left_shoulder (5) and right_shoulder (6)
    out_kpts[1]   = (kpts[5] + kpts[6]) / 2.0
    out_scores[1] = min(scores[5], scores[6])

    return out_kpts, out_scores


def batch_coco17_to_openpose18(
    kpts: np.ndarray,    # (T, M, 17, C)
    scores: np.ndarray   # (T, M, 17)
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised batch conversion — no Python loops over T or M."""
    T, M, _, C = kpts.shape
    out_kpts   = np.zeros((T, M, 18, C), dtype=np.float32)
    out_scores = np.zeros((T, M, 18),    dtype=np.float32)

    co_indices = [DIRECT[i] for i in range(18) if i != 1]  # 17 direct entries
    op_indices = [i          for i in range(18) if i != 1]

    out_kpts  [:, :, op_indices] = kpts  [:, :, co_indices]
    out_scores[:, :, op_indices] = scores[:, :, co_indices]

    # Neck via vectorised mean
    out_kpts  [:, :, 1] = (kpts[:, :, 5] + kpts[:, :, 6]) / 2.0
    out_scores[:, :, 1] = np.minimum(scores[:, :, 5], scores[:, :, 6])

    return out_kpts, out_scores
