"""
Base PyTorch Dataset for skeleton-based multi-label action recognition.
Output item:
    input : Tensor (C=3, T=64, V=18, M=2)   — CTR-GCN format
    label : Tensor (num_classes,)             — binary multi-label
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional
import scipy.interpolate as interp

# OpenPose-18 flip pairs (left ↔ right swap for horizontal flip augmentation)
FLIP_PAIRS = [(2,5),(3,6),(4,7),(8,11),(9,12),(10,13),(14,15),(16,17)]

class SkeletonDataset(Dataset):
    def __init__(self, annotation_dir: str, class_registry,
                 split: str, augment: bool = True,
                 min_quality: float = 0.30,
                 max_samples: Optional[int] = None):
        self.reg     = class_registry
        self.augment = augment
        self.T, self.V, self.M, self.C = 64, 18, 2, 3

        paths = sorted(Path(annotation_dir).rglob("*.npz"))
        samples = []
        for p in paths:
            d = np.load(p, allow_pickle=True)
            if str(d.get("split", "train")) != split:
                continue
            if float(d.get("quality_score", 1)) < min_quality:
                continue
            label = np.array(d["label"], dtype=np.float32)
            if label.sum() == 0:
                continue
            samples.append((str(p), label))

        if max_samples:
            import random; random.shuffle(samples)
            samples = samples[:max_samples]
        self.samples = samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]
        d    = np.load(path, allow_pickle=True)
        kpts = d["keypoints"].copy()   # (T, M, V, C)

        if self.augment:
            kpts = self._augment(kpts)

        # (T, M, V, C) → (C, T, V, M)
        x = torch.from_numpy(
            kpts.transpose(3, 0, 2, 1).astype(np.float32))

        return {
            "input":    x,
            "label":    torch.from_numpy(label),
            "video_id": str(d.get("video_id", "")),
            "index":    idx,
        }

    def _augment(self, kpts: np.ndarray) -> np.ndarray:
        """Applies 6 augmentations to (T, M, V, C) array."""
        # 1. Random temporal shift ± 8 frames (circular)
        shift = np.random.randint(-8, 9)
        kpts  = np.roll(kpts, shift, axis=0)

        # 2. Horizontal flip (p=0.5)
        if np.random.rand() < 0.5:
            kpts[..., 0] = 1.0 - kpts[..., 0]
            for l, r in FLIP_PAIRS:
                kpts[:, :, [l, r]] = kpts[:, :, [r, l]]

        # 3. Speed perturbation — resample T with factor ∈ [0.8, 1.2]
        factor = np.random.uniform(0.8, 1.2)
        new_T  = max(16, int(self.T * factor))
        old_t  = np.linspace(0, 1, self.T)
        new_t  = np.linspace(0, 1, new_T)
        T, M, V, C = kpts.shape
        out = np.zeros((self.T, M, V, C), np.float32)
        for m in range(M):
            for v in range(V):
                for c in range(C):
                    resampled = np.interp(old_t, np.linspace(0, 1, new_T),
                                          np.interp(new_t, old_t, kpts[:, m, v, c]))
                    out[:, m, v, c] = resampled
        kpts = out

        # 4. Gaussian noise on x,y
        kpts[..., :2] += np.random.randn(*kpts[..., :2].shape) * 0.01

        # 5. Random joint masking (drop 1–3 joints with p=0.1)
        if np.random.rand() < 0.1:
            n_drop = np.random.randint(1, 4)
            joints = np.random.choice(self.V, n_drop, replace=False)
            kpts[:, :, joints, :] = 0.0

        # 6. Second-person dropout (p=0.1)
        if self.M > 1 and np.random.rand() < 0.1:
            kpts[:, 1, :, :] = 0.0

        return kpts

    def get_class_pos_weights(self) -> torch.Tensor:
        """
        Per-class positive weights for BCEWithLogitsLoss.
        pos_weight[c] = num_negative / num_positive (clamped to [1, 100]).
        Must be moved to the correct device by the caller.
        """
        counts = np.zeros(self.reg.num_classes, dtype=np.float32)
        N = len(self.samples)
        for _, label in self.samples:
            counts += label
        neg = N - counts
        weights = np.clip(neg / np.maximum(counts, 1), 1.0, 100.0)
        return torch.from_numpy(weights)
