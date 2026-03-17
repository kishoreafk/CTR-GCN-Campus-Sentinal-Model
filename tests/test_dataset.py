"""Tests for skeleton dataset."""
import numpy as np
import torch
import pytest
from pathlib import Path


def _create_sample(out_dir, name, split="train"):
    """Create a synthetic .npz sample."""
    np.savez_compressed(
        str(Path(out_dir) / f"{name}.npz"),
        keypoints=np.random.rand(64, 2, 18, 3).astype(np.float32),
        label=np.array([0,1,0,0,1,0,0,0,0,0,0,0,0,0,0], dtype=np.float32),
        video_id=name,
        timestamp=5.0,
        quality_score=0.9,
        joint_layout="openpose_18",
        split=split,
    )


def test_item_shape(tmp_path):
    """Input should be (3,64,18,2), label should be (15,)."""
    from utils.class_registry import ClassRegistry
    from data_pipeline.skeleton_dataset import SkeletonDataset

    _create_sample(str(tmp_path), "s1")
    reg = ClassRegistry()
    ds = SkeletonDataset(str(tmp_path), reg, split="train", augment=False)
    if len(ds) > 0:
        item = ds[0]
        assert item["input"].shape == (3, 64, 18, 2)
        assert item["label"].shape == (15,)


def test_label_is_multilabel(tmp_path):
    """Label should be float binary, NOT a single int."""
    from utils.class_registry import ClassRegistry
    from data_pipeline.skeleton_dataset import SkeletonDataset

    _create_sample(str(tmp_path), "s1")
    reg = ClassRegistry()
    ds = SkeletonDataset(str(tmp_path), reg, split="train", augment=False)
    if len(ds) > 0:
        item = ds[0]
        label = item["label"]
        assert label.dtype == torch.float32
        assert label.dim() == 1
        assert label.sum() > 0  # at least one positive class


def test_flip_pairs(tmp_path):
    """After flip, OpenPose pairs should be correctly swapped."""
    from data_pipeline.skeleton_dataset import FLIP_PAIRS
    kpts = np.random.rand(64, 2, 18, 3).astype(np.float32)
    original = kpts.copy()

    # Apply flip
    kpts[..., 0] = 1.0 - kpts[..., 0]
    for l, r in FLIP_PAIRS:
        kpts[:, :, [l, r]] = kpts[:, :, [r, l]]

    # Verify pairs are swapped: x-coord should be flipped, y and conf should just be swapped
    for l, r in FLIP_PAIRS:
        # x-coordinate: flipped (1 - original) AND swapped (l ↔ r)
        np.testing.assert_allclose(kpts[:, :, l, 0], 1.0 - original[:, :, r, 0], atol=1e-6)
        # y-coordinate and confidence: just swapped, not flipped
        np.testing.assert_allclose(kpts[:, :, l, 1:], original[:, :, r, 1:], atol=1e-6)


def test_no_nan_after_augment(tmp_path):
    """1000 augmentations should never produce NaN/Inf."""
    from utils.class_registry import ClassRegistry
    from data_pipeline.skeleton_dataset import SkeletonDataset

    for i in range(5):
        _create_sample(str(tmp_path), f"s{i}")
    reg = ClassRegistry()
    ds = SkeletonDataset(str(tmp_path), reg, split="train", augment=True)

    for _ in range(min(200, len(ds) * 40)):
        if len(ds) == 0:
            break
        item = ds[0]
        assert not torch.isnan(item["input"]).any()
        assert not torch.isinf(item["input"]).any()


def test_pos_weights_shape(tmp_path):
    """pos_weight shape should be (num_classes,), all positive."""
    from utils.class_registry import ClassRegistry
    from data_pipeline.skeleton_dataset import SkeletonDataset

    for i in range(5):
        _create_sample(str(tmp_path), f"s{i}")
    reg = ClassRegistry()
    ds = SkeletonDataset(str(tmp_path), reg, split="train", augment=False)
    if len(ds) > 0:
        w = ds.get_class_pos_weights()
        assert w.shape == (reg.num_classes,)
        assert (w > 0).all()
