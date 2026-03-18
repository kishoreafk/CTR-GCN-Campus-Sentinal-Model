"""Tests for utils/dataset_auditor.py"""
import pytest
import numpy as np
from pathlib import Path


def _make_npz(path, action_ids, split="train", quality=0.85, has_nan=False):
    """Helper to create a valid annotation .npz file."""
    kpts = np.random.rand(64, 2, 18, 3).astype(np.float32)
    if has_nan:
        kpts[0, 0, 0, :] = np.nan
    label = np.zeros(15, dtype=np.float32)
    for i, aid in enumerate(action_ids):
        if i < 15:
            label[i] = 1.0
    np.savez_compressed(
        str(path),
        keypoints=kpts,
        label=label,
        video_id="test_vid",
        timestamp=5.0,
        person_bbox=np.array([0.1, 0.2, 0.5, 0.8]),
        action_ids=action_ids,
        quality_score=quality,
        joint_layout="openpose_18",
        split=split,
    )


@pytest.fixture
def class_registry_3():
    from utils.class_registry import ClassRegistry
    classes = [
        {"id": 17, "name": "eat", "category": "object"},
        {"id": 49, "name": "walk", "category": "movement"},
        {"id": 74, "name": "talk to", "category": "interaction"},
    ]
    return ClassRegistry.from_class_list(classes)


@pytest.fixture
def good_dataset(tmp_path, class_registry_3):
    """Create a dataset that passes audit (enough samples)."""
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    # 60 train + 15 val for each class
    for i in range(60):
        for aid in [17, 49, 74]:
            _make_npz(ann_dir / f"train_{aid}_{i}.npz", [aid], split="train")
    for i in range(15):
        for aid in [17, 49, 74]:
            _make_npz(ann_dir / f"val_{aid}_{i}.npz", [aid], split="val")
    return str(ann_dir)


@pytest.fixture
def bad_dataset(tmp_path, class_registry_3):
    """Create a dataset that fails audit (too few samples)."""
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    # Only 5 train samples for class 17
    for i in range(5):
        _make_npz(ann_dir / f"train_{i}.npz", [17], split="train")
    for i in range(5):
        _make_npz(ann_dir / f"val_{i}.npz", [17], split="val")
    return str(ann_dir)


def test_audit_passes_good_dataset(good_dataset, class_registry_3):
    from utils.dataset_auditor import DatasetAuditor
    auditor = DatasetAuditor()
    report = auditor.audit(good_dataset, class_registry_3)
    assert "class_counts" in report
    assert "quality" in report
    assert report["nan_files"] == 0
    # All classes should have enough samples
    for name, counts in report["class_counts"].items():
        assert counts["train"] >= 50
        assert counts["val"] >= 10


def test_audit_fails_insufficient_samples(bad_dataset, class_registry_3):
    from utils.dataset_auditor import DatasetAuditor, AuditError
    auditor = DatasetAuditor()
    with pytest.raises(AuditError, match="Insufficient samples"):
        auditor.audit(bad_dataset, class_registry_3)


def test_audit_no_npz(tmp_path, class_registry_3):
    from utils.dataset_auditor import DatasetAuditor, AuditError
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    auditor = DatasetAuditor()
    with pytest.raises(AuditError, match="No .npz files"):
        auditor.audit(str(empty_dir), class_registry_3)


def test_audit_detects_nan(tmp_path, class_registry_3):
    from utils.dataset_auditor import DatasetAuditor
    ann_dir = tmp_path / "nan_test"
    ann_dir.mkdir()
    # Create enough samples to pass, with some NaN
    for i in range(60):
        has_nan = (i < 3)  # first 3 have NaN
        for aid in [17, 49, 74]:
            _make_npz(ann_dir / f"train_{aid}_{i}.npz", [aid],
                      split="train", has_nan=has_nan)
    for i in range(15):
        for aid in [17, 49, 74]:
            _make_npz(ann_dir / f"val_{aid}_{i}.npz", [aid], split="val")
    auditor = DatasetAuditor()
    report = auditor.audit(str(ann_dir), class_registry_3)
    assert report["nan_files"] > 0


def test_audit_cooccurrence(good_dataset, class_registry_3):
    from utils.dataset_auditor import DatasetAuditor
    auditor = DatasetAuditor()
    report = auditor.audit(good_dataset, class_registry_3)
    # Each class appears with itself on the diagonal
    cooc = report["cooccurrence"]
    assert "eat" in cooc
    assert cooc["eat"]["eat"] > 0
