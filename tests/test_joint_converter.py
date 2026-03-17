"""Tests for COCO-17 → OpenPose-18 joint converter."""
import numpy as np
import pytest
from annotation.joint_converter import coco17_to_openpose18, batch_coco17_to_openpose18


def test_output_shape():
    kpts = np.random.rand(17, 2).astype(np.float32)
    scores = np.random.rand(17).astype(np.float32)
    out_kpts, out_scores = coco17_to_openpose18(kpts, scores)
    assert out_kpts.shape == (18, 2)
    assert out_scores.shape == (18,)


def test_neck_midpoint():
    kpts = np.zeros((17, 2), dtype=np.float32)
    scores = np.ones(17, dtype=np.float32)
    # COCO left_shoulder=5, right_shoulder=6
    kpts[5] = [100, 200]
    kpts[6] = [200, 200]
    out_kpts, out_scores = coco17_to_openpose18(kpts, scores)
    # Neck = midpoint = [150, 200]
    np.testing.assert_allclose(out_kpts[1], [150, 200], atol=1e-5)


def test_neck_confidence():
    kpts = np.zeros((17, 2), dtype=np.float32)
    scores = np.zeros(17, dtype=np.float32)
    scores[5] = 0.9  # left_shoulder
    scores[6] = 0.7  # right_shoulder
    _, out_scores = coco17_to_openpose18(kpts, scores)
    assert out_scores[1] == pytest.approx(0.7)  # min(0.9, 0.7)


def test_direct_remap():
    kpts = np.zeros((17, 2), dtype=np.float32)
    scores = np.zeros(17, dtype=np.float32)
    kpts[0] = [42, 84]  # COCO nose
    scores[0] = 0.95
    out_kpts, out_scores = coco17_to_openpose18(kpts, scores)
    # OpenPose nose is also index 0
    np.testing.assert_allclose(out_kpts[0], [42, 84])
    assert out_scores[0] == pytest.approx(0.95)


def test_batch_shape():
    kpts = np.random.rand(64, 2, 17, 3).astype(np.float32)
    scores = np.random.rand(64, 2, 17).astype(np.float32)
    out_kpts, out_scores = batch_coco17_to_openpose18(kpts, scores)
    assert out_kpts.shape == (64, 2, 18, 3)
    assert out_scores.shape == (64, 2, 18)


def test_no_nan():
    """100 random inputs should never produce NaN."""
    for _ in range(100):
        kpts = np.random.rand(17, 2).astype(np.float32)
        scores = np.random.rand(17).astype(np.float32)
        out_kpts, out_scores = coco17_to_openpose18(kpts, scores)
        assert not np.any(np.isnan(out_kpts))
        assert not np.any(np.isnan(out_scores))
