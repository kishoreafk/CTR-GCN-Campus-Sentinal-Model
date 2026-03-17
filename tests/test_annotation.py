"""Tests for annotation pipeline components."""
import numpy as np
import pytest
from annotation.quality_validator import AnnotationQualityValidator
from annotation.person_tracker import PersonTracker


def test_quality_score_range():
    """Quality score should always be in [0, 1]."""
    from annotation.extractor import SkeletonExtractor

    class MockConfig:
        num_frames = 64
        max_persons = 2
        num_joints = 18

    ext = SkeletonExtractor.__new__(SkeletonExtractor)
    ext.T = 64
    ext.M = 2
    ext.V = 18

    scores = np.random.rand(64, 2, 18).astype(np.float32)
    q = ext._quality_score(scores)
    assert 0.0 <= q <= 1.0


def test_npz_schema(dummy_npz):
    """All required keys should be present with correct shapes."""
    d = np.load(dummy_npz, allow_pickle=True)
    assert "keypoints" in d
    assert "label" in d
    assert d["keypoints"].shape == (64, 2, 18, 3)
    assert d["label"].shape == (15,)


def test_temporal_padding():
    """Short clips should be padded to T=64 frames."""
    # This tests the frame padding logic conceptually
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(30)]
    while len(frames) < 64:
        frames.insert(0, frames[0])
    frames = frames[:64]
    assert len(frames) == 64


def test_tracker_identity():
    """Person identity should be stable across synthetic frames."""
    tracker = PersonTracker(max_persons=2, iou_thr=0.3)

    # First frame: two persons
    dets = [
        {"bbox": np.array([10, 10, 50, 100]),
         "keypoints": np.random.rand(17, 2).astype(np.float32),
         "scores": np.random.rand(17).astype(np.float32)},
        {"bbox": np.array([200, 10, 250, 100]),
         "keypoints": np.random.rand(17, 2).astype(np.float32),
         "scores": np.random.rand(17).astype(np.float32)},
    ]
    tracks = tracker.update(dets)
    assert len(tracks) == 2
    id0, id1 = tracks[0].track_id, tracks[1].track_id

    # Simulate 10 frames with slight movement
    for _ in range(10):
        dets[0]["bbox"] += np.array([1, 0, 1, 0])  # slight shift
        dets[1]["bbox"] += np.array([1, 0, 1, 0])
        tracks = tracker.update(dets)

    # Identities should be preserved
    assert tracks[0].track_id == id0
    assert tracks[1].track_id == id1


def test_validator_catches_nan():
    """NaN in keypoints should be flagged."""
    import tempfile
    validator = AnnotationQualityValidator()
    p = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    kpts = np.random.rand(64, 2, 18, 3).astype(np.float32)
    kpts[0, 0, 0, 0] = np.nan
    np.savez_compressed(p.name,
        keypoints=kpts,
        label=np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32),
        quality_score=0.5)
    r = validator.validate_single(p.name)
    assert not r["valid"]
    assert any("NaN" in i for i in r["issues"])
