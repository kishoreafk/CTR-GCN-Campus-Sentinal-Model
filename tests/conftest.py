import pytest, torch, numpy as np, tempfile, os

@pytest.fixture
def device(): return "cuda:0" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def dummy_skeleton():
    """(batch=2, C=3, T=64, V=18, M=2)"""
    return torch.randn(2, 3, 64, 18, 2)

@pytest.fixture
def dummy_label():
    """Multi-label binary (batch=2, num_classes=15)"""
    t = torch.zeros(2, 15)
    t[0, [3, 7]] = 1; t[1, [1]] = 1
    return t

@pytest.fixture
def dummy_coco_kpts():
    return (np.random.rand(17, 2).astype(np.float32),
            np.random.rand(17).astype(np.float32))

@pytest.fixture
def dummy_npz(tmp_path):
    p = tmp_path / "sample.npz"
    np.savez_compressed(str(p),
        keypoints    = np.random.rand(64, 2, 18, 3).astype(np.float32),
        label        = np.array([0,1,0,0,1,0,0,0,0,0,0,0,0,0,0], dtype=np.float32),
        video_id     = "test001",
        timestamp    = 5.0,
        person_bbox  = np.array([0.1,0.2,0.5,0.8]),
        action_ids   = [17, 49],
        quality_score= 0.85,
        joint_layout = "openpose_18",
        split        = "train",
    )
    return str(p)
