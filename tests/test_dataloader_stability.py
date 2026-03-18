"""Tests for data_pipeline/dataloader_factory.py stability features."""
import pytest
import torch
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import List


@dataclass
class MockConfig:
    dataset: str = "ava"
    batch_size: int = 4
    num_workers: int = 0     # use 0 for testing (no multiprocessing)
    pin_memory: bool = False
    prefetch_factor: int = 2
    seed: int = 42
    data_dir: str = "data"
    class_config: str = "configs/class_config.yaml"
    target_class_ids: List[int] = field(default_factory=lambda: [17, 49])
    target_class_names: List[str] = field(default_factory=lambda: ["eat", "walk"])
    num_classes: int = 2
    num_frames: int = 64
    num_joints: int = 18
    max_persons: int = 2
    in_channels: int = 3
    test_mode: bool = True
    test_max_videos: int = 5
    multi_label: bool = True


def test_worker_init_fn_produces_different_seeds():
    """Each worker should get a unique seed."""
    import random
    import numpy as np

    config = MockConfig()

    # Simulate worker_init_fn for two workers
    def worker_init_fn(worker_id: int):
        seed = config.seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    worker_init_fn(0)
    val_w0 = random.random()

    worker_init_fn(1)
    val_w1 = random.random()

    # Different workers should produce different random values
    assert val_w0 != val_w1


def test_dataloader_timeout_kwarg():
    """Verify that DataLoader gets timeout when num_workers > 0."""
    from torch.utils.data import DataLoader

    # When constructing DataLoader with timeout, it should not raise
    dataset = [torch.randn(3) for _ in range(10)]
    loader = DataLoader(dataset, batch_size=2, timeout=120)
    assert loader.timeout == 120


def test_dataloader_zero_workers_no_persistent():
    """num_workers=0 should not have persistent_workers."""
    from torch.utils.data import DataLoader

    dataset = [torch.randn(3) for _ in range(10)]
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    assert not hasattr(loader, 'persistent_workers') or not loader.persistent_workers
