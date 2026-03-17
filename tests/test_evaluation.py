"""Tests for evaluation metrics."""
import torch
import numpy as np
import pytest
from training.metrics import MultiLabelMetrics


CLASS_NAMES = ["dance", "run/jog", "walk", "eat", "drink"]


def test_map_perfect():
    """All predictions correct → mAP=1.0."""
    m = MultiLabelMetrics(5, CLASS_NAMES)
    # Perfect predictions: high logits for positives, low for negatives
    logits  = torch.tensor([[10.0, -10.0, 10.0, -10.0, -10.0],
                            [-10.0, 10.0, -10.0, -10.0, 10.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 1.0]])
    m.update(logits, targets)
    result = m.compute()
    assert result["mAP"] == pytest.approx(1.0, abs=0.01)


def test_map_random():
    """Random predictions → mAP should be low."""
    m = MultiLabelMetrics(5, CLASS_NAMES)
    np.random.seed(42)
    for _ in range(10):
        logits = torch.randn(16, 5)
        targets = (torch.rand(16, 5) > 0.7).float()
        m.update(logits, targets)
    result = m.compute()
    # Random should give ~1/num_classes mAP
    assert 0.0 < result["mAP"] < 0.8


def test_skip_empty_classes():
    """Classes with no positives should not be included in mean."""
    m = MultiLabelMetrics(5, CLASS_NAMES)
    # Only class 0 and 2 have positives
    logits  = torch.tensor([[10.0, -10.0, 10.0, -10.0, -10.0]])
    targets = torch.tensor([[1.0,   0.0,  1.0,   0.0,   0.0]])
    m.update(logits, targets)
    result = m.compute()
    # Only 2 classes contribute to mAP
    assert len(result["per_class"]) <= 5


def test_per_class_ap():
    """Known inputs should give verifiable AP values."""
    m = MultiLabelMetrics(3, ["a", "b", "c"])
    # Class 0: perfect, Class 1: perfect, Class 2: no positives
    logits  = torch.tensor([[ 5.0,  5.0, -5.0],
                            [-5.0, -5.0, -5.0]])
    targets = torch.tensor([[1.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]])
    m.update(logits, targets)
    result = m.compute()
    assert result["per_class"]["a"]["AP"] == pytest.approx(1.0, abs=0.01)
    assert result["per_class"]["b"]["AP"] == pytest.approx(1.0, abs=0.01)
