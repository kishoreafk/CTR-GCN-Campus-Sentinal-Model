"""Tests for training/bn_calibration.py"""
import torch
import torch.nn as nn
import pytest
from training.bn_calibration import (
    recalibrate_bn_statistics, should_recalibrate, _reset_bn_stats
)


class BNModel(nn.Module):
    """Simple model with BatchNorm for testing."""
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(10)
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(self.bn(x))


def test_should_recalibrate_true():
    schedule = {"5": ["layer1"], "10": ["layer2"]}
    assert should_recalibrate(5, schedule) is True
    assert should_recalibrate(10, schedule) is True


def test_should_recalibrate_false():
    schedule = {"5": ["layer1"], "10": ["layer2"]}
    assert should_recalibrate(0, schedule) is False
    assert should_recalibrate(3, schedule) is False
    assert should_recalibrate(7, schedule) is False


def test_reset_bn_stats():
    model = BNModel()
    # Run some data to update BN stats
    x = torch.randn(32, 10)
    model(x)
    assert not torch.allclose(model.bn.running_mean, torch.zeros(10))

    _reset_bn_stats(model)
    assert torch.allclose(model.bn.running_mean, torch.zeros(10))
    assert torch.allclose(model.bn.running_var, torch.ones(10))


def test_recalibrate_updates_stats():
    model = BNModel()
    # Create a simple dataloader
    dataset = [{"input": torch.randn(10)} for _ in range(20)]
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4)

    # Reset stats first
    _reset_bn_stats(model)
    old_mean = model.bn.running_mean.clone()

    # Recalibrate
    recalibrate_bn_statistics(model, loader, "cpu", num_batches=5)

    # Stats should have changed
    assert not torch.allclose(model.bn.running_mean, old_mean)


def test_recalibrate_preserves_grad_state():
    model = BNModel()
    # Set specific requires_grad state
    model.fc.weight.requires_grad_(True)
    model.fc.bias.requires_grad_(False)

    dataset = [{"input": torch.randn(10)} for _ in range(10)]
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4)

    recalibrate_bn_statistics(model, loader, "cpu", num_batches=2)

    # Should restore original requires_grad state
    assert model.fc.weight.requires_grad is True
    assert model.fc.bias.requires_grad is False
