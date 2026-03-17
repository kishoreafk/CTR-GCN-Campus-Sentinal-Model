"""Tests for loss functions."""
import torch
import pytest
from training.losses import AsymmetricLoss, WeightedBCELoss


def test_asymmetric_perfect():
    """All correct predictions → near-zero loss."""
    loss_fn = AsymmetricLoss()
    # Perfect predictions: large positive logits for positives, large negative for negatives
    logits  = torch.tensor([[10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]])
    targets = torch.tensor([[1.0,   0.0,  1.0], [  0.0,  1.0,   0.0]])
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.01


def test_asymmetric_all_wrong():
    """All wrong predictions → high loss."""
    loss_fn = AsymmetricLoss()
    # Wrong predictions: large positive logits for negatives, large negative for positives
    logits  = torch.tensor([[-10.0, 10.0, -10.0], [10.0, -10.0, 10.0]])
    targets = torch.tensor([[ 1.0,   0.0,  1.0], [ 0.0,  1.0,  0.0]])
    loss = loss_fn(logits, targets)
    assert loss.item() > 1.0


def test_bce_pos_weight_device(device):
    """pos_weight on same device as logits should not crash."""
    pos_weight = torch.ones(15).to(device)
    loss_fn = WeightedBCELoss(pos_weight=pos_weight)
    logits = torch.randn(4, 15).to(device)
    targets = torch.zeros(4, 15).to(device)
    targets[0, [3, 7]] = 1.0
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)


def test_loss_finite():
    """Random logits/targets should never produce NaN/Inf loss."""
    loss_fn = AsymmetricLoss()
    for _ in range(100):
        logits = torch.randn(8, 15)
        targets = (torch.rand(8, 15) > 0.7).float()
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss), f"Non-finite loss: {loss}"
