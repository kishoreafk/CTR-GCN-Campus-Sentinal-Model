"""Tests for training/gradient_monitor.py"""
import torch
import torch.nn as nn
import pytest
from training.gradient_monitor import (
    GradientMonitor, EXPLOSION_THRESHOLD, VANISHING_THRESHOLD
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


def test_normal_gradient_returns_record_on_log_step():
    model = TinyModel()
    monitor = GradientMonitor(model, log_every_n_steps=1)
    x = torch.randn(4, 10)
    y = model(x)
    y.sum().backward()

    result = monitor.check(step=1)
    assert result is not None
    assert "total_norm" in result
    assert result["total_norm"] > 0


def test_normal_gradient_returns_none_on_non_log_step():
    model = TinyModel()
    monitor = GradientMonitor(model, log_every_n_steps=50)
    x = torch.randn(4, 10)
    y = model(x)
    y.sum().backward()

    result = monitor.check(step=3)  # not a multiple of 50
    assert result is None


def test_explosion_detection():
    model = TinyModel()
    monitor = GradientMonitor(model, log_every_n_steps=1)

    # Manually set large gradients to trigger explosion
    x = torch.randn(4, 10)
    y = model(x)
    y.sum().backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.fill_(1000.0)

    result = monitor.check(step=1)
    assert result is not None
    assert result["total_norm"] > EXPLOSION_THRESHOLD


def test_vanishing_detection():
    model = TinyModel()
    monitor = GradientMonitor(model, log_every_n_steps=1)

    # Manually set tiny gradients
    x = torch.randn(4, 10)
    y = model(x)
    y.sum().backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.fill_(1e-10)

    result = monitor.check(step=1)
    assert result is not None
    assert result["total_norm"] < VANISHING_THRESHOLD


def test_get_summary():
    model = TinyModel()
    monitor = GradientMonitor(model, log_every_n_steps=1)

    for step in range(5):
        x = torch.randn(4, 10)
        y = model(x)
        y.sum().backward()
        monitor.check(step)
        model.zero_grad()

    summary = monitor.get_summary()
    assert "mean_grad_norm" in summary
    assert "max_grad_norm" in summary
    assert "explosion_steps" in summary
    assert "vanishing_steps" in summary
    assert summary["mean_grad_norm"] > 0


def test_per_layer_norms():
    model = TinyModel()
    monitor = GradientMonitor(model, log_every_n_steps=1)

    x = torch.randn(4, 10)
    y = model(x)
    y.sum().backward()

    result = monitor.check(step=1)
    assert "layer_norms" in result
    assert len(result["layer_norms"]) > 0
