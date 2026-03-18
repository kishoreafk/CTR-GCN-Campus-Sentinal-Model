"""Tests for training/scheduler_factory.py"""
import torch
import pytest
from dataclasses import dataclass


@dataclass
class MockConfig:
    scheduler: str = "one_cycle"
    epochs: int = 10
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    warmup_epochs: int = 0
    T_0: int = 10
    T_mult: int = 2
    pct_start: float = 0.3


def _make_optimizer():
    model = torch.nn.Linear(10, 5)
    return torch.optim.AdamW([
        {"params": [model.weight], "lr": 1e-4},
        {"params": [model.bias],   "lr": 1e-3},
    ])


def test_one_cycle_fresh():
    from training.scheduler_factory import build_scheduler
    cfg = MockConfig(scheduler="one_cycle")
    opt = _make_optimizer()
    sched = build_scheduler(opt, cfg, steps_per_epoch=100, resume_global_step=0)
    assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)


def test_one_cycle_resume_fast_forward():
    """After fast-forward, LR should match the schedule at that step."""
    from training.scheduler_factory import build_scheduler
    cfg = MockConfig(scheduler="one_cycle", epochs=10)
    steps_per_epoch = 100

    # Create fresh scheduler and step it 250 times to get target LR
    opt1 = _make_optimizer()
    sched1 = build_scheduler(opt1, cfg, steps_per_epoch, resume_global_step=0)
    for _ in range(250):
        sched1.step()
    target_lrs = [g["lr"] for g in opt1.param_groups]

    # Create resumed scheduler fast-forwarded to step 250
    opt2 = _make_optimizer()
    sched2 = build_scheduler(opt2, cfg, steps_per_epoch, resume_global_step=250)
    resumed_lrs = [g["lr"] for g in opt2.param_groups]

    # LRs should match
    for t, r in zip(target_lrs, resumed_lrs):
        assert abs(t - r) < 1e-8, f"LR mismatch: {t} vs {r}"


def test_cosine_scheduler():
    from training.scheduler_factory import build_scheduler
    cfg = MockConfig(scheduler="cosine")
    opt = _make_optimizer()
    sched = build_scheduler(opt, cfg, steps_per_epoch=100)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_cosine_warm_restarts():
    from training.scheduler_factory import build_scheduler
    cfg = MockConfig(scheduler="cosine_warm_restarts", warmup_epochs=0)
    opt = _make_optimizer()
    sched = build_scheduler(opt, cfg, steps_per_epoch=100)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)


def test_cosine_warm_restarts_with_warmup():
    from training.scheduler_factory import build_scheduler
    cfg = MockConfig(scheduler="cosine_warm_restarts", warmup_epochs=2)
    opt = _make_optimizer()
    sched = build_scheduler(opt, cfg, steps_per_epoch=100)
    # Should be wrapped in SequentialLR
    assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)


def test_unknown_scheduler_raises():
    from training.scheduler_factory import build_scheduler
    cfg = MockConfig(scheduler="nonexistent")
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="Unknown scheduler"):
        build_scheduler(opt, cfg, steps_per_epoch=100)
