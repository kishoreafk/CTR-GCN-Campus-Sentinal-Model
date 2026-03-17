"""Tests for training components."""
import torch
import pytest
from models.ctrgcn_ava import CTRGCNForAVA
from training.losses import AsymmetricLoss
from training.ema import ModelEMA
from training.early_stopping import EarlyStopping
from training.checkpoint_manager import CheckpointManager


def test_single_step(device):
    """1 batch, loss should be finite and positive."""
    model = CTRGCNForAVA(num_classes=15).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = AsymmetricLoss()

    x = torch.randn(2, 3, 64, 18, 2).to(device)
    y = torch.zeros(2, 15).to(device)
    y[0, [3, 7]] = 1.0
    y[1, [1]] = 1.0

    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_ema_diverges_from_model(device):
    """EMA should differ from model after updates."""
    model = CTRGCNForAVA(num_classes=15).to(device)
    ema = ModelEMA(model, decay=0.99)

    # Do a training step
    x = torch.randn(2, 3, 64, 18, 2).to(device)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Modify model
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    ema.update(model)

    # EMA and model should differ
    model_params = list(model.parameters())
    ema_params = list(ema.ema.parameters())
    differ = False
    for mp, ep in zip(model_params, ema_params):
        if not torch.allclose(mp, ep, atol=1e-6):
            differ = True
            break
    assert differ


def test_early_stopping_triggers():
    """No improvement for patience epochs → stop."""
    es = EarlyStopping(patience=3, metric="val_mAP", mode="max")

    es({"val_mAP": 0.5})  # best
    assert not es.should_stop

    es({"val_mAP": 0.4})  # no improvement 1
    es({"val_mAP": 0.3})  # no improvement 2
    assert not es.should_stop

    es({"val_mAP": 0.2})  # no improvement 3 → should stop
    assert es.should_stop


def test_checkpoint_round_trip(device, tmp_path):
    """Save + load, all state dicts should be equal."""
    from dataclasses import dataclass

    @dataclass
    class MockConfig:
        num_classes: int = 15
        save_every_n_epochs: int = 1
        keep_last_k_checkpoints: int = 3

    model = CTRGCNForAVA(num_classes=15).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ema = ModelEMA(model, decay=0.999)
    es = EarlyStopping(patience=5)

    ckpt_mgr = CheckpointManager(str(tmp_path / "run"), keep_k=3)
    metrics = {"mAP": 0.42, "val_loss": 0.3}
    cfg = MockConfig()

    ckpt_mgr.save(0, model, optimizer, None, ema, es, metrics, cfg)

    # Reload
    model2 = CTRGCNForAVA(num_classes=15).to(device)
    epoch, loaded_metrics = ckpt_mgr.load(
        str(tmp_path / "run" / "last.pth"), model2)

    assert epoch == 0
    # State should match
    for k in model.state_dict():
        torch.testing.assert_close(
            model.state_dict()[k].cpu(),
            model2.state_dict()[k].cpu())
