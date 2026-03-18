"""
After gradual unfreezing of backbone layers, recalibrate BatchNorm
running statistics using a pass over the training data in eval mode
(no gradients, just forward passes to update running_mean/running_var).

Call this at the epoch boundary where new layers are unfrozen.
"""

import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader

log = logging.getLogger("bn_calibration")


def recalibrate_bn_statistics(model: nn.Module,
                              train_loader: DataLoader,
                              device: str,
                              num_batches: int = 100):
    """
    Update BatchNorm running statistics without updating weights.

    Steps:
      1. Set model to train mode (so BN uses batch stats, not running stats)
      2. Disable all gradient computation
      3. Run num_batches forward passes → running_mean/var updated
      4. Set model back to eval mode

    num_batches=100 with batch_size=64 = 6400 samples, sufficient for
    stable statistics on AVA-scale datasets.
    """
    log.info(
        f"Recalibrating BatchNorm statistics over "
        f"{num_batches} batches..."
    )

    # Save current requires_grad state
    grad_states = {name: p.requires_grad
                   for name, p in model.named_parameters()}

    # Freeze weights but allow BN stats to update
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    # Reset running stats before recalibration
    _reset_bn_stats(model)

    count = 0
    with torch.no_grad():
        for batch in train_loader:
            if count >= num_batches:
                break
            x = batch["input"].to(device, non_blocking=True)
            model(x)    # forward pass updates running_mean/var
            count += 1

    model.eval()

    # Restore requires_grad to previous state
    for name, p in model.named_parameters():
        if name in grad_states:
            p.requires_grad_(grad_states[name])

    log.info("BatchNorm recalibration complete.")


def _reset_bn_stats(model: nn.Module):
    """Reset all BN running statistics to default (mean=0, var=1)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()


def should_recalibrate(epoch: int, unfreeze_schedule: dict) -> bool:
    """
    Return True if this epoch just unfroze new layers.
    BN recalibration should run immediately after any unfreeze event.
    """
    return epoch in {int(k) for k in unfreeze_schedule.keys()}
