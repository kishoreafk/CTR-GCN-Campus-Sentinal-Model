"""
Builds schedulers and handles the OneCycleLR resume edge case.

OneCycleLR is different from every other scheduler — it precomputes the
entire LR schedule at construction time based on total_steps. If you
resume and construct a new OneCycleLR for the remaining epochs, it will
start the cosine cycle from step 0 again, not from where it left off.

This module fast-forwards the scheduler to the correct position on resume.
"""

import torch
import logging

log = logging.getLogger("scheduler_factory")


def build_scheduler(optimizer, config, steps_per_epoch: int,
                    resume_global_step: int = 0):
    """
    Build the correct scheduler, accounting for resume position.

    Parameters
    ----------
    optimizer           : torch.optim.Optimizer
    config              : TrainingConfig
    steps_per_epoch     : int, number of optimizer steps per epoch
    resume_global_step  : int, optimizer steps already completed (0 = fresh start)
    """
    if config.scheduler == "cosine_warm_restarts":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0    = getattr(config, "T_0",    10),
            T_mult = getattr(config, "T_mult",  2),
        )
        if config.warmup_epochs > 0:
            sched = _wrap_with_warmup(sched, optimizer, config, steps_per_epoch)
        return sched

    elif config.scheduler == "one_cycle":
        total_steps = config.epochs * steps_per_epoch

        if resume_global_step > 0:
            # ── Resume fix for OneCycleLR ─────────────────────────────────
            # Construct for total_steps, then fast-forward the internal
            # state by calling scheduler.step() resume_global_step times.
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr           = [config.lr_backbone, config.lr_head],
                total_steps      = total_steps,
                pct_start        = getattr(config, "pct_start", 0.3),
                anneal_strategy  = "cos",
            )
            log.info(
                f"Fast-forwarding OneCycleLR by {resume_global_step} steps..."
            )
            with torch.no_grad():
                for _ in range(resume_global_step):
                    sched.step()
            # Log where we landed
            new_lrs = [g["lr"] for g in optimizer.param_groups]
            log.info(
                f"OneCycleLR fast-forwarded to step {resume_global_step}. "
                f"LR: {[f'{lr:.2e}' for lr in new_lrs]}"
            )
            return sched
        else:
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr      = [config.lr_backbone, config.lr_head],
                total_steps = total_steps,
                pct_start   = getattr(config, "pct_start", 0.3),
            )

    elif config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )

    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


def _wrap_with_warmup(scheduler, optimizer, config, steps_per_epoch: int):
    """
    Prepend a linear warmup to any scheduler using
    SequentialLR (PyTorch built-in).
    """
    warmup_steps = config.warmup_epochs * steps_per_epoch
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 0.01,
        end_factor   = 1.0,
        total_iters  = warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers  = [warmup, scheduler],
        milestones  = [warmup_steps],
    )
