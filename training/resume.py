"""
Resume logic and configuration compatibility checking.
Ensures that a restarted training run doesn't silently break due to
changed YAML config hyperparameters.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
from training.checkpoint_manager import CheckpointManager
from models.model_factory import build_model
from training.ema import ModelEMA
from training.early_stopping import EarlyStopping

log = logging.getLogger("resume")


class ConfigMismatchError(Exception):
    pass


def check_config_compatibility(saved_config: dict, current_config,
                             current_registry) -> None:
    """
    Compare previous run's config against current config.
    Raises ConfigMismatchError for fatal differences.
    Logs warnings for non-fatal differences.
    """
    if not saved_config:
        return

    # Fatal mismatch: Architecture/Data changes
    saved_layout = saved_config.get("joint_layout")
    curr_layout  = getattr(current_config, "joint_layout", None)
    if saved_layout != curr_layout:
        raise ConfigMismatchError(
            f"Cannot resume: joint_layout changed from '{saved_layout}' to '{curr_layout}'"
        )

    # Note: num_classes in config might be from previous phase (e.g. Kinetics).
    # The actual authoritative num_classes comes from the checkpoint metadata or ClassRegistry.
    # But if it's identical phase resume, they must match.

    # Warnings: Hyperparameter changes (learning rate, batch size, etc.)
    for key in ["batch_size", "lr_backbone", "lr_head",
                "scheduler", "weight_decay"]:
        saved_val = saved_config.get(key)
        curr_val  = getattr(current_config, key, None)
        if saved_val is not None and saved_val != curr_val:
            log.warning(
                f"Config Change warning: '{key}' changed from "
                f"{saved_val} to {curr_val}. "
                f"Optimizer/Scheduler state will be loaded, which may override "
                f"this new value depending on the component."
            )


def build_components(config, class_registry):
    """Build core training components."""
    model = build_model(config)
    
    param_groups = model.get_param_groups(config.lr_backbone, config.lr_head)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    
    ema = ModelEMA(model, decay=config.ema_decay) if config.use_ema else None
    
    es = EarlyStopping(
        patience=config.early_stopping_patience,
        metric=config.early_stopping_metric,
        mode=config.early_stopping_mode
    )
    
    return model, optimizer, ema, es


def setup_training_run(config, class_registry, phase_name: str,
                       train_loader_len: int) -> Tuple[
                           Any, Any, Any, Any, Any, CheckpointManager, int, int]:
    """
    Orchestrate training setup:
      1. Build components
      2. Check for resume checkpoint
      3. Verify config compatibility
      4. Load state + RNG
      5. Build/Restore scheduler
    """
    model, optimizer, ema, es = build_components(config, class_registry)
    
    run_dir = Path(config.checkpoint_dir) / "runs" / f"{phase_name}_{config.experiment_name}"
    ckpt_mgr = CheckpointManager(
        str(run_dir),
        keep_k=config.keep_last_k_checkpoints,
        metric=config.early_stopping_metric,
        mode=config.early_stopping_mode
    )
    
    start_epoch = 0
    global_step = 0
    
    # Check for resume
    resume_path = None
    if config.auto_resume:
        resume_path = ckpt_mgr.find_resume_checkpoint()

    # Pretrained override takes precedence if no resume checkpoint
    if not resume_path and getattr(config, "pretrained_ckpt", None):
        resume_path = Path(config.pretrained_ckpt)
        log.info(f"Using pretrained weights from: {getattr(config, 'pretrained_ckpt')}")

    if resume_path and resume_path.exists():
        # Load header to check compatibility before loading weights
        ckpt = torch.load(resume_path, map_location="cpu")
        saved_config = ckpt.get("config", {})
        
        # Only check compatibility if resuming same phase (has checkpoint metadata)
        if "metadata" in ckpt and config.auto_resume:
            check_config_compatibility(saved_config, config, class_registry)

        epoch, step, metrics = ckpt_mgr.load(
            str(resume_path), model, optimizer=optimizer,
            ema=ema, early_stopping=es, device=config.device
        )
        
        if "metadata" in ckpt and config.auto_resume:
            start_epoch = epoch + 1
            global_step = step
            log.info(f"Resuming {phase_name} from epoch {start_epoch}, step {global_step}")

    # Build scheduler AFTER optimizer is optionally restored.
    # Uses scheduler_factory which handles OneCycleLR resume fast-forward.
    from training.scheduler_factory import build_scheduler
    scheduler = build_scheduler(
        optimizer, config, train_loader_len,
        resume_global_step=global_step
    )

    return model, optimizer, scheduler, ema, es, ckpt_mgr, start_epoch, global_step
