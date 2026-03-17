"""
Phase 1: pretrained Kinetics-400 CTR-GCN → fine-tune on AVA-Kinetics (multi-label)
Phase 2: Phase-1 best.pth → fine-tune on AVA (multi-label)

Both phases share the same Trainer; the dataset and config differ.
Head is rebuilt when num_classes changes between phases.
"""
import logging
from pathlib import Path
from utils.config_loader import load_config
from utils.seed import set_seed
from utils.system_checks import preflight
from models.model_factory import build_model
from data_pipeline.dataloader_factory import create_dataloaders
from training.losses import build_loss
from training.ema import ModelEMA
from training.early_stopping import EarlyStopping
from training.checkpoint_manager import CheckpointManager
from training.trainer import Trainer
import torch

log = logging.getLogger("pipeline")


def _build_scheduler(optimizer, config, steps_per_epoch: int):
    if config.scheduler == "cosine_warm_restarts":
        base = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2)
    elif config.scheduler == "one_cycle":
        base = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[config.lr_backbone, config.lr_head],
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch)
    elif config.scheduler == "cosine":
        base = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs)
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
    return base


def run_phase(config_path: str, phase_name: str,
              pretrained_override: str = None) -> dict:
    config = load_config(config_path)
    if pretrained_override:
        config.pretrained_ckpt = pretrained_override

    set_seed(config.seed)
    preflight(required_disk_gb=20.0)

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, train_dataset = create_dataloaders(config)
    config.num_classes = train_dataset.reg.num_classes  # authoritative source

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(config)

    # ── Optimiser ───────────────────────────────────────────────────────────
    param_groups = model.get_param_groups(config.lr_backbone, config.lr_head)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    scheduler = _build_scheduler(optimizer, config, len(train_loader))

    # ── Loss ────────────────────────────────────────────────────────────────
    loss_fn = build_loss(config, train_dataset, device=config.device)

    # ── Support objects ─────────────────────────────────────────────────────
    ema = ModelEMA(model, decay=config.ema_decay) if config.use_ema else None
    es  = EarlyStopping(patience=config.early_stopping_patience,
                        metric=config.early_stopping_metric,
                        mode=config.early_stopping_mode)

    # ── Resume logic ────────────────────────────────────────────────────────
    run_dir = Path(config.checkpoint_dir) / "runs" / \
              f"{phase_name}_{config.experiment_name}"
    ckpt_mgr = CheckpointManager(str(run_dir),
                                  keep_k=config.keep_last_k_checkpoints)
    start_epoch = 0
    best_metric = 0.0

    if config.auto_resume:
        last = run_dir / "last.pth"
        if last.exists():
            start_epoch, metrics = ckpt_mgr.load(
                str(last), model, optimizer, scheduler, ema, es)
            start_epoch += 1
            best_metric = ckpt_mgr.best_mAP
            log.info(f"Resuming {phase_name} from epoch {start_epoch}")

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model, train_loader, val_loader,
        optimizer, scheduler, loss_fn,
        ema, es, ckpt_mgr, config,
        class_names=train_dataset.reg.class_names,
        start_epoch=start_epoch,
        best_metric=best_metric,
    )
    return trainer.fit()


def run_full_pipeline(base_config: str = "configs/base_config.yaml"):
    log.info("=" * 60)
    log.info("PHASE 1: AVA-Kinetics fine-tuning")
    log.info("=" * 60)
    p1_metrics = run_phase("configs/phase1_ava_kinetics.yaml", "phase1")

    # Pass Phase 1 best checkpoint to Phase 2
    phase1_best = str(Path("checkpoints/runs/phase1_phase1_ava_kinetics/best.pth"))

    log.info("=" * 60)
    log.info("PHASE 2: AVA fine-tuning")
    log.info("=" * 60)
    p2_metrics = run_phase("configs/phase2_ava.yaml", "phase2",
                            pretrained_override=phase1_best)

    log.info("Pipeline complete.")
    log.info(f"Phase 1 best mAP: {p1_metrics.get('mAP', 0):.4f}")
    log.info(f"Phase 2 best mAP: {p2_metrics.get('mAP', 0):.4f}")
    return p1_metrics, p2_metrics
