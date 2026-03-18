"""
Phase 1: pretrained Kinetics-400 CTR-GCN → fine-tune on AVA-Kinetics (multi-label)
Phase 2: Phase-1 best.pth → fine-tune on AVA (multi-label)

Both phases share the same Trainer; the dataset and config differ.
Head is rebuilt when num_classes changes between phases.

Pipeline now includes:
  - Pre-training dataset audit
  - Experiment registry (register run, update final metrics)
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


# Scheduler building moved to scheduler_factory.py


def run_phase(config_path: str, phase_name: str,
              pretrained_override: str = None,
              class_registry=None) -> dict:
    """
    Run a single training phase.

    Parameters
    ----------
    config_path        : path to phase config YAML
    phase_name         : "phase1" or "phase2"
    pretrained_override: optional path to override pretrained checkpoint
    class_registry     : optional ClassRegistry subset for class-selective training
    """
    from utils.config_loader import apply_class_selection

    config = load_config(config_path)
    if pretrained_override:
        config.pretrained_ckpt = pretrained_override

    # Apply class selection if provided
    if class_registry is not None:
        config = apply_class_selection(config, class_registry)

    set_seed(config.seed)
    preflight(required_disk_gb=20.0)

    log.info(f"  Num classes : {config.num_classes}")
    if class_registry:
        log.info(f"  Classes     : {class_registry.class_names}")

    # ── Dataset Audit ──────────────────────────────────────────────────────
    _run_audit_if_available(config, class_registry)

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, train_dataset = create_dataloaders(
        config, class_registry=class_registry)
    config.num_classes = train_dataset.reg.num_classes  # authoritative source

    from training.resume import setup_training_run

    model, optimizer, scheduler, ema, es, ckpt_mgr, start_epoch, global_step = setup_training_run(
        config, class_registry, phase_name, len(train_loader)
    )

    # ── Experiment Registry ─────────────────────────────────────────────────
    run_id = _register_experiment(config, phase_name, class_registry)

    # ── Loss ────────────────────────────────────────────────────────────────
    loss_fn = build_loss(config, train_dataset, device=config.device)

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model, train_loader, val_loader,
        optimizer, scheduler, loss_fn,
        ema, es, ckpt_mgr, config,
        class_names=train_dataset.reg.class_names,
        start_epoch=start_epoch,
        best_metric=ckpt_mgr.best_metric,
        global_step=global_step,
    )
    best_metrics = trainer.fit()

    # ── Update Experiment Registry ──────────────────────────────────────────
    _update_experiment(run_id, best_metrics, ckpt_mgr, config, phase_name)

    return best_metrics


def _run_audit_if_available(config, class_registry):
    """Run dataset audit if class_registry is available."""
    if class_registry is None:
        return
    try:
        from utils.dataset_auditor import DatasetAuditor, AuditError
        auditor = DatasetAuditor()
        annotation_dir = str(Path(config.data_dir) / "annotations")
        min_quality = getattr(config, "min_quality", 0.30)
        auditor.audit(annotation_dir, class_registry, min_quality)
    except AuditError as e:
        log.error(f"Dataset audit failed:\n{e}")
        raise
    except FileNotFoundError:
        log.warning("Annotation directory not found — skipping audit")
    except Exception as e:
        log.warning(f"Dataset audit skipped: {e}")


def _register_experiment(config, phase_name, class_registry):
    """Register this run in the experiment registry."""
    try:
        from utils.experiment_registry import ExperimentRegistry
        import datetime
        registry = ExperimentRegistry()
        run_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{phase_name}_{config.experiment_name}"
        config_dict = config.__dict__ if hasattr(config, '__dict__') else dict(config)
        registry.register(
            run_id=run_id,
            phase=phase_name,
            config=config_dict,
            class_names=class_registry.class_names if class_registry else [],
            class_ids=class_registry.class_ids if class_registry else [],
        )
        return run_id
    except Exception as e:
        log.warning(f"Could not register experiment: {e}")
        return None


def _update_experiment(run_id, best_metrics, ckpt_mgr, config, phase_name):
    """Update experiment registry with final metrics."""
    if run_id is None:
        return
    try:
        from utils.experiment_registry import ExperimentRegistry
        registry = ExperimentRegistry()
        per_class_ap = {}
        if "per_class" in best_metrics:
            per_class_ap = {name: d.get("AP", 0.0)
                           for name, d in best_metrics["per_class"].items()}
        registry.update_final(
            run_id=run_id,
            best_mAP=best_metrics.get("mAP", 0.0),
            per_class_AP=per_class_ap,
            best_epoch=best_metrics.get("epoch", 0),
            total_epochs=config.epochs,
            checkpoint_path=str(ckpt_mgr.run_dir / "best.pth"),
        )
    except Exception as e:
        log.warning(f"Could not update experiment registry: {e}")


def run_full_pipeline(base_config: str = "configs/base_config.yaml",
                      class_registry=None):
    """
    Run the full 2-phase training pipeline.

    Parameters
    ----------
    base_config    : path to base config YAML
    class_registry : optional ClassRegistry subset for class-selective training
    """
    log.info("=" * 60)
    log.info("PHASE 1: AVA-Kinetics fine-tuning")
    log.info("=" * 60)
    p1_metrics = run_phase("configs/phase1_ava_kinetics.yaml", "phase1",
                            class_registry=class_registry)

    # Pass Phase 1 best checkpoint to Phase 2
    phase1_best = str(Path("checkpoints/runs/phase1_phase1_ava_kinetics/best.pth"))

    log.info("=" * 60)
    log.info("PHASE 2: AVA fine-tuning")
    log.info("=" * 60)
    p2_metrics = run_phase("configs/phase2_ava.yaml", "phase2",
                            pretrained_override=phase1_best,
                            class_registry=class_registry)

    log.info("Pipeline complete.")
    log.info(f"Phase 1 best mAP: {p1_metrics.get('mAP', 0):.4f}")
    log.info(f"Phase 2 best mAP: {p2_metrics.get('mAP', 0):.4f}")
    return p1_metrics, p2_metrics
