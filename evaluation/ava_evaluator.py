"""
AVA-style evaluator: loads best checkpoint, runs through validation set,
computes per-class AP and mean AP (mAP).
"""
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict

from training.metrics import MultiLabelMetrics
from utils.class_registry import ClassRegistry
from utils.config_loader import load_config

log = logging.getLogger("ava_evaluator")


class AVAEvaluator:
    """
    AVA-style evaluator: loads best checkpoint, runs through
    validation set, computes per-class AP and mAP.

    Accepts optional class_registry for class-selective evaluation.
    """

    def __init__(self, config, class_registry=None):
        self.cfg = config
        self.class_registry = class_registry
        self.reg = class_registry or ClassRegistry(config.class_config)
        self.device = torch.device(config.device)

    def evaluate(self, checkpoint_path: str = None, use_tta: bool = False) -> Dict:
        """Run evaluation using best checkpoint."""
        from models.model_factory import build_model
        from data_pipeline.dataloader_factory import create_dataloaders

        # Find checkpoint
        if checkpoint_path is None:
            # Auto-find best checkpoint
            runs_dir = Path(self.cfg.checkpoint_dir) / "runs"
            best_paths = list(runs_dir.rglob("best.pth"))
            if not best_paths:
                log.error("No best.pth found")
                return {}
            checkpoint_path = str(best_paths[-1])

        log.info(f"Evaluating checkpoint: {checkpoint_path}")

        # Load model
        ckpt = torch.load(checkpoint_path, map_location="cpu",
                          weights_only=False)
        self.cfg.num_classes = ckpt.get("num_classes", self.reg.num_classes)

        # Don't torch.compile for eval
        self.cfg.use_compile = False
        model = build_model(self.cfg)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval()

        # Load EMA if available
        if ckpt.get("ema_state"):
            try:
                model.load_state_dict(ckpt["ema_state"], strict=False)
                log.info("Loaded EMA weights for evaluation")
            except Exception as e:
                log.warning(f"Could not load EMA: {e}")

        # Data — pass class_registry if available
        _, val_loader, _ = create_dataloaders(
            self.cfg, class_registry=self.class_registry)

        # Evaluate — optionally with Test-Time Augmentation
        metrics = MultiLabelMetrics(self.cfg.num_classes, self.reg.class_names)

        if use_tta:
            from evaluation.tta import TTAEvaluator
            tta = TTAEvaluator(model, str(self.device))
            log.info("Using Test-Time Augmentation (4-way)")

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(self.device, non_blocking=True)
                y = batch["label"].to(self.device, non_blocking=True)
                if use_tta:
                    probs = tta.predict(x)
                    # Convert probs back to logits for metrics.update
                    # (which expects raw logits and applies sigmoid internally)
                    logits = torch.log(probs / (1 - probs + 1e-8))
                else:
                    logits = model(x)
                metrics.update(logits, y)

        result = metrics.compute()

        # Log results
        log.info(f"\n{'='*60}")
        log.info(f"EVALUATION RESULTS")
        log.info(f"{'='*60}")
        log.info(f"mAP: {result['mAP']:.4f}")
        log.info(f"\nPer-class results:")
        log.info(f"{'Class':30s}  {'AP':>6s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}")
        log.info("-" * 60)
        for name, d in result["per_class"].items():
            log.info(f"{name:30s}  {d['AP']:6.3f}  {d['P']:6.3f}  "
                     f"{d['R']:6.3f}  {d['F1']:6.3f}")

        # Save results
        import json
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"\nResults saved to {output_dir / 'eval_results.json'}")

        return result
