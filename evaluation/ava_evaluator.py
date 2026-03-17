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
    def __init__(self, config):
        self.cfg = config
        self.reg = ClassRegistry(config.class_config)
        self.device = torch.device(config.device)

    def evaluate(self, checkpoint_path: str = None) -> Dict:
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

        # Data
        _, val_loader, _ = create_dataloaders(self.cfg)

        # Evaluate
        metrics = MultiLabelMetrics(self.cfg.num_classes, self.reg.class_names)

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(self.device, non_blocking=True)
                y = batch["label"].to(self.device, non_blocking=True)
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
