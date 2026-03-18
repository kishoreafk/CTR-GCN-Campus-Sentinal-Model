"""
Checkpoint Manager with atomic saves and full state tracking.
Ensures we can resume training exactly where it left off,
including RNG state, scheduler, and early stopping.
"""

import sys, datetime, logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import numpy as np
import random

log = logging.getLogger("checkpoint_manager")


class CheckpointManager:
    def __init__(self, run_dir: str, keep_k: int = 3,
                 metric: str = "mAP", mode: str = "max"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.keep_k = keep_k
        self.metric = metric
        self.mode = mode

        # Track history
        self.best_metric = -float('inf') if mode == "max" else float('inf')
        self._metrics_history = []
        self._saved_epochs = []

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "max":
            return current > best
        return current < best

    def _unwrap_model(self, model):
        """Handle torch.compile wrapper if present."""
        return model._orig_mod if hasattr(model, '_orig_mod') else model

    def _get_rng_state(self) -> dict:
        """Capture all PRNG states for deterministic resume."""
        state = {
            "python": random.getstate(),
            "numpy":  np.random.get_state(),
            "torch":  torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            # torch.cuda.get_rng_state_all() returns a list of ByteTensors
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _set_rng_state(self, state: dict):
        """Restore all PRNG states."""
        try:
            if "python" in state:
                random.setstate(state["python"])
            if "numpy" in state:
                np.random.set_state(state["numpy"])
            if "torch" in state:
                torch.random.set_rng_state(state["torch"])
            if "cuda" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda"])
        except Exception as e:
            log.warning(f"Could not cleanly restore RNG state: {e}")

    def _atomic_save(self, state: dict, path: Path):
        """Write to .tmp then rename to prevent corrupt files on crash."""
        tmp_path = path.with_suffix(".tmp.pth")
        try:
            torch.save(state, tmp_path)
            tmp_path.replace(path)
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            log.error(f"Failed to save checkpoint {path.name}: {e}")
            raise

    def save(self, model, optimizer, scheduler, ema, early_stopping,
             epoch: int, global_step: int, metrics: dict,
             config, class_registry) -> bool:
        """
        Save periodic epoch checkpoint, update best.pth, and update last.pth.
        Returns True if this was a new best checkpoint.
        """
        model_ut = self._unwrap_model(model)
        
        # Build payload matching CHECKPOINT_SCHEMA
        state = {
            "metadata": {
                "epoch": epoch,
                "global_step": global_step,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "torch_version": torch.__version__,
                "num_classes": class_registry.num_classes if class_registry else None,
                "class_names": class_registry.class_names if class_registry else None,
            },
            "model": model_ut.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "ema": ema.state_dict() if ema else None,
            "early_stopping": early_stopping.state_dict() if early_stopping else None,
            "rng": self._get_rng_state(),
            "metrics": metrics,
            "metrics_history": self._metrics_history,
            "config": config.__dict__ if hasattr(config, '__dict__') else dict(config),
        }
        
        self._metrics_history.append(metrics)

        # ── 1. Save last.pth (Emergency/Resume default) ──────────────
        self._atomic_save(state, self.run_dir / "last.pth")

        # ── 2. Save epoch_NNNN.pth ───────────────────────────────────
        epoch_path = self.run_dir / f"epoch_{epoch:04d}.pth"
        self._atomic_save(state, epoch_path)
        self._saved_epochs.append(epoch_path)

        # Prune old epoch checkpoints
        while len(self._saved_epochs) > self.keep_k:
            oldest = self._saved_epochs.pop(0)
            oldest.unlink(missing_ok=True)

        # ── 3. Save best.pth ─────────────────────────────────────────
        current_val = metrics.get(self.metric)
        is_best = False
        if current_val is not None:
            if self._is_better(current_val, self.best_metric):
                self.best_metric = current_val
                is_best = True
                self._atomic_save(state, self.run_dir / "best.pth")
                log.info(f"New best {self.metric}: {current_val:.4f}")

        return is_best

    def save_emergency(self, model, optimizer, scheduler, ema, early_stopping,
                       epoch: int, global_step: int, config, class_registry):
        """Save a quick last.pth on exceptions/signals without pruning logic."""
        model_ut = self._unwrap_model(model)
        state = {
            "metadata": {
                "epoch": epoch,
                "global_step": global_step,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "torch_version": torch.__version__,
                "num_classes": class_registry.num_classes if class_registry else None,
            },
            "model": model_ut.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "ema": ema.state_dict() if ema else None,
            "early_stopping": early_stopping.state_dict() if early_stopping else None,
            "rng": self._get_rng_state(),
            "metrics_history": self._metrics_history,
            "config": config.__dict__ if hasattr(config, '__dict__') else dict(config),
        }
        self._atomic_save(state, self.run_dir / "last.pth")
        log.info(f"Emergency checkpoint saved at epoch {epoch}, step {global_step}.")

    def find_resume_checkpoint(self) -> Optional[Path]:
        """Priority: last.pth > latest epoch_*.pth > best.pth"""
        last = self.run_dir / "last.pth"
        if last.exists():
            return last

        # Find latest epoch_*.pth
        epoch_ckpts = sorted(self.run_dir.glob("epoch_*.pth"))
        if epoch_ckpts:
            return epoch_ckpts[-1]

        best = self.run_dir / "best.pth"
        if best.exists():
            return best

        return None

    def load(self, path: str, model, optimizer=None, scheduler=None,
             ema=None, early_stopping=None, device="cpu") -> Tuple[int, int, dict]:
        """
        Load checkpoint into provided objects.
        Returns: (epoch, global_step, metrics)
        """
        log.info(f"Loading checkpoint: {path}")
        # Use map_location=device to load directly to the target device
        ckpt = torch.load(path, map_location=device)

        model_ut = self._unwrap_model(model)
        model_ut.load_state_dict(ckpt["model"])

        if optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            
        if scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
            scheduler.load_state_dict(ckpt["scheduler"])
            
        if ema and "ema" in ckpt and ckpt["ema"]:
            ema.load_state_dict(ckpt["ema"])
            
        if early_stopping and "early_stopping" in ckpt and ckpt["early_stopping"]:
            early_stopping.load_state_dict(ckpt["early_stopping"])

        if "rng" in ckpt:
            self._set_rng_state(ckpt["rng"])

        if "metrics_history" in ckpt:
            self._metrics_history = ckpt["metrics_history"]
            
        epoch = ckpt.get("metadata", {}).get("epoch", 0)
        global_step = ckpt.get("metadata", {}).get("global_step", 0)
        metrics = ckpt.get("metrics", {})

        # Restore best metric tracker if history exists
        if self._metrics_history:
            vals = [m.get(self.metric) for m in self._metrics_history 
                    if m.get(self.metric) is not None]
            if vals:
                self.best_metric = max(vals) if self.mode == "max" else min(vals)

        log.info(f"Loaded epoch {epoch}, step {global_step}")
        return epoch, global_step, metrics
