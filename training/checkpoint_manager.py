"""
Saves best.pth, last.pth, and epoch_N.pth.
Checkpoint contains ALL state for exact reconstruction.
"""
import torch, json, shutil, logging
from pathlib import Path
from dataclasses import asdict
log = logging.getLogger("checkpoint")

class CheckpointManager:
    def __init__(self, run_dir: str, keep_k: int = 3):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.keep_k   = keep_k
        self.best_mAP = 0.0

    def save(self, epoch, model, optimizer, scheduler,
             ema, early_stopping, metrics, config):
        payload = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "ema_state":       ema.state_dict() if ema else None,
            "early_stopping":  early_stopping.state_dict() if early_stopping else None,
            "metrics":         {k: v for k, v in metrics.items()
                                if not isinstance(v, dict)},
            "config":          asdict(config) if hasattr(config, "__dataclass_fields__")
                               else vars(config),
            "joint_layout":    "openpose_18",
            "num_classes":     config.num_classes,
        }
        # Always save last
        torch.save(payload, self.run_dir / "last.pth")

        # Save best
        current_mAP = metrics.get("mAP", 0.0)
        if current_mAP > self.best_mAP:
            self.best_mAP = current_mAP
            torch.save(payload, self.run_dir / "best.pth")
            log.info(f"New best mAP={current_mAP:.4f} — saved best.pth")

        # Periodic checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0:
            p = self.run_dir / f"epoch_{epoch:04d}.pth"
            torch.save(payload, p)
            self._prune_old()

    def _prune_old(self):
        old = sorted(self.run_dir.glob("epoch_*.pth"))
        for p in old[:-self.keep_k]:
            p.unlink()

    def load(self, path: str, model, optimizer=None, scheduler=None,
             ema=None, early_stopping=None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if optimizer and ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ema and ckpt.get("ema_state"):
            ema.load_state_dict(ckpt["ema_state"])
        if early_stopping and ckpt.get("early_stopping"):
            early_stopping.load_state_dict(ckpt["early_stopping"])
        log.info(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
        return ckpt["epoch"], ckpt.get("metrics", {})
