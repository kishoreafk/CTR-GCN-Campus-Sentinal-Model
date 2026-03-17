"""
Resume utilities for training checkpoint recovery.
"""
import logging
from pathlib import Path

log = logging.getLogger("resume")


def find_latest_checkpoint(run_dir: str) -> str:
    """Find the latest checkpoint in a run directory."""
    run_path = Path(run_dir)
    if not run_path.exists():
        return None

    # Prefer last.pth for exact state restoration
    last = run_path / "last.pth"
    if last.exists():
        return str(last)

    # Fallback to latest epoch checkpoint
    epoch_ckpts = sorted(run_path.glob("epoch_*.pth"))
    if epoch_ckpts:
        return str(epoch_ckpts[-1])

    # Fallback to best.pth
    best = run_path / "best.pth"
    if best.exists():
        return str(best)

    return None


def find_phase1_best(checkpoint_dir: str = "checkpoints") -> str:
    """Find the best checkpoint from Phase 1 for Phase 2 initialization."""
    runs_dir = Path(checkpoint_dir) / "runs"
    if not runs_dir.exists():
        return None

    # Look for phase1 run directories
    for d in sorted(runs_dir.iterdir()):
        if d.is_dir() and "phase1" in d.name:
            best = d / "best.pth"
            if best.exists():
                log.info(f"Found Phase 1 best checkpoint: {best}")
                return str(best)

    return None
