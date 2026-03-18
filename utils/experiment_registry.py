"""
Lightweight experiment registry backed by a single JSON file.
Records every training run and its final metrics.
No external dependencies — just JSON + cross-platform file locking.
"""

import json, logging, os, sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

log = logging.getLogger("experiment_registry")

REGISTRY_PATH = "outputs/experiment_registry.json"


def _lock_file(f):
    """Acquire an exclusive file lock (cross-platform)."""
    if sys.platform == "win32":
        import msvcrt
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl
        fcntl.flock(f, fcntl.LOCK_EX)


def _unlock_file(f):
    """Release file lock (cross-platform)."""
    if sys.platform == "win32":
        import msvcrt
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass  # already unlocked
    else:
        import fcntl
        fcntl.flock(f, fcntl.LOCK_UN)


class ExperimentRegistry:

    def __init__(self, registry_path: str = REGISTRY_PATH):
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def register(self, run_id: str, phase: str,
                 config: dict, class_names: List[str],
                 class_ids: List[int]) -> str:
        """
        Register a new run at start of training.
        Returns the run_id for reference.
        """
        entry = {
            "run_id":      run_id,
            "phase":       phase,
            "started_at":  datetime.now().isoformat(),
            "finished_at": None,
            "status":      "running",
            "class_names": class_names,
            "class_ids":   class_ids,
            "num_classes": len(class_ids),
            "config": {
                "lr_backbone":   config.get("lr_backbone"),
                "lr_head":       config.get("lr_head"),
                "batch_size":    config.get("batch_size"),
                "scheduler":     config.get("scheduler"),
                "loss_type":     config.get("loss_type"),
                "epochs":        config.get("epochs"),
                "dataset":       config.get("dataset"),
            },
            "best_mAP":        None,
            "per_class_AP":    {},
            "best_epoch":      None,
            "total_epochs":    None,
            "checkpoint_path": None,
        }
        runs = self._read()
        runs.append(entry)
        self._write(runs)
        log.info(f"Registered run: {run_id}")
        return run_id

    def update_final(self, run_id: str, best_mAP: float,
                     per_class_AP: dict, best_epoch: int,
                     total_epochs: int, checkpoint_path: str,
                     status: str = "completed"):
        """Update a run with final metrics at end of training."""
        runs = self._read()
        for run in runs:
            if run["run_id"] == run_id:
                run.update({
                    "finished_at":    datetime.now().isoformat(),
                    "status":         status,
                    "best_mAP":       round(best_mAP, 4),
                    "per_class_AP":   {k: round(v, 4)
                                      for k, v in per_class_AP.items()},
                    "best_epoch":     best_epoch,
                    "total_epochs":   total_epochs,
                    "checkpoint_path": checkpoint_path,
                })
                break
        self._write(runs)

    def find_best_run(self, class_names: List[str] = None,
                      phase: str = None,
                      min_mAP: float = 0.0) -> Optional[dict]:
        """
        Find the best completed run matching optional filters.
        Returns the run dict or None.
        """
        runs = self._read()
        candidates = [
            r for r in runs
            if r["status"] == "completed"
            and (r["best_mAP"] or 0) >= min_mAP
            and (phase is None or r["phase"] == phase)
            and (class_names is None or
                 set(class_names).issubset(set(r["class_names"])))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r["best_mAP"] or 0)

    def print_summary(self, last_n: int = 10):
        """Print a table of recent runs."""
        runs = sorted(self._read(),
                      key=lambda r: r.get("started_at", ""),
                      reverse=True)[:last_n]
        print(f"\n{'Run ID':<36} {'Classes':<20} {'mAP':>7} {'Status':<12}")
        print("─" * 80)
        for r in runs:
            classes = ", ".join(r["class_names"][:3])
            if len(r["class_names"]) > 3:
                classes += f" +{len(r['class_names'])-3}"
            mAP = f"{r['best_mAP']:.4f}" if r["best_mAP"] else "—"
            print(f"{r['run_id']:<36} {classes:<20} {mAP:>7} {r['status']:<12}")
        print()

    def _read(self) -> list:
        try:
            with open(self.path) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, runs: list):
        """Write with file lock to prevent concurrent corruption."""
        with open(self.path, "w") as f:
            _lock_file(f)
            json.dump(runs, f, indent=2)
            _unlock_file(f)
