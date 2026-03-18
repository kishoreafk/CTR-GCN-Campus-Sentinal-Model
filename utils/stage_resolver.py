"""
Resolves --start_from / --skip_* flags into an ordered list of stages.

Stage order is always:  download -> annotate -> train -> evaluate
Skipping removes stages from the front or middle of this list.

Also validates that prerequisites exist when stages are skipped.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List
import argparse

log = logging.getLogger("stage_resolver")

STAGE_ORDER = ["download", "annotate", "train", "evaluate"]


def resolve_stages(args: argparse.Namespace) -> List[str]:
    """
    Returns ordered list of stages to execute.

    --start_from X      : skip all stages before X (inclusive of X)
    --skip_download     : remove 'download' from list
    --skip_annotate     : remove 'annotate' from list
    --skip_train        : remove 'train' from list
    --mode download     : run only that one stage
    --mode full_pipeline: run all stages (subject to skip flags)
    """
    mode = args.mode

    # Single-stage modes: no resolution needed
    if mode in ("test_run", "status", "export"):
        return [mode]

    if mode != "full_pipeline":
        # Explicit single stage requested
        return [mode]

    # full_pipeline: start from all, then apply skips
    stages = list(STAGE_ORDER)

    if getattr(args, "start_from", None):
        idx = STAGE_ORDER.index(args.start_from)
        stages = STAGE_ORDER[idx:]
        log.info(f"--start_from {args.start_from}: "
                 f"skipping {STAGE_ORDER[:idx]}")

    if getattr(args, "skip_download", False) and "download" in stages:
        stages.remove("download")
    if getattr(args, "skip_annotate", False) and "annotate" in stages:
        stages.remove("annotate")
    if getattr(args, "skip_train", False) and "train" in stages:
        stages.remove("train")

    if not stages:
        raise ValueError("All stages skipped — nothing to do.")

    log.info(f"Resolved stages: {stages}")
    return stages


def validate_stage_prerequisites(stages: List[str], config,
                                  selected_registry) -> List[str]:
    """
    Checks that data required by each stage already exists when that
    stage's prerequisite has been skipped.

    Returns a list of warning strings (non-fatal) — prints them before running.
    Raises RuntimeError on fatal missing prerequisites.
    """
    warnings = []

    if "download" not in stages and "annotate" in stages:
        # Videos must already exist
        video_dir = Path(config.data_dir) / "raw" / config.dataset / "videos"
        videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        if not videos:
            raise RuntimeError(
                f"--skip_download specified but no videos found in {video_dir}.\n"
                f"Run: python main.py --mode download "
                f"--classes {' '.join(selected_registry.class_names)}"
            )
        warnings.append(
            f"Found {len(videos)} videos in {video_dir} — using existing downloads."
        )

    if "annotate" not in stages and "train" in stages:
        # Skeletons must already exist
        skel_dir = Path(config.data_dir) / "processed" / config.dataset / "skeletons"
        skeletons = list(skel_dir.rglob("*.npz"))
        if not skeletons:
            raise RuntimeError(
                f"--skip_annotate specified but no .npz files found in {skel_dir}.\n"
                f"Run: python main.py --mode annotate "
                f"--classes {' '.join(selected_registry.class_names)}"
            )

        # Check that existing skeletons cover the requested classes
        covered_ids = _get_annotated_class_ids(skeletons, config)
        missing = [cid for cid in selected_registry.class_ids
                   if cid not in covered_ids]
        if missing:
            missing_names = [c["name"] for c in selected_registry._classes
                             if c["id"] in missing]
            raise RuntimeError(
                f"Skeleton files exist but are missing annotations for: "
                f"{missing_names}\n"
                f"Run annotation for these classes first."
            )
        warnings.append(
            f"Found {len(skeletons)} skeleton files covering all requested classes."
        )

    return warnings


def _get_annotated_class_ids(npz_paths, config) -> set:
    """
    Scan a random sample of .npz files to determine which
    AVA class IDs have been annotated.
    Reads at most 200 files to avoid slow startup.
    """
    ids = set()
    sample = npz_paths[:200]  # fast check

    for p in sample:
        try:
            d = np.load(str(p), allow_pickle=True)
            aids = list(d.get("action_ids", []))
            ids.update(aids)
        except Exception:
            continue
    return ids
