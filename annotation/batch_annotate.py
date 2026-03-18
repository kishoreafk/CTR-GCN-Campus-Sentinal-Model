"""
Batch annotation runner. Processes all downloaded videos for a dataset,
extracts skeleton samples, and tracks progress in the state DB.

Supports class-selective annotation and idempotent re-runs:
- Accepts a ClassRegistry subset to only annotate selected classes
- Checks DB + file existence before processing each sample
- Groups all annotations for a video so it's loaded only once
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from annotation.pose_estimator import PoseEstimator
from annotation.joint_converter import batch_coco17_to_openpose18
from annotation.extractor import SkeletonExtractor
from annotation.quality_validator import AnnotationQualityValidator
from utils.db_manager import DBManager
from utils.class_registry import ClassRegistry

log = logging.getLogger("batch_annotate")


class BatchAnnotator:
    def __init__(self, config, selected_registry=None):
        """
        Parameters
        ----------
        config             : TrainingConfig
        selected_registry  : ClassRegistry subset (only selected classes).
                             If None, uses all classes from config.
        """
        self.cfg = config
        self.db = DBManager(config.state_db)
        self.reg = selected_registry or ClassRegistry(config.class_config)
        self.pose = PoseEstimator(config)
        self.extractor = SkeletonExtractor(
            self.pose, batch_coco17_to_openpose18, self.reg, config)
        self.validator = AnnotationQualityValidator()
        self.min_quality = 0.30

    def _load_annotations(self, dataset: str) -> pd.DataFrame:
        """Load AVA-style annotation CSV, filtered to selected classes."""
        ann_dir = Path(self.cfg.data_dir) / "annotations" / dataset

        # Try standard AVA annotation file names
        for name in [f"{dataset}_train_v2.2.csv",
                     f"{dataset}_val_v2.2.csv",
                     f"{dataset}_train_v1.0.csv",
                     f"{dataset}_val_v1.0.csv",
                     f"{dataset}_annotations.csv",
                     "train.csv", "val.csv"]:
            p = ann_dir / name
            if p.exists():
                df = pd.read_csv(p, header=None,
                                 names=["video_id", "timestamp", "x1", "y1",
                                        "x2", "y2", "action_id", "person_id"])
                return self.reg.filter_annotations(df)

        # Try all CSVs
        frames = []
        for csv_file in ann_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, header=None,
                                 names=["video_id", "timestamp", "x1", "y1",
                                        "x2", "y2", "action_id", "person_id"])
                frames.append(self.reg.filter_annotations(df))
            except Exception:
                continue
        if frames:
            return pd.concat(frames, ignore_index=True)

        log.warning(f"No annotation CSV found in {ann_dir}")
        return pd.DataFrame()

    def _find_video_file(self, video_dir: Path, video_id: str) -> Optional[Path]:
        """
        Find a video file for video_id in video_dir.
        Tries extensions: .mp4, .avi, .mkv, .webm
        """
        for ext in (".mp4", ".avi", ".mkv", ".webm"):
            p = video_dir / f"{video_id}{ext}"
            if p.exists():
                return p
        # Also try files that start with video_id (yt-dlp may add suffix)
        matches = list(video_dir.glob(f"{video_id}*"))
        return matches[0] if matches else None

    def _npz_path(self, out_dir: Path, video_id: str,
                  timestamp: float, person_id: str) -> Path:
        """Deterministic path for consistent idempotency checks."""
        ts_str = f"{timestamp:.1f}".replace(".", "_")
        return out_dir / f"{video_id}__{ts_str}__{person_id}.npz"

    def _parse_npz_filename(self, filename: str):
        """
        Parse deterministic filename: {video_id}__{ts}__{person_id}.npz
        Returns (video_id, timestamp_float, person_id) or (None, None, None).
        """
        try:
            stem   = filename.replace(".npz", "")
            parts  = stem.split("__")
            if len(parts) != 3:
                return None, None, None
            vid_id    = parts[0]
            ts        = float(parts[1].replace("_", "."))
            person_id = parts[2]
            return vid_id, ts, person_id
        except Exception:
            return None, None, None

    def recover_interrupted_annotations(self, out_dir: Path, dataset: str):
        """
        Run once before any annotation session begins.

        Actions:
          1. Delete .tmp.npz files (partial writes from previous crash)
          2. Find .npz files on disk not marked done in DB → validate + register
          3. Find DB entries stuck in 'annotating' state → reset to pending
          4. Find DB entries marked 'done' but .npz missing → reset to pending
        """
        summary = {
            "tmp_deleted":       0,
            "orphan_registered": 0,
            "stale_reset":       0,
            "missing_reset":     0,
        }

        # ── 1. Delete partial .npz files ──────────────────────────────────────
        for tmp in out_dir.rglob("*.tmp.npz"):
            tmp.unlink()
            summary["tmp_deleted"] += 1
            log.info(f"Deleted partial annotation: {tmp.name}")

        # ── 2. Register orphan .npz files ─────────────────────────────────────
        for npz in out_dir.rglob("*.npz"):
            vid_id, ts, person_id = self._parse_npz_filename(npz.name)
            if vid_id is None:
                continue
            if not self.db.is_annotated(vid_id, dataset, ts, person_id):
                result = self.validator.validate_single(str(npz))
                if result["valid"]:
                    d = np.load(str(npz), allow_pickle=True)
                    self.db.mark_annotation_done(
                        vid_id, dataset, ts, person_id,
                        str(npz), float(d.get("quality_score", 0.5))
                    )
                    summary["orphan_registered"] += 1
                else:
                    npz.unlink()
                    log.warning(
                        f"Deleted invalid orphan annotation: {npz.name} "
                        f"({result['issues']})"
                    )

        # ── 3. Reset stale 'annotating' DB entries ─────────────────────────────
        stale = self.db.get_stale_annotations(
            dataset, max_age_minutes=60
        )
        for key in stale:
            self.db.reset_annotation_to_pending(*key, dataset)
            summary["stale_reset"] += 1

        # ── 4. Reset 'done' entries whose .npz disappeared ─────────────────────
        done_entries = self.db.get_done_annotation_entries(dataset)
        for vid_id, ts, person_id, npz_path in done_entries:
            if not Path(npz_path).exists():
                self.db.reset_annotation_to_pending(
                    vid_id, dataset, ts, person_id
                )
                summary["missing_reset"] += 1
                log.warning(
                    f".npz disappeared for {vid_id} t={ts} p={person_id} "
                    f"→ reset to pending"
                )

        log.info(
            f"Annotation recovery: "
            f"tmp_deleted={summary['tmp_deleted']}  "
            f"orphan_registered={summary['orphan_registered']}  "
            f"stale_reset={summary['stale_reset']}  "
            f"missing_reset={summary['missing_reset']}"
        )
        return summary

    def _get_pending_samples(self, video_id: str, video_df: pd.DataFrame,
                              dataset: str, out_dir: Path) -> list:
        """
        For each (timestamp, person_id) in this video's annotations:
          - Check DBManager: if status='done' AND .npz exists -> skip
          - Check .npz path directly: if file exists and is valid -> mark done + skip
          - Otherwise: add to pending list
        """
        pending = []

        for (ts, pid), group in video_df.groupby(["timestamp", "person_id"]):
            person_id = str(pid)
            action_ids = group["action_id"].tolist()

            # Filter to selected classes only
            action_ids = [a for a in action_ids if a in self.reg._id_to_idx]
            if not action_ids:
                continue

            person_bbox = group[["x1", "y1", "x2", "y2"]].iloc[0].values
            npz_path = self._npz_path(out_dir, video_id, float(ts), person_id)

            # Idempotency check 1: DB says done
            if self.db.is_annotated(video_id, dataset, float(ts), person_id):
                if npz_path.exists():
                    continue  # genuinely done
                else:
                    log.warning(
                        f"DB says done but .npz missing: "
                        f"{video_id} t={ts} p={person_id}. Re-annotating."
                    )

            # Idempotency check 2: file exists on disk
            if npz_path.exists():
                v = self.validator.validate_single(str(npz_path))
                if v["valid"]:
                    self.db.mark_annotation_done(
                        video_id, dataset, float(ts), person_id,
                        str(npz_path), quality_score=1.0
                    )
                    continue
                else:
                    log.warning(
                        f"Existing .npz invalid — re-annotating: {npz_path.name}"
                    )
                    npz_path.unlink()

            pending.append({
                "timestamp":   float(ts),
                "person_id":   person_id,
                "person_bbox": np.array(person_bbox, dtype=np.float32),
                "action_ids":  action_ids,
            })

        return pending

    def _process_group(self, video_id: str, group: pd.DataFrame,
                       dataset: str) -> dict:
        """Process all annotations for a single video."""
        video_dir = Path(self.cfg.data_dir) / "raw" / dataset / "videos"
        out_dir = Path(self.cfg.data_dir) / "processed" / dataset / "skeletons"
        out_dir.mkdir(parents=True, exist_ok=True)

        result = {"done": 0, "skipped": 0, "low_quality": 0, "failed": 0}

        video_path = self._find_video_file(video_dir, video_id)
        if video_path is None:
            log.warning(f"Video not found: {video_id}")
            result["failed"] = 1
            return result

        # Get only the pending (not-yet-annotated) samples
        pending = self._get_pending_samples(video_id, group, dataset, out_dir)
        if not pending:
            result["skipped"] = len(group.groupby(["timestamp", "person_id"]))
            return result

        for sample_info in pending:
            ts = sample_info["timestamp"]
            person_id = sample_info["person_id"]
            npz_path = self._npz_path(out_dir, video_id, ts, person_id)

            try:
                self.db.mark_annotation_start(video_id, dataset, ts, person_id)
                sample = self.extractor.extract_ava_sample(
                    str(video_path), ts,
                    sample_info["person_bbox"],
                    sample_info["action_ids"])

                if sample is None:
                    self.db.mark_annotation_failed(
                        video_id, dataset, ts, person_id,
                        "Extraction returned None")
                    continue

                if sample.get("quality_score", 1.0) < self.min_quality:
                    log.debug(
                        f"Low quality ({sample['quality_score']:.3f}) "
                        f"{video_id} t={ts} — skipping"
                    )
                    result["low_quality"] += 1
                    self.db.mark_annotation_failed(
                        video_id, dataset, ts, person_id,
                        f"quality={sample['quality_score']:.3f} < {self.min_quality}")
                    continue

                self.extractor.save_sample(sample, str(npz_path))
                self.db.mark_annotation_done(
                    video_id, dataset, ts, person_id,
                    str(npz_path), sample["quality_score"])
                result["done"] += 1

            except Exception as e:
                log.error(f"Failed: {video_id} ts={ts} pid={person_id}: {e}")
                self.db.mark_annotation_failed(
                    video_id, dataset, ts, person_id, str(e))

        return result

    def run(self, dataset: str = None):
        """Run batch annotation for the configured dataset."""
        dataset = dataset or self.cfg.dataset
        log.info(f"Starting batch annotation for {dataset}")
        log.info(f"Classes: {self.reg.class_names}")

        out_dir = Path(self.cfg.data_dir) / "processed" / dataset / "skeletons"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.recover_interrupted_annotations(out_dir, dataset)

        df = self._load_annotations(dataset)
        if df.empty:
            log.warning("No annotations to process")
            return

        # Group by video_id to load each video only once
        grouped = df.groupby("video_id")

        # Filter to downloaded videos only
        available_groups = []
        for video_id, video_df in grouped:
            if self.db.is_downloaded(video_id, dataset):
                available_groups.append((video_id, video_df))

        if not available_groups:
            log.warning("No downloaded videos found. Run download first.")
            return

        log.info(f"Processing {len(available_groups)} videos with "
                 f"{len(df)} annotation rows")

        total_done = total_skipped = total_low_qual = total_failed = 0

        for video_id, video_df in tqdm(available_groups, desc="Annotating"):
            r = self._process_group(video_id, video_df, dataset)
            total_done     += r["done"]
            total_skipped  += r["skipped"]
            total_low_qual += r["low_quality"]
            total_failed   += r["failed"]

        log.info(
            f"\nAnnotation complete for {dataset}:\n"
            f"  Videos processed : {len(available_groups)}\n"
            f"  Samples done     : {total_done}\n"
            f"  Samples skipped  : {total_skipped} (already existed)\n"
            f"  Low quality      : {total_low_qual}\n"
            f"  Failed           : {total_failed}"
        )

        # Run quality validation
        skel_dir = str(Path(self.cfg.data_dir) / "processed" / dataset / "skeletons")
        qr = self.validator.validate_dir(skel_dir)
        log.info(f"Quality: {qr['valid']}/{qr['total']} valid, "
                 f"{qr['invalid']} quarantined")

        # Print DB summary
        summary = self.db.annotation_summary(dataset)
        log.info(f"DB summary: {summary}")

