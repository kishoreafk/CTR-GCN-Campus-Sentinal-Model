"""
Batch annotation runner. Processes all downloaded videos for a dataset,
extracts skeleton samples, and tracks progress in the state DB.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from annotation.pose_estimator import PoseEstimator
from annotation.joint_converter import batch_coco17_to_openpose18
from annotation.extractor import SkeletonExtractor
from annotation.quality_validator import AnnotationQualityValidator
from utils.db_manager import DBManager
from utils.class_registry import ClassRegistry

log = logging.getLogger("batch_annotate")


class BatchAnnotator:
    def __init__(self, config):
        self.cfg = config
        self.db = DBManager(config.state_db)
        self.reg = ClassRegistry(config.class_config)
        self.pose = PoseEstimator(config)
        self.extractor = SkeletonExtractor(
            self.pose, batch_coco17_to_openpose18, self.reg, config)
        self.validator = AnnotationQualityValidator()

    def _load_annotations(self, dataset: str) -> pd.DataFrame:
        """Load AVA-style annotation CSV."""
        ann_dir = Path(self.cfg.data_dir) / "annotations" / dataset

        # Try standard AVA annotation file names
        for name in [f"{dataset}_train_v2.2.csv",
                     f"{dataset}_val_v2.2.csv",
                     f"{dataset}_annotations.csv",
                     "train.csv", "val.csv"]:
            p = ann_dir / name
            if p.exists():
                df = pd.read_csv(p, header=None,
                                 names=["video_id", "timestamp", "x1", "y1",
                                        "x2", "y2", "action_id", "person_id"])
                return self.reg.filter_annotations(df)

        log.warning(f"No annotation CSV found in {ann_dir}")
        return pd.DataFrame()

    def _process_group(self, video_id: str, group: pd.DataFrame,
                       dataset: str) -> int:
        """Process all annotations for a single video."""
        video_dir = Path(self.cfg.data_dir) / "raw" / dataset / "videos"
        out_dir = Path(self.cfg.data_dir) / "processed" / dataset / "skeletons"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Find the video file
        video_path = None
        for ext in [".mp4", ".mkv", ".webm", ".avi"]:
            p = video_dir / f"{video_id}{ext}"
            if p.exists():
                video_path = str(p)
                break

        if not video_path:
            log.warning(f"Video not found: {video_id}")
            return 0

        count = 0
        # Group by (timestamp, person_id) to get all actions for each person at each time
        for (ts, pid), subgroup in group.groupby(["timestamp", "person_id"]):
            person_id = str(pid)

            # Check if already annotated
            if self.db.is_annotated(video_id, dataset, float(ts), person_id):
                continue

            action_ids = subgroup["action_id"].tolist()
            bbox = np.array([
                subgroup.iloc[0]["x1"], subgroup.iloc[0]["y1"],
                subgroup.iloc[0]["x2"], subgroup.iloc[0]["y2"]
            ], dtype=np.float32)

            try:
                sample = self.extractor.extract_ava_sample(
                    video_path, float(ts), bbox, action_ids)

                if sample is None:
                    self.db.mark_annotation_failed(
                        video_id, dataset, float(ts), person_id,
                        "Extraction returned None")
                    continue

                out_name = f"{video_id}_{int(ts):06d}_{person_id}.npz"
                out_path = str(out_dir / out_name)
                self.extractor.save_sample(sample, out_path)

                self.db.mark_annotation_done(
                    video_id, dataset, float(ts), person_id,
                    out_path, sample["quality_score"])
                count += 1

            except Exception as e:
                log.error(f"Failed: {video_id} ts={ts} pid={person_id}: {e}")
                self.db.mark_annotation_failed(
                    video_id, dataset, float(ts), person_id, str(e))

        return count

    def run(self):
        """Run batch annotation for the configured dataset."""
        dataset = self.cfg.dataset
        log.info(f"Starting batch annotation for {dataset}")

        df = self._load_annotations(dataset)
        if df.empty:
            log.warning("No annotations to process")
            return

        # Filter to downloaded videos only
        video_ids = df["video_id"].unique()
        available = [vid for vid in video_ids
                     if self.db.is_downloaded(vid, dataset)]

        if not available:
            log.warning("No downloaded videos found. Run download first.")
            return

        log.info(f"Processing {len(available)} videos with "
                 f"{len(df)} annotation rows")

        total_samples = 0
        for vid in tqdm(available, desc="Annotating"):
            group = df[df["video_id"] == vid]
            n = self._process_group(vid, group, dataset)
            total_samples += n

        log.info(f"Annotation complete: {total_samples} samples created")

        # Run quality validation
        skel_dir = str(Path(self.cfg.data_dir) / "processed" / dataset / "skeletons")
        qr = self.validator.validate_dir(skel_dir)
        log.info(f"Quality: {qr['valid']}/{qr['total']} valid, "
                 f"{qr['invalid']} quarantined")

        # Print DB summary
        summary = self.db.annotation_summary(dataset)
        log.info(f"DB summary: {summary}")
