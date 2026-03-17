"""
AVA dataset wrapper.
Inherits SkeletonDataset behavior for multi-label skeleton action recognition.
"""
from data_pipeline.skeleton_dataset import SkeletonDataset
from utils.class_registry import ClassRegistry
from pathlib import Path


class AVADataset(SkeletonDataset):
    """Dataset for AVA skeleton data (Phase 2)."""

    def __init__(self, config, split: str = "train"):
        reg = ClassRegistry(config.class_config)
        annotation_dir = str(
            Path(config.data_dir) / "processed" / "ava" / "skeletons")

        max_samples = None
        if config.test_mode:
            max_samples = config.test_max_videos * 10

        super().__init__(
            annotation_dir=annotation_dir,
            class_registry=reg,
            split=split,
            augment=(split == "train"),
            min_quality=0.30,
            max_samples=max_samples,
        )
