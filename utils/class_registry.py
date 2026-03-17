"""Maps AVA class IDs ↔ model output indices. Handles multi-label vectors."""
import yaml, numpy as np, pandas as pd
from typing import List

class ClassRegistry:
    def __init__(self, config_path: str = "configs/class_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self._classes = cfg["target_classes"]    # list of {id, name, category}
        self._id_to_idx = {c["id"]: i for i, c in enumerate(self._classes)}
        self._idx_to_id = {i: c["id"] for i, c in enumerate(self._classes)}

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def class_names(self) -> List[str]:
        return [c["name"] for c in self._classes]

    @property
    def class_ids(self) -> List[int]:
        return [c["id"] for c in self._classes]

    def ava_id_to_index(self, ava_id: int) -> int:
        return self._id_to_idx[ava_id]

    def filter_annotations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows whose action_id is in our target set."""
        return df[df["action_id"].isin(self._id_to_idx)].copy()

    def get_multilabel_vector(self, ava_ids: List[int]) -> np.ndarray:
        """
        Convert a list of AVA class IDs to a binary (num_classes,) vector.
        Unknown IDs are silently ignored.
        """
        vec = np.zeros(self.num_classes, dtype=np.float32)
        for ava_id in ava_ids:
            if ava_id in self._id_to_idx:
                vec[self._id_to_idx[ava_id]] = 1.0
        return vec
