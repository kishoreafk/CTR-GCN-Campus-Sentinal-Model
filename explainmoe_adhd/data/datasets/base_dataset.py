"""
Base Dataset classes for ExplainMoE-ADHD v2.13.

Provides the torch.utils.data.Dataset implementations for all modalities.
Each subject is represented as a dict with standardized keys.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SubjectRecord:
    """A single subject's data across all available fields."""
    subject_id: str
    modality: str                   # e.g. "child_eeg_19ch", "clinical", "actigraphy"
    dataset_id: str                 # e.g. "D1", "D5"
    label: int                      # 0=control, 1=ADHD
    data: Dict[str, Any] = field(default_factory=dict)
    # Optional metadata
    subtype: Optional[int] = None   # 0=Combined, 1=HI, 2=Inattentive (ADHD-200 only)
    severity_scores: Optional[np.ndarray] = None  # [inattentive, hyperactive]
    has_subtype: bool = False
    has_severity: bool = False
    age: Optional[float] = None
    sex: Optional[int] = None
    site_id: Optional[int] = None
    dataset_source: Optional[int] = None
    hardware_token_id: Optional[int] = None
    mean_fd: Optional[float] = None  # fMRI head motion


class BaseADHDDataset(Dataset):
    """
    Base dataset for ExplainMoE-ADHD.

    Wraps a list of SubjectRecord objects and returns standardized dicts
    suitable for all training phases.
    """

    def __init__(self, records: List[SubjectRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        item: Dict[str, Any] = {
            "subject_id": rec.subject_id,
            "modality": rec.modality,
            "dataset_id": rec.dataset_id,
            "label": torch.tensor(rec.label, dtype=torch.float32),
        }

        # Modality-specific data tensors
        for key, value in rec.data.items():
            if isinstance(value, np.ndarray):
                item[key] = torch.from_numpy(value).float()
            elif isinstance(value, torch.Tensor):
                item[key] = value.float()
            else:
                item[key] = value

        # Subtype (ADHD-200 only)
        item["has_subtype"] = torch.tensor(rec.has_subtype, dtype=torch.bool)
        if rec.has_subtype and rec.subtype is not None:
            item["subtype"] = torch.tensor(rec.subtype, dtype=torch.long)
        else:
            item["subtype"] = torch.tensor(0, dtype=torch.long)

        # Severity scores
        item["has_severity"] = torch.tensor(rec.has_severity, dtype=torch.bool)
        if rec.has_severity and rec.severity_scores is not None:
            item["severity"] = torch.from_numpy(rec.severity_scores).float()
        else:
            item["severity"] = torch.zeros(2, dtype=torch.float32)

        # Covariates
        item["age"] = torch.tensor(rec.age if rec.age is not None else float("nan"), dtype=torch.float32)
        item["sex"] = torch.tensor(rec.sex if rec.sex is not None else 0, dtype=torch.long)
        item["site_id"] = torch.tensor(rec.site_id if rec.site_id is not None else 0, dtype=torch.long)
        item["dataset_source"] = torch.tensor(
            rec.dataset_source if rec.dataset_source is not None else 0, dtype=torch.long
        )
        item["hardware_token_id"] = torch.tensor(
            rec.hardware_token_id if rec.hardware_token_id is not None else 0, dtype=torch.long
        )
        item["mean_fd"] = torch.tensor(
            rec.mean_fd if rec.mean_fd is not None else 0.0, dtype=torch.float32
        )

        return item

    def get_labels(self) -> np.ndarray:
        return np.array([r.label for r in self.records])

    def get_subject_ids(self) -> List[str]:
        return [r.subject_id for r in self.records]

    def get_modalities(self) -> List[str]:
        return [r.modality for r in self.records]

    def get_indices_by_modality(self) -> Dict[str, List[int]]:
        indices: Dict[str, List[int]] = {}
        for i, rec in enumerate(self.records):
            if rec.modality not in indices:
                indices[rec.modality] = []
            indices[rec.modality].append(i)
        return indices

    def get_indices_by_modality_label(self) -> Dict[tuple, List[int]]:
        """Returns {(modality, label): [indices]}."""
        indices: Dict[tuple, List[int]] = {}
        for i, rec in enumerate(self.records):
            key = (rec.modality, rec.label)
            if key not in indices:
                indices[key] = []
            indices[key].append(i)
        return indices

    def subset(self, indices: List[int]) -> "BaseADHDDataset":
        return BaseADHDDataset([self.records[i] for i in indices])


class ModalityDataset(Dataset):
    """
    Single-modality dataset wrapper.

    Filters a BaseADHDDataset to a single modality for Phase 2
    per-encoder training.
    """

    def __init__(self, base_dataset: BaseADHDDataset, modality: str):
        self.base_dataset = base_dataset
        self.modality = modality
        self.indices = [
            i for i, r in enumerate(base_dataset.records) if r.modality == modality
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.base_dataset[self.indices[idx]]

    def get_labels(self) -> np.ndarray:
        return np.array([self.base_dataset.records[i].label for i in self.indices])


class SyntheticADHDDataset(BaseADHDDataset):
    """
    Synthetic dataset for testing.

    Generates random data matching expected shapes for all modalities.
    """

    def __init__(
        self,
        n_subjects_per_modality: int = 20,
        eeg_time_samples: int = 256,
        actigraphy_time_samples: int = 1000,
        eye_tracking_time_samples: int = 100,
    ):
        records = []
        modality_specs = [
            ("child_eeg_19ch", "D1", 0, {"eeg": (19, eeg_time_samples)}),
            ("child_eeg_10ch", "D2", 1, {"eeg": (10, eeg_time_samples)}),
            ("adult_eeg_5ch", "D3", 2, {"eeg": (5, eeg_time_samples)}),
            ("clinical", "D5", None, {
                "tabular": (6,),
                "fmri": (4005,),
            }),
            ("actigraphy", "D7", None, {"timeseries": (4, actigraphy_time_samples)}),
        ]

        subj_counter = 0
        for modality, dataset_id, hw_token, data_shapes in modality_specs:
            for i in range(n_subjects_per_modality):
                label = i % 2  # Alternate ADHD/control
                data = {}
                for key, shape in data_shapes.items():
                    data[key] = np.random.randn(*shape).astype(np.float32)

                rec = SubjectRecord(
                    subject_id=f"sub_{subj_counter:04d}",
                    modality=modality,
                    dataset_id=dataset_id,
                    label=label,
                    data=data,
                    age=float(np.random.uniform(7, 60)),
                    sex=np.random.randint(0, 2),
                    site_id=np.random.randint(0, 8),
                    dataset_source=0,
                    hardware_token_id=hw_token,
                    mean_fd=float(np.random.uniform(0.1, 0.8)) if modality == "clinical" else None,
                    has_subtype=(modality == "clinical" and label == 1),
                    subtype=np.random.randint(0, 3) if (modality == "clinical" and label == 1) else None,
                    has_severity=(modality == "clinical" and np.random.random() > 0.3),
                    severity_scores=np.random.randn(2).astype(np.float32) if (modality == "clinical") else None,
                )
                records.append(rec)
                subj_counter += 1

        super().__init__(records)
