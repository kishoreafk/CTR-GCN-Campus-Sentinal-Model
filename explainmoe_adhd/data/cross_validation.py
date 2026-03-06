"""
Cross-Validation Protocol for ExplainMoE-ADHD v2.13.

This module implements the 5-fold GroupKFold protocol (Section 7):
- Stratified by (ADHD_label, modality, dataset_source)
- Group by subject_id to prevent leakage
- Test set: NEVER used for early stopping or hyperparameter selection
"""

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedShuffleSplit
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import torch


@dataclass
class CVSplit:
    """A single CV split."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    
    train_subjects: List[str]
    val_subjects: List[str]
    test_subjects: List[str]


@dataclass
class FoldResult:
    """Result from a single fold."""
    fold: int
    metrics: Dict[str, float]
    predictions: Optional[Dict[str, np.ndarray]] = None


class CrossValidator:
    """
    Cross-validator for ExplainMoE-ADHD.
    
    Implements 5-fold GroupKFold with stratified splits.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        val_split: float = 0.2,
        stratify_by: List[str] = None,
        group_by: str = 'subject_id',
        seed: int = 42,
    ):
        self.n_folds = n_folds
        self.val_split = val_split
        self.stratify_by = stratify_by or ['label', 'modality', 'dataset_source']
        self.group_by = group_by
        self.seed = seed
        
        np.random.seed(seed)
    
    def create_splits(
        self,
        subjects: List[str],
        labels: np.ndarray,
        groups: np.ndarray,
        stratify_labels: np.ndarray,
    ) -> List[CVSplit]:
        """
        Create CV splits.
        
        Args:
            subjects: List of subject IDs
            labels: Binary labels (0/1)
            groups: Group IDs for GroupKFold (subject_id)
            stratify_labels: Labels for stratification
        
        Returns:
            List of CVSplit objects
        """
        n_subjects = len(subjects)
        
        # GroupKFold for test set
        gkf = GroupKFold(n_splits=self.n_folds)
        
        splits = []
        
        for fold, (remaining_idx, test_idx) in enumerate(gkf.split(
            np.arange(n_subjects), 
            groups=groups
        )):
            # Further split remaining into train/val (stratified)
            remaining_labels = stratify_labels[remaining_idx]
            
            # Stratified split for train/val to preserve label balance
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.val_split,
                random_state=self.seed + fold,
            )
            train_local_idx, val_local_idx = next(sss.split(
                remaining_idx, remaining_labels
            ))
            train_idx = remaining_idx[train_local_idx]
            val_idx = remaining_idx[val_local_idx]
            
            # Verify no overlap
            assert len(set(train_idx) & set(val_idx)) == 0
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(set(val_idx) & set(test_idx)) == 0
            
            # Create split
            split = CVSplit(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                train_subjects=[subjects[i] for i in train_idx],
                val_subjects=[subjects[i] for i in val_idx],
                test_subjects=[subjects[i] for i in test_idx],
            )
            
            splits.append(split)
        
        return splits
    
    def get_subject_counts(
        self,
        splits: List[CVSplit],
        labels: np.ndarray,
    ) -> Dict[str, Dict[str, int]]:
        """
        Get subject counts per split.
        
        Returns:
            Dict with train/val/test counts per fold
        """
        counts = {}
        
        for fold, split in enumerate(splits):
            train_labels = labels[split.train_idx]
            val_labels = labels[split.val_idx]
            test_labels = labels[split.test_idx]
            
            counts[f'fold_{fold}'] = {
                'train': {
                    'total': len(split.train_idx),
                    'adhd': int((train_labels == 1).sum()),
                    'control': int((train_labels == 0).sum()),
                },
                'val': {
                    'total': len(split.val_idx),
                    'adhd': int((val_labels == 1).sum()),
                    'control': int((val_labels == 0).sum()),
                },
                'test': {
                    'total': len(split.test_idx),
                    'adhd': int((test_labels == 1).sum()),
                    'control': int((test_labels == 0).sum()),
                },
            }
        
        return counts


class ModalityBalancedSampler:
    """
    Sampler for modality-balanced batches.
    
    Used in Phases 4 and 5. Generates batches indefinitely
    for step-based training (3000 steps in Phase 4, 1000 in Phase 5).
    """
    
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        modality_indices: Dict[str, List[int]],
        samples_per_modality: int = 16,
        batch_size: int = 80,
        num_steps: int = 3000,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.modality_indices = modality_indices
        self.samples_per_modality = samples_per_modality
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.shuffle = shuffle
        
        # Calculate batches per modality
        self.n_modalities = len(modality_indices)
        assert batch_size % self.n_modalities == 0, \
            f"Batch size {batch_size} not divisible by {self.n_modalities}"
    
    def _sample_one_batch(self) -> np.ndarray:
        """Sample a single modality-balanced batch."""
        indices = []
        for modality, mod_indices in self.modality_indices.items():
            if self.shuffle:
                selected = np.random.choice(
                    mod_indices,
                    size=min(self.samples_per_modality, len(mod_indices)),
                    replace=len(mod_indices) < self.samples_per_modality,
                )
            else:
                selected = mod_indices[:self.samples_per_modality]
            indices.extend(selected)
        
        indices = np.array(indices)
        if self.shuffle:
            indices = np.random.permutation(indices)
        return indices
    
    def __iter__(self):
        """Generate batches for num_steps iterations."""
        for _ in range(self.num_steps):
            yield self._sample_one_batch()
    
    def __len__(self):
        return self.num_steps


class MMDBatchSampler:
    """
    Sampler for MMD batches.
    
    Creates balanced batches for MMD alignment (Phase 3).
    """
    
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        indices_by_modality_label: Dict[Tuple[str, int], List[int]],
        min_batch_per_class: int = 16,
    ):
        self.dataset = dataset
        self.indices_by_modality_label = indices_by_modality_label
        self.min_batch_per_class = min_batch_per_class
    
    def sample_batch(
        self,
        modality_a: str,
        modality_b: str,
    ) -> Tuple[List[int], List[int]]:
        """
        Sample a batch for MMD computation between two modalities.
        
        Returns:
            (adhd_indices_a, adhd_indices_b), (control_indices_a, control_indices_b)
        """
        # Get ADHD and Control indices for each modality
        adhd_a = self.indices_by_modality_label.get((modality_a, 1), [])
        adhd_b = self.indices_by_modality_label.get((modality_b, 1), [])
        control_a = self.indices_by_modality_label.get((modality_a, 0), [])
        control_b = self.indices_by_modality_label.get((modality_b, 0), [])
        
        # Sample with replacement if needed
        n = self.min_batch_per_class
        
        adhd_a_sample = np.random.choice(
            adhd_a, size=n, replace=len(adhd_a) < n
        ).tolist() if adhd_a else []
        
        adhd_b_sample = np.random.choice(
            adhd_b, size=n, replace=len(adhd_b) < n
        ).tolist() if adhd_b else []
        
        control_a_sample = np.random.choice(
            control_a, size=n, replace=len(control_a) < n
        ).tolist() if control_a else []
        
        control_b_sample = np.random.choice(
            control_b, size=n, replace=len(control_b) < n
        ).tolist() if control_b else []
        
        return (adhd_a_sample, adhd_b_sample), (control_a_sample, control_b_sample)


# =============================================================================
# SUBJECT LEAKAGE PREVENTION
# =============================================================================

def verify_no_subject_leakage(
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
) -> bool:
    """
    Verify no subject leakage across splits.
    
    Returns:
        True if no leakage, raises AssertionError otherwise
    """
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)
    
    assert train_set.isdisjoint(val_set), "Train/Val subject overlap!"
    assert train_set.isdisjoint(test_set), "Train/Test subject overlap!"
    assert val_set.isdisjoint(test_set), "Val/Test subject overlap!"
    
    return True


# =============================================================================
# APPROXIMATE SAMPLE SIZES (Section 7)
# =============================================================================

APPROXIMATE_SIZES = {
    "clinical": {"total": 1030, "train": 659, "val": 165, "test": 206},
    "child_eeg_19ch": {"total": 121, "train": 77, "val": 20, "test": 24},
    "child_eeg_10ch": {"total": 103, "train": 66, "val": 16, "test": 21},
    "actigraphy": {"total": 103, "train": 66, "val": 16, "test": 21},
    "adult_eeg": {"total": 79, "train": 51, "val": 12, "test": 16},
}


# =============================================================================
# TEST
# =============================================================================

def test_cross_validation():
    """Test cross-validation splits."""
    np.random.seed(42)
    
    # Create dummy data
    n = 100
    subjects = [f"sub_{i}" for i in range(n)]
    labels = np.random.randint(0, 2, n)
    groups = np.arange(n)  # Each subject in own group
    
    # Create validator
    validator = CrossValidator(n_folds=5, val_split=0.2)
    
    # Create splits
    splits = validator.create_splits(
        subjects=subjects,
        labels=labels,
        groups=groups,
        stratify_labels=labels,
    )
    
    assert len(splits) == 5, f"Expected 5 splits, got {len(splits)}"
    
    # Verify no leakage
    for fold, split in enumerate(splits):
        verify_no_subject_leakage(
            split.train_subjects,
            split.val_subjects,
            split.test_subjects,
        )
        print(f"Fold {fold}: train={len(split.train_subjects)}, "
              f"val={len(split.val_subjects)}, test={len(split.test_subjects)}")
    
    print("\nCross-validation tests passed!")


if __name__ == '__main__':
    test_cross_validation()
