"""
Training configuration for ExplainMoE-ADHD v2.13.

This module defines all training hyperparameters, phase schedules,
loss weights, and learning rates as specified in Section 6 of the technical specification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class Phase1Config:
    """Phase 1: Self-Supervised Pretraining (Section 6, Phase 1)."""
    # Duration
    epochs: int = 30
    
    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    
    # Batch
    batch_size: int = 64
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    
    # Pretraining task
    mask_span_min: int = 16     # 125ms at 128Hz
    mask_span_max: int = 64     # 500ms at 128Hz
    mask_prob: float = 0.15     # ~15% of positions masked
    
    # Fold dependency: NONE (run once, shared across all folds)


@dataclass
class Phase2Config:
    """Phase 2: Supervised Encoder Training (Section 6, Phase 2)."""
    # Duration
    max_epochs: int = 50
    early_stopping_patience: int = 10
    
    # Optimizer
    lr: float = 5e-4
    weight_decay: float = 1e-3
    betas: tuple = (0.9, 0.999)
    
    # Batch
    batch_size: int = 32
    
    # Loss
    diagnosis_loss_weight: float = 1.0
    
    # Fold dependency: RE-RUN per fold
    # Early stopping: per-encoder val_auroc


@dataclass
class Phase3Config:
    """Phase 3: Latent Space Alignment (MMD) (Section 6, Phase 3)."""
    # Duration
    max_steps: int = 1000
    eval_every: int = 50
    early_stopping_patience: int = 10  # eval cycles
    
    # Optimizer
    lr: float = 1e-4
    
    # MMD specific
    kernel_type: str = "multibandwidth_rbf"
    bandwidth_heuristic: str = "adaptive_median"
    min_batch_per_class: int = 16
    
    # Loss weight
    mmd_loss_weight: float = 1.0
    
    # Fold dependency: RE-RUN per fold
    # Early stopping: mean MMD loss on val_subjects


@dataclass
class Phase4Config:
    """Phase 4: FuseMoE Training (Section 6, Phase 4)."""
    # Duration
    max_steps: int = 3000
    eval_every: int = 100
    early_stopping_patience: int = 10  # eval cycles
    
    # Optimizer
    lr_main: float = 3e-4           # FuseMoE, experts, heads
    lr_projection: float = 3e-5     # 0.1├ù for projection heads (preserve alignment)
    weight_decay: float = 1e-4
    
    # Batch
    batch_size: int = 80
    batch_modality_balance: bool = True
    
    # Loss weights
    diagnosis_loss_weight: float = 1.0
    subtype_loss_weight: float = 0.25
    severity_loss_weight: float = 0.0   # INACTIVE in Phase 4
    mmd_loss_weight: float = 0.0       # INACTIVE in Phase 4
    load_balance_weight: float = 0.01
    
    # Modality sampling
    samples_per_modality: int = 16
    
    # Fold dependency: RE-RUN per fold
    # Early stopping: val_auroc_macro


@dataclass
class Phase5Config:
    """Phase 5: Joint Fine-Tuning (Section 6, Phase 5)."""
    # Duration
    max_steps: int = 1000
    eval_every: int = 50
    early_stopping_patience: int = 10  # eval cycles
    
    # Optimizer - all at 1e-5
    lr: float = 1e-5
    weight_decay: float = 1e-4
    
    # Batch
    batch_size: int = 80
    batch_modality_balance: bool = True
    
    # Loss weights
    diagnosis_loss_weight: float = 1.0
    subtype_loss_weight: float = 0.25
    severity_loss_weight: float = 0.1   # ACTIVE in Phase 5
    mmd_loss_weight: float = 0.3       # ACTIVE in Phase 5
    load_balance_weight: float = 0.01
    
    # Fold dependency: RE-RUN per fold
    # Early stopping: val_auroc_macro


@dataclass
class CrossValidationConfig:
    """Cross-Validation Protocol (Section 7)."""
    n_folds: int = 5
    val_split: float = 0.2  # 80/20 split of remaining folds
    stratify_by: List[str] = field(default_factory=lambda: ["label", "modality", "dataset_source"])
    
    # Group by subject_id to prevent leakage
    group_by: str = "subject_id"
    
    # Test set: NEVER used for early stopping, hyperparameter selection, etc.


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    phase4: Phase4Config = field(default_factory=Phase4Config)
    phase5: Phase5Config = field(default_factory=Phase5Config)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    
    # Seed for reproducibility
    seed: int = 42
    
    # Device (auto-detect CUDA availability)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 4
    
    # Gradient clipping
    grad_clip: float = 1.0
    
    # Logging
    log_every: int = 10
    save_checkpoint_every: int = 500
    
    # Eye-tracking: permanently frozen after Phase 1
    freeze_eye_tracking_after_phase: int = 1


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_TRAINING_CONFIG = TrainingConfig()


# =============================================================================
# LOSS WEIGHT SUMMARY (Sections 6.4 and 6.5)
# =============================================================================

LOSS_WEIGHTS = {
    "phase2": {
        "diagnosis": 1.0,
        "subtype": 0.0,
        "severity": 0.0,
        "mmd": 0.0,
        "load_balance": 0.0,
    },
    "phase3": {
        "diagnosis": 0.0,
        "subtype": 0.0,
        "severity": 0.0,
        "mmd": 1.0,
        "load_balance": 0.0,
    },
    "phase4": {
        "diagnosis": 1.0,
        "subtype": 0.25,
        "severity": 0.0,
        "mmd": 0.0,
        "load_balance": 0.01,
    },
    "phase5": {
        "diagnosis": 1.0,
        "subtype": 0.25,
        "severity": 0.1,
        "mmd": 0.3,
        "load_balance": 0.01,
    },
}


# =============================================================================
# LEARNING RATE SCHEDULE (Sections 6.4 and 6.5)
# =============================================================================

LR_SCHEDULE = {
    "phase2": {
        "encoder": 5e-4,
        "projection": 0.0,  # Does not exist in Phase 2
        "fusemoe": 0.0,  # Does not exist in Phase 2
        "diagnosis_head": 5e-4,  # Temporary MLP head
        "subtype_head": 0.0,  # Does not exist in Phase 2
        "severity_head": 0.0,  # Does not exist in Phase 2
    },
    "phase3": {
        "encoder": 0.0,  # Frozen
        "projection": 1e-4,
        "fusemoe": 0.0,  # Not created yet
        "diagnosis_head": 0.0,
        "subtype_head": 0.0,
        "severity_head": 0.0,
    },
    "phase4": {
        "encoder": 0.0,  # Frozen
        "projection": 3e-5,  # 0.1├ù
        "fusemoe": 3e-4,
        "diagnosis_head": 3e-4,
        "subtype_head": 3e-4,
        "severity_head": 3e-4,  # Exists but inactive
    },
    "phase5": {
        "encoder": 1e-5,
        "projection": 1e-5,
        "fusemoe": 1e-5,
        "diagnosis_head": 1e-5,
        "subtype_head": 1e-5,
        "severity_head": 1e-5,
    },
}


# =============================================================================
# EARLY STOPPING MONITORS
# =============================================================================

EARLY_STOPPING_MONITORS = {
    "phase2": "val_auroc",           # Per-encoder modality-specific
    "phase3": "mean_mmd_loss",       # Mean across 4 MMD pairs
    "phase4": "val_auroc_macro",     # Macro-average across 5 modalities
    "phase5": "val_auroc_macro",     # Macro-average across 5 modalities
}


# =============================================================================
# MODALITY-BALANCED SAMPLING (Phase 4/5)
# =============================================================================

# Default: 16 subjects per modality ├ù 5 modalities = 80
MODALITY_BALANCED_BATCH = {
    "total": 80,
    "per_modality": 16,
    "per_class_per_modality": 8,  # 8 ADHD + 8 Control
}


# =============================================================================
# APPROXIMATE SAMPLE SIZES PER FOLD (Section 7)
# =============================================================================

SAMPLE_SIZES_PER_FOLD = {
    "clinical": {"total": 1030, "train": 659, "val": 165, "test": 206},
    "child_eeg_19ch": {"total": 121, "train": 77, "val": 20, "test": 24},
    "child_eeg_10ch": {"total": 103, "train": 66, "val": 16, "test": 21},
    "actigraphy": {"total": 103, "train": 66, "val": 16, "test": 21},
    "adult_eeg": {"total": 79, "train": 51, "val": 12, "test": 16},
}
