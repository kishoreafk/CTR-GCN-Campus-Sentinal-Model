"""Dataset-specific configurations for ExplainMoE-ADHD v2.13."""

from explainmoe_adhd.data.constants import DATASETS, Modality

# Group A datasets (ADHD-labeled, used in supervised phases)
GROUP_A_DATASETS = {k: v for k, v in DATASETS.items() if v.is_group_a}

# Group B datasets (pretraining only, no ADHD labels)
GROUP_B_DATASETS = {k: v for k, v in DATASETS.items() if v.is_group_b}

# Datasets per modality
DATASETS_BY_MODALITY = {}
for k, v in DATASETS.items():
    mod = v.modality.value
    if mod not in DATASETS_BY_MODALITY:
        DATASETS_BY_MODALITY[mod] = []
    DATASETS_BY_MODALITY[mod].append(k)

# MMD pair definitions (Section 6, Phase 3)
MMD_PAIRS = [
    ("child_eeg_19ch", "child_eeg_10ch"),   # Pair 1: clinical vs consumer child EEG
    ("child_eeg_19ch", "clinical"),           # Pair 2: child EEG ↔ clinical/fMRI
    ("child_eeg_10ch", "clinical"),           # Pair 3: consumer EEG ↔ clinical/fMRI
    ("adult_eeg_5ch", "actigraphy"),          # Pair 4: adult EEG ↔ actigraphy
]

# Router index mapping
MODALITY_ROUTER_INDEX = {
    "child_eeg_19ch": 0,
    "child_eeg_10ch": 1,
    "adult_eeg_5ch": 2,
    "clinical": 3,
    "actigraphy": 4,
}
