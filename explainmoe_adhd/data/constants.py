"""
Data constants for ExplainMoE-ADHD v2.13.

This module defines all dataset specifications, feature tiers, modality mappings,
and licensing information as specified in the v2.13 technical specification.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set


# =============================================================================
# MODALITY ENUMS
# =============================================================================

class Modality(Enum):
    """All modality pathways in the system."""
    CHILD_EEG_19CH = "child_eeg_19ch"    # D1: Clinical-grade EEG
    CHILD_EEG_10CH = "child_eeg_10ch"    # D2: EMOTIV EPOC (10 retained channels)
    ADULT_EEG_5CH = "adult_eeg_5ch"       # D3: Mendeley adult EEG
    CLINICAL_FMRI = "clinical"            # D5+D6: ADHD-200 + ds002424
    ACTIGRAPHY = "actigraphy"              # D7: Hyperaktiv
    EYE_TRACKING = "eye_tracking"          # D8: Wainstein (proof-of-concept)


class HardwareToken(Enum):
    """Hardware domain tokens for EEG encoders (Section 5.3)."""
    CLINICAL_19CH_WET = 0   # D1: 19-channel clinical wet electrodes
    EMOTIV_10CH_SALINE = 1  # D2: EMOTIV EPOC (10 of 14 channels retained)
    MENDELEY_5CH = 2        # D3: 5-channel adult EEG
    REPOD_VARIABLE = 3      # D4: RepOD (validation only, variable channels)


# =============================================================================
# DATASET SPECIFICATIONS (Section 2)
# =============================================================================

@dataclass
class DatasetSpec:
    """Specification for a single dataset."""
    id: str                    # D1, D2, etc.
    name: str                  # Full title
    modality: Modality
    num_subjects: int          # Total N
    num_adhd: int              # ADHD count
    num_control: int           # Control count
    age_range: tuple           # (min, max) in years
    license: str               # License type
    is_group_a: bool          # Has ADHD labels (supervised phases)
    is_group_b: bool          # Pretraining-only (no ADHD labels)
    citation: str              # Required citation
    url: str                  # Download URL
    # Hardware token (for EEG datasets)
    hardware_token: Optional[HardwareToken] = None


DATASETS: Dict[str, DatasetSpec] = {
    # -------------------------------------------------------------------------
    # Group A: ADHD-Labeled Datasets (Used in Supervised Phases + MMD)
    # -------------------------------------------------------------------------
    "D1": DatasetSpec(
        id="D1",
        name="IEEE DataPort ADHD EEG Children",
        modality=Modality.CHILD_EEG_19CH,
        num_subjects=121,
        num_adhd=61,
        num_control=60,
        age_range=(7, 12),
        license="CC BY",
        is_group_a=True,
        is_group_b=False,
        citation="Nasrabadi et al.",
        url="https://ieee-dataport.org/open-access/eeg-data-adhd-control-children",
        hardware_token=HardwareToken.CLINICAL_19CH_WET,
    ),
    "D2": DatasetSpec(
        id="D2",
        name="FOCUS EEG Gameplay",
        modality=Modality.CHILD_EEG_10CH,
        num_subjects=103,
        num_adhd=49,
        num_control=54,
        age_range=(7, 14),  # Children (exact range in metadata)
        license="CC BY 4.0",
        is_group_a=True,
        is_group_b=False,
        citation="Alchalabi et al.",
        url="https://ieee-dataport.org/open-access/focus-eeg-brain-recordings-adhd-and-non-adhd-individuals-during-gameplay",
        hardware_token=HardwareToken.EMOTIV_10CH_SALINE,
    ),
    "D3": DatasetSpec(
        id="D3",
        name="Mendeley ADHD EEG Adults",
        modality=Modality.ADULT_EEG_5CH,
        num_subjects=79,
        num_adhd=37,
        num_control=42,
        age_range=(20, 68),
        license="CC BY 4.0",
        is_group_a=True,
        is_group_b=False,
        citation="Dataset authors",
        url="https://data.mendeley.com/datasets/6k4g25fhzg/1",
        hardware_token=HardwareToken.MENDELEY_5CH,
    ),
    "D4": DatasetSpec(
        id="D4",
        name="RepOD ADHD EEG Adults",
        modality=Modality.ADULT_EEG_5CH,
        num_subjects=16,
        num_adhd=8,
        num_control=8,
        age_range=(18, 50),  # Variable
        license="CC BY 4.0",
        is_group_a=False,  # Supplementary validation only
        is_group_b=False,
        citation="Dataset authors",
        url="https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/YHSZR3",
        hardware_token=HardwareToken.REPOD_VARIABLE,
    ),
    "D5": DatasetSpec(
        id="D5",
        name="ADHD-200 Sample",
        modality=Modality.CLINICAL_FMRI,
        num_subjects=973,
        num_adhd=491,
        num_control=482,
        age_range=(7, 21),  # Overwhelmingly pediatric
        license="Non-commercial research only",
        is_group_a=True,
        is_group_b=False,
        citation="Bellec et al., 2017",
        url="http://fcon_1000.projects.nitrc.org/indi/adhd200/",
        hardware_token=None,
    ),
    "D6": DatasetSpec(
        id="D6",
        name="OpenNeuro ds002424",
        modality=Modality.CLINICAL_FMRI,
        num_subjects=57,
        num_adhd=27,
        num_control=30,
        age_range=(7, 14),  # Children
        license="CC0",
        is_group_a=True,
        is_group_b=False,
        citation="Lytle et al., 2020",
        url="https://openneuro.org/datasets/ds002424",
        hardware_token=None,
    ),
    "D7": DatasetSpec(
        id="D7",
        name="Hyperaktiv",
        modality=Modality.ACTIGRAPHY,
        num_subjects=103,
        num_adhd=51,
        num_control=52,
        age_range=(18, 65),  # Adults
        license="CC BY-NC 4.0",
        is_group_a=True,
        is_group_b=False,
        citation="Hicks et al.",
        url="https://datasets.simula.no/hyperaktiv/",
        hardware_token=None,
    ),
    "D8": DatasetSpec(
        id="D8",
        name="Wainstein Eye-tracking",
        modality=Modality.EYE_TRACKING,
        num_subjects=50,
        num_adhd=28,
        num_control=22,
        age_range=(6, 16),  # Children
        license="CC BY 4.0",
        is_group_a=True,
        is_group_b=False,
        citation="Wainstein et al.",
        url="https://figshare.com/articles/dataset/ADHD_Pupil_Size_Dataset/7218725",
        hardware_token=None,
    ),
    # -------------------------------------------------------------------------
    # Group B: Pretraining-Only Datasets (No ADHD Labels)
    # -------------------------------------------------------------------------
    "D9": DatasetSpec(
        id="D9",
        name="TUH EEG Corpus",
        modality=Modality.CHILD_EEG_19CH,  # Mapped to child EEG for pretraining
        num_subjects=15000,  # ~15,000 unique patients
        num_adhd=0,
        num_control=0,
        age_range=(0, 100),  # All ages
        license="Free with registration",
        is_group_a=False,
        is_group_b=True,
        citation="Obeid & Picone, 2016",
        url="https://isip.piconepress.com/projects/tuh_eeg/",
        hardware_token=HardwareToken.CLINICAL_19CH_WET,
    ),
    "D10": DatasetSpec(
        id="D10",
        name="GazeBase",
        modality=Modality.EYE_TRACKING,
        num_subjects=322,
        num_adhd=0,
        num_control=0,
        age_range=(18, 60),  # Adults
        license="CC BY 4.0",
        is_group_a=False,
        is_group_b=True,
        citation="Dataset authors",
        url="https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257",
        hardware_token=None,
    ),
    "D11": DatasetSpec(
        id="D11",
        name="CAPTURE-24",
        modality=Modality.ACTIGRAPHY,
        num_subjects=151,
        num_adhd=0,
        num_control=0,
        age_range=(18, 65),
        license="Oxford research-only",
        is_group_a=False,
        is_group_b=True,
        citation="Dataset authors",
        url="https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001",
        hardware_token=None,
    ),
}


# =============================================================================
# SUBJECT COUNTS SUMMARY (Section 2.4)
# =============================================================================

MODALITY_COUNTS: Dict[Modality, Dict[str, int]] = {
    Modality.CLINICAL_FMRI: {
        "total": 1030,  # D5 + D6
        "adhd": 518,
        "control": 512,
    },
    Modality.CHILD_EEG_19CH: {
        "total": 121,   # D1
        "adhd": 61,
        "control": 60,
    },
    Modality.CHILD_EEG_10CH: {
        "total": 103,   # D2
        "adhd": 49,
        "control": 54,
    },
    Modality.ACTIGRAPHY: {
        "total": 103,   # D7
        "adhd": 51,
        "control": 52,
    },
    Modality.ADULT_EEG_5CH: {
        "total": 79,    # D3 (smallest)
        "adhd": 37,
        "control": 42,
    },
    Modality.EYE_TRACKING: {
        "total": 50,    # D8 (proof-of-concept only)
        "adhd": 28,
        "control": 22,
    },
}


# =============================================================================
# FEATURE TIERS (Section 3.1)
# =============================================================================

class FeatureTier(Enum):
    """Three-tier feature classification to prevent label leakage."""
    TIER_1_SAFE = 1        # Known before assessment
    TIER_2_CLINICAL = 2    # Correlates with diagnosis (ablation only)
    TIER_3_FORBIDDEN = 3   # IS the label or directly determines it


TIER_1_FEATURES: Dict[str, Dict] = {
    "age": {"type": "continuous", "missing_rate": "~0%", "handling": "numerical_embedding"},
    "sex": {"type": "categorical", "missing_rate": "~0%", "handling": "Embedding(2)"},
    "handedness": {"type": "categorical", "missing_rate": "5-30%", "handling": "Embedding(4) with missing=3"},
    "IQ": {"type": "continuous", "missing_rate": "10-40%", "handling": "learned_miss_token"},
    "site_id": {"type": "categorical", "missing_rate": "0%", "handling": "Embedding(N_sites)"},
    "dataset_source": {"type": "categorical", "missing_rate": "0%", "handling": "Embedding(N_datasets)"},
}

TIER_2_FEATURES: Dict[str, Dict] = {
    "medication_status": {"risk": "medicated Γëê ADHD", "source": "D5"},
    "WURS": {"risk": "retrospective ADHD rating", "source": "D7"},
    "ASRS": {"risk": "ADHD screening instrument (score Γëê label)", "source": "D7"},
    "MDRS": {"risk": "depression comorbidity measure", "source": "D7"},
}

TIER_3_FEATURES: Dict[str, Dict] = {
    "DX": {"why_forbidden": "IS the diagnosis code", "allowed_use": "prediction target, evaluation stratifier"},
    "ADHD_index": {"why_forbidden": "quantitative severity tracking diagnosis", "allowed_use": "evaluation stratifier"},
    "inattentive_score": {"why_forbidden": "dimensional score ΓåÆ ADHD-I diagnosis", "allowed_use": "severity head TARGET, stratifier"},
    "hyperactive_score": {"why_forbidden": "dimensional score ΓåÆ ADHD-HI diagnosis", "allowed_use": "severity head TARGET, stratifier"},
}


# =============================================================================
# EEG CHANNEL CONFIGURATIONS (Section 4.1)
# =============================================================================

# Standard 10-20 system channels (19 channels)
CHANNELS_10_20 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2"
]

# EMOTIV EPOC 14 channels (D2)
CHANNELS_EMOTIV_14 = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
    "P8", "T8", "FC6", "F4", "F8", "AF4"
]

# EMOTIV channels retained (10 channels matching 10-20)
CHANNELS_EMOTIV_RETAINED = [
    "F7", "F3", "T7", "P7", "O1", "O2", "P8", "T8", "F4", "F8"
]

# Adult EEG 5 channels (D3)
CHANNELS_ADULT_5 = ["O1", "F3", "F4", "Cz", "Fz"]

# Channel sets per encoder
CHANNEL_CONFIGS: Dict[Modality, List[str]] = {
    Modality.CHILD_EEG_19CH: CHANNELS_10_20,
    Modality.CHILD_EEG_10CH: CHANNELS_EMOTIV_RETAINED,
    Modality.ADULT_EEG_5CH: CHANNELS_ADULT_5,
}


# =============================================================================
# SAMPLING RATES
# =============================================================================

SAMPLING_RATES: Dict[str, int] = {
    "D1": 128,   # IEEE DataPort
    "D2": 128,   # FOCUS EMOTIV
    "D3": 256,   # Mendeley
    "D4": 256,   # RepOD
    "D5": None,  # fMRI (not applicable)
    "D6": None,  # fMRI (not applicable)
    "D7": 30,    # Hyperaktiv (typical actigraphy)
    "D8": 100,   # Wainstein (typical eye-tracking)
    "D9": 256,   # TUH
    "D10": 1000, # GazeBase
    "D11": 30,   # CAPTURE-24
}


# =============================================================================
# MODALITY TO DATASET MAPPING
# =============================================================================

MODALITY_TO_DATASETS: Dict[Modality, List[str]] = {
    Modality.CHILD_EEG_19CH: ["D1"],
    Modality.CHILD_EEG_10CH: ["D2"],
    Modality.ADULT_EEG_5CH: ["D3", "D4"],  # D4 is validation only
    Modality.CLINICAL_FMRI: ["D5", "D6"],
    Modality.ACTIGRAPHY: ["D7"],
    Modality.EYE_TRACKING: ["D8"],
}


# =============================================================================
# HARDWARE TOKEN MAPPING
# =============================================================================

DATASET_TO_HARDWARE_TOKEN: Dict[str, HardwareToken] = {
    "D1": HardwareToken.CLINICAL_19CH_WET,
    "D2": HardwareToken.EMOTIV_10CH_SALINE,
    "D3": HardwareToken.MENDELEY_5CH,
    "D4": HardwareToken.REPOD_VARIABLE,
}


# =============================================================================
# MODALITY INDEX MAPPING (For FuseMoE routers)
# =============================================================================

MODALITY_INDICES: Dict[Modality, int] = {
    Modality.CHILD_EEG_19CH: 0,
    Modality.CHILD_EEG_10CH: 1,
    Modality.ADULT_EEG_5CH: 2,
    Modality.CLINICAL_FMRI: 3,
    Modality.ACTIGRAPHY: 4,
    # Eye-tracking is excluded from FuseMoE (evaluated separately)
}


# =============================================================================
# DEFAULT MMD PAIRS (Section 6, Phase 3)
# =============================================================================

# Each tuple is (Modality_A, Modality_B)
MMD_DEFAULT_PAIRS: List[tuple] = [
    (Modality.CHILD_EEG_19CH, Modality.CHILD_EEG_10CH),  # Child EEG alignment
    (Modality.CHILD_EEG_19CH, Modality.CLINICAL_FMRI),  # Child EEG Γåö clinical
    (Modality.CHILD_EEG_10CH, Modality.CLINICAL_FMRI),  # EMOTIV Γåö clinical
    (Modality.ADULT_EEG_5CH, Modality.ACTIGRAPHY),      # Adult EEG Γåö actigraphy
]


# =============================================================================
# ADHD-200 DX CODES (Section 2.1, D5)
# =============================================================================

DX_CODES: Dict[int, str] = {
    0: "Control",
    1: "ADHD-Combined",
    2: "ADHD-Hyperactive/Impulsive",
    3: "ADHD-Inattentive",
}

DX_BINARIZATION: Dict[int, int] = {
    0: 0,  # Control ΓåÆ 0
    1: 1,  # ADHD-C ΓåÆ 1
    2: 1,  # ADHD-HI ΓåÆ 1
    3: 1,  # ADHD-I ΓåÆ 1
}


# =============================================================================
# MODEL WEIGHT LICENSE (Section 1.7)
# =============================================================================

MODEL_LICENSE = "Non-commercial research only"
# Constrained by: D5 (ADHD-200), D7 (Hyperaktiv), D11 (CAPTURE-24)


# =============================================================================
# LICENSING TIERS (Section 2.3)
# =============================================================================

LICENSE_TIER_1_OPEN = ["D1", "D2", "D3", "D4", "D6", "D8", "D10"]
LICENSE_TIER_2_RESTRICTED = ["D5", "D7", "D11"]
LICENSE_TIER_3_REGISTRATION = ["D9"]
