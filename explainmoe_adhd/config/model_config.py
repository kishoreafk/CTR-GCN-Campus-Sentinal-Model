"""
Model configuration for ExplainMoE-ADHD v2.13.

This module defines all model hyperparameters, architectural choices,
and component specifications as specified in Section 5 of the technical specification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EEGEncoderConfig:
    """Configuration for EEG encoders (Section 5.2.1-5.2.3)."""
    # EEGNet parameters
    temporal_filters: int = 8          # F1
    depth_multiplier: int = 2         # D
    separable_filters: int = 16        # F2
    
    # Temporal convolution
    temporal_kernel_size: int = 64
    
    # Separable convolution
    separable_kernel_size: int = 16
    
    # Pooling
    pool1_size: int = 4
    pool2_size: int = 8
    
    # Dropout
    dropout_rate: float = 0.5
    
    # Transformer parameters
    transformer_d_model: int = 256
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 512
    transformer_dropout: float = 0.1
    
    # Output dimension
    output_dim: int = 256
    
    # Number of channels (set per variant)
    num_channels: int = 19
    
    # Hardware token ID
    hardware_token_id: int = 0


@dataclass
class ClinicalEncoderConfig:
    """Configuration for Clinical/fMRI encoder (Section 5.2.4)."""
    # FT-Transformer for tabular
    ft_num_categories: List[int] = field(default_factory=lambda: [2, 4, 20, 10])  # sex, handedness, site, source
    ft_num_continuous: int = 2  # age, IQ
    ft_dim: int = 64
    ft_depth: int = 2
    ft_heads: int = 4
    ft_dim_head: int = 32
    ft_ff_dim: int = 256
    ft_attn_dropout: float = 0.1
    ft_ff_dropout: float = 0.1
    
    # fMRI MLP
    fmri_input_dim: int = 4006  # 4005 connectivity + 1 mean_FD
    fmri_hidden_dim: int = 512
    fmri_output_dim: int = 128
    fmri_dropout: float = 0.3
    
    # Tabular projection
    tabular_output_dim: int = 128
    
    # Final merge
    merge_hidden_dim: int = 256
    merge_output_dim: int = 256
    
    # Output dimension
    output_dim: int = 256


@dataclass
class ActigraphyEncoderConfig:
    """Configuration for Actigraphy encoder (Section 5.2.5)."""
    # ResNet1D
    resnet_in_channels: int = 4  # 3 accel + 1 HR
    resnet_out_features: int = 224
    
    # BiLSTM
    bilstm_hidden_size: int = 112
    bilstm_num_layers: int = 2
    bilstm_bidirectional: bool = True
    
    # Auxiliary MLP
    aux_input_dim: int = 2  # age, sex
    aux_output_dim: int = 32
    
    # Merge MLP (256ΓåÆ256 by default, no bottleneck)
    merge_input_dim: int = 256  # 224 + 32
    merge_hidden_dim: int = 256
    merge_output_dim: int = 256
    
    # Output dimension
    output_dim: int = 256


@dataclass
class EyeTrackingEncoderConfig:
    """Configuration for Eye-tracking encoder (Section 5.2.6)."""
    # BiLSTM
    bilstm_input_size: int = 3  # x, y, pupil
    bilstm_hidden_size: int = 128
    bilstm_num_layers: int = 2
    bilstm_bidirectional: bool = True
    
    # Attention
    attn_embed_dim: int = 256
    attn_num_heads: int = 4
    
    # Output dimension
    output_dim: int = 256


@dataclass
class ProjectionHeadConfig:
    """Configuration for Projection Heads (Section 5.4)."""
    input_dim: int = 256
    output_dim: int = 256
    depth: int = 2  # Default: 2 layers


@dataclass
class FuseMoEConfig:
    """Configuration for FuseMoE module (Section 5.5)."""
    # Expert configuration
    num_experts: int = 4
    expert_hidden_dim: int = 256
    expert_output_dim: int = 256
    expert_activation: str = "gelu"
    
    # Routing
    top_k: int = 2  # Select top-K experts
    num_routers: int = 5  # One per modality pathway
    
    # Temperature (learned per router)
    init_temperature: float = 1.0
    
    # Residual connection
    use_residual: bool = True
    
    # Load balancing
    balance_coeff: float = 0.01
    
    # Output dimension
    output_dim: int = 256


@dataclass
class TaskHeadConfig:
    """Configuration for Task Heads (Section 5.6)."""
    input_dim: int = 256
    
    # Diagnosis head
    diagnosis_hidden_dim: int = 128
    diagnosis_output_dim: int = 1
    diagnosis_dropout: float = 0.3
    
    # Subtype head
    subtype_hidden_dim: int = 128
    subtype_output_dim: int = 3  # C, HI, I
    subtype_dropout: float = 0.3
    
    # Severity head
    severity_hidden_dim: int = 128
    severity_output_dim: int = 2  # inattentive, hyperactive
    severity_dropout: float = 0.5  # Higher dropout for regression


@dataclass
class ModelConfig:
    """Complete model configuration."""
    # Encoder configs
    eeg_19ch: EEGEncoderConfig = field(default_factory=lambda: EEGEncoderConfig(
        num_channels=19,
        hardware_token_id=0
    ))
    eeg_10ch: EEGEncoderConfig = field(default_factory=lambda: EEGEncoderConfig(
        num_channels=10,
        hardware_token_id=1
    ))
    eeg_5ch: EEGEncoderConfig = field(default_factory=lambda: EEGEncoderConfig(
        num_channels=5,
        hardware_token_id=2
    ))
    clinical: ClinicalEncoderConfig = field(default_factory=ClinicalEncoderConfig)
    actigraphy: ActigraphyEncoderConfig = field(default_factory=ActigraphyEncoderConfig)
    eye_tracking: EyeTrackingEncoderConfig = field(default_factory=EyeTrackingEncoderConfig)
    
    # Shared components
    projection: ProjectionHeadConfig = field(default_factory=ProjectionHeadConfig)
    fusemoe: FuseMoEConfig = field(default_factory=FuseMoEConfig)
    task_heads: TaskHeadConfig = field(default_factory=TaskHeadConfig)
    
    # Global
    latent_dim: int = 256
    
    # Eye-tracking is permanently frozen after Phase 1
    freeze_eye_tracking: bool = True


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_MODEL_CONFIG = ModelConfig()


# =============================================================================
# ABLATION CONFIGURATIONS
# =============================================================================

# A1: Projection head depth variants
PROJECTION_DEPTH_VARIANTS = [1, 2, 3]

# A4: Top-K routing variants
TOP_K_VARIANTS = [1, 2, 3]

# A14: Temperature variants
TEMPERATURE_VARIANTS = [0.5, 1.0, 2.0]


# =============================================================================
# DIMENSION VERIFICATION (Section 5.7)
# =============================================================================

DIMENSION_CHAIN = {
    "eeg_19ch": {
        "input": (None, 19, 256),  # (B, 19, T) - T varies
        "eegnet_out": (None, 16, 8),  # After separable conv and pooling
        "transformer_out": (None, 256),
        "output": (None, 256),
    },
    "eeg_10ch": {
        "input": (None, 10, 256),
        "output": (None, 256),
    },
    "eeg_5ch": {
        "input": (None, 5, 256),
        "output": (None, 256),
    },
    "clinical": {
        "tabular_input": (None, 6),  # 6 Tier 1 features
        "fmri_input": (None, 4006),  # 4005 connectivity + mean_FD
        "tabular_out": (None, 128),
        "fmri_out": (None, 128),
        "merge_out": (None, 256),
        "output": (None, 256),
    },
    "actigraphy": {
        "timeseries_input": (None, 4, 1000),  # 4 channels, T samples
        "timeseries_out": (None, 224),
        "aux_input": (None, 2),
        "aux_out": (None, 32),
        "merge_out": (None, 256),
        "output": (None, 256),
    },
    "eye_tracking": {
        "input": (None, 100, 3),  # (B, T, 3) - x, y, pupil
        "lstm_out": (None, 100, 256),
        "attn_out": (None, 256),
        "output": (None, 256),
    },
    "projection": {
        "input": (None, 256),
        "output": (None, 256),
    },
    "fusemoe": {
        "input": (None, 256),
        "expert_out": (None, 256),
        "output": (None, 256),
    },
    "diagnosis_head": {
        "input": (None, 256),
        "output": (None, 1),
    },
    "subtype_head": {
        "input": (None, 256),
        "output": (None, 3),
    },
    "severity_head": {
        "input": (None, 256),
        "output": (None, 2),
    },
}
