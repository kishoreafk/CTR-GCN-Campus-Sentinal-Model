"""Pre-built model configurations for ExplainMoE-ADHD v2.13."""

from explainmoe_adhd.config.model_config import (
    EEGEncoderConfig,
    ClinicalEncoderConfig,
    ActigraphyEncoderConfig,
    EyeTrackingEncoderConfig,
)

# Default model configurations per spec section 5
DEFAULT_CHILD_EEG_19CH = EEGEncoderConfig(channels=19, sampling_rate=128)
DEFAULT_CHILD_EEG_10CH = EEGEncoderConfig(channels=10, sampling_rate=128)
DEFAULT_ADULT_EEG_5CH = EEGEncoderConfig(channels=5, sampling_rate=256)
DEFAULT_CLINICAL = ClinicalEncoderConfig()
DEFAULT_ACTIGRAPHY = ActigraphyEncoderConfig()
DEFAULT_EYE_TRACKING = EyeTrackingEncoderConfig()
