"""Unified configuration entry point for ExplainMoE-ADHD v2.13."""

from dataclasses import dataclass, field
from explainmoe_adhd.config.model_config import ModelConfig, DEFAULT_MODEL_CONFIG
from explainmoe_adhd.config.training_config import TrainingConfig, DEFAULT_TRAINING_CONFIG


@dataclass
class ExplainMoEConfig:
    """Top-level configuration combining model and training configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


DEFAULT_CONFIG = ExplainMoEConfig()
