"""
Load YAML configs, apply inheritance, validate, produce a typed dataclass.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import yaml, logging
log = logging.getLogger("config_loader")

@dataclass
class TrainingConfig:
    # Identity
    experiment_name: str = "ctrgcn_ava"
    dataset: str = "ava_kinetics"      # "ava_kinetics" | "ava"
    seed: int = 42

    # ── CRITICAL: always True for both AVA and AVA-Kinetics ──
    multi_label: bool = True
    num_classes: int = 15

    # Model
    pretrained_ckpt: str = ""
    joint_layout: str = "openpose_18"
    num_joints: int = 18
    num_frames: int = 64
    max_persons: int = 2
    in_channels: int = 3

    # Fine-tuning
    finetune_mode: str = "gradual"
    unfreeze_schedule: Dict[int, List[str]] = field(default_factory=dict)
    dropout: float = 0.1

    # Loss
    loss_type: str = "asymmetric"
    asymmetric_gamma_neg: float = 4.0
    asymmetric_gamma_pos: float = 0.0
    asymmetric_clip: float = 0.05

    # Training
    epochs: int = 50
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine_warm_restarts"
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    gradient_accumulation: int = 1

    # Precision — BF16 by default, no GradScaler needed
    precision: str = "bf16"
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_mAP"
    early_stopping_mode: str = "max"

    # Checkpointing
    save_every_n_epochs: int = 5
    keep_last_k_checkpoints: int = 3
    auto_resume: bool = True
    resume_run_id: Optional[str] = None

    # Logging
    use_wandb: bool = False
    use_tensorboard: bool = True

    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    output_dir: str = "outputs"
    state_db: str = "data/state.db"
    class_config: str = "configs/class_config.yaml"
    joint_mappings: str = "configs/joint_mappings.yaml"

    # Hardware
    device: str = "cuda:0"
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 3

    # Test mode
    test_mode: bool = False
    test_max_videos: int = 10
    test_max_epochs: int = 3


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(yaml_path: str, overrides: dict = None) -> TrainingConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    if "inherit" in raw:
        with open(raw.pop("inherit")) as f:
            base = yaml.safe_load(f) or {}
        raw = _deep_merge(base, raw)
    if overrides:
        raw = _deep_merge(raw, overrides)

    # Flatten nested keys into TrainingConfig fields
    flat = {}
    for section in ("hardware", "skeleton", "annotation", "paths", "project"):
        flat.update(raw.pop(section, {}))
    flat.update(raw)

    cfg = TrainingConfig()
    for k, v in flat.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    validate_config(cfg)
    return cfg

def validate_config(cfg: TrainingConfig) -> None:
    """Raise on fatal misconfigurations."""
    assert cfg.multi_label, \
        "multi_label must be True for both AVA and AVA-Kinetics"
    assert cfg.joint_layout == "openpose_18", \
        "joint_layout must be openpose_18 to match Kinetics-400 pretrained weights"
    assert cfg.precision in ("bf16", "fp32"), \
        "precision must be 'bf16' or 'fp32' (fp16 removed — use bf16 on Ada)"
    if cfg.precision == "bf16":
        try:
            import torch
            if torch.cuda.is_available():
                assert torch.cuda.is_bf16_supported(), \
                    "BF16 not supported on this GPU"
        except ImportError:
            pass  # torch not installed yet
    log.info("Config validated OK")
