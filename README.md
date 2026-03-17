# CTR-GCN AVA Action Recognition Pipeline

Multi-label skeleton-based action recognition using **CTR-GCN** (Channel-wise Topology Refinement Graph Convolution) fine-tuned on the **AVA** and **AVA-Kinetics** datasets.

## Overview

This project implements a complete pipeline from data download through model training and evaluation:

- **Phase 1**: Fine-tune Kinetics-400 pretrained CTR-GCN on AVA-Kinetics (multi-label)
- **Phase 2**: Further fine-tune on AVA with ground-truth bounding boxes (multi-label)

### Key Features
- OpenPose-18 skeleton graph (matching Kinetics-400 pretrained weights)
- COCO-17 → OpenPose-18 joint conversion for RTMPose compatibility
- Asymmetric Loss for multi-label imbalance
- BF16 mixed precision for Ada Lovelace GPUs
- Gradual backbone unfreezing schedule
- EMA model weights for stable evaluation
- IoU-based person tracking across frames
- Thread-safe SQLite state tracking (WAL mode)
- `torch.compile` for ~25% throughput gain

## Quick Start

### 1. Environment Setup

```bash
# Using conda + pip (requires NVIDIA GPU with SM 8.x+)
bash scripts/setup_env.sh
```

### 2. Download Pretrained Models

```bash
python scripts/download_pretrained.py
```

### 3. Smoke Test (< 15 minutes)

```bash
python main.py --mode test_run --test_mode
```

This generates synthetic skeleton data and runs a mini training loop.

### 4. Full Pipeline

```bash
# Download datasets
python main.py --mode download

# Annotate (extract skeletons)
python main.py --mode annotate

# Train (Phase 1 → Phase 2)
python main.py --mode train --phase both

# Evaluate
python main.py --mode evaluate

# Export to ONNX
python main.py --mode export
```

## Project Structure

```
├── configs/                  # YAML configuration files
│   ├── base_config.yaml      # Base hardware, skeleton, paths
│   ├── class_config.yaml     # Target AVA action classes (15 selected)
│   ├── joint_mappings.yaml   # COCO-17 ↔ OpenPose-18 joint indices
│   ├── phase1_ava_kinetics.yaml
│   └── phase2_ava.yaml
├── models/                   # CTR-GCN architecture
│   ├── ctrgcn/              # Core GCN modules
│   ├── ctrgcn_ava.py        # AVA-adapted model head
│   └── model_factory.py     # Model builder with pretrained loading
├── annotation/               # Pose estimation & skeleton extraction
│   ├── joint_converter.py   # COCO-17 → OpenPose-18
│   ├── person_tracker.py    # IoU-based tracking
│   ├── pose_estimator.py    # RTMDet + RTMPose wrapper
│   └── extractor.py         # Full extraction pipeline
├── data_pipeline/            # PyTorch datasets & dataloaders
├── training/                 # Training engine
│   ├── losses.py            # Asymmetric Loss + BCE
│   ├── trainer.py           # Full training loop (BF16, EMA, unfreeze)
│   ├── pipeline.py          # Two-phase orchestration
│   └── ...
├── evaluation/               # AVA-style mAP evaluation
├── scripts/                  # Setup, download, smoke test, ONNX export
├── tests/                    # Comprehensive test suite
├── main.py                   # CLI entry point
└── requirements.txt          # Pinned dependencies
```

## Hardware Requirements

- **GPU**: NVIDIA RTX 4500 Ada (or any SM 8.x+ GPU with BF16 support)
- **VRAM**: 16+ GB recommended
- **Disk**: 500+ GB for full AVA dataset (5 GB for smoke test)
- **Driver**: >= 525.xx
- **CUDA**: 12.1

## Testing

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run specific test modules
pytest tests/test_joint_converter.py -v
pytest tests/test_model.py -v
pytest tests/test_integration.py -v
```

## Target Classes (15 AVA actions)

| Category | Actions |
|----------|---------|
| Movement | dance, run/jog, walk |
| Object | eat, drink, ride (bike/horse), smoke, use a computer |
| Interaction | hand shake, hug, kick, punch/slap, push |
| Pose | bend/bow, crouch/kneel |

## Non-Negotiable Constraints

1. Both phases use **multi-label** classification (BCE/Asymmetric Loss, mAP metric)
2. **OpenPose-18** joint layout preserved throughout (Kinetics-400 compatibility)
3. **BF16** precision only — no FP16/GradScaler on Ada Lovelace
4. AVA uses **GT bounding boxes** for pose estimation
5. All validation uses **EMA model weights**
