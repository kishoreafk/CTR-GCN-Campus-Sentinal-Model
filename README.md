# ExplainMoE-ADHD v2.13

Cross-modal aligned unified ADHD classifier using Mixture-of-Experts.

## Prerequisites

- Python 3.10+
- pip
- Git
- (Recommended) NVIDIA GPU with CUDA support

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/kishoreafk/ADHD-MoE-Model-Pipeline.git
cd ADHD-MoE-Model-Pipeline
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages: `torch>=1.9.0`, `numpy>=1.21.0`, `scikit-learn>=1.0.0`, `scipy>=1.7.0`

For running tests, also install pytest:

```bash
pip install pytest
```

## Running Tests

Run the full test suite (81 tests) to verify the installation:

```bash
python -m pytest tests/test_all.py -v
```

Run a specific test class:

```bash
# Encoder tests only
python -m pytest tests/test_all.py::TestEEGEncoders -v

# FuseMoE tests only
python -m pytest tests/test_all.py::TestFuseMoE -v

# Loss function tests
python -m pytest tests/test_all.py::TestLosses -v

# Full model forward pass tests
python -m pytest tests/test_all.py::TestExplainMoEModel -v

# Phase configuration tests
python -m pytest tests/test_all.py::TestPhaseConfiguration -v

# Ablation baseline tests (Conditions A, B, D)
python -m pytest tests/test_all.py::TestAblationBaselines -v

# End-to-end dimension chain verification
python -m pytest tests/test_all.py::TestDimensionChain -v
```

All 81 tests must pass before proceeding with training.

## Architecture

**Encoders** (5 modalities → ℝ²⁵⁶):
- Child EEG 19-ch (EEGNet + Transformer + hardware token)
- Child EEG 10-ch (EEGNet + Transformer + hardware token)
- Adult EEG 5-ch (EEGNet + Transformer + hardware token)
- Clinical/fMRI (FT-Transformer + fMRI MLP)
- Actigraphy (ResNet1D + BiLSTM + aux MLP)
- Eye-tracking (BiLSTM + attention, proof-of-concept, permanently frozen)

**FuseMoE**: 4 shared experts, 5 per-modality Laplace-kernel routers, top-K=2, residual + LayerNorm.

**Task Heads**: Diagnosis (binary), Subtype (3-class), Severity (2-regression).

## 5-Phase Training Pipeline

| Phase | What trains | LR | Max | Early stop metric |
|-------|-------------|-----|-----|-------------------|
| 1 | Self-supervised encoders (Group B) | 1e-3 | 30 ep | reconstruction loss |
| 2 | Supervised encoders (Group A) + temp MLP | 5e-4 | 50 ep | per-encoder val_auroc |
| 3 | Projection heads (MMD alignment) | 1e-4 | 1000 steps | mean_mmd_loss |
| 4 | FuseMoE + task heads | 3e-4 / 3e-5 | 3000 steps | val_auroc_macro |
| 5 | Joint fine-tuning (all except eye-tracking) | 1e-5 | 1000 steps | val_auroc_macro |

### Running the pipeline

The training script orchestrates all 5 phases across 5-fold cross-validation:

```bash
python -m explainmoe_adhd.scripts.train --folds 0 1 2 3 4 --seed 42
```

Train a single fold:

```bash
python -m explainmoe_adhd.scripts.train --folds 0
```

**Note:** Before running training, you must implement a `data_module` that provides:
- `data_module.get_model(phase_name, fold)` → `nn.Module`
- `data_module.get_loaders(phase_name, fold)` → `(train_loader, val_loader)`

See `explainmoe_adhd/scripts/train.py` for the expected interface.

### Resume from interruption

Training automatically resumes from the last checkpoint. Each phase writes a `COMPLETED`
marker on success; completed phases are skipped on re-run.

### Checkpoint layout

```
checkpoints/
├── phase1/                        # Shared across folds (run once)
│   ├── best_model.pt
│   ├── last_model.pt
│   └── COMPLETED
└── fold_{i}/
    └── phase{p}/
        ├── best_model.pt
        ├── last_model.pt
        ├── COMPLETED
        └── checkpoint_step_{N}.pt
```

## Evaluation

5-fold GroupKFold cross-validation stratified by (label, modality, dataset_source).

Mandatory metrics (Section 8.5):
- AUROC per modality (5-fold mean ± std)
- Macro AUROC across modality pathways
- Per-subtype AUROC
- Calibration (ECE, Brier score)
- Confound probes

```python
from explainmoe_adhd.scripts.evaluate import evaluate_model, compute_ece, bootstrap_ci
```

## Inference

Load a trained model and run predictions on new data:

```python
from explainmoe_adhd.scripts.inference import load_model, predict

model = load_model("checkpoints/fold_0/phase5/best_model.pt", device="cuda")
results = predict(model, batch, modality="clinical")
# results["diagnosis_prob"]   → P(ADHD) per subject
# results["subtype_probs"]    → P(Combined, HI, Inattentive)
# results["severity_scores"]  → [inattentive, hyperactive] scores
```

## Ablation Baselines (Section 10)

Three experimental conditions for comparison:

| Condition | Description | Module |
|-----------|-------------|--------|
| A — Baseline_M | Encoder + MLP, no FuseMoE, no alignment | `BaselineModel` |
| B — MoE_M_only | FuseMoE without Phase 3 alignment | `MoEWithoutAlignmentModel` |
| D — Simple_M | Classical ML (logistic regression / gradient boosting) | `SimpleMLBaseline` |

```python
from explainmoe_adhd.models.ablation_baselines import (
    BaselineModel,              # Condition A
    MoEWithoutAlignmentModel,   # Condition B
    SimpleMLBaseline,           # Condition D
)
```

## Datasets

11 datasets (D1–D11), split into two groups:

- **Group A** (D1–D3, D5–D8): ADHD-labeled, used in supervised phases + MMD alignment
- **Group B** (D9–D11): Pretraining only (no ADHD labels)

See `explainmoe_adhd/data/constants.py` for full dataset specifications, feature tiers,
hardware tokens, and licensing information.

## Project Structure

```
explainmoe_adhd/
├── config/           # Training, model, dataset configs
├── data/             # Constants, CV, samplers, preprocessing
├── evaluation/       # Probes
├── models/
│   ├── components/   # Encoders, FuseMoE, projection heads, task heads
│   ├── explainmoe_model.py
│   └── ablation_baselines.py   # Conditions A, B, D
├── scripts/          # train, evaluate, inference
├── training/
│   ├── losses.py
│   └── phases/       # Phases 1–5
└── utils/
tests/
└── test_all.py       # 81 unit tests
```
