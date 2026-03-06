# ExplainMoE-ADHD v2.13

Cross-modal aligned unified ADHD classifier using Mixture-of-Experts.

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

| Phase | What trains | LR | Max |
|-------|-------------|-----|-----|
| 1 | Self-supervised encoders (Group B) | 1e-3 | 30 ep |
| 2 | Supervised encoders (Group A) + temp MLP | 5e-4 | 50 ep |
| 3 | Projection heads (MMD alignment) | 1e-4 | 1000 steps |
| 4 | FuseMoE + task heads | 3e-4 / 3e-5 | 3000 steps |
| 5 | Joint fine-tuning (all except eye-tracking) | 1e-5 | 1000 steps |

## Evaluation

5-fold GroupKFold cross-validation stratified by (label, modality, dataset_source).

## Setup

```bash
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest tests/test_all.py -v
```

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
└── test_all.py
```
