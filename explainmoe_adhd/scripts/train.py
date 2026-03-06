"""
Training orchestration for ExplainMoE-ADHD v2.13.

Runs the 5-phase training pipeline across 5-fold cross-validation.
Saves best model (by early stopping) and last model per phase per fold.

Resume logic:
    - Each phase writes a COMPLETED marker on success.
    - On re-run, completed phases are skipped automatically.
    - Interrupted phases resume from their last_model.pt checkpoint.

Checkpoint layout:
    checkpoints/
        phase1/                        (shared across folds)
            best_model.pt
            last_model.pt
            COMPLETED
        fold_{i}/
            phase{p}/
                best_model.pt
                last_model.pt
                COMPLETED
                checkpoint_step_{N}.pt
"""

import os
import argparse
import torch

from explainmoe_adhd.config.training_config import (
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG,
    LR_SCHEDULE,
    LOSS_WEIGHTS,
    EARLY_STOPPING_MONITORS,
)
from explainmoe_adhd.training.phases.phase1 import Phase1Pretraining
from explainmoe_adhd.training.phases.phase2 import Phase2SupervisedEncoder
from explainmoe_adhd.training.phases.phase3 import Phase3MMDAlignment
from explainmoe_adhd.training.phases.phase4 import Phase4FuseMoE
from explainmoe_adhd.training.phases.phase5 import Phase5FineTuning


def run_phase(phase, train_loader, val_loader, optimizer, scheduler=None,
              max_epochs=None, max_steps=None, config=None):
    """Run a single training phase with automatic resume, return PhaseOutput.

    If the phase was already completed (COMPLETED marker exists), returns None.
    If last_model.pt exists, training resumes from that checkpoint.
    """
    # Skip if already finished in a prior run
    if phase.is_phase_complete():
        print(f"  [Skip] Phase already completed ({phase.checkpoint_dir}/COMPLETED found)")
        return None

    save_every = config.save_checkpoint_every if config else 500
    grad_clip = config.grad_clip if config else 1.0
    log_every = config.log_every if config else 10

    # Auto-detect resume checkpoint
    resume_path = os.path.join(phase.checkpoint_dir, 'last_model.pt')
    resume_from = resume_path if os.path.exists(resume_path) else None

    return phase.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        max_epochs=max_epochs,
        max_steps=max_steps,
        grad_clip=grad_clip,
        log_every=log_every,
        save_every=save_every,
        resume_from=resume_from,
    )


def train_fold(fold: int, config: TrainingConfig, data_module):
    """
    Train all 5 phases for a single fold.

    Args:
        fold: Fold index (0-4)
        config: TrainingConfig instance
        data_module: Object that provides train_loader/val_loader per phase
    """
    base_dir = os.path.join('checkpoints', f'fold_{fold}')
    device = config.device

    # ------------------------------------------------------------------
    # Phase 1: Self-Supervised Pretraining (shared across folds, run once)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\nPhase 1: Self-Supervised Pretraining\n{'='*60}")
    phase1_dir = os.path.join('checkpoints', 'phase1')
    model_p1 = data_module.get_model('phase1')
    phase1 = Phase1Pretraining(model_p1, vars(config.phase1), device, phase1_dir)
    train_dl, val_dl = data_module.get_loaders('phase1', fold)
    opt = torch.optim.AdamW(
        model_p1.parameters(), lr=config.phase1.lr,
        weight_decay=config.phase1.weight_decay, betas=config.phase1.betas,
    )
    run_phase(phase1, train_dl, val_dl, opt,
              max_epochs=config.phase1.epochs, config=config)

    # ------------------------------------------------------------------
    # Phase 2: Supervised Encoder Training (per fold)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\nFold {fold} | Phase 2: Supervised Encoder Training\n{'='*60}")
    phase2_dir = os.path.join(base_dir, 'phase2')
    model_p2 = data_module.get_model('phase2', fold)
    phase2 = Phase2SupervisedEncoder(model_p2, vars(config.phase2), device, phase2_dir)
    train_dl, val_dl = data_module.get_loaders('phase2', fold)
    opt = torch.optim.AdamW(
        model_p2.parameters(), lr=config.phase2.lr,
        weight_decay=config.phase2.weight_decay, betas=config.phase2.betas,
    )
    output_p2 = run_phase(phase2, train_dl, val_dl, opt,
                          max_epochs=config.phase2.max_epochs, config=config)
    if output_p2 is not None:
        print(f"  Phase 2 best: epoch={output_p2.best_epoch}, val_auroc={output_p2.best_value:.4f}")

    # ------------------------------------------------------------------
    # Phase 3: MMD Latent Space Alignment (per fold)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\nFold {fold} | Phase 3: MMD Alignment\n{'='*60}")
    phase3_dir = os.path.join(base_dir, 'phase3')
    model_p3 = data_module.get_model('phase3', fold)
    phase3 = Phase3MMDAlignment(model_p3, vars(config.phase3), device, phase3_dir)
    train_dl, val_dl = data_module.get_loaders('phase3', fold)
    opt = torch.optim.Adam(
        [p for p in model_p3.parameters() if p.requires_grad],
        lr=config.phase3.lr,
    )
    output_p3 = run_phase(phase3, train_dl, val_dl, opt,
                          max_steps=config.phase3.max_steps, config=config)
    if output_p3 is not None:
        print(f"  Phase 3 best: step={output_p3.best_epoch}, mmd={output_p3.best_value:.4f}")

    # ------------------------------------------------------------------
    # Phase 4: FuseMoE Training (per fold)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\nFold {fold} | Phase 4: FuseMoE Training\n{'='*60}")
    phase4_dir = os.path.join(base_dir, 'phase4')
    model_p4 = data_module.get_model('phase4', fold)
    phase4 = Phase4FuseMoE(model_p4, vars(config.phase4), device, phase4_dir)
    train_dl, val_dl = data_module.get_loaders('phase4', fold)
    opt = torch.optim.AdamW(
        [p for p in model_p4.parameters() if p.requires_grad],
        lr=config.phase4.lr_main, weight_decay=config.phase4.weight_decay,
    )
    output_p4 = run_phase(phase4, train_dl, val_dl, opt,
                          max_steps=config.phase4.max_steps, config=config)
    if output_p4 is not None:
        print(f"  Phase 4 best: step={output_p4.best_epoch}, val_auroc_macro={output_p4.best_value:.4f}")

    # ------------------------------------------------------------------
    # Phase 5: Joint Fine-Tuning (per fold)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\nFold {fold} | Phase 5: Joint Fine-Tuning\n{'='*60}")
    phase5_dir = os.path.join(base_dir, 'phase5')
    model_p5 = data_module.get_model('phase5', fold)
    phase5 = Phase5FineTuning(model_p5, vars(config.phase5), device, phase5_dir)
    train_dl, val_dl = data_module.get_loaders('phase5', fold)
    opt = torch.optim.AdamW(
        model_p5.parameters(), lr=config.phase5.lr,
        weight_decay=config.phase5.weight_decay,
    )
    output_p5 = run_phase(phase5, train_dl, val_dl, opt,
                          max_steps=config.phase5.max_steps, config=config)
    if output_p5 is not None:
        print(f"  Phase 5 best: step={output_p5.best_epoch}, val_auroc_macro={output_p5.best_value:.4f}")

    return output_p5


def main():
    parser = argparse.ArgumentParser(description='Train ExplainMoE-ADHD v2.13')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Which folds to train (default: all 5)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = DEFAULT_TRAINING_CONFIG
    config.seed = args.seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # NOTE: data_module must be implemented to provide get_model() and get_loaders()
    # for each phase and fold. This is dataset-specific and left for the user.
    # Example interface:
    #   data_module.get_model(phase_name, fold) -> nn.Module
    #   data_module.get_loaders(phase_name, fold) -> (train_loader, val_loader)
    raise NotImplementedError(
        "Provide a data_module that implements get_model(phase, fold) and "
        "get_loaders(phase, fold). See train_fold() for the expected interface."
    )

    for fold in args.folds:
        print(f"\n{'#'*60}\n# FOLD {fold}\n{'#'*60}")
        train_fold(fold, config, data_module=None)


if __name__ == '__main__':
    main()
