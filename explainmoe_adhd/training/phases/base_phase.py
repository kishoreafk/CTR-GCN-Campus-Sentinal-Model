"""
Base Phase class for ExplainMoE-ADHD v2.13.

This module defines the base class for all training phases:
- Phase 1: Self-Supervised Pretraining
- Phase 2: Supervised Encoder Training
- Phase 3: MMD Latent Space Alignment
- Phase 4: FuseMoE Training
- Phase 5: Joint Fine-Tuning
"""

import os
import signal
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class PhaseOutput:
    """Output from a training phase."""
    model_state: Dict[str, Any]
    metrics: Dict[str, float]
    best_epoch: int
    best_value: float


class BasePhase(ABC):
    """
    Base class for all training phases.
    
    Each phase implements:
    - forward(): Training step
    - evaluate(): Evaluation step
    - train(): Full training loop with early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
    ):
        self.model = model
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_value = float('-inf') if self.is_maximize() else float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        
        # Metrics history
        self.metrics_history = []
        
        # Best model state kept in memory for final restore
        self._best_model_state: Optional[Dict[str, Any]] = None
        
        # Interrupt flag for graceful shutdown
        self._interrupted = False
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    @abstractmethod
    def is_maximize(self) -> bool:
        """Whether the monitor metric should be maximized."""
        pass
    
    @abstractmethod
    def get_monitor_metric(self, metrics: Dict[str, float]) -> float:
        """Extract the monitor metric from metrics dict."""
        pass
    
    @abstractmethod
    def forward_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step."""
        pass
    
    @abstractmethod
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step."""
        pass
    
    def check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """
        Check if early stopping criteria are met.
        Saves best model checkpoint when improvement is detected.
        
        Returns:
            True if training should stop
        """
        monitor_value = self.get_monitor_metric(metrics)
        improved = False
        
        if self.is_maximize():
            if monitor_value > self.best_value:
                improved = True
        else:
            if monitor_value < self.best_value:
                improved = True
        
        if improved:
            self.best_value = monitor_value
            self.best_epoch = self.current_epoch
            self.patience_counter = 0
            # Save best model to disk and keep state in memory
            self._best_model_state = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            self.save_checkpoint(best_path)
            return False
        
        self.patience_counter += 1
        patience = self.config.get('early_stopping_patience', 10)
        
        return self.patience_counter >= patience
    
    def save_checkpoint(self, path: str, additional_state: Optional[Dict] = None):
        """Save model checkpoint atomically (write to tmp then rename)."""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'epoch': self.current_epoch,
            'step': self.current_step,
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'metrics_history': self.metrics_history,
            'patience_counter': self.patience_counter,
        }
        if self.optimizer is not None:
            checkpoint['optimizer_state'] = self.optimizer.state_dict()
        if self.scheduler is not None and hasattr(self.scheduler, 'state_dict'):
            checkpoint['scheduler_state'] = self.scheduler.state_dict()
        if additional_state:
            checkpoint.update(additional_state)
        # Atomic write: save to temp file, then rename to prevent corruption
        tmp_path = path + '.tmp'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint and restore optimizer/scheduler state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.best_value = checkpoint['best_value']
        self.best_epoch = checkpoint['best_epoch']
        self.metrics_history = checkpoint.get('metrics_history', [])
        self.patience_counter = checkpoint.get('patience_counter', 0)
        if self.optimizer is not None and 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.scheduler is not None and 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        return checkpoint
    
    def restore_best_model(self):
        """Restore the best model weights (in-memory or from disk)."""
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
            return
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
    
    def mark_phase_complete(self):
        """Write a marker file indicating this phase finished successfully."""
        marker = os.path.join(self.checkpoint_dir, 'COMPLETED')
        with open(marker, 'w') as f:
            f.write(f'best_epoch={self.best_epoch}\nbest_value={self.best_value}\n')
    
    def is_phase_complete(self) -> bool:
        """Check if this phase was already completed in a prior run."""
        return os.path.exists(os.path.join(self.checkpoint_dir, 'COMPLETED'))
    
    def _install_signal_handlers(self):
        """Install handlers so Ctrl+C / SIGTERM triggers a graceful save."""
        def _handler(signum, frame):
            print(f'\n[!] Signal {signum} received ΓÇö saving emergency checkpoint...')
            self._interrupted = True
        # Only install on main thread; workers may call this too
        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
        except (OSError, ValueError):
            pass  # Not main thread or unsupported platform
    
    def train(
        self,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        max_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        grad_clip: float = 1.0,
        log_every: int = 10,
        save_every: Optional[int] = None,
        resume_from: Optional[str] = None,
    ) -> PhaseOutput:
        """
        Full training loop with early stopping, best-model saving, and periodic checkpoints.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer instance
            scheduler: Optional LR scheduler
            max_epochs: Maximum epochs (epoch-based phases)
            max_steps: Maximum steps (step-based phases like 4/5)
            grad_clip: Max gradient norm for clipping
            log_every: Log metrics every N steps
            save_every: Save periodic checkpoint every N steps (None = disabled)
            resume_from: Path to checkpoint to resume from
        
        Returns:
            PhaseOutput with best model state and metrics
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._interrupted = False
        self._install_signal_handlers()
        
        # Resume from checkpoint if available
        resume_step = 0
        if resume_from is not None and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
            resume_step = self.current_step
            # Reload best model state from disk so it isn't lost
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            if os.path.exists(best_path):
                best_ckpt = torch.load(best_path, map_location=self.device)
                self._best_model_state = best_ckpt['model_state']
            print(f"[Resume] Loaded checkpoint ΓÇö epoch={self.current_epoch}, "
                  f"step={self.current_step}, best_value={self.best_value:.4f}")
        
        if max_epochs is None:
            max_epochs = self.config.get('max_epochs', 50)
        
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_metrics: Dict[str, float] = {}
            n_batches = 0
            
            for batch in train_loader:
                # On resume, skip batches that were already processed in this epoch
                if self.current_step < resume_step:
                    self.current_step += 1
                    continue
                # After we've caught up, stop skipping
                resume_step = 0
                
                # Graceful interrupt: save and exit
                if self._interrupted:
                    print('[!] Saving interrupt checkpoint...')
                    interrupt_path = os.path.join(self.checkpoint_dir, 'last_model.pt')
                    self.save_checkpoint(interrupt_path)
                    print(f'[!] Checkpoint saved to {interrupt_path}. Exiting.')
                    return self._make_output()
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                
                optimizer.zero_grad()
                loss, step_metrics = self.forward_step(batch)
                loss.backward()
                
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
                optimizer.step()
                self.current_step += 1
                epoch_loss += loss.item()
                n_batches += 1
                
                # Accumulate step metrics
                for k, v in step_metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                
                if self.current_step % log_every == 0:
                    avg_loss = epoch_loss / n_batches
                    print(f"  Step {self.current_step} | loss={avg_loss:.4f}")
                
                # Periodic checkpoint
                if save_every and self.current_step % save_every == 0:
                    periodic_path = os.path.join(
                        self.checkpoint_dir, f'checkpoint_step_{self.current_step}.pt'
                    )
                    self.save_checkpoint(periodic_path)
                
                # Step-based termination
                if max_steps is not None and self.current_step >= max_steps:
                    break
            
            # Average epoch metrics
            if n_batches > 0:
                epoch_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
                epoch_metrics['train_loss'] = epoch_loss / n_batches
            
            # Validation
            val_metrics = self._run_validation(val_loader)
            epoch_metrics.update(val_metrics)
            self.metrics_history.append(epoch_metrics)
            
            print(f"Epoch {epoch} | train_loss={epoch_metrics.get('train_loss', 0):.4f}"
                  f" | {self._format_val_metrics(val_metrics)}")
            
            if scheduler is not None:
                scheduler.step()
            
            # Early stopping (saves best model internally)
            if self.check_early_stopping(epoch_metrics):
                print(f"Early stopping at epoch {epoch}. "
                      f"Best epoch: {self.best_epoch}, best value: {self.best_value:.4f}")
                break
            
            # Save last_model.pt at end of every epoch for safe resume
            last_path = os.path.join(self.checkpoint_dir, 'last_model.pt')
            self.save_checkpoint(last_path)
            
            # Step-based termination
            if max_steps is not None and self.current_step >= max_steps:
                print(f"Reached max steps ({max_steps}). "
                      f"Best epoch: {self.best_epoch}, best value: {self.best_value:.4f}")
                break
            
            # Check interrupt between epochs
            if self._interrupted:
                print('[!] Interrupted between epochs ΓÇö checkpoint already saved.')
                return self._make_output()
        
        # Save final checkpoint (last epoch state)
        final_path = os.path.join(self.checkpoint_dir, 'last_model.pt')
        self.save_checkpoint(final_path)
        
        # Mark phase as fully completed
        self.mark_phase_complete()
        
        # Restore best model weights for downstream use
        self.restore_best_model()
        
        return PhaseOutput(
            model_state=self.model.state_dict(),
            metrics=self.metrics_history[-1] if self.metrics_history else {},
            best_epoch=self.best_epoch,
            best_value=self.best_value,
        )
    
    def _make_output(self) -> PhaseOutput:
        """Build PhaseOutput from current state (used on interrupt)."""
        self.restore_best_model()
        return PhaseOutput(
            model_state=self.model.state_dict(),
            metrics=self.metrics_history[-1] if self.metrics_history else {},
            best_epoch=self.best_epoch,
            best_value=self.best_value,
        )
    
    def _run_validation(self, val_loader) -> Dict[str, float]:
        """Run validation loop and return aggregated metrics."""
        self.model.eval()
        val_metrics: Dict[str, float] = {}
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                step_metrics = self.evaluate_step(batch)
                for k, v in step_metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v
                n_batches += 1
        
        if n_batches > 0:
            val_metrics = {k: v / n_batches for k, v in val_metrics.items()}
        
        self.model.train()
        return val_metrics
    
    @staticmethod
    def _format_val_metrics(val_metrics: Dict[str, float]) -> str:
        """Format validation metrics for logging."""
        parts = [f"{k}={v:.4f}" for k, v in val_metrics.items()]
        return " | ".join(parts) if parts else "no val metrics"


class Phase1Pretraining(BasePhase):
    """
    Phase 1: Self-Supervised Pretraining.
    
    Per-encoder, independent, Group B (unlabeled) data.
    Run once, shared across all folds.
    """
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cuda',
                 checkpoint_dir: str = 'checkpoints/phase1'):
        super().__init__(model, config, device, checkpoint_dir)
    
    def is_maximize(self) -> bool:
        return True  # Monitor pretraining loss (minimize)
    
    def get_monitor_metric(self, metrics: Dict[str, float]) -> float:
        return -metrics.get('loss', 0)  # Negative for maximization
    
    def forward_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step."""
        raise NotImplementedError("Phase1Pretraining.forward_step must be implemented")
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step."""
        raise NotImplementedError("Phase1Pretraining.evaluate_step must be implemented")


class Phase2SupervisedEncoder(BasePhase):
    """
    Phase 2: Supervised Encoder Training.
    
    Per-encoder, independent, Group A train_subjects.
    Re-run per fold.
    """
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cuda',
                 checkpoint_dir: str = 'checkpoints/phase2'):
        super().__init__(model, config, device, checkpoint_dir)
    
    def is_maximize(self) -> bool:
        return True  # Monitor val AUROC
    
    def get_monitor_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get('val_auroc', 0)


class Phase3MMDAlignment(BasePhase):
    """
    Phase 3: MMD Latent Space Alignment.
    
    Cross-modal, projection heads only.
    Re-run per fold.
    """
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cuda',
                 checkpoint_dir: str = 'checkpoints/phase3'):
        super().__init__(model, config, device, checkpoint_dir)
    
    def is_maximize(self) -> bool:
        return False  # Monitor MMD loss (minimize)
    
    def get_monitor_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get('mmd_loss', float('inf'))


class Phase4FuseMoE(BasePhase):
    """
    Phase 4: FuseMoE Training.
    
    Cross-modal, FuseMoE + task heads.
    Re-run per fold.
    """
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cuda',
                 checkpoint_dir: str = 'checkpoints/phase4'):
        super().__init__(model, config, device, checkpoint_dir)
    
    def is_maximize(self) -> bool:
        return True  # Monitor val AUROC macro
    
    def get_monitor_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get('val_auroc_macro', 0)


class Phase5FineTuning(BasePhase):
    """
    Phase 5: Joint Fine-Tuning.
    
    End-to-end, all components, low LR.
    Re-run per fold.
    """
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cuda',
                 checkpoint_dir: str = 'checkpoints/phase5'):
        super().__init__(model, config, device, checkpoint_dir)
    
    def is_maximize(self) -> bool:
        return True  # Monitor val AUROC macro
    
    def get_monitor_metric(self, metrics: Dict[str, float]) -> float:
        return metrics.get('val_auroc_macro', 0)
