"""
Full training loop.
Key implementation notes:
  - BF16 autocast only — NO GradScaler (not needed / incorrect for BF16)
  - EMA updated every step after optimizer.step()
  - Gradual unfreeze applied at epoch boundaries
  - mAP used as primary metric everywhere
  - Gradient health monitoring (explosion/vanishing detection)
  - BN recalibration after unfreeze events
  - Robust interruption recovery (global_step tracking, OOM handler, SIGTERM handler)
"""
import time, torch, logging, sys, signal
from tqdm import tqdm
from torch.cuda.amp import autocast
from training.metrics import MultiLabelMetrics
from training.gradient_monitor import GradientMonitor
from training.bn_calibration import recalibrate_bn_statistics, should_recalibrate

log = logging.getLogger("trainer")

class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 optimizer, scheduler, loss_fn,
                 ema, early_stopping, checkpoint_manager,
                 config, class_names,
                 start_epoch=0, best_metric=0.0, global_step=0):
        self.model   = model
        self.tl, self.vl = train_loader, val_loader
        self.opt     = optimizer
        self.sched   = scheduler
        self.loss_fn = loss_fn
        self.ema     = ema
        self.es      = early_stopping
        self.ckpt    = checkpoint_manager
        self.cfg     = config
        self.names   = class_names
        self.start_epoch  = start_epoch
        self.global_step  = global_step
        self.best_metric  = best_metric
        self.device  = torch.device(config.device)
        # BF16 context — note: NO GradScaler with BF16
        self.use_bf16 = (config.precision == "bf16" and
                         torch.cuda.is_available() and
                         torch.cuda.is_bf16_supported())
        # Gradient health monitor
        self.grad_monitor = GradientMonitor(
            model, log_every_n_steps=50, history_len=200
        )

    def _autocast(self):
        if self.use_bf16:
            return autocast(dtype=torch.bfloat16)
        return autocast(enabled=False)
        
    def _install_signal_handlers(self, current_epoch: int):
        """Install SIGTERM handler for cluster preemption."""
        def handler(signum, frame):
            log.warning(f"Received signal {signum}. Saving emergency checkpoint and exiting...")
            self.ckpt.save_emergency(
                self.model, self.opt, self.sched, self.ema, self.es,
                current_epoch, self.global_step, self.cfg,
                getattr(self.tl.dataset, 'reg', None)
            )
            sys.exit(0)
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGHUP, handler)

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()

        # Apply gradual unfreeze schedule
        sched_int = {int(k): v
                     for k, v in self.cfg.unfreeze_schedule.items()}
        if epoch in sched_int:
            self.model.unfreeze_layers(sched_int[epoch])
            log.info(f"Epoch {epoch}: unfreezing {sched_int[epoch]}")

        # Recalibrate BN statistics after unfreeze events
        if should_recalibrate(epoch, self.cfg.unfreeze_schedule):
            recalibrate_bn_statistics(
                self.model, self.tl, str(self.device), num_batches=100
            )
            # Restore requires_grad based on current unfreeze state
            if epoch in sched_int:
                self.model.unfreeze_layers(sched_int[epoch])
            self.model.train()

        accumulate = max(1, getattr(self.cfg, 'gradient_accumulation', 1))
        total_loss = 0.0
        self.opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(self.tl, desc=f"Train E{epoch}")):
            x = batch["input"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)

            with self._autocast():
                logits = self.model(x)
                loss   = self.loss_fn(logits, y) / accumulate

            loss.backward()

            # Gradient health check (after backward, before optimizer step)
            self.grad_monitor.check(self.global_step)

            total_loss += loss.item() * accumulate

            if (step + 1) % accumulate == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), getattr(self.cfg, 'gradient_clip', 1.0))
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                self.global_step += 1
                
                if self.ema:
                    self.ema.update(self.model)

            # Support step-based schedulers (e.g. OneCycleLR)
            if hasattr(self.sched, 'step') and not isinstance(
                self.sched, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            ):
                self.sched.step()

        # Support epoch-based boundary schedulers
        if isinstance(self.sched, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            self.sched.step(epoch)

        # Include gradient health summary in epoch metrics
        train_metrics = {"train_loss": total_loss / max(len(self.tl), 1)}
        grad_summary = self.grad_monitor.get_summary()
        train_metrics.update({"grad_" + k: v for k, v in grad_summary.items()})
        return train_metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        eval_model = self.ema.ema if self.ema else self.model
        eval_model.eval()
        metrics = MultiLabelMetrics(self.cfg.num_classes, self.names)
        total_loss = 0.0

        for batch in tqdm(self.vl, desc=f"Val   E{epoch}"):
            x = batch["input"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)
            with self._autocast():
                logits = eval_model(x)
                total_loss += self.loss_fn(logits, y).item()
            metrics.update(logits, y)

        result = metrics.compute()
        result["val_loss"] = total_loss / max(len(self.vl), 1)
        return result

    def fit(self) -> dict:
        best_metrics = {}
        for epoch in range(self.start_epoch, self.cfg.epochs):
            self._install_signal_handlers(epoch)
            t0 = time.time()
            
            try:
                train_m = self.train_epoch(epoch)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                log.error(
                    f"GPU OOM during epoch {epoch}, step {self.global_step}. "
                    f"Try reducing batch_size (currently {self.cfg.batch_size}) "
                    f"or increasing gradient_accumulation."
                )
                raise e
            except KeyboardInterrupt:
                log.warning("\nKeyboardInterrupt received! Saving emergency checkpoint...")
                class_registry = getattr(self.tl.dataset, 'reg', None)
                self.ckpt.save_emergency(
                    self.model, self.opt, self.sched, self.ema, self.es,
                    epoch, self.global_step, self.cfg, class_registry
                )
                log.info("Emergency checkpoint saved. Resume training by re-running the same command.")
                sys.exit(130)

            val_m = self.validate(epoch)

            all_m   = {**train_m, **val_m,
                       "epoch": epoch,
                       "epoch_time_s": time.time() - t0,
                       "gpu_mem_mb": torch.cuda.max_memory_allocated() // 1e6
                       if torch.cuda.is_available() else 0}

            log.info(
                f"E{epoch:03d}  loss={train_m['train_loss']:.4f}  "
                f"val_loss={val_m['val_loss']:.4f}  "
                f"mAP={val_m['mAP']:.4f}")
            for name, d in val_m["per_class"].items():
                log.info(f"  {name:30s}  AP={d['AP']:.3f}  "
                         f"P={d['P']:.3f}  R={d['R']:.3f}  F1={d['F1']:.3f}")

            class_registry = getattr(self.tl.dataset, 'reg', None)
            self.ckpt.save(
                self.model, self.opt, self.sched, self.ema, self.es,
                epoch, self.global_step, all_m, self.cfg, class_registry
            )

            if val_m["mAP"] > self.best_metric:
                self.best_metric = val_m["mAP"]
                best_metrics = all_m.copy()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            if self.es(val_m):
                log.info(f"Early stopping at epoch {epoch}")
                break

        return best_metrics
