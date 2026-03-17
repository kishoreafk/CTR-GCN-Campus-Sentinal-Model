"""
Full training loop.
Key implementation notes:
  - BF16 autocast only — NO GradScaler (not needed / incorrect for BF16)
  - EMA updated every step after optimizer.step()
  - Gradual unfreeze applied at epoch boundaries
  - mAP used as primary metric everywhere
"""
import time, torch, logging
from tqdm import tqdm
from torch.cuda.amp import autocast
from training.metrics import MultiLabelMetrics
log = logging.getLogger("trainer")

class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 optimizer, scheduler, loss_fn,
                 ema, early_stopping, checkpoint_manager,
                 config, class_names,
                 start_epoch=0, best_metric=0.0):
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
        self.best_metric  = best_metric
        self.device  = torch.device(config.device)
        # BF16 context — note: NO GradScaler with BF16
        self.use_bf16 = (config.precision == "bf16" and
                         torch.cuda.is_available() and
                         torch.cuda.is_bf16_supported())

    def _autocast(self):
        if self.use_bf16:
            return autocast(dtype=torch.bfloat16)
        return autocast(enabled=False)

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()

        # Apply gradual unfreeze schedule
        sched_int = {int(k): v
                     for k, v in self.cfg.unfreeze_schedule.items()}
        if epoch in sched_int:
            self.model.unfreeze_layers(sched_int[epoch])
            log.info(f"Epoch {epoch}: unfreezing {sched_int[epoch]}")

        accumulate = max(1, self.cfg.gradient_accumulation)
        total_loss = 0.0
        self.opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(self.tl, desc=f"Train E{epoch}")):
            x = batch["input"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)

            with self._autocast():
                logits = self.model(x)
                loss   = self.loss_fn(logits, y) / accumulate

            loss.backward()
            total_loss += loss.item() * accumulate

            if (step + 1) % accumulate == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.gradient_clip)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                if self.ema:
                    self.ema.update(self.model)

        if isinstance(self.sched,
                      torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            self.sched.step()

        return {"train_loss": total_loss / max(len(self.tl), 1)}

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
            t0 = time.time()
            train_m = self.train_epoch(epoch)
            val_m   = self.validate(epoch)

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

            self.ckpt.save(epoch, self.model, self.opt, self.sched,
                           self.ema, self.es, all_m, self.cfg)

            if val_m["mAP"] > self.best_metric:
                self.best_metric = val_m["mAP"]
                best_metrics = all_m.copy()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            if self.es(val_m):
                log.info(f"Early stopping at epoch {epoch}")
                break

        return best_metrics
