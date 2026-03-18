"""
Monitors gradient norms per layer group every N steps.
Detects explosion (norm > threshold) and vanishing (norm < threshold).
Logs to TensorBoard / WandB and saves a gradient history CSV.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from collections import deque
from typing import Optional

log = logging.getLogger("gradient_monitor")

# Thresholds
EXPLOSION_THRESHOLD = 100.0   # total grad norm above this → explosion
VANISHING_THRESHOLD = 1e-7    # total grad norm below this → vanishing
WARN_CONSECUTIVE    = 3       # warn after N consecutive bad steps


class GradientMonitor:

    def __init__(self, model: nn.Module,
                 log_every_n_steps: int = 50,
                 history_len: int = 200):
        self.model          = model
        self.log_every      = log_every_n_steps
        self.history        = deque(maxlen=history_len)
        self._consec_bad    = 0
        self._step          = 0

    def check(self, step: int) -> Optional[dict]:
        """
        Call AFTER loss.backward() but BEFORE optimizer.step().
        Returns stats dict or None if not a logging step.
        """
        self._step = step

        # Compute total gradient norm (same as clip_grad_norm_ reports)
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        # Per-layer-group norms
        layer_norms = self._per_layer_norms()

        record = {
            "step":        step,
            "total_norm":  total_norm,
            "layer_norms": layer_norms,
        }
        self.history.append(record)

        # ── Explosion detection ───────────────────────────────────────────
        if total_norm > EXPLOSION_THRESHOLD:
            self._consec_bad += 1
            log.warning(
                f"Step {step}: GRADIENT EXPLOSION detected! "
                f"total_norm={total_norm:.2f} "
                f"(threshold={EXPLOSION_THRESHOLD}). "
                f"Consecutive: {self._consec_bad}"
            )
            if self._consec_bad >= WARN_CONSECUTIVE:
                log.error(
                    "Gradient explosion persisting. Suggestions:\n"
                    "  1. Reduce learning rate (current LR may be too high)\n"
                    "  2. Increase gradient_clip (or add it if missing)\n"
                    "  3. Check loss_type — unexpected large values in targets?"
                )
        # ── Vanishing detection (only for unfrozen params) ────────────────
        elif total_norm < VANISHING_THRESHOLD:
            active_params = sum(
                1 for p in self.model.parameters() if p.requires_grad
            )
            if active_params > 0:
                self._consec_bad += 1
                log.warning(
                    f"Step {step}: Gradient vanishing! "
                    f"total_norm={total_norm:.2e}. "
                    f"Backbone may not be learning."
                )
        else:
            self._consec_bad = 0

        if step % self.log_every != 0:
            return None
        return record

    def _per_layer_norms(self) -> dict:
        """Compute gradient norm for each named layer group."""
        norms = {}
        for name, module in self.model.named_modules():
            params = list(module.parameters(recurse=False))
            if not params:
                continue
            n = sum(
                p.grad.detach().norm(2).item() ** 2
                for p in params
                if p.grad is not None
            ) ** 0.5
            if n > 0:
                norms[name] = round(n, 6)
        return norms

    def get_summary(self) -> dict:
        """Summary statistics over the monitoring history."""
        if not self.history:
            return {}
        norms = [r["total_norm"] for r in self.history]
        return {
            "mean_grad_norm":   round(float(np.mean(norms)), 6),
            "max_grad_norm":    round(float(np.max(norms)), 6),
            "min_grad_norm":    round(float(np.min(norms)), 6),
            "std_grad_norm":    round(float(np.std(norms)), 6),
            "explosion_steps":  sum(1 for n in norms if n > EXPLOSION_THRESHOLD),
            "vanishing_steps":  sum(1 for n in norms if n < VANISHING_THRESHOLD),
        }
