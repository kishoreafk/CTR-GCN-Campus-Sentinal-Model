"""
LR Finder: sweeps learning rate to find optimal range.
Reference: 'Cyclical Learning Rates for Training Neural Networks' (Smith, 2017)
"""
import torch
import numpy as np
import logging
from pathlib import Path

log = logging.getLogger("lr_finder")


class LRFinder:
    """
    Linear LR sweep from start_lr to end_lr over num_steps,
    recording loss at each step.
    """
    def __init__(self, model, optimizer, loss_fn, device="cuda",
                 start_lr=1e-7, end_lr=1.0, num_steps=200):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_steps = num_steps

    def run(self, train_loader) -> dict:
        """Run the LR range test. Returns dict with lrs and losses."""
        # Save initial state
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        initial_opt_state = self.optimizer.state_dict()

        lrs, losses = [], []
        lr_mult = (self.end_lr / self.start_lr) ** (1 / self.num_steps)

        # Set initial LR
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.start_lr

        self.model.train()
        step = 0
        best_loss = float("inf")

        for batch in train_loader:
            if step >= self.num_steps:
                break

            x = batch["input"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()

            lr = self.optimizer.param_groups[0]["lr"]
            lrs.append(lr)
            losses.append(loss.item())

            # Stop if loss diverges
            if loss.item() > best_loss * 10:
                break
            best_loss = min(best_loss, loss.item())

            # Update LR
            for pg in self.optimizer.param_groups:
                pg["lr"] *= lr_mult

            step += 1

        # Restore initial state
        self.model.load_state_dict(initial_state)
        self.optimizer.load_state_dict(initial_opt_state)

        result = {"lrs": lrs, "losses": losses}

        # Find suggested LR (steepest negative gradient)
        if len(losses) > 10:
            grads = np.gradient(losses)
            suggested_idx = int(np.argmin(grads))
            result["suggested_lr"] = lrs[suggested_idx]
            log.info(f"Suggested LR: {result['suggested_lr']:.2e}")

        return result

    def plot(self, result: dict, output_path: str = "outputs/lr_finder.png"):
        """Save LR finder plot."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(result["lrs"], result["losses"])
            ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Loss")
            ax.set_title("LR Finder")
            if "suggested_lr" in result:
                ax.axvline(x=result["suggested_lr"], color="r",
                          linestyle="--", label=f"Suggested: {result['suggested_lr']:.2e}")
                ax.legend()

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(f"LR finder plot saved to {output_path}")
        except ImportError:
            log.warning("matplotlib not available, skipping plot")
