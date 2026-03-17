import logging
log = logging.getLogger("early_stopping")

class EarlyStopping:
    def __init__(self, patience: int = 15, metric: str = "val_mAP",
                 mode: str = "max", min_delta: float = 1e-4):
        self.patience   = patience
        self.metric     = metric
        self.mode       = mode
        self.min_delta  = min_delta
        self.best       = float("-inf") if mode == "max" else float("inf")
        self.counter    = 0
        self.should_stop = False

    def __call__(self, metrics: dict) -> bool:
        v = metrics.get(self.metric)
        if v is None:
            log.warning(f"Metric '{self.metric}' missing"); return False
        improved = (v > self.best + self.min_delta if self.mode == "max"
                    else v < self.best - self.min_delta)
        if improved:
            self.best = v; self.counter = 0
        else:
            self.counter += 1
            log.info(f"No improvement {self.counter}/{self.patience} "
                     f"({self.metric}={v:.4f}, best={self.best:.4f})")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def state_dict(self):
        return {"best": self.best, "counter": self.counter,
                "should_stop": self.should_stop}

    def load_state_dict(self, d):
        self.best = d["best"]; self.counter = d["counter"]
        self.should_stop = d["should_stop"]
