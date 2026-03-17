"""
Primary metric for both phases: mean Average Precision (mAP).
This is the official AVA evaluation metric; it also applies to AVA-Kinetics
because both datasets are multi-label.
"""
import numpy as np, torch
from sklearn.metrics import average_precision_score
from typing import List, Optional, Dict

class MultiLabelMetrics:
    def __init__(self, num_classes: int, class_names: List[str]):
        self.class_names = class_names
        self.reset()

    def reset(self):
        self._logits  : List[np.ndarray] = []
        self._targets : List[np.ndarray] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        self._logits.append(probs)
        self._targets.append(targets.detach().cpu().numpy())

    def compute(self) -> Dict:
        preds   = np.concatenate(self._logits,  axis=0)
        targets = np.concatenate(self._targets, axis=0)

        ap_per_class = {}
        for c, name in enumerate(self.class_names):
            if targets[:, c].sum() == 0:
                continue    # skip classes with no positives in val set
            ap = float(average_precision_score(targets[:, c], preds[:, c]))
            ap_per_class[name] = ap

        mAP = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0

        # Per-class P/R/F1 at threshold 0.5
        binary = (preds >= 0.5).astype(int)
        per_class_prf = {}
        for c, name in enumerate(self.class_names):
            tp = int(((binary[:, c] == 1) & (targets[:, c] == 1)).sum())
            fp = int(((binary[:, c] == 1) & (targets[:, c] == 0)).sum())
            fn = int(((binary[:, c] == 0) & (targets[:, c] == 1)).sum())
            p  = tp / max(tp + fp, 1)
            r  = tp / max(tp + fn, 1)
            f1 = 2*p*r / max(p+r, 1e-8)
            per_class_prf[name] = {"AP": ap_per_class.get(name, 0.),
                                   "P": p, "R": r, "F1": f1}

        return {"mAP": mAP, "per_class": per_class_prf}
