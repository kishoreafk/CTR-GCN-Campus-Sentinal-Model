"""Post-annotation QA. Run before training to catch broken .npz files."""
import numpy as np
from pathlib import Path
from typing import Dict, List

class AnnotationQualityValidator:
    REQUIRED_SHAPE = (64, 2, 18, 3)
    MIN_QUALITY    = 0.30
    MIN_POSITIVE_CLASSES = 1

    def validate_single(self, path: str) -> Dict:
        issues = []
        try:
            d = np.load(path, allow_pickle=True)
        except Exception as e:
            return {"valid": False, "issues": [f"Load failed: {e}"]}

        kpts  = d["keypoints"]
        label = d["label"]
        qual  = float(d.get("quality_score", 0))

        if kpts.shape != self.REQUIRED_SHAPE:
            issues.append(f"keypoints shape {kpts.shape} != {self.REQUIRED_SHAPE}")
        if np.any(np.isnan(kpts)) or np.any(np.isinf(kpts)):
            issues.append("NaN/Inf in keypoints")
        if label.sum() < self.MIN_POSITIVE_CLASSES:
            issues.append("No positive classes in label vector")
        if qual < self.MIN_QUALITY:
            issues.append(f"quality_score {qual:.3f} < {self.MIN_QUALITY}")

        return {"valid": len(issues) == 0, "issues": issues}

    def validate_dir(self, annotation_dir: str) -> Dict:
        paths = list(Path(annotation_dir).rglob("*.npz"))
        valid, invalid = 0, 0
        for p in paths:
            r = self.validate_single(str(p))
            if r["valid"]:
                valid += 1
            else:
                invalid += 1
                p.rename(p.with_suffix(".bad"))  # quarantine
        return {"total": len(paths), "valid": valid, "invalid": invalid}
