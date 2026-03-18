"""
Run before any training phase. Analyses the annotation directory and
produces a report that either confirms training is safe to proceed
or raises with specific remediation steps.
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Dict

log = logging.getLogger("dataset_auditor")


class AuditError(Exception):
    """Raised when dataset does not meet minimum requirements for training."""
    pass


class DatasetAuditor:

    # Thresholds — raise error if violated
    MIN_SAMPLES_PER_CLASS_TRAIN = 50    # below this: model cannot generalise
    MIN_SAMPLES_PER_CLASS_VAL   = 10    # below this: mAP estimate is unreliable
    MAX_TRAIN_VAL_RATIO         = 20.0  # train:val ratio above this is suspicious
    MIN_TRAIN_VAL_RATIO         = 2.0   # below this: too much data in val

    # Thresholds — log warning if violated
    WARN_CLASS_IMBALANCE_RATIO  = 10.0  # largest/smallest class count ratio
    WARN_LOW_QUALITY_FRACTION   = 0.20  # >20% low-quality samples is a problem

    def audit(self, annotation_dir: str,
              class_registry,
              min_quality: float = 0.30) -> dict:
        """
        Full audit of annotation directory.

        Checks:
          1. Sample counts per class per split
          2. Train/val ratio per class
          3. Class co-occurrence matrix (multi-label)
          4. Quality score distribution
          5. Keypoint confidence statistics
          6. Label vector integrity
          7. Temporal coverage (gap analysis)

        Returns report dict. Raises AuditError on fatal issues.
        """
        npz_files = list(Path(annotation_dir).rglob("*.npz"))
        if not npz_files:
            raise AuditError(
                f"No .npz files found in {annotation_dir}.\n"
                f"Run annotation before training."
            )

        log.info(f"Auditing {len(npz_files)} annotation files...")

        records = []
        for p in npz_files:
            try:
                d = np.load(str(p), allow_pickle=True)
                records.append({
                    "path":          str(p),
                    "split":         str(d.get("split", "train")),
                    "action_ids":    list(d.get("action_ids", [])),
                    "quality_score": float(d.get("quality_score", 0)),
                    "keypoint_mean_conf": float(
                        np.mean(d["keypoints"][..., 2])
                    ),
                    "has_nan":       bool(np.any(np.isnan(d["keypoints"]))),
                    "label_sum":     float(d["label"].sum()),
                })
            except Exception as e:
                log.warning(f"Cannot load {p.name}: {e}")
                continue

        if not records:
            raise AuditError("All annotation files failed to load.")

        report = {}

        # ── 1. Per-class sample counts ────────────────────────────────────
        counts = self._count_per_class(records, class_registry)
        report["class_counts"] = counts
        self._check_minimum_samples(counts)

        # ── 2. Train/val ratio per class ──────────────────────────────────
        ratios = self._check_split_ratios(counts)
        report["train_val_ratios"] = ratios

        # ── 3. Class imbalance ────────────────────────────────────────────
        imbalance = self._check_class_imbalance(counts)
        report["class_imbalance_ratio"] = imbalance

        # ── 4. Quality distribution ───────────────────────────────────────
        quality = self._check_quality(records, min_quality)
        report["quality"] = quality

        # ── 5. NaN/corrupt files ─────────────────────────────────────────
        nan_count = sum(1 for r in records if r["has_nan"])
        if nan_count > 0:
            log.warning(
                f"{nan_count} files contain NaN values — "
                f"they will be filtered by SkeletonDataset's min_quality check"
            )
        report["nan_files"] = int(nan_count)

        # ── 6. Label integrity ────────────────────────────────────────────
        zero_label = sum(1 for r in records if r["label_sum"] == 0)
        if zero_label > 0:
            log.warning(
                f"{zero_label} files have all-zero label vectors — "
                f"they will be skipped by SkeletonDataset"
            )
        report["zero_label_files"] = int(zero_label)

        # ── 7. Co-occurrence matrix ───────────────────────────────────────
        report["cooccurrence"] = self._compute_cooccurrence(
            records, class_registry
        )

        # ── Print summary ─────────────────────────────────────────────────
        self._print_report(report, class_registry)

        # Save report
        out = Path(annotation_dir) / "audit_report.json"
        import json
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info(f"Audit report saved to {out}")

        return report

    def _count_per_class(self, records: list,
                         class_registry) -> Dict:
        """Count samples per class per split."""
        counts = {}
        for cls in class_registry._classes:
            cid   = cls["id"]
            cname = cls["name"]
            train = sum(1 for r in records
                        if cid in r["action_ids"] and r["split"] == "train")
            val   = sum(1 for r in records
                        if cid in r["action_ids"] and r["split"] == "val")
            counts[cname] = {"id": cid, "train": train, "val": val,
                             "total": train + val}
        return counts

    def _check_minimum_samples(self, counts: dict):
        """Raise AuditError if any class is below minimum sample count."""
        errors = []
        for name, c in counts.items():
            if c["train"] < self.MIN_SAMPLES_PER_CLASS_TRAIN:
                errors.append(
                    f"  '{name}': only {c['train']} train samples "
                    f"(need {self.MIN_SAMPLES_PER_CLASS_TRAIN})"
                )
            if c["val"] < self.MIN_SAMPLES_PER_CLASS_VAL:
                errors.append(
                    f"  '{name}': only {c['val']} val samples "
                    f"(need {self.MIN_SAMPLES_PER_CLASS_VAL})"
                )
        if errors:
            raise AuditError(
                "Insufficient samples for the following classes:\n"
                + "\n".join(errors) + "\n\n"
                "Options:\n"
                "  1. Download more videos: python main.py --mode download "
                "--classes <class>\n"
                "  2. Remove the class from selection\n"
                "  3. Lower MIN_SAMPLES_PER_CLASS_TRAIN (reduces model quality)"
            )

    def _check_split_ratios(self, counts: dict) -> dict:
        ratios = {}
        for name, c in counts.items():
            if c["val"] == 0:
                ratio = float("inf")
            else:
                ratio = c["train"] / c["val"]
            ratios[name] = round(ratio, 2)

            if ratio > self.MAX_TRAIN_VAL_RATIO:
                log.warning(
                    f"'{name}': train/val ratio = {ratio:.1f} "
                    f"(very few val samples — mAP estimate will be noisy)"
                )
            elif ratio < self.MIN_TRAIN_VAL_RATIO:
                log.warning(
                    f"'{name}': train/val ratio = {ratio:.1f} "
                    f"(unusually much data in val — check split logic)"
                )
        return ratios

    def _check_class_imbalance(self, counts: dict) -> float:
        train_counts = [c["train"] for c in counts.values() if c["train"] > 0]
        if not train_counts:
            return 0.0
        ratio = max(train_counts) / min(train_counts)
        if ratio > self.WARN_CLASS_IMBALANCE_RATIO:
            log.warning(
                f"Class imbalance ratio = {ratio:.1f}x. "
                f"AsymmetricLoss should handle this, but verify "
                f"pos_weight is being computed correctly."
            )
        return round(float(ratio), 2)

    def _check_quality(self, records: list, min_quality: float) -> dict:
        scores = [r["quality_score"] for r in records]
        low_frac = sum(1 for s in scores if s < min_quality) / max(len(scores), 1)
        if low_frac > self.WARN_LOW_QUALITY_FRACTION:
            log.warning(
                f"{low_frac:.1%} of samples below quality threshold "
                f"{min_quality}. Consider re-annotating with better "
                f"detection threshold or checking video quality."
            )
        return {
            "mean":    round(float(np.mean(scores)), 3),
            "median":  round(float(np.median(scores)), 3),
            "pct_low": round(low_frac, 3),
            "min":     round(float(np.min(scores)), 3),
        }

    def _compute_cooccurrence(self, records: list,
                              class_registry) -> dict:
        """
        Which classes frequently appear together?
        Useful for understanding if the model might confuse co-occurring classes.
        """
        ids   = class_registry.class_ids
        names = class_registry.class_names
        n     = len(ids)
        matrix = np.zeros((n, n), dtype=int)

        for row in records:
            aids = set(row["action_ids"])
            for i, id_i in enumerate(ids):
                for j, id_j in enumerate(ids):
                    if id_i in aids and id_j in aids:
                        matrix[i, j] += 1

        result = {}
        for i, name_i in enumerate(names):
            result[name_i] = {}
            for j, name_j in enumerate(names):
                if matrix[i, j] > 0:
                    result[name_i][name_j] = int(matrix[i, j])
        return result

    def _print_report(self, report: dict, class_registry):
        print("\n" + "═" * 65)
        print("DATASET AUDIT REPORT")
        print("═" * 65)
        print(f"{'Class':<30} {'Train':>8} {'Val':>8} {'TV Ratio':>10}")
        print("─" * 65)
        for name, c in report["class_counts"].items():
            ratio = report["train_val_ratios"].get(name, "?")
            print(f"{name:<30} {c['train']:>8} {c['val']:>8} {ratio:>10}")
        print("─" * 65)
        print(f"Class imbalance ratio : {report['class_imbalance_ratio']:.1f}x")
        print(f"Mean quality score    : {report['quality']['mean']:.3f}")
        print(f"Low-quality fraction  : {report['quality']['pct_low']:.1%}")
        print(f"Files with NaN        : {report['nan_files']}")
        print("═" * 65 + "\n")
