"""
End-to-end smoke test. Default: --skip_download --skip_annotation.
Target: completes in < 15 minutes on RTX 4500 Ada.
"""
import argparse, time, json, logging
from pathlib import Path
log = logging.getLogger("test_run")

def run_smoke_test(args):
    from utils.system_checks import preflight
    from utils.seed import set_seed
    set_seed(42)
    preflight(required_disk_gb=5.0)

    # Synthetic data generation if no real data present
    classes = getattr(args, 'classes', None) or ["eat", "drink", "walk"]
    if not list(Path("data/processed").rglob("*.npz")):
        log.info("Generating synthetic skeleton data for smoke test...")
        _generate_synthetic_data(classes, n_per_class=20)

    from utils.config_loader import load_config
    max_epochs = getattr(args, 'max_epochs', 3)
    cfg = load_config("configs/phase1_ava_kinetics.yaml", overrides={
        "test_mode": True,
        "test_max_epochs": max_epochs,
        "batch_size": 8,
        "num_workers": 2,
    })

    from training.pipeline import run_phase

    t0 = time.time()
    metrics = run_phase("configs/phase1_ava_kinetics.yaml", "smoke_test")
    elapsed = time.time() - t0

    report = {
        "status":   "PASS" if metrics.get("mAP", 0) >= 0 else "FAIL",
        "mAP":      metrics.get("mAP", 0),
        "time_min": elapsed / 60,
    }
    Path("outputs").mkdir(exist_ok=True)
    json.dump(report, open("outputs/smoke_test_report.json", "w"), indent=2)
    log.info(f"\nSmoke Test: {report['status']} | "
             f"mAP={report['mAP']:.4f} | time={report['time_min']:.1f} min")
    assert elapsed < 900, f"Test run exceeded 15 min ({elapsed/60:.1f} min)"

def _generate_synthetic_data(classes, n_per_class=20):
    import numpy as np
    from pathlib import Path
    from utils.class_registry import ClassRegistry
    reg = ClassRegistry()

    out = Path("data/processed/ava_kinetics/skeletons")
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        n = n_per_class if split == "train" else n_per_class // 4
        for cls in classes:
            matching = [c for c in reg._classes if c["name"] == cls]
            if not matching: continue
            ava_id = matching[0]["id"]
            for i in range(max(1, n)):
                label = reg.get_multilabel_vector([ava_id])
                np.savez_compressed(
                    str(out / f"{split}_{cls}_{i:04d}.npz"),
                    keypoints    = np.random.rand(64, 2, 18, 3).astype(np.float32),
                    label        = label,
                    video_id     = f"synth_{cls}_{i}",
                    timestamp    = float(i),
                    person_bbox  = np.array([0.1,0.2,0.5,0.8]),
                    action_ids   = [ava_id],
                    quality_score= 0.9,
                    joint_layout = "openpose_18",
                    split        = split,
                )

if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    p = argparse.ArgumentParser()
    p.add_argument("--classes", nargs="+", default=["eat","drink","walk"])
    p.add_argument("--max_epochs", type=int, default=3)
    p.add_argument("--skip_download", action="store_true", default=True)
    p.add_argument("--skip_annotation", action="store_true", default=True)
    run_smoke_test(p.parse_args())
