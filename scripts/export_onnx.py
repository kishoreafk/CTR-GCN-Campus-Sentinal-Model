"""
Export trained CTR-GCN model to ONNX and TorchScript formats for deployment.
ONNX: works with TensorRT, OpenVINO, ONNX Runtime.
TorchScript: works in C++ / mobile without Python.
Also writes a metadata sidecar JSON.
"""
import torch
import json
import logging
from pathlib import Path

log = logging.getLogger("export_onnx")


def export(config, checkpoint_path: str = None):
    """Export model to ONNX + TorchScript + metadata sidecar."""
    from models.ctrgcn_ava import CTRGCNForAVA
    from utils.class_registry import ClassRegistry

    # Resolve class information
    registry = ClassRegistry(config.class_config)

    # Find checkpoint
    if checkpoint_path is None:
        runs_dir = Path(config.checkpoint_dir) / "runs"
        best_paths = list(runs_dir.rglob("best.pth"))
        if not best_paths:
            log.error("No best.pth found")
            return
        checkpoint_path = str(best_paths[-1])

    log.info(f"Exporting from: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    num_classes = ckpt.get("num_classes", config.num_classes)
    # Try metadata key
    meta = ckpt.get("metadata", {})
    if meta.get("num_classes"):
        num_classes = meta["num_classes"]

    # Build model (without compile)
    model = CTRGCNForAVA(num_classes=num_classes, dropout=0.0)

    # Load weights (prefer EMA)
    state = ckpt.get("ema_state", ckpt.get("ema", ckpt.get("model_state", ckpt.get("model"))))
    if isinstance(state, dict) and "ema" in state:
        # EMA wrapper state — extract inner model state
        from training.ema import ModelEMA
        ema = ModelEMA(model, decay=0.999)
        ema.load_state_dict(state)
        state = ema.ema.state_dict()
    if state:
        model.load_state_dict(state, strict=False)
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, 64, 18, 2)

    # Output directory
    output_dir = Path(config.output_dir) / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── TorchScript ───────────────────────────────────────────────────────
    try:
        traced_path = output_dir / "ctrgcn_ava.pt"
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy)
        traced.save(str(traced_path))
        log.info(f"TorchScript exported: {traced_path}")
        log.info(f"TorchScript size: {traced_path.stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        log.warning(f"TorchScript export failed: {e}")

    # ── ONNX ──────────────────────────────────────────────────────────────
    onnx_path = output_dir / "ctrgcn_ava.onnx"
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["skeleton_input"],
            output_names=["action_logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={
                "skeleton_input": {0: "batch_size"},
                "action_logits": {0: "batch_size"},
            },
        )

        log.info(f"ONNX model exported to {onnx_path}")
        log.info(f"ONNX size: {onnx_path.stat().st_size / 1e6:.1f} MB")

        # Verify
        try:
            import onnx, onnxruntime as ort
            onnx.checker.check_model(str(onnx_path))
            sess = ort.InferenceSession(str(onnx_path))
            out = sess.run(None, {"skeleton_input": dummy.numpy()})
            log.info(f"ONNX verification OK — output shape: {out[0].shape}")
        except ImportError:
            log.warning("onnxruntime not installed — skipping ONNX verification")

    except Exception as e:
        log.warning(f"ONNX export failed: {e}")

    # ── Metadata sidecar ──────────────────────────────────────────────────
    class_names = meta.get("class_names") or registry.class_names
    class_ids = registry.class_ids
    metadata = {
        "class_names":    class_names,
        "class_ids":      class_ids,
        "num_classes":    num_classes,
        "input_shape":    [1, 3, 64, 18, 2],
        "joint_layout":   "openpose_18",
        "output":         "raw logits (apply sigmoid for probabilities)",
        "threshold":      0.3,
        "source_checkpoint": str(checkpoint_path),
    }
    meta_path = output_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Metadata sidecar: {meta_path}")

    log.info("Export complete.")


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    from utils.config_loader import load_config
    cfg = load_config("configs/base_config.yaml")
    export(cfg)
