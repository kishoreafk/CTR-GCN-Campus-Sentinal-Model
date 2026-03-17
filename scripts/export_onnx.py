"""
Export trained CTR-GCN model to ONNX format for deployment.
"""
import torch
import logging
from pathlib import Path

log = logging.getLogger("export_onnx")


def export(config, checkpoint_path: str = None):
    """Export model to ONNX."""
    from models.ctrgcn_ava import CTRGCNForAVA
    from models.ctrgcn.graph import OpenPoseGraph

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

    # Build model (without compile)
    graph = OpenPoseGraph(strategy="spatial")
    model = CTRGCNForAVA(num_classes=num_classes, dropout=0.0)

    # Load weights (prefer EMA)
    state = ckpt.get("ema_state", ckpt.get("model_state"))
    model.load_state_dict(state, strict=False)
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, 64, 18, 2)

    # Export
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "ctrgcn_ava.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["skeleton_input"],
        output_names=["action_logits"],
        opset_version=17,
        dynamic_axes={
            "skeleton_input": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
    )

    log.info(f"ONNX model exported to {onnx_path}")
    log.info(f"Model size: {onnx_path.stat().st_size / 1e6:.1f} MB")

    # Verify
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(onnx_path))
        out = sess.run(None, {"skeleton_input": dummy.numpy()})
        log.info(f"ONNX verification OK — output shape: {out[0].shape}")
    except ImportError:
        log.warning("onnxruntime not installed — skipping verification")


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    from utils.config_loader import load_config
    cfg = load_config("configs/base_config.yaml")
    export(cfg)
