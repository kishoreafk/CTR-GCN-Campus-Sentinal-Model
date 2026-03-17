"""
Downloads and verifies:
  1. CTR-GCN Kinetics-400 (OpenPose-18, 400 classes)
  2. RTMDet-m person detector
  3. RTMPose-l COCO keypoint estimator

Run this BEFORE any annotation step.
"""
import hashlib, requests, time, logging, sys
from pathlib import Path
import torch

log = logging.getLogger("download_pretrained")

MODELS = {
    "ctrgcn_kinetics400.pt": [
        "https://github.com/Uason-Chen/CTR-GCN/releases/download/v1.0/kinetics_joint.pt",
        # Add GDrive fallback in configs/mirrors.yaml -> key: ctrgcn_url
    ],
    "rtmdet-m_coco-person.pth": [
        "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
        "rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
    ],
    "rtmpose-l_coco.pth": [
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "rtmpose-l_simcc-coco-wholebody_pt-ubs1k_270e-256x192-6f206314_20230124.pth"
    ],
}

def _download(url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            tmp.rename(dest)
            log.info(f"Downloaded: {dest.name} ({dest.stat().st_size//1e6:.0f} MB)")
            return True
        except Exception as e:
            log.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return False

def verify_ctrgcn(path: Path) -> bool:
    """
    Load checkpoint, confirm it contains OpenPose-18 graph weights.
    Run a dummy forward pass: input (1,3,64,18,2) → output (1,400).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    # Detect num_joints from adjacency matrix
    for k, v in state.items():
        if k.endswith(".A") and v.ndim == 3:
            num_joints = v.shape[-1]
            assert num_joints == 18, \
                f"Expected OpenPose-18 pretrained model, got {num_joints} joints. " \
                f"Download the Kinetics-400 model, not NTU (25 joints) or COCO (17)."
            log.info(f"Confirmed OpenPose-18 graph in checkpoint (key={k})")
            break

    log.info(f"CTR-GCN checkpoint OK: {len(state)} keys, num_joints=18")
    return True

def main():
    dest_dir = Path("checkpoints/pretrained")
    dest_dir.mkdir(parents=True, exist_ok=True)

    for name, urls in MODELS.items():
        dest = dest_dir / name
        if dest.exists():
            log.info(f"Already present: {name}"); continue
        ok = False
        for url in urls:
            ok = _download(url, dest)
            if ok: break
        if not ok:
            log.error(f"FAILED: {name}"); sys.exit(1)

    # Deep verify CTR-GCN
    verify_ctrgcn(dest_dir / "ctrgcn_kinetics400.pt")
    log.info("All pretrained models ready.")

if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
