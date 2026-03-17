"""Skeleton visualization utilities for debugging annotation quality."""
import numpy as np
import logging
from pathlib import Path

log = logging.getLogger("visualize_skeleton")

# OpenPose-18 bones for drawing
BONES_18 = [
    (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
    (0,14),(14,16),(0,15),(15,17)
]

JOINT_NAMES_18 = [
    "nose","neck","r_shoulder","r_elbow","r_wrist",
    "l_shoulder","l_elbow","l_wrist",
    "r_hip","r_knee","r_ankle",
    "l_hip","l_knee","l_ankle",
    "r_eye","l_eye","r_ear","l_ear"
]


def visualize_skeleton_on_frame(frame: np.ndarray,
                                 keypoints: np.ndarray,
                                 scores: np.ndarray,
                                 conf_thr: float = 0.3,
                                 output_path: str = None) -> np.ndarray:
    """
    Draw OpenPose-18 skeleton on a frame.

    Args:
        frame: (H, W, 3) RGB image
        keypoints: (18, 2) pixel coordinates
        scores: (18,) confidence scores
        conf_thr: minimum confidence to draw a joint
        output_path: optional path to save the visualization

    Returns:
        frame with skeleton drawn on it
    """
    try:
        import cv2
    except ImportError:
        log.warning("cv2 not available, skipping visualization")
        return frame

    vis = frame.copy()
    H, W = vis.shape[:2]

    # Draw bones
    for i, j in BONES_18:
        if scores[i] > conf_thr and scores[j] > conf_thr:
            pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

    # Draw joints
    for k in range(18):
        if scores[k] > conf_thr:
            pt = (int(keypoints[k, 0]), int(keypoints[k, 1]))
            cv2.circle(vis, pt, 4, (255, 0, 0), -1)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        log.info(f"Saved skeleton visualization to {output_path}")

    return vis


def visualize_npz_sample(npz_path: str, output_dir: str = "outputs/vis",
                          frame_indices: list = None):
    """
    Visualize skeleton data from an .npz file.
    Since we don't have the original video frames, draws on a blank canvas.

    Args:
        npz_path: path to .npz skeleton file
        output_dir: directory to save visualizations
        frame_indices: specific frame indices to visualize (default: [0, 16, 32, 48])
    """
    try:
        import cv2
    except ImportError:
        log.warning("cv2 not available, skipping visualization")
        return

    data = np.load(npz_path, allow_pickle=True)
    kpts = data["keypoints"]  # (T, M, V, C) where C=3 (x, y, conf)

    if frame_indices is None:
        frame_indices = [0, 16, 32, 48]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas_size = 512
    for t in frame_indices:
        if t >= kpts.shape[0]:
            continue
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

        for m in range(kpts.shape[1]):
            person_kpts = kpts[t, m, :, :2] * canvas_size  # denormalize
            person_conf = kpts[t, m, :, 2]
            canvas = visualize_skeleton_on_frame(
                canvas, person_kpts, person_conf, conf_thr=0.1)

        stem = Path(npz_path).stem
        out_path = str(out_dir / f"{stem}_frame{t:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    log.info(f"Saved {len(frame_indices)} frames to {output_dir}")
