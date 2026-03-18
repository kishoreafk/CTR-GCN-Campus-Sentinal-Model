"""
Run the trained CTR-GCN AVA model on a new video.

Pipeline:
  1. Load video
  2. Detect persons + estimate poses (RTMDet + RTMPose)
  3. Convert COCO-17 → OpenPose-18
  4. Build sliding window skeleton sequences (64 frames, stride 8)
  5. Run model (with TTA optionally)
  6. Aggregate predictions per person per window
  7. Output: per-second action probabilities per person

Usage:
  python main.py --mode infer --input_video path/to/video.mp4 \\
                 --checkpoint checkpoints/runs/phase2_best.pth \\
                 --classes eat drink walk \\
                 --output_json results.json
"""

import torch
import numpy as np
import json
import cv2
import logging
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("inference")


class VideoInference:

    def __init__(self, checkpoint_path: str,
                 class_registry,
                 config,
                 use_tta: bool = False,
                 stride: int = 8,
                 threshold: float = 0.3):
        self.reg       = class_registry
        self.device    = config.device
        self.stride    = stride
        self.threshold = threshold
        self.T         = 64      # frames per window

        # Load model from checkpoint
        self.model = self._load_model(checkpoint_path, config)

        # Load pose estimator
        from annotation.pose_estimator import PoseEstimator
        from annotation.joint_converter import batch_coco17_to_openpose18
        self.pose_est  = PoseEstimator(config)
        self.jconv     = batch_coco17_to_openpose18

        if use_tta:
            from evaluation.tta import TTAEvaluator
            self.predictor = TTAEvaluator(self.model, self.device).predict
        else:
            self.predictor = self._simple_predict

    def _load_model(self, ckpt_path: str, config):
        from models.model_factory import build_model

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Update config from checkpoint metadata
        saved_cfg = ckpt.get("config", {})
        num_classes = ckpt.get("metadata", {}).get("num_classes")
        if num_classes:
            config.num_classes = num_classes
        elif "num_classes" in saved_cfg:
            config.num_classes = saved_cfg["num_classes"]

        # Don't torch.compile for inference
        config.use_compile = False

        model = build_model(config)

        # Load weights — prefer EMA, fall back to model state
        state = ckpt.get("ema")
        if state and isinstance(state, dict) and "ema" in state:
            # EMA state_dict contains 'ema' key with model parameters
            from training.ema import ModelEMA
            ema = ModelEMA(model, decay=0.999)
            ema.load_state_dict(state)
            # Use EMA model for inference
            target = model._orig_mod if hasattr(model, '_orig_mod') else model
            target.load_state_dict(ema.ema.state_dict(), strict=False)
        else:
            state = ckpt.get("model")
            if state:
                target = model._orig_mod if hasattr(model, '_orig_mod') else model
                target.load_state_dict(state, strict=False)

        model.eval()
        return model

    @torch.no_grad()
    def _simple_predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = self.model(x.to(self.device))
        return torch.sigmoid(logits)

    def run(self, video_path: str,
            output_path: Optional[str] = None) -> List[dict]:
        """
        Run inference on a full video.

        Returns list of result dicts per window:
        [
          {
            "center_time_s":  float,
            "person_id":      int,
            "predictions": {
              "eat":   0.82,
              "drink": 0.04,
              ...
            },
            "active_actions": ["eat"]   # predictions above threshold
          },
          ...
        ]
        """
        frames, fps = self._load_video(video_path)
        T, H, W = len(frames), frames.shape[1], frames.shape[2]

        if T < self.T:
            log.warning(
                f"Video has only {T} frames (need {self.T}). "
                f"Padding with last frame."
            )
            pad = np.stack([frames[-1]] * (self.T - T))
            frames = np.concatenate([frames, pad], axis=0)
            T = len(frames)

        max_persons = 2

        # Run pose estimation + tracking on all frames
        log.info(f"Running pose estimation on {T} frames...")
        all_kpts17, all_scores17 = self.pose_est.process_frames_auto_detect(
            frames, max_persons=max_persons
        )
        # all_kpts17: (T, M, 17, 2), all_scores17: (T, M, 17)

        # Normalise coordinates
        all_kpts17[..., 0] /= W
        all_kpts17[..., 1] /= H

        # Convert to OpenPose-18
        kpts18, sc18 = self.jconv(all_kpts17, all_scores17)
        # (T, M, 18, 2), (T, M, 18)

        # Build channel-3 array: [x, y, conf]
        skeleton = np.concatenate(
            [kpts18, sc18[..., np.newaxis]], axis=-1
        ).astype(np.float32)
        # (T, M, 18, 3)

        # Sliding window inference
        results = []
        for start in range(0, T - self.T + 1, self.stride):
            end     = start + self.T
            window  = skeleton[start:end]   # (64, M, 18, 3)
            center  = (start + end) / 2 / fps

            # (C, T, V, M) format
            x = torch.from_numpy(
                window.transpose(3, 0, 2, 1)
            ).unsqueeze(0)  # (1, 3, 64, 18, M)

            probs = self.predictor(x)[0].cpu().numpy()

            for person_id in range(max_persons):
                # Check if this person is actually present
                # (non-zero confidence in at least half the frames)
                person_conf = skeleton[start:end, person_id, :, 2].mean()
                if person_conf < 0.1:
                    continue

                pred_dict = {
                    name: round(float(probs[i]), 4)
                    for i, name in enumerate(self.reg.class_names)
                }
                active = [name for name, p in pred_dict.items()
                          if p >= self.threshold]

                results.append({
                    "center_time_s":  round(center, 2),
                    "person_id":      person_id,
                    "predictions":    pred_dict,
                    "active_actions": active,
                })

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            log.info(f"Results saved to {output_path}")

        log.info(f"Inference complete: {len(results)} prediction windows")
        return results

    def _load_video(self, path: str):
        cap    = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError(f"No frames read from video: {path}")
        return np.stack(frames), fps
