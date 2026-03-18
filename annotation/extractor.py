"""
Extracts one training sample per (video, timestamp, person_bbox).
Complete pipeline:
  1. Read 64 frames centred on keyframe timestamp
  2. Pose estimate (GT boxes for AVA, detector for AVA-Kinetics)
  3. Person-track to maintain identity across frames
  4. COCO-17 → OpenPose-18 conversion
  5. Normalise coordinates (frame-relative [0,1])
  6. Build multi-label binary vector
  7. Compute quality score
  8. Save compressed .npz
"""
import cv2, numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from annotation.joint_converter import batch_coco17_to_openpose18
from annotation.person_tracker import PersonTracker

class SkeletonExtractor:
    def __init__(self, pose_estimator, joint_converter,
                 class_registry, config):
        self.pose  = pose_estimator
        self.jconv = joint_converter
        self.reg   = class_registry
        self.cfg   = config
        self.T     = config.num_frames      # 64
        self.M     = config.max_persons     # 2
        self.V     = config.num_joints      # 18 (OpenPose)

    def _read_frames(self, video_path: str,
                     center_ts: float) -> Tuple[np.ndarray, float, int, int]:
        """
        Returns (frames, fps, H, W).
        Reads self.T frames centred on center_ts.
        Pads with edge replication if clip boundary reached.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        center_frame = int(center_ts * fps)
        start = max(0, center_frame - self.T // 2)
        end   = start + self.T

        # Clamp and note padding needed
        read_start = max(0, start)
        read_end   = min(total, end)

        cap.set(cv2.CAP_PROP_POS_FRAMES, read_start)
        frames = []
        for _ in range(read_end - read_start):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        # Pad to exactly self.T frames
        while len(frames) < self.T:
            frames.insert(0, frames[0] if frames else np.zeros((H, W, 3), np.uint8))
        frames = frames[:self.T]
        return np.stack(frames), float(fps), H, W

    def extract_ava_sample(self,
                            video_path: str,
                            timestamp: float,
                            person_bbox: np.ndarray,  # normalised (4,)
                            action_ids: List[int]
                            ) -> Optional[dict]:
        """AVA path: uses GT bounding box, single target person."""
        frames, fps, H, W = self._read_frames(video_path, timestamp)

        # Top-down pose with GT box
        kpts17, sc17 = self.pose.process_frames_with_gt_boxes(
            frames, person_bbox, H, W)
        # kpts17: (T, 17, 2), sc17: (T, 17)

        # Expand M dim (AVA has single target person)
        kpts17 = kpts17[:, np.newaxis]   # (T, 1, 17, 2)
        sc17   = sc17[:, np.newaxis]     # (T, 1, 17)

        # Pad to max_persons
        pad_kpts = np.zeros((self.T, self.M, 17, 2), np.float32)
        pad_sc   = np.zeros((self.T, self.M, 17),    np.float32)
        pad_kpts[:, :1] = kpts17
        pad_sc  [:, :1] = sc17

        # COCO-17 → OpenPose-18
        kpts18, sc18 = batch_coco17_to_openpose18(pad_kpts, pad_sc)

        # Normalise to [0,1]
        kpts18[..., 0] /= W
        kpts18[..., 1] /= H

        # Build (T, M, V, C=3) array: [x, y, conf]
        sample_kpts = np.concatenate(
            [kpts18, sc18[..., np.newaxis]], axis=-1).astype(np.float32)

        label = self.reg.get_multilabel_vector(action_ids)
        quality = self._quality_score(sc18)

        return dict(
            keypoints    = sample_kpts,        # (64, 2, 18, 3)
            label        = label,              # (num_classes,)
            video_id     = Path(video_path).stem,
            timestamp    = float(timestamp),
            person_bbox  = person_bbox,
            action_ids   = action_ids,
            quality_score= float(quality),
            joint_layout = "openpose_18",
        )

    def _quality_score(self, scores: np.ndarray) -> float:
        """
        Aggregate quality in [0, 1]:
        - Mean confidence of visible joints
        - Penalise frames where < 50% of joints are above 0.3 threshold
        """
        mean_conf  = float(np.mean(scores))
        frac_good  = float(np.mean(np.mean(scores > 0.3, axis=-1) >= 0.5))
        return 0.5 * mean_conf + 0.5 * frac_good

    def save_sample(self, sample: dict, output_path: str):
        """
        Write .npz atomically:
          1. Write to output_path.tmp.npz
          2. Validate the written file
          3. Rename to output_path (atomic on POSIX)

        If the process is killed between steps 1 and 3,
        a .tmp.npz file is left behind — recovered at next startup.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path + ".tmp.npz"
        try:
            np.savez_compressed(tmp_path, **sample)

            # Validate before finalising
            test = np.load(tmp_path, allow_pickle=True)
            kpts = test["keypoints"]
            expected = (self.T, self.M, self.V, 3)
            assert kpts.shape == expected, \
                f"Bad keypoints shape: {kpts.shape}, expected {expected}"
            assert not np.any(np.isnan(kpts)), "NaN in keypoints"

            # Atomic rename
            Path(tmp_path).rename(output_path)

        except Exception as e:
            Path(tmp_path).unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save sample: {e}")
