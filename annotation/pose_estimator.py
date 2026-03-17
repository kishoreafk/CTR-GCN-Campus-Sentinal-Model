"""
Two-stage top-down pose estimation using RTMDet + RTMPose (MMPose).
RTMPose outputs COCO-17 keypoints; joint_converter converts to OpenPose-18.

AVA mode  : skip detector, use ground-truth bounding boxes from annotation CSV.
AVA-Kinetics mode: run RTMDet detector per frame, then top-down RTMPose.
"""
import numpy as np
from typing import Tuple
import logging

log = logging.getLogger("pose_estimator")


class PoseEstimator:
    def __init__(self, config):
        self.cfg = config
        self.inferencer = None  # Lazy init to avoid import errors without mmpose

    def _init_inferencer(self):
        """Lazy initialization of MMPose inferencer."""
        if self.inferencer is not None:
            return
        try:
            from mmpose.apis import MMPoseInferencer
            det_model = self.cfg.detector_ckpt if not getattr(
                self.cfg, 'use_gt_boxes_for_ava', False) else None
            self.inferencer = MMPoseInferencer(
                pose2d=self.cfg.pose_ckpt,
                det_model=det_model,
                device=self.cfg.device,
            )
            log.info("MMPose inferencer initialized")
        except ImportError:
            log.warning("MMPose not available. Pose estimation will use dummy data.")

    def process_frames_with_gt_boxes(
        self,
        frames: np.ndarray,      # (T, H, W, 3)
        gt_bbox: np.ndarray,     # (4,) normalized [x1,y1,x2,y2] at keyframe
        frame_h: int, frame_w: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        AVA path: use the AVA ground-truth bounding box for top-down pose.
        The GT box is given at the keyframe; we replicate it across frames
        (tracker refines it — see extractor.py for full pipeline).

        Returns:
            keypoints_17  : (T, 17, 2)  pixel coords
            scores_17     : (T, 17)
        """
        self._init_inferencer()

        # Denormalize box
        box_px = np.array([
            gt_bbox[0]*frame_w, gt_bbox[1]*frame_h,
            gt_bbox[2]*frame_w, gt_bbox[3]*frame_h])

        all_kpts, all_scores = [], []
        for frame in frames:
            if self.inferencer is None:
                # Fallback: generate dummy data if mmpose not available
                all_kpts.append(np.random.rand(17, 2).astype(np.float32) *
                               np.array([frame_w, frame_h]))
                all_scores.append(np.random.rand(17).astype(np.float32))
                continue

            result = self.inferencer(
                frame,
                bboxes=[box_px.tolist()],
                bbox_format="xyxy",
                return_vis=False,
            )
            pred = result["predictions"][0]
            if pred:
                all_kpts.append(np.array(pred[0]["keypoints"], dtype=np.float32))
                all_scores.append(np.array(pred[0]["keypoint_scores"], dtype=np.float32))
            else:
                all_kpts.append(np.zeros((17, 2), dtype=np.float32))
                all_scores.append(np.zeros(17, dtype=np.float32))

        return np.stack(all_kpts), np.stack(all_scores)

    def process_frames_auto_detect(
        self,
        frames: np.ndarray,      # (T, H, W, 3)
        max_persons: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        AVA-Kinetics path: run RTMDet on every frame, then RTMPose top-down.
        Returns:
            keypoints_17  : (T, M, 17, 2)
            scores_17     : (T, M, 17)
        """
        self._init_inferencer()

        T = len(frames)
        H, W = frames[0].shape[:2]
        all_kpts   = np.zeros((T, max_persons, 17, 2), dtype=np.float32)
        all_scores = np.zeros((T, max_persons, 17),    dtype=np.float32)

        for t, frame in enumerate(frames):
            if self.inferencer is None:
                # Fallback: generate dummy data
                for m in range(max_persons):
                    all_kpts[t, m] = np.random.rand(17, 2).astype(np.float32) * \
                                     np.array([W, H])
                    all_scores[t, m] = np.random.rand(17).astype(np.float32)
                continue

            result = self.inferencer(frame, return_vis=False)
            preds = result["predictions"][0][:max_persons]
            for m, pred in enumerate(preds):
                all_kpts[t, m]   = np.array(pred["keypoints"], dtype=np.float32)
                all_scores[t, m] = np.array(pred["keypoint_scores"], dtype=np.float32)

        return all_kpts, all_scores
