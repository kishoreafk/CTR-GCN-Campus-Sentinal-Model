"""
fMRI Preprocessing Pipeline for ExplainMoE-ADHD v2.13.

Implements Section 4.2:
- Unified pipeline for D5 (ADHD-200 resting-state) and D6 (ds002424 task fMRI)
- Motion scrubbing (FD > 0.5mm)
- ROI parcellation (AAL-90)
- Pearson correlation → Fisher z-transform → upper-triangle vector
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class FMRIPreprocessingConfig:
    """Configuration for fMRI preprocessing."""
    fd_threshold: float = 0.5          # Framewise displacement threshold in mm
    high_motion_threshold: float = 0.4  # Flag if >40% volumes scrubbed
    n_rois: int = 90                   # AAL atlas, cortical+subcortical
    bandpass_low: float = 0.01
    bandpass_high: float = 0.1
    head_radius: float = 50.0         # mm, for FD computation


class FMRIPreprocessor:
    """
    Unified fMRI preprocessing pipeline.

    Produces a connectivity feature vector of size n_rois*(n_rois-1)/2
    plus metadata (mean_FD, scrub_rate, high_motion_flag).
    """

    def __init__(self, config: Optional[FMRIPreprocessingConfig] = None):
        self.config = config or FMRIPreprocessingConfig()

    def compute_framewise_displacement(self, motion_params: np.ndarray) -> np.ndarray:
        """Compute per-volume framewise displacement.

        Args:
            motion_params: (n_volumes, 6) — 3 translation + 3 rotation

        Returns:
            fd: (n_volumes,) framewise displacement
        """
        if motion_params.shape[0] < 2:
            return np.zeros(motion_params.shape[0])

        diff = np.diff(motion_params, axis=0)
        # Translations (mm)
        fd_trans = np.abs(diff[:, :3]).sum(axis=1)
        # Rotations (radians → mm at head surface)
        fd_rot = np.abs(diff[:, 3:]).sum(axis=1) * self.config.head_radius

        fd = np.concatenate([[0.0], fd_trans + fd_rot])
        return fd

    def scrub_volumes(
        self, timeseries: np.ndarray, fd: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool]:
        """Remove high-motion volumes.

        Args:
            timeseries: (n_volumes, n_rois)
            fd: (n_volumes,) framewise displacement

        Returns:
            (clean_timeseries, scrub_rate, high_motion_flag)
        """
        mask = fd <= self.config.fd_threshold
        scrub_rate = 1.0 - mask.sum() / len(mask)
        high_motion = scrub_rate > self.config.high_motion_threshold
        clean = timeseries[mask]
        return clean, float(scrub_rate), high_motion

    def bandpass_filter_timeseries(
        self, timeseries: np.ndarray, tr: float,
    ) -> np.ndarray:
        """Bandpass filter ROI timeseries.

        Args:
            timeseries: (n_volumes, n_rois)
            tr: repetition time in seconds
        """
        from scipy.signal import butter, filtfilt

        fs = 1.0 / tr
        nyq = fs / 2.0
        low = self.config.bandpass_low / nyq
        high = self.config.bandpass_high / nyq

        # Ensure valid filter parameters
        if low >= 1.0 or high >= 1.0 or low <= 0 or high <= 0 or low >= high:
            return timeseries

        order = min(5, timeseries.shape[0] // 6)
        if order < 1:
            return timeseries

        b, a = butter(order, [low, high], btype="band")
        filtered = np.zeros_like(timeseries)
        for roi in range(timeseries.shape[1]):
            filtered[:, roi] = filtfilt(b, a, timeseries[:, roi])
        return filtered

    def compute_connectivity(self, timeseries: np.ndarray) -> np.ndarray:
        """Compute ROI-to-ROI Pearson correlation → Fisher z-transform → upper triangle.

        Args:
            timeseries: (n_volumes, n_rois)

        Returns:
            features: (n_features,) where n_features = n_rois*(n_rois-1)/2
        """
        n_rois = timeseries.shape[1]
        corr = np.corrcoef(timeseries.T)  # (n_rois, n_rois)

        # Clip to avoid arctanh overflow
        corr = np.clip(corr, -0.9999, 0.9999)

        # Fisher z-transform
        z = np.arctanh(corr)

        # Upper triangle (excluding diagonal)
        triu_idx = np.triu_indices(n_rois, k=1)
        features = z[triu_idx]
        return features

    def preprocess(
        self,
        roi_timeseries: np.ndarray,
        motion_params: np.ndarray,
        tr: float,
        dataset_source: str = "adhd200_resting",
    ) -> Dict[str, np.ndarray]:
        """Full fMRI preprocessing pipeline.

        Args:
            roi_timeseries: (n_volumes, n_rois) — already parcellated
            motion_params: (n_volumes, 6) — motion parameters
            tr: repetition time in seconds
            dataset_source: either 'adhd200_resting' or 'ds002424_residual'

        Returns:
            Dict containing:
                connectivity: (n_features,) Fisher z-transformed connectivity
                mean_fd: scalar
                scrub_rate: scalar
                high_motion_flag: bool
        """
        # Compute FD
        fd = self.compute_framewise_displacement(motion_params)
        mean_fd = float(fd.mean())

        # Motion scrubbing
        clean_ts, scrub_rate, high_motion = self.scrub_volumes(roi_timeseries, fd)

        if clean_ts.shape[0] < 20:
            # Too few volumes remaining; return zeros with flags
            n_features = self.config.n_rois * (self.config.n_rois - 1) // 2
            return {
                "connectivity": np.zeros(n_features, dtype=np.float32),
                "mean_fd": mean_fd,
                "scrub_rate": scrub_rate,
                "high_motion_flag": True,
            }

        # Bandpass filter
        clean_ts = self.bandpass_filter_timeseries(clean_ts, tr)

        # Connectivity
        connectivity = self.compute_connectivity(clean_ts)

        return {
            "connectivity": connectivity.astype(np.float32),
            "mean_fd": mean_fd,
            "scrub_rate": scrub_rate,
            "high_motion_flag": high_motion,
        }
