"""
Eye-Tracking Preprocessing Pipeline for ExplainMoE-ADHD v2.13.

Implements Section 4.4:
- Blink detection and removal
- Short-gap interpolation
- Segmentation and normalization
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class EyeTrackingPreprocessingConfig:
    """Configuration for eye-tracking preprocessing."""
    blink_recovery_ms: float = 50.0     # Recovery period after blink (ms)
    max_interp_ms: float = 100.0        # Max gap to interpolate (ms)
    min_pupil_threshold: float = 0.1    # Pupil diameter below this = blink
    window_seconds: float = 5.0         # Window duration for segmentation
    overlap: float = 0.5


class EyeTrackingPreprocessor:
    """
    Eye-tracking preprocessing pipeline.

    Processes gaze coordinates (x, y) and pupil diameter
    from Wainstein dataset (D8).
    """

    def __init__(self, config: Optional[EyeTrackingPreprocessingConfig] = None):
        self.config = config or EyeTrackingPreprocessingConfig()

    def detect_blinks(self, pupil: np.ndarray, fs: float) -> np.ndarray:
        """Detect blink intervals.

        Args:
            pupil: (samples,) pupil diameter
            fs: sampling rate

        Returns:
            blink_mask: (samples,) boolean, True = blink
        """
        blink_mask = (pupil <= self.config.min_pupil_threshold) | (pupil == 0)

        # Extend blink mask by recovery period
        recovery_samples = int(self.config.blink_recovery_ms / 1000.0 * fs)

        extended = blink_mask.copy()
        blink_offsets = np.where(np.diff(blink_mask.astype(int)) == -1)[0]
        for offset in blink_offsets:
            end = min(offset + recovery_samples + 1, len(extended))
            extended[offset:end] = True

        return extended

    def interpolate_gaps(
        self, signal: np.ndarray, blink_mask: np.ndarray, fs: float,
    ) -> np.ndarray:
        """Interpolate short gaps; leave long gaps as NaN.

        Args:
            signal: (samples,) — one channel of gaze data
            blink_mask: (samples,) boolean

        Returns:
            interpolated signal
        """
        result = signal.copy().astype(np.float64)
        max_gap_samples = int(self.config.max_interp_ms / 1000.0 * fs)

        # Find gap boundaries
        result[blink_mask] = np.nan

        # Identify contiguous NaN runs
        is_nan = np.isnan(result)
        changes = np.diff(is_nan.astype(int))
        gap_starts = np.where(changes == 1)[0] + 1
        gap_ends = np.where(changes == -1)[0] + 1

        # Handle edge cases
        if is_nan[0]:
            gap_starts = np.concatenate([[0], gap_starts])
        if is_nan[-1]:
            gap_ends = np.concatenate([gap_ends, [len(result)]])

        for start, end in zip(gap_starts, gap_ends):
            gap_len = end - start
            if gap_len <= max_gap_samples and start > 0 and end < len(result):
                # Linear interpolation
                left_val = result[start - 1]
                right_val = result[end] if end < len(result) else left_val
                result[start:end] = np.linspace(left_val, right_val, gap_len)

        return result

    def segment_windows(
        self, data: np.ndarray, fs: float,
    ) -> np.ndarray:
        """Segment into fixed-length windows.

        Args:
            data: (samples, features) — [x, y, pupil]

        Returns:
            (n_windows, window_samples, features)
        """
        window_samples = int(self.config.window_seconds * fs)
        stride = int(window_samples * (1 - self.config.overlap))
        n_samples = data.shape[0]

        if n_samples < window_samples:
            padded = np.zeros((window_samples, data.shape[1]), dtype=data.dtype)
            padded[:n_samples] = data
            return padded[np.newaxis, :, :]

        starts = range(0, n_samples - window_samples + 1, stride)
        windows = np.array([data[s:s + window_samples] for s in starts])
        return windows

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Per-subject normalization.

        Args:
            data: (n_windows, window_samples, features) or (samples, features)
        """
        if data.ndim == 3:
            for f in range(data.shape[2]):
                feat_data = data[:, :, f]
                valid = feat_data[~np.isnan(feat_data)]
                if len(valid) > 0:
                    mean = valid.mean()
                    std = valid.std()
                    if std < 1e-8:
                        std = 1.0
                    data[:, :, f] = (feat_data - mean) / std
        return data

    def preprocess(
        self, gaze_x: np.ndarray, gaze_y: np.ndarray,
        pupil: np.ndarray, fs: float,
    ) -> Dict[str, np.ndarray]:
        """Full eye-tracking preprocessing pipeline.

        Args:
            gaze_x: (samples,) x-coordinates
            gaze_y: (samples,) y-coordinates
            pupil: (samples,) pupil diameter
            fs: sampling rate

        Returns:
            Dict with 'gaze_sequence': (n_windows, window_samples, 3)
        """
        # Detect blinks
        blink_mask = self.detect_blinks(pupil, fs)

        # Interpolate short gaps
        gaze_x_clean = self.interpolate_gaps(gaze_x, blink_mask, fs)
        gaze_y_clean = self.interpolate_gaps(gaze_y, blink_mask, fs)
        pupil_clean = self.interpolate_gaps(pupil, blink_mask, fs)

        # Replace remaining NaNs with 0
        gaze_x_clean = np.nan_to_num(gaze_x_clean, nan=0.0)
        gaze_y_clean = np.nan_to_num(gaze_y_clean, nan=0.0)
        pupil_clean = np.nan_to_num(pupil_clean, nan=0.0)

        # Stack: (samples, 3)
        data = np.stack([gaze_x_clean, gaze_y_clean, pupil_clean], axis=1)

        # Segment
        windows = self.segment_windows(data, fs)

        # Normalize
        windows = self.normalize(windows)

        return {
            "gaze_sequence": windows.astype(np.float32),
            "blink_rate": float(blink_mask.sum() / len(blink_mask)),
        }
