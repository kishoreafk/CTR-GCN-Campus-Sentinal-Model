"""
Actigraphy Preprocessing Pipeline for ExplainMoE-ADHD v2.13.

Implements Section 4.3:
- 3-axis accelerometry + heart rate processing
- Magnitude computation, epoch segmentation, normalization
- Non-wear detection
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ActigraphyPreprocessingConfig:
    """Configuration for actigraphy preprocessing."""
    epoch_seconds: float = 30.0   # Epoch/window duration
    overlap: float = 0.0          # No overlap by default for actigraphy
    nonwear_threshold: float = 0.001  # Magnitude below this = non-wear


class ActigraphyPreprocessor:
    """
    Actigraphy preprocessing pipeline.

    Processes 3-axis accelerometry + heart rate data from Hyperaktiv (D7).
    """

    def __init__(self, config: Optional[ActigraphyPreprocessingConfig] = None):
        self.config = config or ActigraphyPreprocessingConfig()

    def compute_magnitude(self, accel: np.ndarray) -> np.ndarray:
        """Compute acceleration magnitude: sqrt(x² + y² + z²).

        Args:
            accel: (3, samples) or (samples, 3)

        Returns:
            mag: (samples,)
        """
        if accel.shape[0] == 3:
            return np.sqrt((accel ** 2).sum(axis=0))
        else:
            return np.sqrt((accel ** 2).sum(axis=1))

    def detect_nonwear(self, magnitude: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Detect non-wear epochs (consecutive low activity).

        Args:
            magnitude: (samples,)

        Returns:
            wear_mask: (samples,) boolean, True = valid wear
        """
        threshold = threshold or self.config.nonwear_threshold
        return magnitude > threshold

    def segment_epochs(
        self, data: np.ndarray, fs: float,
        epoch_sec: Optional[float] = None,
    ) -> np.ndarray:
        """Segment time-series into fixed-length epochs.

        Args:
            data: (channels, samples)
            fs: sampling rate in Hz

        Returns:
            (n_epochs, channels, epoch_samples)
        """
        epoch_sec = epoch_sec or self.config.epoch_seconds
        epoch_samples = int(epoch_sec * fs)
        n_channels, n_samples = data.shape
        n_epochs = n_samples // epoch_samples

        if n_epochs == 0:
            padded = np.zeros((n_channels, epoch_samples), dtype=data.dtype)
            padded[:, :min(n_samples, epoch_samples)] = data[:, :min(n_samples, epoch_samples)]
            return padded[np.newaxis, :, :]

        # Trim to exact epoch boundaries
        trimmed = data[:, :n_epochs * epoch_samples]
        return trimmed.reshape(n_channels, n_epochs, epoch_samples).transpose(1, 0, 2)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Per-subject z-score normalization across all windows.

        Args:
            data: (n_epochs, channels, samples) or (channels, samples)
        """
        if data.ndim == 3:
            # Normalize per channel across all epochs
            for ch in range(data.shape[1]):
                ch_data = data[:, ch, :]
                mean = ch_data.mean()
                std = ch_data.std()
                if std < 1e-8:
                    std = 1.0
                data[:, ch, :] = (ch_data - mean) / std
        elif data.ndim == 2:
            for ch in range(data.shape[0]):
                mean = data[ch].mean()
                std = data[ch].std()
                if std < 1e-8:
                    std = 1.0
                data[ch] = (data[ch] - mean) / std
        return data

    def preprocess(
        self, accel: np.ndarray, heart_rate: np.ndarray, fs: float,
    ) -> Dict[str, np.ndarray]:
        """Full actigraphy preprocessing pipeline.

        Args:
            accel: (3, samples) — x, y, z acceleration
            heart_rate: (samples,) — heart rate signal
            fs: sampling rate

        Returns:
            Dict with 'timeseries' key: (n_epochs, 4, epoch_samples)
        """
        # Stack channels: (4, samples) — [x, y, z, HR]
        if heart_rate.ndim == 1:
            heart_rate = heart_rate[np.newaxis, :]
        data = np.concatenate([accel, heart_rate], axis=0)

        # Detect non-wear (on magnitude of accel)
        mag = self.compute_magnitude(accel)
        wear_mask = self.detect_nonwear(mag)
        wear_rate = wear_mask.sum() / len(wear_mask)

        # Segment into epochs
        epochs = self.segment_epochs(data, fs)

        # Normalize
        epochs = self.normalize(epochs)

        return {
            "timeseries": epochs.astype(np.float32),
            "wear_rate": float(wear_rate),
        }
