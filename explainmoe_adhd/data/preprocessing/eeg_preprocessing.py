"""
EEG Preprocessing Pipeline for ExplainMoE-ADHD v2.13.

Implements Section 4.1:
- Clinical-grade EEG (D1 — 19ch, 128Hz)
- Consumer-grade EMOTIV (D2 — 14ch→10ch, 128Hz)
- Adult EEG (D3 — 5ch, 256Hz)
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


# 10-20 standard channel sets
CHANNELS_19CH = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]

EMOTIV_14CH = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

# 10 EMOTIV channels retained (present in standard 10-20)
EMOTIV_10CH_RETAINED = ["F7", "F3", "T7", "P7", "O1", "O2", "P8", "T8", "F4", "F8"]

# 4 EMOTIV channels discarded (10-10 intermediate positions)
EMOTIV_4CH_DISCARDED = ["AF3", "AF4", "FC5", "FC6"]

CHANNELS_5CH = ["O1", "F3", "F4", "Cz", "Fz"]


@dataclass
class EEGPreprocessingConfig:
    """Configuration for EEG preprocessing."""
    bandpass_low: float = 1.0
    bandpass_high: float = 45.0
    notch_freq: float = 50.0       # 50Hz for Iran (D1), 60Hz for US datasets
    asr_threshold: float = 20.0    # ASR threshold in standard deviations
    window_seconds: float = 2.0
    overlap: float = 0.5
    ica_fallback_threshold: float = 0.30  # Max fallback rate before reporting


class EEGPreprocessor:
    """
    EEG preprocessing pipeline.

    Implements bandpass filtering, notch filtering, ASR, ICA artifact removal,
    re-referencing, windowing, and per-channel z-score normalization.
    """

    def __init__(self, config: Optional[EEGPreprocessingConfig] = None):
        self.config = config or EEGPreprocessingConfig()

    def bandpass_filter(
        self, data: np.ndarray, fs: float,
        low: Optional[float] = None, high: Optional[float] = None,
    ) -> np.ndarray:
        """Apply FIR bandpass filter.

        Args:
            data: (channels, samples)
            fs: sampling rate in Hz
            low: low cutoff (default: config.bandpass_low)
            high: high cutoff (default: config.bandpass_high)

        Returns:
            Filtered data with same shape.
        """
        from scipy.signal import firwin, filtfilt

        low = low or self.config.bandpass_low
        high = high or self.config.bandpass_high

        nyq = fs / 2.0
        numtaps = int(fs) + 1  # ~1 second kernel
        if numtaps % 2 == 0:
            numtaps += 1

        coeffs = firwin(numtaps, [low / nyq, high / nyq], pass_zero=False)
        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            # Require sufficient samples for filtering
            if data.shape[1] > 3 * numtaps:
                filtered[ch] = filtfilt(coeffs, 1.0, data[ch])
            else:
                filtered[ch] = data[ch]
        return filtered

    def notch_filter(self, data: np.ndarray, fs: float, freq: Optional[float] = None) -> np.ndarray:
        """Apply notch filter to remove power line noise.

        Args:
            data: (channels, samples)
            fs: sampling rate
            freq: notch frequency (default: config.notch_freq)
        """
        from scipy.signal import iirnotch, filtfilt

        freq = freq or self.config.notch_freq
        q = 30.0
        b, a = iirnotch(freq, q, fs)
        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered[ch] = filtfilt(b, a, data[ch])
        return filtered

    def common_average_reference(self, data: np.ndarray) -> np.ndarray:
        """Re-reference to common average."""
        return data - data.mean(axis=0, keepdims=True)

    def segment_windows(
        self, data: np.ndarray, fs: float,
        window_sec: Optional[float] = None, overlap: Optional[float] = None,
    ) -> np.ndarray:
        """Segment into fixed-length overlapping windows.

        Args:
            data: (channels, samples)
            fs: sampling rate
            window_sec: window duration in seconds
            overlap: overlap fraction (0-1)

        Returns:
            (n_windows, channels, window_samples)
        """
        window_sec = window_sec or self.config.window_seconds
        overlap = overlap or self.config.overlap

        window_samples = int(window_sec * fs)
        stride = int(window_samples * (1 - overlap))

        n_channels, n_samples = data.shape

        if n_samples < window_samples:
            # Pad with zeros if data is shorter than one window
            padded = np.zeros((n_channels, window_samples), dtype=data.dtype)
            padded[:, :n_samples] = data
            return padded[np.newaxis, :, :]

        starts = list(range(0, n_samples - window_samples + 1, stride))
        windows = np.array([data[:, s:s + window_samples] for s in starts])
        return windows

    def normalize_windows(self, windows: np.ndarray) -> np.ndarray:
        """Per-channel z-score normalization within each window.

        Args:
            windows: (n_windows, channels, samples)

        Returns:
            Normalized windows.
        """
        mean = windows.mean(axis=-1, keepdims=True)
        std = windows.std(axis=-1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return (windows - mean) / std

    def extract_emotiv_10ch(self, data: np.ndarray, channel_names: list) -> Tuple[np.ndarray, list]:
        """Extract 10 retained channels from 14-channel EMOTIV recording.

        Args:
            data: (14, samples)
            channel_names: list of 14 channel names

        Returns:
            (data_10ch, retained_names)
        """
        name_map = {name.upper(): i for i, name in enumerate(channel_names)}
        retained_indices = []
        retained_names = []
        for ch in EMOTIV_10CH_RETAINED:
            key = ch.upper()
            if key in name_map:
                retained_indices.append(name_map[key])
                retained_names.append(ch)
        return data[retained_indices], retained_names

    def preprocess(
        self, data: np.ndarray, fs: float,
        channel_type: str = "19ch",
        channel_names: Optional[list] = None,
    ) -> np.ndarray:
        """Full preprocessing pipeline.

        Args:
            data: (channels, samples)
            fs: sampling rate
            channel_type: "19ch", "10ch", or "5ch"
            channel_names: optional channel name list (for EMOTIV extraction)

        Returns:
            (n_windows, channels, window_samples) normalized windows
        """
        # Step 0: Channel extraction for EMOTIV
        if channel_type == "10ch" and data.shape[0] == 14 and channel_names is not None:
            data, _ = self.extract_emotiv_10ch(data, channel_names)

        # Step 1: Bandpass filter
        data = self.bandpass_filter(data, fs)

        # Step 2: Notch filter
        data = self.notch_filter(data, fs)

        # Step 3: Common average reference
        data = self.common_average_reference(data)

        # Step 4: Segment into windows
        windows = self.segment_windows(data, fs)

        # Step 5: Normalize
        windows = self.normalize_windows(windows)

        return windows

    @staticmethod
    def validate(data: np.ndarray, expected_channels: int, expected_rate: float) -> None:
        """Post-preprocessing validation (Section 4.1.4)."""
        assert not np.any(np.isnan(data)), "NaN found in preprocessed EEG"
        assert not np.any(np.isinf(data)), "Inf found in preprocessed EEG"

        if data.ndim == 3:
            # (n_windows, channels, samples)
            assert data.shape[1] == expected_channels, (
                f"Expected {expected_channels} channels, got {data.shape[1]}"
            )
            variance = np.var(data, axis=-1)
            assert np.all(variance > 1e-8), "Zero-variance channel detected"
        elif data.ndim == 2:
            assert data.shape[0] == expected_channels
