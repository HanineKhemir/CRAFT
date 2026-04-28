"""
preprocess.py — audio preprocessing pipeline.

Steps applied to every ICBHI cycle:
  1. Load .wav  (any sample rate)
  2. Resample to target_sr (22,050 Hz)
  3. Convert to mono
  4. Bandpass filter  50–2000 Hz  (respiratory sound range)
  5. Trim leading/trailing silence
  6. Pad or truncate to exactly `duration_seconds`
  7. Per-sample z-normalisation

All functions are stateless and operate on numpy arrays so they
are easy to unit-test without a Dataset.
"""
from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt


# ─────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load a .wav file, resample, and convert to mono.

    Returns:
        waveform: float32 array shape (n_samples,)
        sample_rate: always target_sr
    """
    waveform, sr = librosa.load(path, sr=target_sr, mono=True)
    return waveform.astype(np.float32), target_sr


def bandpass_filter(
    waveform: np.ndarray,
    sr: int,
    low_hz: float = 50.0,
    high_hz: float = 2000.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter.

    Removes sub-50 Hz movement artifacts and frequencies above
    the respiratory sound range (>2 kHz).
    """
    nyq = sr / 2.0
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.999)  # must be < 1
    sos = butter(order, [low, high], btype="band", output="sos")
    filtered = sosfilt(sos, waveform)
    return filtered.astype(np.float32)


def trim_silence(
    waveform: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
) -> np.ndarray:
    """Remove leading and trailing silence below threshold_db."""
    _, intervals = librosa.effects.trim(waveform, top_db=-threshold_db)
    start, end = int(intervals[0]), int(intervals[1])
    trimmed = waveform[start:end]
    # Guard: if everything was trimmed (pure silence), return original
    return trimmed if len(trimmed) > 0 else waveform


def pad_or_truncate(
    waveform: np.ndarray,
    sr: int,
    duration_seconds: float = 8.0,
) -> np.ndarray:
    """Pad with zeros or truncate to exactly `duration_seconds`.

    Truncation: takes the first `target_len` samples.
    Padding: zero-pads at the end.
    """
    target_len = int(sr * duration_seconds)
    n = len(waveform)
    if n >= target_len:
        return waveform[:target_len]
    pad_width = target_len - n
    return np.pad(waveform, (0, pad_width), mode="constant")


def z_normalize(waveform: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-sample z-normalisation: (x - mean) / std.

    Neutralises device-level amplitude differences between the
    four ICBHI recording devices.
    """
    mean = waveform.mean()
    std = waveform.std()
    return ((waveform - mean) / (std + eps)).astype(np.float32)


# ─────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────

def preprocess_waveform(
    waveform: np.ndarray,
    sr: int,
    bandpass_low: float = 50.0,
    bandpass_high: float = 2000.0,
    bandpass_order: int = 4,
    silence_threshold_db: float = -40.0,
    duration_seconds: float = 8.0,
) -> np.ndarray:
    """Apply the full preprocessing pipeline to a waveform.

    Assumes input has already been resampled to `sr`.
    Returns float32 array of shape (sr * duration_seconds,).
    """
    waveform = bandpass_filter(waveform, sr, bandpass_low, bandpass_high, bandpass_order)
    waveform = trim_silence(waveform, sr, silence_threshold_db)
    waveform = pad_or_truncate(waveform, sr, duration_seconds)
    waveform = z_normalize(waveform)
    return waveform


def load_and_preprocess(
    path: str,
    target_sr: int = 22050,
    bandpass_low: float = 50.0,
    bandpass_high: float = 2000.0,
    bandpass_order: int = 4,
    silence_threshold_db: float = -40.0,
    duration_seconds: float = 8.0,
) -> np.ndarray:
    """Convenience: load from disk and apply full preprocessing pipeline."""
    waveform, sr = load_audio(path, target_sr)
    return preprocess_waveform(
        waveform, sr, bandpass_low, bandpass_high,
        bandpass_order, silence_threshold_db, duration_seconds,
    )