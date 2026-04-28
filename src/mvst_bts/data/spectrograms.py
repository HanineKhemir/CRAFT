"""
spectrograms.py — dual-resolution mel-spectrogram computation.

Fine branch  : high temporal resolution → catches crackles (5–15 ms events)
Coarse branch: high frequency resolution → catches wheezes (>80 ms tonal events)

Both branches output a float32 tensor of shape (1, H, W) normalized to [0, 1].
"""
from __future__ import annotations

import numpy as np
import librosa
import torch
import torch.nn.functional as F


def _mel_spectrogram(
    waveform: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """Compute a log-power mel spectrogram. Returns shape (n_mels, T)."""
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=50.0,
        fmax=2000.0,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def _normalize_to_unit(spec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize spectrogram values to [0, 1]."""
    spec_min = spec.min()
    spec_max = spec.max()
    return (spec - spec_min) / (spec_max - spec_min + eps)


def _resize_spec(spec: np.ndarray, target_h: int, target_w: int) -> torch.Tensor:
    """Resize spectrogram to (target_h, target_w) using bilinear interpolation."""
    t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze(0)  # (1, target_h, target_w)


def compute_fine_spectrogram(
    waveform: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 64,
    target_h: int = 128,
    target_w: int = 1024,
) -> torch.Tensor:
    """
    Fine-resolution spectrogram branch.
    hop_length=64 → ~3 ms per frame → captures crackles (5–15 ms bursts).
    Returns float32 tensor of shape (1, target_h, target_w).
    """
    spec = _mel_spectrogram(waveform, sr, n_mels, n_fft, hop_length)
    spec = _normalize_to_unit(spec)
    return _resize_spec(spec, target_h, target_w)


def compute_coarse_spectrogram(
    waveform: np.ndarray,
    sr: int = 22050,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 256,
    target_h: int = 128,
    target_w: int = 1024,
) -> torch.Tensor:
    """
    Coarse-resolution spectrogram branch.
    hop_length=256 → ~12 ms per frame → captures wheezes (>80 ms sustained tones).
    Returns float32 tensor of shape (1, target_h, target_w).
    """
    spec = _mel_spectrogram(waveform, sr, n_mels, n_fft, hop_length)
    spec = _normalize_to_unit(spec)
    return _resize_spec(spec, target_h, target_w)


def compute_dual_spectrograms(
    waveform: np.ndarray,
    sr: int = 22050,
    fine_cfg: dict | None = None,
    coarse_cfg: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute both spectrogram branches from the same preprocessed waveform.

    Args:
        waveform:   Preprocessed float32 waveform (already resampled + z-normed).
        sr:         Sample rate.
        fine_cfg:   Dict of kwargs for compute_fine_spectrogram (overrides defaults).
        coarse_cfg: Dict of kwargs for compute_coarse_spectrogram (overrides defaults).

    Returns:
        fine_spec:   (1, H, W) tensor
        coarse_spec: (1, H, W) tensor
    """
    fine_kwargs = fine_cfg or {}
    coarse_kwargs = coarse_cfg or {}
    fine_spec = compute_fine_spectrogram(waveform, sr, **fine_kwargs)
    coarse_spec = compute_coarse_spectrogram(waveform, sr, **coarse_kwargs)
    return fine_spec, coarse_spec