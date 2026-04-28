"""
spec_augment.py — SpecAugment for mel-spectrograms.

Applies random time masking and frequency masking to force the model to
be robust to partial signal — prevents over-fitting to device-specific
spectral signatures.
"""
from __future__ import annotations

import random
import torch


class SpecAugment:
    """
    SpecAugment augmentation applied to a spectrogram tensor.

    Args:
        time_mask_max_ratio:  Max fraction of time frames to mask (default 0.20).
        freq_mask_max_ratio:  Max fraction of mel bins to mask (default 0.20).
        num_time_masks:       Number of independent time masks (default 2).
        num_freq_masks:       Number of independent freq masks (default 2).
    """

    def __init__(
        self,
        time_mask_max_ratio: float = 0.20,
        freq_mask_max_ratio: float = 0.20,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ):
        self.time_mask_max_ratio = time_mask_max_ratio
        self.freq_mask_max_ratio = freq_mask_max_ratio
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def __call__(self, sample: dict) -> dict:
        """Apply SpecAugment to both fine and coarse spectrograms in the sample."""
        sample["fine_spec"]   = self._augment(sample["fine_spec"])
        sample["coarse_spec"] = self._augment(sample["coarse_spec"])
        return sample

    def _augment(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: (1, H, W) or (H, W) float tensor.
        Returns:
            Masked spectrogram of the same shape.
        """
        spec = spec.clone()
        _, H, W = spec.shape  # (channels, freq_bins, time_frames)

        # Time masking (W dimension)
        max_t = int(W * self.time_mask_max_ratio)
        for _ in range(self.num_time_masks):
            if max_t > 0:
                t = random.randint(0, max_t)
                t0 = random.randint(0, max(0, W - t))
                spec[:, :, t0:t0 + t] = 0.0

        # Frequency masking (H dimension)
        max_f = int(H * self.freq_mask_max_ratio)
        for _ in range(self.num_freq_masks):
            if max_f > 0:
                f = random.randint(0, max_f)
                f0 = random.randint(0, max(0, H - f))
                spec[:, f0:f0 + f, :] = 0.0

        return spec