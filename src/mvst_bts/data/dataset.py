"""
dataset.py — ICBHI PyTorch Dataset, weighted sampler, and collate_fn.

The Dataset reads from metadata.csv (built by scripts/extract_cycles.py) and
returns a dict with:
    fine_spec   : (1, H, W) float32 tensor — fine-resolution spectrogram
    coarse_spec : (1, H, W) float32 tensor — coarse-resolution spectrogram
    metadata    : dict of encoded metadata fields
    label       : int in {0, 1, 2, 3}
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from mvst_bts.data.preprocess import load_and_preprocess
from mvst_bts.data.spectrograms import compute_dual_spectrograms


# ── Label mapping ───────────────────────────────────────
LABEL_MAP = {"normal": 0, "crackle": 1, "wheeze": 2, "both": 3}

# ── Device and location vocabularies ────────────────────
DEVICE_MAP = {"AKGC417L": 0, "LittC2SE": 1, "Litt3200": 2, "Meditron": 3}
SEX_MAP = {"M": 0, "F": 1, "unknown": 0}
# ICBHI chest locations (9 positions)
LOCATION_MAP = {
    "Tc": 0, "Al": 1, "Ar": 2, "Pl": 3, "Pr": 4,
    "Ll": 5, "Lr": 6, "Ml": 7, "Mr": 8,
}
# Age bins: child <18, young adult 18–40, adult 40–65, elderly 65+
def _age_bin(age: float) -> int:
    if age < 18:
        return 0
    elif age < 40:
        return 1
    elif age < 65:
        return 2
    else:
        return 3


class ICBHIDataset(Dataset):
    """
    ICBHI 2017 respiratory sound dataset.

    Args:
        metadata_csv : Path to metadata.csv produced by extract_cycles.py.
        cycles_dir   : Directory containing extracted cycle .wav files.
        split        : "train" or "test".
        audio_cfg    : Dict with audio preprocessing settings.
        spec_cfg     : Dict with spectrogram settings (fine + coarse sub-dicts).
        augmentations: Optional list of augmentation transforms applied to spectrograms.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        cycles_dir: str | Path,
        split: str,
        audio_cfg: dict,
        spec_cfg: dict,
        augmentations=None,
    ):
        super().__init__()
        df = pd.read_csv(metadata_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.cycles_dir = Path(cycles_dir)
        self.audio_cfg = audio_cfg
        self.spec_cfg = spec_cfg
        self.augmentations = augmentations  # applied in __getitem__ (training only)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Load and preprocess waveform ────────────────
        wav_path = str(self.cycles_dir / row["filename"])
        waveform = load_and_preprocess(
            wav_path,
            target_sr=self.audio_cfg.get("sample_rate", 22050),
            bandpass_low=self.audio_cfg.get("bandpass_low_hz", 50.0),
            bandpass_high=self.audio_cfg.get("bandpass_high_hz", 2000.0),
            bandpass_order=self.audio_cfg.get("bandpass_order", 4),
            silence_threshold_db=self.audio_cfg.get("silence_threshold_db", -40.0),
            duration_seconds=self.audio_cfg.get("duration_seconds", 8.0),
        )

        # ── Compute dual spectrograms ───────────────────
        fine_cfg = dict(self.spec_cfg.get("fine", {}))
        coarse_cfg = dict(self.spec_cfg.get("coarse", {}))
        # Remove non-spectrogram keys
        for cfg in (fine_cfg, coarse_cfg):
            cfg.pop("target_height", None); cfg.pop("target_width", None)

        fine_spec, coarse_spec = compute_dual_spectrograms(
            waveform,
            sr=self.audio_cfg.get("sample_rate", 22050),
            fine_cfg=fine_cfg,
            coarse_cfg=coarse_cfg,
        )

        # ── Encode metadata ─────────────────────────────
        age = float(row.get("age", 40.0))
        metadata = {
            "age_bin":  torch.tensor(_age_bin(age), dtype=torch.long),
            "sex":      torch.tensor(SEX_MAP.get(str(row.get("sex", "M")), 0), dtype=torch.long),
            "device":   torch.tensor(DEVICE_MAP.get(str(row.get("device", "AKGC417L")), 0), dtype=torch.long),
            "location": torch.tensor(LOCATION_MAP.get(str(row.get("location", "Tc")), 0), dtype=torch.long),
        }

        label = int(LABEL_MAP.get(str(row["label"]), 0))

        sample = {
            "fine_spec":   fine_spec,    # (1, H, W)
            "coarse_spec": coarse_spec,  # (1, H, W)
            "metadata":    metadata,
            "label":       torch.tensor(label, dtype=torch.long),
            "cycle_id":    str(row["cycle_id"]),
        }

        # ── Augmentations (train split only) ────────────
        if self.augmentations is not None:
            for aug in self.augmentations:
                sample = aug(sample)

        return sample


# ── Weighted Sampler ─────────────────────────────────────

def build_weighted_sampler(dataset: ICBHIDataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that over-samples minority classes.

    Weight per sample = 1 / class_count.
    This ensures each mini-batch has a balanced class distribution
    regardless of the raw dataset imbalance.
    """
    labels = dataset.df["label"].map(LABEL_MAP).values
    class_counts = np.bincount(labels, minlength=4)
    # Avoid division by zero for absent classes
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )


# ── Collate function ─────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """Stack a list of samples into a batched dict."""
    fine_specs   = torch.stack([s["fine_spec"]   for s in batch])   # (B, 1, H, W)
    coarse_specs = torch.stack([s["coarse_spec"] for s in batch])   # (B, 1, H, W)
    labels       = torch.stack([s["label"]       for s in batch])   # (B,)

    # Stack metadata fields
    metadata = {
        key: torch.stack([s["metadata"][key] for s in batch])
        for key in batch[0]["metadata"]
    }

    return {
        "fine_spec":   fine_specs,
        "coarse_spec": coarse_specs,
        "metadata":    metadata,
        "label":       labels,
        "cycle_ids":   [s["cycle_id"] for s in batch],
    }