"""
test_dataset.py — Test ICBHIDataset __getitem__ using a tiny mock dataset
(no real ICBHI files needed).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
import tempfile
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import pytest

from mvst_bts.data.dataset import ICBHIDataset, build_weighted_sampler, collate_fn, LABEL_MAP


SR = 22050
DURATION = 8.0


def make_mock_dataset(tmp_dir: Path, n_samples: int = 12) -> Path:
    """Create a minimal mock ICBHI dataset for testing."""
    cycles_dir = tmp_dir / "cycles"
    cycles_dir.mkdir()

    labels = ["normal", "crackle", "wheeze", "both"] * (n_samples // 4)
    splits = (["train"] * 8 + ["test"] * 4)[:n_samples]

    rows = []
    for i in range(n_samples):
        cycle_id = f"mock_{i:03d}"
        wav = np.random.randn(SR * 2).astype(np.float32)  # 2-second random audio
        sf.write(str(cycles_dir / f"{cycle_id}.wav"), wav, SR)
        rows.append({
            "cycle_id":    cycle_id,
            "filename":    f"{cycle_id}.wav",
            "patient_id":  f"{i:03d}",
            "label":       labels[i],
            "age":         30.0 + i,
            "sex":         "M" if i % 2 == 0 else "F",
            "device":      "AKGC417L",
            "location":    "Tc",
            "split":       splits[i],
        })

    csv_path = tmp_dir / "metadata.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, cycles_dir


AUDIO_CFG = {
    "sample_rate": SR,
    "duration_seconds": DURATION,
    "bandpass_low_hz": 50.0,
    "bandpass_high_hz": 2000.0,
    "bandpass_order": 4,
    "silence_threshold_db": -40.0,
}
SPEC_CFG = {
    "fine":   {"n_mels": 64, "n_fft": 512, "hop_length": 64},
    "coarse": {"n_mels": 32, "n_fft": 1024, "hop_length": 256},
}


def test_dataset_len():
    with tempfile.TemporaryDirectory() as tmp:
        csv, cycles = make_mock_dataset(Path(tmp))
        ds = ICBHIDataset(csv, cycles, "train", AUDIO_CFG, SPEC_CFG)
        assert len(ds) == 8


def test_dataset_getitem_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        csv, cycles = make_mock_dataset(Path(tmp))
        ds = ICBHIDataset(csv, cycles, "train", AUDIO_CFG, SPEC_CFG)
        sample = ds[0]
        assert sample["fine_spec"].shape[0] == 1,   "fine_spec should have 1 channel"
        assert sample["coarse_spec"].shape[0] == 1, "coarse_spec should have 1 channel"
        assert sample["fine_spec"].ndim == 3,   "fine_spec should be 3D (C,H,W)"
        assert sample["coarse_spec"].ndim == 3


def test_dataset_getitem_label_type():
    with tempfile.TemporaryDirectory() as tmp:
        csv, cycles = make_mock_dataset(Path(tmp))
        ds = ICBHIDataset(csv, cycles, "train", AUDIO_CFG, SPEC_CFG)
        sample = ds[0]
        assert isinstance(sample["label"], torch.Tensor)
        assert sample["label"].dtype == torch.long
        assert sample["label"].item() in [0, 1, 2, 3]


def test_dataset_metadata_keys():
    with tempfile.TemporaryDirectory() as tmp:
        csv, cycles = make_mock_dataset(Path(tmp))
        ds = ICBHIDataset(csv, cycles, "train", AUDIO_CFG, SPEC_CFG)
        sample = ds[0]
        for key in ["age_bin", "sex", "device", "location"]:
            assert key in sample["metadata"], f"Missing metadata key: {key}"


def test_collate_fn_batching():
    with tempfile.TemporaryDirectory() as tmp:
        csv, cycles = make_mock_dataset(Path(tmp))
        ds = ICBHIDataset(csv, cycles, "train", AUDIO_CFG, SPEC_CFG)
        samples = [ds[i] for i in range(4)]
        batch = collate_fn(samples)
        assert batch["fine_spec"].shape[0] == 4
        assert batch["coarse_spec"].shape[0] == 4
        assert batch["label"].shape == (4,)


def test_weighted_sampler_length():
    with tempfile.TemporaryDirectory() as tmp:
        csv, cycles = make_mock_dataset(Path(tmp))
        ds = ICBHIDataset(csv, cycles, "train", AUDIO_CFG, SPEC_CFG)
        sampler = build_weighted_sampler(ds)
        assert len(sampler) == len(ds)