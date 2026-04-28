"""Unit tests for the preprocessing pipeline."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mvst_bts.data.preprocess import (
    bandpass_filter, trim_silence, pad_or_truncate, z_normalize, preprocess_waveform
)

SR = 22050
DURATION = 8.0
TARGET_LEN = int(SR * DURATION)


def make_sine(freq=440, duration=3.0, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_bandpass_output_shape():
    wav = make_sine()
    out = bandpass_filter(wav, SR)
    assert out.shape == wav.shape

def test_pad_short_signal():
    wav = make_sine(duration=2.0)
    out = pad_or_truncate(wav, SR, DURATION)
    assert len(out) == TARGET_LEN

def test_truncate_long_signal():
    wav = make_sine(duration=12.0)
    out = pad_or_truncate(wav, SR, DURATION)
    assert len(out) == TARGET_LEN

def test_z_normalize_zero_mean():
    wav = make_sine(duration=1.0)
    out = z_normalize(wav)
    assert abs(out.mean()) < 1e-5

def test_z_normalize_unit_std():
    wav = make_sine(duration=1.0)
    out = z_normalize(wav)
    assert abs(out.std() - 1.0) < 0.01

def test_full_pipeline_output_shape():
    wav = make_sine(duration=5.0)
    out = preprocess_waveform(wav, SR)
    assert out.shape == (TARGET_LEN,), f"Expected ({TARGET_LEN},), got {out.shape}"

def test_full_pipeline_dtype():
    wav = make_sine(duration=5.0)
    out = preprocess_waveform(wav, SR)
    assert out.dtype == np.float32