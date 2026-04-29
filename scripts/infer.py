"""
infer.py — Single-file inference.

Usage:
    python scripts/infer.py \
        --audio path/to/cycle.wav \
        --checkpoint experiments/runs/<run>/best_sensitivity.pt \
        --config     experiments/runs/<run>/config.yaml \
        [--age 45 --sex M --device Meditron --location Tc]

Prints the predicted class, confidence scores for all 4 classes,
and the branch gate weights (useful for interpretability).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from mvst_bts.utils.config import load_config
from mvst_bts.data.preprocess import load_and_preprocess
from mvst_bts.data.spectrograms import compute_dual_spectrograms
from mvst_bts.data.dataset import DEVICE_MAP, LOCATION_MAP, SEX_MAP, _age_bin
from mvst_bts.models.mvst_bts_plus import MVSTBTSPlus


CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]


def infer_single(
    audio_path: str,
    checkpoint_path: str,
    config_path: str,
    age: float = 40.0,
    sex: str = "M",
    device_name: str = "AKGC417L",
    location: str = "Tc",
) -> dict:
    """
    Run inference on a single audio file.

    Returns a dict with:
        predicted_class:  str  — e.g. "crackle"
        probabilities:    dict — {class_name: float}
        gate_fine:        float — weight given to the fine branch
        gate_coarse:      float — weight given to the coarse branch
    """
    cfg = load_config(config_path)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────
    model = MVSTBTSPlus(cfg.model)
    state = torch.load(checkpoint_path, map_location=torch_device)
    model.load_state_dict(state["model_state_dict"])
    model.to(torch_device)
    model.eval()

    # ── Preprocess audio ──────────────────────────────
    waveform = load_and_preprocess(
        audio_path,
        target_sr=cfg.audio.sample_rate,
        bandpass_low=cfg.audio.bandpass_low_hz,
        bandpass_high=cfg.audio.bandpass_high_hz,
        bandpass_order=cfg.audio.bandpass_order,
        silence_threshold_db=cfg.audio.silence_threshold_db,
        duration_seconds=cfg.audio.duration_seconds,
    )

    # ── Compute spectrograms ──────────────────────────
    fine_spec, coarse_spec = compute_dual_spectrograms(
        waveform, sr=cfg.audio.sample_rate
    )
    fine_spec   = fine_spec.unsqueeze(0).to(torch_device)   # (1, 1, H, W)
    coarse_spec = coarse_spec.unsqueeze(0).to(torch_device)

    # ── Encode metadata ───────────────────────────────
    metadata = {
        "age_bin":  torch.tensor([_age_bin(age)], dtype=torch.long).to(torch_device),
        "sex":      torch.tensor([SEX_MAP.get(sex, 0)], dtype=torch.long).to(torch_device),
        "device":   torch.tensor([DEVICE_MAP.get(device_name, 0)], dtype=torch.long).to(torch_device),
        "location": torch.tensor([LOCATION_MAP.get(location, 0)], dtype=torch.long).to(torch_device),
    }

    # ── Inference ─────────────────────────────────────
    with torch.no_grad():
        outputs = model(fine_spec, coarse_spec, metadata)
        probs   = torch.softmax(outputs["logits"], dim=-1).squeeze(0).cpu().numpy()
        gates   = outputs["gates"].squeeze(0).cpu().numpy()

    predicted_class = CLASS_NAMES[int(np.argmax(probs))]

    return {
        "predicted_class": predicted_class,
        "probabilities":   {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)},
        "gate_fine":       float(gates[0]),
        "gate_coarse":     float(gates[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Single-file ICBHI inference")
    parser.add_argument("--audio",      type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--age",        type=float, default=40.0)
    parser.add_argument("--sex",        type=str,   default="M", choices=["M", "F"])
    parser.add_argument("--device",     type=str,   default="AKGC417L",
                        choices=list(DEVICE_MAP.keys()))
    parser.add_argument("--location",   type=str,   default="Tc",
                        choices=list(LOCATION_MAP.keys()))
    args = parser.parse_args()

    result = infer_single(
        args.audio, args.checkpoint, args.config,
        age=args.age, sex=args.sex,
        device_name=args.device, location=args.location,
    )

    print("\n" + "="*40)
    print(f"  Prediction: {result['predicted_class'].upper()}")
    print("="*40)
    print("  Class probabilities:")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"    {cls:10s} {prob:.3f}  {bar}")
    print(f"\n  Branch gates:")
    print(f"    Fine   (crackle-sensitive): {result['gate_fine']:.3f}")
    print(f"    Coarse (wheeze-sensitive):  {result['gate_coarse']:.3f}")


if __name__ == "__main__":
    main()