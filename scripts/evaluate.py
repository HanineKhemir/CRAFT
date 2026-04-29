"""
evaluate.py — Evaluate a trained checkpoint on the test split.

Usage:
    python scripts/evaluate.py \
        --checkpoint experiments/runs/<run>/best_sensitivity.pt \
        --config     experiments/runs/<run>/config.yaml

Outputs (in the checkpoint's run directory):
    eval_report.txt       — full per-class classification report
    confusion_matrix.png  — 4×4 confusion matrix heatmap
    gate_weights.csv      — per-sample gating weights (fine vs coarse branch)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

from mvst_bts.utils.config import load_config
from mvst_bts.utils.seed import set_seed
from mvst_bts.utils.metrics import compute_icbhi_metrics, format_metrics, sklearn_report
from mvst_bts.data.dataset import ICBHIDataset, collate_fn
from mvst_bts.models.mvst_bts_plus import MVSTBTSPlus


CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]


def evaluate(checkpoint_path: str, config_path: str) -> None:
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent

    cfg = load_config(config_path)
    set_seed(cfg.data.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ────────────────────────────────────────
    val_dataset = ICBHIDataset(
        metadata_csv=cfg.data.metadata_csv,
        cycles_dir=cfg.data.cycles_dir,
        split="test",
        audio_cfg=dict(cfg.audio),
        spec_cfg=dict(cfg.spectrogram),
        augmentations=None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Test samples: {len(val_dataset)}")

    # ── Model ──────────────────────────────────────────
    model = MVSTBTSPlus(cfg.model)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # ── Inference ──────────────────────────────────────
    all_preds, all_labels, all_gates, all_ids = [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            fine   = batch["fine_spec"].to(device)
            coarse = batch["coarse_spec"].to(device)
            meta   = {k: v.to(device) for k, v in batch["metadata"].items()}
            labels = batch["label"].to(device)

            outputs = model(fine, coarse, meta)
            preds   = outputs["logits"].argmax(dim=-1)
            gates   = outputs["gates"]  # (B, 2)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_gates.extend(gates.cpu().tolist())
            all_ids.extend(batch["cycle_ids"])

    # ── Metrics ────────────────────────────────────────
    metrics = compute_icbhi_metrics(all_labels, all_preds)
    report_str = sklearn_report(np.array(all_labels), np.array(all_preds))

    print("\n" + "="*55)
    print("EVALUATION RESULTS")
    print("="*55)
    print(format_metrics(metrics))
    print("\nFull classification report:")
    print(report_str)

    # ── Save text report ───────────────────────────────
    report_path = run_dir / "eval_report.txt"
    with open(report_path, "w") as f:
        f.write("MVST-BTS+ Evaluation Report\n")
        f.write("="*55 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        f.write(format_metrics(metrics) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
    print(f"\nReport saved to {report_path}")

    # ── Confusion matrix ───────────────────────────────
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_ylabel("True label")
    axes[0].set_xlabel("Predicted label")

    # Normalised (recall per cell)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("Confusion Matrix (row-normalised recall)")
    axes[1].set_ylabel("True label")
    axes[1].set_xlabel("Predicted label")

    plt.suptitle(
        f"ICBHI Score: {metrics['icbhi_score']:.2f}%  |  "
        f"Sensitivity: {metrics['sensitivity']:.2f}%  |  "
        f"Specificity: {metrics['specificity']:.2f}%",
        fontsize=12,
    )
    plt.tight_layout()
    cm_path = run_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # ── Gate weights ───────────────────────────────────
    gates_arr = np.array(all_gates)  # (N, 2)
    gate_df = pd.DataFrame({
        "cycle_id":    all_ids,
        "true_label":  [CLASS_NAMES[l] for l in all_labels],
        "pred_label":  [CLASS_NAMES[p] for p in all_preds],
        "gate_fine":   gates_arr[:, 0],
        "gate_coarse": gates_arr[:, 1],
    })
    gate_csv = run_dir / "gate_weights.csv"
    gate_df.to_csv(gate_csv, index=False)
    print(f"Gate weights saved to {gate_csv}")

    # Print mean gates per class (useful for analysis)
    print("\nMean gate weights per true class:")
    print(f"  {'Class':<12} {'Fine branch':>14} {'Coarse branch':>15}")
    for cls_name in CLASS_NAMES:
        subset = gate_df[gate_df["true_label"] == cls_name]
        if len(subset) == 0:
            continue
        print(f"  {cls_name:<12} {subset['gate_fine'].mean():>14.3f} {subset['gate_coarse'].mean():>15.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config.yaml (snapshot saved during training)")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config)


if __name__ == "__main__":
    main()