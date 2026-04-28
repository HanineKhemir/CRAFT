"""
metrics.py — ICBHI evaluation metrics.

The official ICBHI score is the average of:
  - Specificity (Se): fraction of true negatives correctly identified
  - Sensitivity (Sp): fraction of true positives correctly identified

In multi-class setting (normal / crackle / wheeze / both):
  - Sensitivity = macro-average recall across ALL four classes
  - Specificity = recall on the "normal" class only
  - ICBHI score = (Sensitivity + Specificity) / 2

This matches the evaluation used in the reference paper.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]


def compute_icbhi_metrics(
    y_true: list[int] | np.ndarray | torch.Tensor,
    y_pred: list[int] | np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """
    Compute the full ICBHI metric suite.

    Returns a dict with:
        sensitivity     — macro recall across all 4 classes
        specificity     — recall on class 0 (normal)
        icbhi_score     — (sensitivity + specificity) / 2
        recall_normal   — per-class recall
        recall_crackle
        recall_wheeze
        recall_both
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    # Per-class recall = TP / (TP + FN) = cm[i,i] / cm[i,:].sum()
    per_class_recall = np.zeros(4)
    for i in range(4):
        row_sum = cm[i, :].sum()
        per_class_recall[i] = cm[i, i] / row_sum if row_sum > 0 else 0.0

    sensitivity = float(np.mean(per_class_recall))   # macro recall
    specificity = float(per_class_recall[0])          # normal class recall
    icbhi_score = (sensitivity + specificity) / 2.0

    return {
        "sensitivity":    round(sensitivity * 100, 2),
        "specificity":    round(specificity * 100, 2),
        "icbhi_score":    round(icbhi_score * 100, 2),
        "recall_normal":  round(float(per_class_recall[0]) * 100, 2),
        "recall_crackle": round(float(per_class_recall[1]) * 100, 2),
        "recall_wheeze":  round(float(per_class_recall[2]) * 100, 2),
        "recall_both":    round(float(per_class_recall[3]) * 100, 2),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """Pretty-print metrics for logging."""
    lines = [
        f"  ICBHI score:   {metrics['icbhi_score']:.2f}%",
        f"  Sensitivity:   {metrics['sensitivity']:.2f}%",
        f"  Specificity:   {metrics['specificity']:.2f}%",
        f"  Recall normal:  {metrics['recall_normal']:.2f}%",
        f"  Recall crackle: {metrics['recall_crackle']:.2f}%",
        f"  Recall wheeze:  {metrics['recall_wheeze']:.2f}%",
        f"  Recall both:    {metrics['recall_both']:.2f}%",
    ]
    return "\n".join(lines)


def sklearn_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """Full sklearn classification report for the report appendix."""
    return classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)