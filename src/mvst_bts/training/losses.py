"""
losses.py — Recall-aware loss functions.

FocalLoss:     Multi-class focal loss with per-class alpha weights.
               Penalizes easy examples and down-samples the majority class.
SoftFocalLoss: Variant that accepts soft (mixed) labels from PatchMix.
CrossEntropyLoss: Standard CE with class weights (baseline).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma:        Focusing parameter. 0 = cross-entropy. Recommended: 2.0.
        class_weights: Per-class alpha weights as a list or tensor.
                       Should be inversely proportional to class frequency.
                       Default: [1.0, 1.9, 3.9, 7.2] for ICBHI.
        reduction:    'mean' (default) or 'sum'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: list[float] | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None  # type: ignore[assignment]

    def forward(
        self,
        logits: torch.Tensor,  # (B, C) raw logits
        targets: torch.Tensor, # (B,) long class indices
    ) -> torch.Tensor:
        """Compute focal loss for a batch."""
        # Get class probabilities
        log_probs = F.log_softmax(logits, dim=-1)         # (B, C)
        probs     = torch.exp(log_probs)                   # (B, C)

        # Gather the probability for the true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # (B,)

        # Focal modulation
        focal_weight = (1.0 - pt) ** self.gamma

        # Class alpha weights
        if self.class_weights is not None:
            alpha_t = self.class_weights.to(logits.device)[targets]  # (B,)
        else:
            alpha_t = torch.ones_like(pt)

        loss = -alpha_t * focal_weight * log_pt  # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SoftFocalLoss(nn.Module):
    """
    Focal loss variant that accepts soft (mixed) labels from PatchMix.

    Targets are (B, C) float tensors summing to 1 instead of integer indices.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None  # type: ignore[assignment]

    def forward(
        self,
        logits: torch.Tensor,       # (B, C) raw logits
        soft_targets: torch.Tensor, # (B, C) float — soft labels
    ) -> torch.Tensor:
        """Compute soft focal loss."""
        probs    = F.softmax(logits, dim=-1)         # (B, C)
        log_prob = F.log_softmax(logits, dim=-1)     # (B, C)

        # Focal modulation per class
        focal = (1.0 - probs) ** self.gamma          # (B, C)

        if self.class_weights is not None:
            alpha = self.class_weights.to(logits.device).unsqueeze(0)  # (1, C)
        else:
            alpha = torch.ones_like(probs)

        # Soft cross-entropy with focal weight
        loss = -(alpha * focal * log_prob * soft_targets).sum(dim=-1)  # (B,)
        return loss.mean()


def build_loss(cfg) -> nn.Module:
    """Build the loss function from config."""
    loss_type = cfg.loss.type.lower()
    class_weights = list(cfg.classes.weights) if hasattr(cfg, "classes") else None

    if loss_type == "focal":
        return FocalLoss(gamma=cfg.loss.gamma, class_weights=class_weights)
    elif loss_type == "soft_focal":
        return SoftFocalLoss(gamma=cfg.loss.gamma, class_weights=class_weights)
    elif loss_type == "cross_entropy":
        if class_weights:
            w = torch.tensor(class_weights)
            return nn.CrossEntropyLoss(weight=w)
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")