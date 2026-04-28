"""
rep_augment.py — RepAugment: latent-space minority class augmentation.

Two operations applied to the fused embedding before classification:
  1. Rep-Mask: randomly zeros out a fraction of embedding dimensions.
  2. Rep-Gen:  generates synthetic minority embeddings by interpolating
               between two real minority-class embeddings (latent SMOTE).

Applied during training only (model.training == True).
"""
from __future__ import annotations

import random
import torch
import torch.nn as nn


class RepAugment(nn.Module):
    """
    RepAugment applied in the latent space.

    Args:
        mask_rate:    Fraction of embedding dims to zero out (default 0.2).
        gen_alpha:    Beta(alpha, alpha) parameter for interpolation (default 0.4).
        minority_classes: Class indices to target for Rep-Gen (default [1,2,3]).
    """

    def __init__(
        self,
        mask_rate: float = 0.20,
        gen_alpha: float = 0.40,
        minority_classes: list[int] | None = None,
    ):
        super().__init__()
        self.mask_rate = mask_rate
        self.gen_alpha = gen_alpha
        self.minority_classes = minority_classes or [1, 2, 3]  # crackle, wheeze, both

    def forward(
        self,
        embeddings: torch.Tensor,   # (B, D)
        labels: torch.Tensor,       # (B,) long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Rep-Mask and optionally Rep-Gen to the batch.

        Returns:
            aug_embeddings: (B + n_synth, D) — original + synthetic
            aug_labels:     (B + n_synth,) long
        """
        if not self.training:
            return embeddings, labels

        # ── Rep-Mask ──────────────────────────────────
        masked = self._rep_mask(embeddings)

        # ── Rep-Gen ───────────────────────────────────
        synth_embs, synth_labels = self._rep_gen(masked, labels)

        if synth_embs is not None:
            aug_embeddings = torch.cat([masked, synth_embs], dim=0)
            aug_labels     = torch.cat([labels, synth_labels], dim=0)
        else:
            aug_embeddings = masked
            aug_labels     = labels

        return aug_embeddings, aug_labels

    def _rep_mask(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Randomly zero out mask_rate fraction of each embedding vector."""
        mask = torch.bernoulli(
            torch.full_like(embeddings, 1.0 - self.mask_rate)
        )
        return embeddings * mask

    def _rep_gen(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Generate synthetic embeddings by interpolating between two minority samples.
        Returns None if no minority class has at least 2 samples in the batch.
        """
        synth_list, label_list = [], []

        for cls in self.minority_classes:
            idx = (labels == cls).nonzero(as_tuple=True)[0]
            if len(idx) < 2:
                continue

            # Sample two distinct indices from this class
            i, j = random.sample(idx.tolist(), 2)
            alpha = float(torch.distributions.Beta(self.gen_alpha, self.gen_alpha).sample())
            synth = alpha * embeddings[i] + (1.0 - alpha) * embeddings[j]
            synth_list.append(synth.unsqueeze(0))
            label_list.append(torch.tensor([cls], device=labels.device, dtype=labels.dtype))

        if not synth_list:
            return None, None

        return torch.cat(synth_list, dim=0), torch.cat(label_list, dim=0)