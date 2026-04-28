"""
gated_fusion.py — Gated multi-view fusion module.

Combines CLS embeddings from the fine and coarse AST encoder branches
using learned scalar gates. The gates are computed from the concatenation
of both embeddings so the model can decide dynamically:
  - "this sample has sharp transients → weight the fine branch more"
  - "this sample has sustained tones → weight the coarse branch more"
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Fuse N view embeddings using learned softmax gates.

    Args:
        input_dim: Dimensionality of each view embedding (e.g. 768).
        num_views: Number of branches to fuse (default 2: fine + coarse).
    """

    def __init__(self, input_dim: int = 768, num_views: int = 2):
        super().__init__()
        self.num_views = num_views
        self.input_dim = input_dim

        # Gate network: concatenated embeddings → num_views scalars
        self.gate = nn.Sequential(
            nn.Linear(input_dim * num_views, num_views),
        )

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: List of `num_views` tensors, each (B, input_dim).

        Returns:
            fused: (B, input_dim) weighted sum of the view embeddings.
            gates: (B, num_views) normalized gate weights (for analysis).
        """
        assert len(embeddings) == self.num_views, (
            f"Expected {self.num_views} embeddings, got {len(embeddings)}"
        )

        concat = torch.cat(embeddings, dim=-1)          # (B, input_dim * num_views)
        raw_gates = self.gate(concat)                   # (B, num_views)
        gates = torch.softmax(raw_gates, dim=-1)        # (B, num_views) — sums to 1

        # Weighted sum of embeddings
        stacked = torch.stack(embeddings, dim=1)        # (B, num_views, input_dim)
        gates_expanded = gates.unsqueeze(-1)            # (B, num_views, 1)
        fused = (stacked * gates_expanded).sum(dim=1)   # (B, input_dim)

        return fused, gates