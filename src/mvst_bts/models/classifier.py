"""
classifier.py — Final classification head.

Takes the fused embedding (B, hidden_dim) and outputs (B, num_classes) logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """
    LayerNorm → Dropout → Linear classification head.

    Args:
        hidden_dim:  Input dimension (should match AST hidden_size = 768).
        dropout:     Dropout probability before the linear layer.
        num_classes: Number of output classes (4 for ICBHI).
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        dropout: float = 0.3,
        num_classes: int = 4,
    ):
        super().__init__()
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, hidden_dim) fused embedding.
        Returns:
            logits: (B, num_classes) raw class scores.
        """
        x = self.norm(x)
        x = self.dropout(x)
        return self.linear(x)