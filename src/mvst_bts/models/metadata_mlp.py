"""
metadata_mlp.py — BTS-inspired metadata fusion module.

Encodes ICBHI patient/device metadata (age, sex, device, chest location)
into a vector that is added to the fused spectrogram representation.

This exploits information the baseline model completely ignores:
  - Recording device  → affects frequency response
  - Chest location    → correlates with expected pathology
  - Patient age/sex   → priors on disease prevalence
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MetadataMLP(nn.Module):
    """
    Encode patient/device metadata and project to `output_dim`.

    All categorical inputs are embedded with learned embedding tables.
    The embeddings are concatenated and passed through a 2-layer MLP.

    Args:
        age_bins:         Number of age bin categories (default 4).
        device_vocab_size: Number of recording devices (default 4).
        location_vocab_size: Number of chest positions (default 9).
        sex_vocab_size:   2 (M/F).
        embedding_dim:    Dimension of each categorical embedding (default 64).
        hidden_size:      MLP hidden layer size (default 256).
        output_dim:       Output dimension — should match AST hidden_size (768).
    """

    def __init__(
        self,
        age_bins: int = 4,
        device_vocab_size: int = 4,
        location_vocab_size: int = 9,
        sex_vocab_size: int = 2,
        embedding_dim: int = 64,
        hidden_size: int = 256,
        output_dim: int = 768,
    ):
        super().__init__()

        self.age_emb      = nn.Embedding(age_bins, embedding_dim)
        self.device_emb   = nn.Embedding(device_vocab_size, embedding_dim)
        self.location_emb = nn.Embedding(location_vocab_size, embedding_dim)
        self.sex_emb      = nn.Embedding(sex_vocab_size, embedding_dim)

        in_dim = 4 * embedding_dim  # 4 fields concatenated

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, metadata: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            metadata: dict with keys 'age_bin', 'sex', 'device', 'location'.
                      Each value is a (B,) long tensor.

        Returns:
            (B, output_dim) float tensor — metadata representation.
        """
        age_e  = self.age_emb(metadata["age_bin"])       # (B, E)
        dev_e  = self.device_emb(metadata["device"])     # (B, E)
        loc_e  = self.location_emb(metadata["location"]) # (B, E)
        sex_e  = self.sex_emb(metadata["sex"])           # (B, E)

        x = torch.cat([age_e, dev_e, loc_e, sex_e], dim=-1)  # (B, 4E)
        return self.mlp(x)                                     # (B, output_dim)