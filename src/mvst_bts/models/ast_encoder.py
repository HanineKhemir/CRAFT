"""
ast_encoder.py — Audio Spectrogram Transformer encoder wrapper.

Wraps the HuggingFace ASTModel (MIT/ast-finetuned-audioset-10-10-0.4593)
to accept our 128×1024 mel-spectrograms and return a CLS token embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import ASTConfig, ASTModel


class ASTEncoder(nn.Module):
    """
    Wraps the pretrained AST model to encode a single mel-spectrogram.

    The HuggingFace AST expects input_values of shape
    (batch, time_frames, freq_bins) with time_frames=1024, freq_bins=128.
    We reshape our (B, 1, H, W) spectrogram accordingly.

    Args:
        pretrained_checkpoint: HuggingFace model ID or local path.
        hidden_size:           Expected to be 768 for ViT-base AST.
        freeze_first_n_layers: Freeze the first N transformer encoder layers
                               to reduce overfitting on small ICBHI dataset.
                               0 means fine-tune all layers (recommended).
    """

    def __init__(
        self,
        pretrained_checkpoint: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        hidden_size: int = 768,
        freeze_first_n_layers: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.model = ASTModel.from_pretrained(pretrained_checkpoint)

        if freeze_first_n_layers > 0:
            self._freeze_layers(freeze_first_n_layers)

    def _freeze_layers(self, n: int) -> None:
        """Freeze the first n encoder layers of the AST."""
        for i, layer in enumerate(self.model.encoder.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: (B, 1, H, W) float32 mel-spectrogram,
                  where H=128 (mel bands) and W=1024 (time frames).

        Returns:
            cls_embedding: (B, hidden_size) CLS token embedding.
        """
        B = spec.shape[0]
        # AST expects (B, time_frames, freq_bins) = (B, 1024, 128)
        # Our spec is (B, 1, 128, 1024) → squeeze channel → transpose
        x = spec.squeeze(1)          # (B, 128, 1024)
        x = x.permute(0, 2, 1)      # (B, 1024, 128) = (B, time, freq)

        outputs = self.model(input_values=x)
        # outputs.last_hidden_state: (B, seq_len, hidden_size)
        # seq_len = n_patches + 1 (CLS token is at index 0)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        return cls_embedding