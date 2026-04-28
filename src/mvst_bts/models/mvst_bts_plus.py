"""
mvst_bts_plus.py — MVST-BTS+ full model.

Assembly of:
  Fine AST encoder   (high temporal resolution branch)
  Coarse AST encoder (high frequency resolution branch)
  MetadataMLP        (patient/device context)
  GatedFusion        (learned branch weighting)
  ClassifierHead     (4-class output)

Forward pass returns logits (B, 4) and gate weights (B, 2) for analysis.
"""
from __future__ import annotations

from omegaconf import DictConfig
import torch
import torch.nn as nn

from mvst_bts.models.ast_encoder import ASTEncoder
from mvst_bts.models.metadata_mlp import MetadataMLP
from mvst_bts.models.gated_fusion import GatedFusion
from mvst_bts.models.classifier import ClassifierHead


class MVSTBTSPlus(nn.Module):
    """
    Multi-View Spectrogram Transformer with BTS metadata fusion.

    Args:
        cfg: model config sub-dict (cfg.model from the full config).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        ast_cfg   = cfg.ast
        meta_cfg  = cfg.metadata_mlp
        fuse_cfg  = cfg.gated_fusion
        cls_cfg   = cfg.classifier

        # ── Fine branch encoder ────────────────────────
        self.encoder_fine = ASTEncoder(
            pretrained_checkpoint=ast_cfg.pretrained_checkpoint,
            hidden_size=ast_cfg.hidden_size,
            freeze_first_n_layers=ast_cfg.freeze_first_n_layers,
        )

        # ── Coarse branch encoder ──────────────────────
        # Independent weights by default (shared_weights=False)
        if ast_cfg.shared_weights:
            self.encoder_coarse = self.encoder_fine   # share weights
        else:
            self.encoder_coarse = ASTEncoder(
                pretrained_checkpoint=ast_cfg.pretrained_checkpoint,
                hidden_size=ast_cfg.hidden_size,
                freeze_first_n_layers=ast_cfg.freeze_first_n_layers,
            )

        num_views = fuse_cfg.get("num_views", 2)

        # ── Metadata MLP (optional) ────────────────────
        self.use_metadata = meta_cfg.get("enabled", True) and num_views >= 2
        if self.use_metadata:
            self.metadata_mlp = MetadataMLP(
                age_bins=meta_cfg.get("age_bins", 4),
                device_vocab_size=meta_cfg.get("device_vocab_size", 4),
                location_vocab_size=meta_cfg.get("location_vocab_size", 9),
                sex_vocab_size=meta_cfg.get("sex_vocab_size", 2),
                embedding_dim=meta_cfg.get("embedding_dim", 64),
                hidden_size=meta_cfg.get("hidden_size", 256),
                output_dim=meta_cfg.get("output_dim", ast_cfg.hidden_size),
            )
            # Projection to align metadata with embedding dim
            self.meta_proj = nn.Linear(
                meta_cfg.get("output_dim", ast_cfg.hidden_size),
                ast_cfg.hidden_size,
            )

        # ── Gated fusion ────────────────────────────────
        self.gated_fusion = GatedFusion(
            input_dim=fuse_cfg.input_dim,
            num_views=num_views,
        )

        # ── Classifier ──────────────────────────────────
        self.classifier = ClassifierHead(
            hidden_dim=cls_cfg.hidden_dim,
            dropout=cls_cfg.dropout,
            num_classes=cls_cfg.num_classes,
        )

    def forward(
        self,
        fine_spec: torch.Tensor,
        coarse_spec: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            fine_spec:   (B, 1, H, W) fine-resolution mel-spectrogram.
            coarse_spec: (B, 1, H, W) coarse-resolution mel-spectrogram.
            metadata:    Dict of (B,) long tensors: age_bin, sex, device, location.

        Returns:
            dict with:
                logits: (B, 4) raw class scores
                gates:  (B, 2) learned branch weights (for analysis/logging)
                h_fine, h_coarse: (B, 768) per-branch embeddings
        """
        # ── Encode both branches ────────────────────────
        h_fine   = self.encoder_fine(fine_spec)     # (B, 768)
        h_coarse = self.encoder_coarse(coarse_spec) # (B, 768)

        # ── Add metadata residual ───────────────────────
        if self.use_metadata and metadata is not None:
            meta_vec = self.metadata_mlp(metadata)  # (B, 768)
            # Residual addition: enriches the average embedding with context
            meta_ctx = self.meta_proj(meta_vec)     # (B, 768)
            h_fine   = h_fine   + 0.5 * meta_ctx
            h_coarse = h_coarse + 0.5 * meta_ctx

        # ── Gated fusion ────────────────────────────────
        fused, gates = self.gated_fusion([h_fine, h_coarse])  # (B, 768), (B, 2)

        # ── Classify ────────────────────────────────────
        logits = self.classifier(fused)  # (B, 4)

        return {
            "logits":   logits,
            "gates":    gates,
            "h_fine":   h_fine,
            "h_coarse": h_coarse,
            "fused":    fused,
        }