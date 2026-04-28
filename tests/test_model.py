"""
test_model.py — Verify model components produce correct output shapes
without needing real data or GPU.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from omegaconf import OmegaConf

from mvst_bts.models.gated_fusion import GatedFusion
from mvst_bts.models.metadata_mlp import MetadataMLP
from mvst_bts.models.classifier import ClassifierHead


B = 4   # batch size
D = 768 # hidden dim


def test_gated_fusion_output_shape():
    fusion = GatedFusion(input_dim=D, num_views=2)
    h1 = torch.randn(B, D)
    h2 = torch.randn(B, D)
    fused, gates = fusion([h1, h2])
    assert fused.shape == (B, D), f"Expected ({B},{D}), got {fused.shape}"
    assert gates.shape == (B, 2), f"Expected ({B},2), got {gates.shape}"
    # Gates should sum to 1
    assert torch.allclose(gates.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_metadata_mlp_output_shape():
    mlp = MetadataMLP(output_dim=D)
    meta = {
        "age_bin":  torch.randint(0, 4, (B,)),
        "sex":      torch.randint(0, 2, (B,)),
        "device":   torch.randint(0, 4, (B,)),
        "location": torch.randint(0, 9, (B,)),
    }
    out = mlp(meta)
    assert out.shape == (B, D), f"Expected ({B},{D}), got {out.shape}"


def test_classifier_head_output_shape():
    head = ClassifierHead(hidden_dim=D, dropout=0.1, num_classes=4)
    head.eval()
    x = torch.randn(B, D)
    logits = head(x)
    assert logits.shape == (B, 4), f"Expected ({B},4), got {logits.shape}"


def test_gated_fusion_single_view():
    fusion = GatedFusion(input_dim=D, num_views=1)
    h = torch.randn(B, D)
    fused, gates = fusion([h])
    assert fused.shape == (B, D)
    assert torch.allclose(fused, h, atol=1e-5)  # single view → passthrough