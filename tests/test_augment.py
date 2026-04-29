"""Unit tests for augmentation modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytest

from mvst_bts.augmentation.spec_augment import SpecAugment
from mvst_bts.augmentation.patch_mix import PatchMixBatch
from mvst_bts.augmentation.rep_augment import RepAugment


B, H, W = 8, 128, 1024


def make_specs(b=B):
    fine   = torch.rand(b, 1, H, W)
    coarse = torch.rand(b, 1, H, W)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3][:b])
    return fine, coarse, labels


# ── SpecAugment ─────────────────────────────────────

def test_spec_augment_output_shape():
    aug = SpecAugment()
    spec = torch.rand(1, H, W)
    sample = {"fine_spec": spec.clone(), "coarse_spec": spec.clone()}
    out = aug(sample)
    assert out["fine_spec"].shape == (1, H, W)
    assert out["coarse_spec"].shape == (1, H, W)

def test_spec_augment_introduces_zeros():
    """After augmentation, some values should be zeroed out."""
    aug = SpecAugment(time_mask_max_ratio=0.5, freq_mask_max_ratio=0.5,
                      num_time_masks=3, num_freq_masks=3)
    spec = torch.ones(1, H, W)
    sample = {"fine_spec": spec.clone(), "coarse_spec": spec.clone()}
    out = aug(sample)
    assert out["fine_spec"].min().item() == 0.0


# ── PatchMix ─────────────────────────────────────────

def test_patchmix_output_shapes():
    mixer = PatchMixBatch(patch_size=16, prob=1.0)
    fine, coarse, labels = make_specs()
    f_out, c_out, soft = mixer(fine, coarse, labels)
    assert f_out.shape == fine.shape
    assert c_out.shape == coarse.shape
    assert soft.shape == (B, 4)

def test_patchmix_soft_labels_sum_to_one():
    mixer = PatchMixBatch(patch_size=16, prob=1.0)
    fine, coarse, labels = make_specs()
    _, _, soft = mixer(fine, coarse, labels)
    sums = soft.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), \
        f"Soft label rows should sum to 1, got {sums}"

def test_patchmix_no_mix_at_zero_prob():
    mixer = PatchMixBatch(patch_size=16, prob=0.0)
    fine, coarse, labels = make_specs()
    f_out, c_out, soft = mixer(fine, coarse, labels)
    assert torch.allclose(f_out, fine)
    # With prob=0, soft labels should be one-hot
    expected_soft = torch.nn.functional.one_hot(labels, 4).float()
    assert torch.allclose(soft, expected_soft)


# ── RepAugment ───────────────────────────────────────

def test_rep_augment_training_mode():
    aug = RepAugment(mask_rate=0.2)
    aug.train()
    embs = torch.randn(B, 768)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    out_embs, out_labels = aug(embs, labels)
    # Should produce at least B embeddings (B + synthetics)
    assert out_embs.shape[0] >= B
    assert out_labels.shape[0] == out_embs.shape[0]

def test_rep_augment_eval_mode_passthrough():
    aug = RepAugment(mask_rate=0.2)
    aug.eval()
    embs = torch.randn(B, 768)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    out_embs, out_labels = aug(embs, labels)
    # In eval mode: no augmentation, no shape change
    assert out_embs.shape == embs.shape
    assert torch.equal(out_labels, labels)

def test_rep_augment_mask_changes_values():
    aug = RepAugment(mask_rate=0.9)   # very high mask rate
    aug.train()
    embs = torch.ones(B, 768)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    out_embs, _ = aug(embs, labels)
    # At mask_rate=0.9, most values should be 0
    zero_frac = (out_embs[:B] == 0).float().mean().item()
    assert zero_frac > 0.5, f"Expected >50% zeros at mask_rate=0.9, got {zero_frac:.2f}"