"""Unit tests for loss functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from mvst_bts.training.losses import FocalLoss, SoftFocalLoss


def test_focal_loss_perfect_prediction():
    """With perfect predictions, focal loss should be near zero."""
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
    labels = torch.tensor([0])
    loss = loss_fn(logits, labels)
    assert loss.item() < 0.01, f"Expected near-zero loss, got {loss.item()}"


def test_focal_loss_shape():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 4)
    labels = torch.randint(0, 4, (8,))
    loss = loss_fn(logits, labels)
    assert loss.shape == (), "Loss should be a scalar"


def test_focal_loss_gradient_flows():
    loss_fn = FocalLoss(gamma=2.0, class_weights=[1.0, 1.9, 3.9, 7.2])
    logits = torch.randn(8, 4, requires_grad=True)
    labels = torch.randint(0, 4, (8,))
    loss = loss_fn(logits, labels)
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_soft_focal_loss_shape():
    loss_fn = SoftFocalLoss(gamma=2.0)
    logits = torch.randn(8, 4)
    soft_labels = torch.softmax(torch.randn(8, 4), dim=-1)
    loss = loss_fn(logits, soft_labels)
    assert loss.shape == ()


def test_focal_weight_higher_for_minority():
    """Loss should be higher for minority class (weight=7.2) than majority (weight=1.0)."""
    weights = [1.0, 1.9, 3.9, 7.2]
    loss_fn = FocalLoss(gamma=2.0, class_weights=weights)

    # Same logits, different target class
    logits = torch.zeros(1, 4)
    loss_majority = loss_fn(logits, torch.tensor([0]))  # normal, weight=1.0
    loss_minority = loss_fn(logits, torch.tensor([3]))  # both, weight=7.2

    assert loss_minority.item() > loss_majority.item(), (
        "Minority class loss should be higher than majority class loss"
    )