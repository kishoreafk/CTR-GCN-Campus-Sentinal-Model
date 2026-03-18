"""Tests for evaluation/tta.py"""
import torch
import torch.nn as nn
import pytest
from evaluation.tta import (
    apply_horizontal_flip, apply_temporal_reverse, TTAEvaluator, FLIP_PAIRS
)


class DummyModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(3 * 64 * 18 * 2, num_classes)

    def forward(self, x):
        # Flatten input and produce logits
        return self.fc(x.reshape(x.shape[0], -1))


def test_horizontal_flip_swaps_pairs():
    x = torch.randn(2, 3, 64, 18, 2)
    flipped = apply_horizontal_flip(x)

    # Check that left/right pairs are swapped for non-x channels (y, conf)
    for l, r in FLIP_PAIRS:
        # Y and confidence channels should be swapped directly
        torch.testing.assert_close(x[:, 1:, :, l, :], flipped[:, 1:, :, r, :])
        torch.testing.assert_close(x[:, 1:, :, r, :], flipped[:, 1:, :, l, :])

    # X coordinate should be inverted (1.0 - original)
    # For a joint not in any flip pair (joint 0 = nose), check x-flip
    torch.testing.assert_close(
        flipped[:, 0, :, 0, :],
        1.0 - x[:, 0, :, 0, :]
    )


def test_horizontal_flip_double_is_identity():
    x = torch.randn(2, 3, 64, 18, 2)
    double_flip = apply_horizontal_flip(apply_horizontal_flip(x))
    torch.testing.assert_close(x, double_flip)


def test_temporal_reverse():
    x = torch.randn(2, 3, 64, 18, 2)
    rev = apply_temporal_reverse(x)

    # First frame of reversed should be last frame of original
    torch.testing.assert_close(rev[:, :, 0, :, :], x[:, :, -1, :, :])
    torch.testing.assert_close(rev[:, :, -1, :, :], x[:, :, 0, :, :])


def test_temporal_reverse_double_is_identity():
    x = torch.randn(2, 3, 64, 18, 2)
    double_rev = apply_temporal_reverse(apply_temporal_reverse(x))
    torch.testing.assert_close(x, double_rev)


def test_tta_evaluator_output_shape():
    model = DummyModel(num_classes=5)
    model.eval()
    tta = TTAEvaluator(model, "cpu")
    x = torch.randn(4, 3, 64, 18, 2)
    probs = tta.predict(x)
    assert probs.shape == (4, 5)
    # Should be probabilities (0-1 range)
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_tta_differs_from_single_forward():
    model = DummyModel(num_classes=5)
    model.eval()

    x = torch.randn(2, 3, 64, 18, 2)

    # Single forward
    with torch.no_grad():
        single = torch.sigmoid(model(x))

    # TTA forward
    tta = TTAEvaluator(model, "cpu")
    tta_probs = tta.predict(x)

    # TTA should give different results from single forward
    assert not torch.allclose(single, tta_probs, atol=1e-4)


def test_tta_subset_augmentations():
    model = DummyModel(num_classes=3)
    model.eval()
    tta = TTAEvaluator(model, "cpu", augmentations=["original", "flip"])
    x = torch.randn(2, 3, 64, 18, 2)
    probs = tta.predict(x)
    assert probs.shape == (2, 3)
