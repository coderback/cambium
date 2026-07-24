"""Contract tests for the tabular FeatureEncoder (gbe.features)."""

from __future__ import annotations

import torch

from gbe.features.encoder import TabularMLPEncoder
from gbe.run.seeding import seed_everything


def test_shape_and_out_dim():
    enc = TabularMLPEncoder(in_dim=165, hidden_dim=64, out_dim=32, num_layers=2)
    x = torch.randn(10, 165)
    out = enc(x)
    assert out.shape == (10, 32)
    assert enc.out_dim == 32


def test_single_layer_is_plain_projection():
    enc = TabularMLPEncoder(in_dim=8, hidden_dim=16, out_dim=4, num_layers=1)
    assert enc(torch.randn(5, 8)).shape == (5, 4)


def test_deterministic_under_seed():
    seed_everything(0)
    enc1 = TabularMLPEncoder(8, 16, 4)
    seed_everything(0)
    enc2 = TabularMLPEncoder(8, 16, 4)
    enc1.eval(), enc2.eval()  # drop dropout noise
    x = torch.randn(3, 8)
    assert torch.equal(enc1(x), enc2(x))


def test_rejects_zero_layers():
    try:
        TabularMLPEncoder(8, 16, 4, num_layers=0)
    except ValueError:
        return
    raise AssertionError("num_layers=0 must be rejected")
