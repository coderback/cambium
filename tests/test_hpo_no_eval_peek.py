"""Integrity guard: the HPO sweep can never tune against the 35-49 held-out eval.

The constitution forbids tuning against the held-out test window. The whole defense is that
the sweep's inner split stops at step 34 — this test pins that so a later edit to the split
cannot silently start leaking the eval into hyperparameter selection.
"""

from __future__ import annotations

import torch

from gbe.eval import split_masks
from adapters.ell1.hpo import INNER_SPLIT


def test_inner_split_stops_at_34():
    assert INNER_SPLIT.test_max <= 34
    assert (INNER_SPLIT.train_max, INNER_SPLIT.test_min, INNER_SPLIT.test_max) == (29, 30, 34)


def test_no_hpo_mask_touches_the_35_49_window():
    ts = torch.arange(1, 50)  # steps 1..49
    train_mask, val_mask = split_masks(ts, INNER_SPLIT)
    assert bool((ts[train_mask] <= 29).all())
    assert bool((ts[val_mask] >= 30).all()) and bool((ts[val_mask] <= 34).all())
    # nothing at step >= 35 is used for either tuning-train or validation
    assert not bool((train_mask | val_mask)[ts >= 35].any()), "HPO reaches the 35-49 eval window"
