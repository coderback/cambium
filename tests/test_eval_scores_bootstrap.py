"""Guards for score persistence and the paired bootstrap (ADR-012 clause 10).

Written in the same session as the code. Synthetic data only. The properties that matter: a score
file carries the ids its scores belong to and a hash of the *content*; and the paired bootstrap
preserves prevalence, reproduces from its pinned seed, and is tighter than the unpaired form
because both arms see the same resample.
"""

from __future__ import annotations

import numpy as np
import pytest

from gbe.eval import (
    assert_aligned,
    content_hash,
    load_scores,
    logit_interval,
    paired_bootstrap_difference,
    save_scores,
    straddles_zero,
)
from gbe.eval.bootstrap import resample_index


def _scored(n=4000, n_pos=200, seed=0, gap=0.6):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.int64)
    y[rng.choice(n, n_pos, replace=False)] = 1
    floor = rng.normal(size=n) + 1.2 * y
    model = 0.85 * floor + 0.53 * rng.normal(size=n) + gap * y   # correlated, slightly better
    ids = np.arange(n, dtype=np.int64)
    return ids, y, floor, model


# -- score files ------------------------------------------------------------------------------
def test_scores_round_trip_with_their_ids(tmp_path):
    ids, y, floor, _ = _scored()
    digest = save_scores(tmp_path / "r.npz", node_ids=ids, y_true=y, proba=floor)
    back = load_scores(tmp_path / "r.npz")
    assert np.array_equal(back["node_ids"], ids) and np.array_equal(back["y_true"], y)
    assert np.allclose(back["proba"], floor.astype(np.float32))
    assert len(digest) == 64


def test_hash_is_over_content_so_a_rewrite_reproduces_it(tmp_path):
    """A file hash would change on every rewrite (.npz is a zip and carries timestamps)."""
    ids, y, floor, _ = _scored()
    first = save_scores(tmp_path / "a.npz", node_ids=ids, y_true=y, proba=floor)
    second = save_scores(tmp_path / "b.npz", node_ids=ids, y_true=y, proba=floor)
    assert first == second


def test_hash_changes_when_any_value_changes():
    ids, y, floor, _ = _scored()
    base = content_hash(ids, y, floor)
    moved = floor.copy()
    moved[7] = np.float32(moved[7]) + np.float32(1e-3)
    assert content_hash(ids, y, moved) != base
    assert content_hash(ids[::-1], y, floor) != base


def test_mismatched_shapes_are_rejected(tmp_path):
    ids, y, floor, _ = _scored()
    with pytest.raises(ValueError, match="same length"):
        save_scores(tmp_path / "x.npz", node_ids=ids[:-1], y_true=y, proba=floor)


def test_alignment_is_asserted_before_any_pairing():
    ids, y, floor, model = _scored()
    a = {"node_ids": ids, "y_true": y, "proba": floor}
    assert_aligned(a, {"node_ids": ids, "y_true": y, "proba": model})
    with pytest.raises(AssertionError, match="different nodes"):
        assert_aligned(a, {"node_ids": ids[::-1], "y_true": y, "proba": model})
    with pytest.raises(AssertionError, match="labels"):
        assert_aligned(a, {"node_ids": ids, "y_true": 1 - y, "proba": model})


# -- paired bootstrap -------------------------------------------------------------------------
def test_reproduces_from_its_pinned_seed():
    _, y, floor, model = _scored()
    kw = dict(n_resamples=100, seed=0)
    first = paired_bootstrap_difference(y, floor, model, **kw)
    assert first == paired_bootstrap_difference(y, floor, model, **kw)
    assert first != paired_bootstrap_difference(y, floor, model, n_resamples=100, seed=1)


def test_stratified_resampling_holds_the_positive_count_exactly():
    """AUPRC's chance level *is* the prevalence, so a replicate that changes the positive count
    changes the quantity being estimated. Tested on the resampler directly: an end-to-end test
    cannot see this, because at 5% prevalence an unstratified resample almost never loses a class
    and so produces no visible symptom (a mutation check caught that weakness)."""
    _, y, _, _ = _scored()
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(0)

    counts = {np.isin(resample_index(rng, pos, neg), pos).sum() for _ in range(50)}
    assert counts == {pos.size}, "stratified resampling must carry exactly the original positives"

    loose = {np.isin(resample_index(rng, pos, neg, stratified=False), pos).sum() for _ in range(50)}
    assert len(loose) > 1 and loose != {pos.size}, "fixture cannot tell the two schemes apart"


def test_every_replicate_contributes_under_stratification():
    _, y, floor, model = _scored()
    out = paired_bootstrap_difference(y, floor, model, n_resamples=50, seed=0)
    assert out["auprc"]["n_resamples"] == 50


def test_pairing_is_tighter_than_scoring_the_arms_independently():
    """The point of pairing: the shared 'which users are hard' component cancels."""
    _, y, floor, model = _scored()
    paired = paired_bootstrap_difference(y, floor, model, n_resamples=200, seed=0)["auprc"]["se"]

    rng = np.random.default_rng(0)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    from sklearn.metrics import average_precision_score
    a_vals, b_vals = [], []
    for _ in range(200):                       # independent resamples per arm
        ia = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        ib = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        a_vals.append(average_precision_score(y[ia], floor[ia]))
        b_vals.append(average_precision_score(y[ib], model[ib]))
    unpaired = float(np.sqrt(np.var(a_vals, ddof=1) + np.var(b_vals, ddof=1)))
    assert paired < unpaired


def test_identical_arms_give_a_zero_difference():
    _, y, floor, _ = _scored()
    out = paired_bootstrap_difference(y, floor, floor, n_resamples=50, seed=0)
    for metric in ("auc", "auprc"):
        assert out[metric]["mean"] == 0.0 and out[metric]["se"] == 0.0
        assert straddles_zero(out[metric])


def test_sign_convention_is_model_minus_floor():
    _, y, floor, model = _scored(gap=1.5)      # model clearly better
    out = paired_bootstrap_difference(y, floor, model, n_resamples=100, seed=0)
    assert out["auprc"]["mean"] > 0
    assert not straddles_zero(out["auprc"])
    flipped = paired_bootstrap_difference(y, model, floor, n_resamples=100, seed=0)
    assert flipped["auprc"]["mean"] < 0


def test_degenerate_inputs_are_rejected():
    _, y, floor, model = _scored()
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_difference(y[:-1], floor, model)
    with pytest.raises(ValueError, match="at least 2 positives"):
        paired_bootstrap_difference(np.zeros_like(y), floor, model)


# -- Boyd logit interval ------------------------------------------------------------------------
def test_logit_interval_brackets_the_estimate_and_stays_in_range():
    lo, hi = logit_interval(0.30, n_pos=2717)
    assert 0.0 < lo < 0.30 < hi < 1.0


def test_logit_interval_widens_as_positives_get_scarcer():
    wide = logit_interval(0.30, n_pos=50)
    narrow = logit_interval(0.30, n_pos=2717)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_logit_interval_rejects_impossible_inputs():
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="strictly in"):
            logit_interval(bad, n_pos=100)
    with pytest.raises(ValueError, match="n_pos"):
        logit_interval(0.3, n_pos=0)
