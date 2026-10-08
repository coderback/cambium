"""DGF-1 input transform guards — ADR-011 clause 3, clause-5 test 7 (fit-on-train).

Synthetic DGraph-shaped fixture (shared with test_dgf1_datasource). The load-bearing property: the
fitted statistics depend on users existing by ``train_max`` and on edges dated up to it — **nothing
a later user or a later edge does can move them** — and they are fitted on the training view, then
frozen for every scoring view.
"""

from __future__ import annotations

import torch

from gbe.features import Standardizer
from adapters.dgf1.datasource_dgraph import graph_view
from adapters.dgf1.features import (
    INPUT_FEATURE_NAMES,
    constant_input_features,
    fit_input_transform,
    node_inputs,
)
from tests.test_dgf1_datasource import SPLIT_VAL, T, _data


def _same(a: Standardizer, b: Standardizer) -> bool:
    return torch.equal(a.mean, b.mean) and torch.equal(a.std, b.std) and torch.equal(a.constant, b.constant)


def test_inputs_are_raw_features_next_to_the_views_statistics():
    data = _data()
    view = graph_view(data, T)
    inp = node_inputs(data, view)
    assert inp.shape == (8, len(INPUT_FEATURE_NAMES)) == (8, 30)
    assert torch.equal(inp[:, :17], data.x)
    assert torch.equal(inp[:, 17:], view.node_features)


def test_perturbing_post_cutoff_users_does_not_move_the_fitted_statistics():
    """Clause-5 test 7: a later user's raw features cannot reach the transform."""
    data = _data()
    before = fit_input_transform(data, SPLIT_VAL)
    data.x[data.node_time > T] = data.x[data.node_time > T] * 1000 - 5
    assert _same(before, fit_input_transform(data, SPLIT_VAL))


def test_rewriting_post_cutoff_edges_does_not_move_the_fitted_statistics():
    """…nor can a later edge, through the view features of an earlier user."""
    data = _data()
    before = fit_input_transform(data, SPLIT_VAL)
    later = data.edge_time > T
    data.edge_type[later] = 9
    data.edge_time[later] = data.edge_time[later] + 1
    assert _same(before, fit_input_transform(data, SPLIT_VAL))


def test_the_transform_is_fitted_on_the_training_view_not_a_scoring_view():
    data = _data()
    fitted = fit_input_transform(data, SPLIT_VAL)
    on_train_view = Standardizer.fit(node_inputs(data, graph_view(data, T)), data.node_time <= T)
    on_end_view = Standardizer.fit(node_inputs(data, graph_view(data, 8)), data.node_time <= T)
    assert _same(fitted, on_train_view)
    assert not _same(fitted, on_end_view), "fixture cannot tell the two views apart"


def test_an_edge_type_absent_from_training_is_zeroed_at_scoring():
    """The DGraph type-8 case on the fixture: types 2 and 5 first appear after T, so their columns
    are constant in training and must read 0 in the scoring view — not (count - 0) / 1e-6."""
    data = _data()
    fitted = fit_input_transform(data, SPLIT_VAL)
    assert {"edge_type_2", "edge_type_5"} <= set(constant_input_features(fitted))

    scoring = node_inputs(data, graph_view(data, 8))
    col2 = INPUT_FEATURE_NAMES.index("edge_type_2")
    assert float(scoring[:, col2].max()) > 0                  # the raw count is alive at scoring
    out = fitted.transform(scoring)
    assert bool((out[:, col2] == 0).all())
    assert torch.isfinite(out).all()
