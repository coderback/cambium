"""DGF-1 trainer guards — ADR-011 clauses 2-4, ADR-012 clauses 1/3/10.

Synthetic DGraph-shaped fixture (shared with the floor tests). The load-bearing properties:
training sees only the training view, the three scoring views really differ from **one** set of
weights, scored vectors line up with the ids they are saved against, and the reported views can
never be confused with the gated one in a registry row.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from gbe.eval import load_scores
from gbe.features.scaling import Standardizer
from gbe.gnn import GNNHParams, build_model
from gbe.run import resolve_device
from gbe.run.seeding import seed_everything
from adapters.dgf1.datasource_dgraph import graph_view, window_target_mask
from adapters.dgf1.eval import dgf1_base_config
from adapters.dgf1.features import fit_input_transform, node_inputs
from adapters.dgf1.train_gnn import (
    REPORTED_VIEWS,
    SCORING_BATCH_SIZE,
    evaluate_view,
    run_dgf1,
    score_view,
    train_dgf1,
    window_only_edges,
)
from tests.test_dgf1_floor import SPLIT_TEST, SPLIT_VAL, _synthetic

HP = GNNHParams(num_layers=2, hidden_dim=16, fan_out=(5, 5), epochs=2, batch_size=64)
DEV = resolve_device("cpu")


def _trained(data, split=SPLIT_VAL, seed=0):
    seed_everything(seed)
    return train_dgf1(data, split, HP, DEV)


# -- training sees only the training view -----------------------------------------------------
def test_training_uses_the_training_view_and_its_own_transform():
    data = _synthetic()
    _, transform, train_view = _trained(data)
    expected = fit_input_transform(data, SPLIT_VAL)
    assert torch.equal(transform.mean, expected.mean) and torch.equal(transform.std, expected.std)
    assert int(train_view.edge_time.max()) <= SPLIT_VAL.train_max, "training view reached later edges"


def test_later_edges_and_later_users_cannot_change_training_inputs():
    """The whole ADR-011 protocol in one assertion: perturb everything after the cutoff, and the
    transform the model trains on must not move."""
    data = _synthetic()
    _, before, _ = _trained(data)
    later = data.edge_time > SPLIT_VAL.train_max
    data.edge_type[later] = 9
    data.x[data.node_time > SPLIT_VAL.train_max] += 100.0
    _, after, _ = _trained(data)
    assert torch.equal(before.mean, after.mean) and torch.equal(before.std, after.std)


# -- the three views --------------------------------------------------------------------------
def test_scoring_uses_the_transform_it_is_handed_and_never_refits_on_the_view():
    """ADR-011 clause 3's other half. `train_dgf1` returning a frozen transform is worth nothing
    if `score_view` quietly refits on the scoring view — a mutation check found exactly that hole,
    because every other test only inspects the transform, not its use."""
    data = _synthetic()
    model, frozen, _ = _trained(data, SPLIT_TEST)
    scored = score_view(model, data, SPLIT_TEST, frozen, HP, DEV, "gated", batch_size=16)

    # a transform fitted on the scoring view is a *different* object; scores must move with it,
    # which is only possible if score_view actually applies the one it is given
    leaky = Standardizer.fit(
        node_inputs(data, graph_view(data, SPLIT_TEST.test_max)),
        data.node_time <= SPLIT_TEST.test_max,
    )
    assert not torch.equal(frozen.mean, leaky.mean), "fixture cannot tell the transforms apart"
    other = score_view(model, data, SPLIT_TEST, leaky, HP, DEV, "gated", batch_size=16)
    assert not np.allclose(scored["proba"], other["proba"])

    # and the frozen path is stable across calls
    again = score_view(model, data, SPLIT_TEST, frozen, HP, DEV, "gated", batch_size=16)
    assert np.array_equal(scored["proba"], again["proba"])


def test_each_probability_belongs_to_its_own_node_id():
    """Clause 10's point: ids are stored so a score can be attributed to a user. Checking that the
    id *vector* is right does not check that the probabilities are in the same order — permuting
    the targets must permute the scores identically."""
    data = _synthetic()
    model, transform, _ = _trained(data, SPLIT_TEST)
    scored = score_view(model, data, SPLIT_TEST, transform, HP, DEV, "gated", batch_size=16)

    order = np.argsort(-scored["proba"])[:20]          # a deterministic, non-trivial subset
    subset = torch.as_tensor(scored["node_ids"][order])
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.data import Data as PygData

    view = graph_view(data, SPLIT_TEST.test_max)
    x = transform.transform(node_inputs(data, view))
    loader = NeighborLoader(
        PygData(x=x, edge_index=view.edge_index, num_nodes=data.num_nodes),
        num_neighbors=[-1] * HP.num_layers, input_nodes=subset, batch_size=len(subset),
        shuffle=False, num_workers=0,
    )
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        _, logits = model(batch.x, batch.edge_index)
        direct = torch.softmax(logits[: batch.batch_size], dim=-1)[:, 1].numpy()
    assert np.allclose(direct, scored["proba"][order], atol=1e-6), "scores are not aligned to ids"


def test_all_three_views_score_the_same_targets_in_the_same_order():
    data = _synthetic()
    model, transform, _ = _trained(data, SPLIT_TEST)
    targets = window_target_mask(data, SPLIT_TEST).nonzero().view(-1).numpy()
    for kind in ("gated",) + REPORTED_VIEWS:
        scored = score_view(model, data, SPLIT_TEST, transform, HP, DEV, kind, batch_size=16)
        assert np.array_equal(scored["node_ids"], targets), f"{kind} mis-aligned its ids"
        assert scored["proba"].shape == targets.shape


def test_the_views_actually_differ_from_identical_weights():
    """If they agreed, the sensitivity rows would be decoration."""
    data = _synthetic()
    model, transform, _ = _trained(data, SPLIT_TEST)
    out = {k: score_view(model, data, SPLIT_TEST, transform, HP, DEV, k, batch_size=16)["proba"]
           for k in ("gated",) + REPORTED_VIEWS}
    assert not np.allclose(out["gated"], out["window_only"])
    assert not np.allclose(out["gated"], out["first_appearance"])


def test_window_only_keeps_just_the_edges_inside_the_window():
    data = _synthetic()
    view = graph_view(data, SPLIT_TEST.test_max)
    kept = window_only_edges(data, view, SPLIT_TEST)
    t = data.node_time
    assert bool(((t[kept[0]] >= SPLIT_TEST.test_min) & (t[kept[1]] >= SPLIT_TEST.test_min)).all())
    assert kept.size(1) < view.edge_index.size(1), "fixture has no cross-window edges to drop"


def test_an_unknown_view_is_rejected():
    data = _synthetic()
    model, transform, _ = _trained(data)
    with pytest.raises(ValueError, match="unknown view"):
        score_view(model, data, SPLIT_VAL, transform, HP, DEV, "as_of_paper_submission")


def test_scoring_batch_size_is_independent_of_the_tuned_training_knob():
    """ADR-012 tunes `batch_size` for training; scoring must not inherit it silently."""
    assert SCORING_BATCH_SIZE == 1024
    data = _synthetic()
    model, transform, _ = _trained(data, SPLIT_TEST)
    a = score_view(model, data, SPLIT_TEST, transform, HP, DEV, "gated", batch_size=8)["proba"]
    b = score_view(model, data, SPLIT_TEST, transform, HP, DEV, "gated", batch_size=64)["proba"]
    assert np.allclose(a, b, atol=1e-6), "scores depend on the scoring batch size"


# -- metrics and the registry row ---------------------------------------------------------------
def test_metrics_carry_prevalence_and_the_positive_count():
    data = _synthetic()
    model, transform, _ = _trained(data, SPLIT_TEST)
    m = evaluate_view(score_view(model, data, SPLIT_TEST, transform, HP, DEV, "gated", batch_size=16))
    for key in ("fraud_auc", "fraud_auprc", "prevalence", "n_score", "n_score_fraud"):
        assert key in m
    assert m["prevalence"] == pytest.approx(m["n_score_fraud"] / m["n_score"])


def test_one_run_writes_one_row_with_gated_and_prefixed_reported_metrics(tmp_path):
    registry, scores = tmp_path / "registry.csv", tmp_path / "scores"
    seed_everything(0)
    run_id, logged = run_dgf1(0, _synthetic(), SPLIT_VAL, HP, dgf1_base_config(),
                              registry_path=registry, device="cpu", scores_dir=scores)
    rows = list(csv.DictReader(registry.open(encoding="utf-8", newline="")))
    assert len(rows) == 1 and rows[0]["model"] == "dgf1"
    row = json.loads(rows[0]["metrics_json"])

    # The expected keys are written out rather than derived from REPORTED_VIEWS: looping the same
    # constant the code uses makes this assertion vacuous the moment that constant is emptied,
    # which a mutation check demonstrated.
    assert "fraud_auprc" in row and "fraud_auc" in row          # gated, bare
    for key in ("window_only_fraud_auprc", "first_appearance_fraud_auprc",
                "window_only_prevalence", "first_appearance_prevalence"):
        assert key in row, f"reported view key {key} missing — a sensitivity would be unreadable"
    assert set(REPORTED_VIEWS) == {"window_only", "first_appearance"}
    assert row["arm"] == "dgf1-parity" and row["window"] == "9-12"

    saved = load_scores(scores / f"{run_id}.npz")
    assert np.array_equal(saved["node_ids"], score_ids(_synthetic(), SPLIT_VAL))
    assert row["scores_sha256"] and len(row["scores_sha256"]) == 64


def score_ids(data, split):
    return window_target_mask(data, split).nonzero().view(-1).numpy()
