"""DGF-1 parity floor guards — ADR-011 clause 4, clause-5 test 8; ADR-007 metric discipline.

Synthetic fixtures only (the snapshot is gitignored). The load-bearing property is **parity**: the
floor's columns are the GNN's input columns — same function, same views, same node sets — so any
GNN–floor gap is message passing, not information one arm was denied.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from gbe.eval import TemporalSplit
from adapters.dgf1.baselines_tabular import (
    FEATURE_SETS,
    evaluate,
    fit_predict,
    floor_design,
    run_floor,
)
from adapters.dgf1.datasource_dgraph import graph_view, prepare_dgraph, train_seed_mask, window_target_mask
from adapters.dgf1.eval import dgf1_base_config
from adapters.dgf1.features import INPUT_FEATURE_NAMES, fit_input_transform, node_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_dgf1_floor import require_accepted_preregistration  # noqa: E402

SPLIT_VAL = TemporalSplit(train_max=8, test_min=9, test_max=12)
SPLIT_TEST = TemporalSplit(train_max=8, test_min=13, test_max=20)


def _synthetic(n: int = 400, seed: int = 0):
    """A DGraph-shaped random graph: 20 steps, types 1..11, two background values, -1 sentinels,
    a ring so nobody is isolated, and a weak fraud signal in x[:, 0] so metrics are non-trivial."""
    g = torch.Generator().manual_seed(seed)
    ring_src = torch.arange(n)
    ring_dst = (ring_src + 1) % n
    extra = torch.randint(0, n, (2, 2 * n), generator=g)
    extra = extra[:, extra[0] != extra[1]]
    ei = torch.cat([torch.stack([ring_src, ring_dst]), extra], dim=1)
    et = torch.randint(1, 21, (ei.size(1),), generator=g)
    ety = torch.randint(1, 12, (ei.size(1),), generator=g)
    y = (torch.rand(n, generator=g) < 0.35).long()
    bg = torch.rand(n, generator=g) < 0.15
    y[bg] = torch.randint(2, 4, (int(bg.sum()),), generator=g)
    x = torch.randn(n, 17, generator=g)
    x[:, 0] += 1.5 * (y == 1).float()
    x[torch.rand(n, 17, generator=g) < 0.3] = -1.0
    data = prepare_dgraph(x, ei, et, ety, y)
    for split in (SPLIT_VAL, SPLIT_TEST):   # the fixture must exercise both classes everywhere
        for mask in (train_seed_mask(data, split), window_target_mask(data, split)):
            assert set(data.y[mask].tolist()) == {0, 1}
    return data


# -- clause-5 test 8: parity ------------------------------------------------------------------
@pytest.mark.parametrize("split", [SPLIT_VAL, SPLIT_TEST])
def test_parity_floor_columns_are_exactly_the_gnn_input_columns(split):
    data = _synthetic()
    d = floor_design(data, split, "parity")
    seeds, targets = train_seed_mask(data, split), window_target_mask(data, split)
    assert np.array_equal(d.X_train, node_inputs(data, graph_view(data, split.train_max))[seeds].numpy())
    assert np.array_equal(d.X_score, node_inputs(data, graph_view(data, split.test_max))[targets].numpy())
    assert d.feature_names == INPUT_FEATURE_NAMES


def test_raw17_floor_is_the_parity_floor_minus_every_view_column():
    data = _synthetic()
    parity, raw = floor_design(data, SPLIT_VAL, "parity"), floor_design(data, SPLIT_VAL, "raw17")
    assert np.array_equal(raw.X_train, parity.X_train[:, :17])
    assert np.array_equal(raw.X_score, parity.X_score[:, :17])
    assert set(FEATURE_SETS["parity"]) - set(FEATURE_SETS["raw17"]) == set(INPUT_FEATURE_NAMES[17:])


def test_training_rows_use_the_training_view_not_the_scoring_view():
    data = _synthetic()
    d = floor_design(data, SPLIT_TEST, "parity")
    seeds = train_seed_mask(data, SPLIT_TEST)
    scoring_view_rows = node_inputs(data, graph_view(data, SPLIT_TEST.test_max))[seeds].numpy()
    assert not np.array_equal(d.X_train, scoring_view_rows), "fixture cannot tell the views apart"


def test_background_users_are_never_trained_on_or_scored():
    d = floor_design(_synthetic(), SPLIT_TEST, "parity")
    assert set(np.unique(d.y_train)) <= {0, 1}
    assert set(np.unique(d.y_score)) <= {0, 1}


def test_lr_path_uses_the_gnns_own_fitted_transform():
    data = _synthetic()
    d = floor_design(data, SPLIT_VAL, "parity", standardise=True)
    tf = fit_input_transform(data, SPLIT_VAL)
    seeds = train_seed_mask(data, SPLIT_VAL)
    expected = tf.transform(node_inputs(data, graph_view(data, SPLIT_VAL.train_max)))[seeds].numpy()
    assert np.array_equal(d.X_train, expected)


# -- ADR-007 metric discipline ----------------------------------------------------------------
def test_metrics_are_core_metrics_with_prevalence_and_positive_count():
    data = _synthetic()
    d = floor_design(data, SPLIT_VAL, "parity")
    proba, pred = fit_predict("rf", d, seed=0)
    m = evaluate(d, proba, pred)
    for key in ("fraud_auc", "fraud_auprc", "fraud_f1", "prevalence", "n_score_fraud", "n_score"):
        assert key in m
    assert m["fraud_auprc"] == pytest.approx(average_precision_score(d.y_score, proba, pos_label=1))
    assert m["prevalence"] == pytest.approx(m["n_score_fraud"] / m["n_score"])


def test_an_unavailable_model_names_the_open_decision():
    d = floor_design(_synthetic(), SPLIT_VAL, "parity")
    with pytest.raises(ValueError, match="XGBoost"):
        fit_predict("xgboost", d, seed=0)


def test_run_floor_writes_one_row_with_provenance(tmp_path):
    registry = tmp_path / "registry.csv"
    run_id, m = run_floor("lr", "parity", SPLIT_VAL, 0, _synthetic(), dgf1_base_config(),
                          registry_path=registry)
    rows = list(csv.DictReader(registry.open(encoding="utf-8", newline="")))
    assert len(rows) == 1 and rows[0]["run_id"] == run_id and rows[0]["model"] == "dgf1"
    logged = json.loads(rows[0]["metrics_json"])
    assert logged["arm"] == "lr-parity" and logged["features"] == "parity"
    assert logged["window"] == "9-12" and logged["deterministic"] is True
    assert "fraud_auprc" in logged and "prevalence" in logged


# -- the test-window guard --------------------------------------------------------------------
def test_test_window_is_refused_without_a_preregistration():
    with pytest.raises(SystemExit, match="no --preregistration"):
        require_accepted_preregistration(None)


@pytest.mark.parametrize("status", ["proposed", "rejected", None])
def test_test_window_is_refused_unless_the_adr_is_accepted(tmp_path, status):
    adr = tmp_path / "ADR-999-x.md"
    adr.write_text("# ADR-999\n\n" + (f"**Status:** {status}\n" if status else "no status\n"),
                   encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing --window test"):
        require_accepted_preregistration(adr)


def test_an_accepted_preregistration_lets_the_test_window_through(tmp_path):
    adr = tmp_path / "ADR-999-x.md"
    adr.write_text("# ADR-999\n\n**Status:** accepted\n", encoding="utf-8")
    require_accepted_preregistration(adr)   # no exit
