"""DGF-1 parity floor guards — ADR-011 clause 4, clause-5 test 8; ADR-007 metric discipline; and
the shared test-window guard (ADR-016).

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
import preregistration  # noqa: E402
from preregistration import require_gated_preregistration  # noqa: E402

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


def test_an_unknown_model_is_rejected():
    """Was a guard for XGBoost being unavailable; XGBoost is the gated model since ADR-012, so what
    remains to pin is that a typo cannot silently select something else."""
    d = floor_design(_synthetic(), SPLIT_VAL, "parity")
    with pytest.raises(ValueError, match="unknown floor model"):
        fit_predict("xgbost", d, seed=0)


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


# -- the shared test-window guard (ADR-016) ---------------------------------------------------------
# The real map is empty, so every acceptance below goes through a fixture entry, and `git_dirty` is
# pinned so that no result depends on the state of the checkout.
ADR_012 = REPO_ROOT / "decisions" / "ADR-012-dgf1-gate1-preregistration.md"
ADR_015 = REPO_ROOT / "decisions" / "ADR-015-dgf1-official-split-positioning.md"
ADR_016 = REPO_ROOT / "decisions" / "ADR-016-close-gate1-test-window.md"


def _doc(path: Path, status: str | None = "accepted",
         seeds_line: str = "**Stage-1 seeds:** 12\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# ADR-999\n\n" + (f"**Status:** {status}\n" if status else "no status\n")
                    + seeds_line, encoding="utf-8")
    return path


@pytest.fixture
def mapped(tmp_path, monkeypatch):
    """A clean tree, and one fixture mode mapped to an accepted document that carries its count."""
    doc = _doc(tmp_path / "decisions" / "ADR-999-x.md")
    monkeypatch.setattr(preregistration, "git_dirty", lambda *a, **k: False)
    monkeypatch.setattr(preregistration, "GATED_MODES", {"fixture": doc})
    return doc


def test_a_mapped_mode_opens_with_its_own_document_and_returns_its_count(mapped, tmp_path,
                                                                         monkeypatch):
    assert require_gated_preregistration("fixture", mapped, allow_dirty=False) == 12
    # compared in resolved form, so another spelling of the same file is the same document
    monkeypatch.chdir(tmp_path)
    respelled = "decisions/../decisions/ADR-999-x.md"
    assert require_gated_preregistration("fixture", respelled, allow_dirty=False) == 12


def test_another_document_is_refused_even_when_accepted_with_a_count(mapped, tmp_path):
    """ADR-015 clause 6's defect: any accepted file with a seed line used to open the window."""
    other = _doc(tmp_path / "elsewhere" / "ADR-999-x.md")
    with pytest.raises(SystemExit, match="not the document mapped"):
        require_gated_preregistration("fixture", other, allow_dirty=False)
    with pytest.raises(SystemExit, match="no --preregistration"):
        require_gated_preregistration("fixture", None, allow_dirty=False)


def test_a_dirty_tree_and_its_override_are_both_refused_in_a_gated_mode(mapped, monkeypatch):
    """The guard checks the tree itself (ADR-016 clause 2), and no flag waives it."""
    with pytest.raises(SystemExit, match="--allow-dirty"):
        require_gated_preregistration("fixture", mapped, allow_dirty=True)       # clean tree
    monkeypatch.setattr(preregistration, "git_dirty", lambda *a, **k: True)
    with pytest.raises(SystemExit, match="uncommitted"):
        require_gated_preregistration("fixture", mapped, allow_dirty=False)
    with pytest.raises(SystemExit, match="--allow-dirty"):
        require_gated_preregistration("fixture", mapped, allow_dirty=True)       # dirty tree


@pytest.mark.parametrize("status", ["proposed", "rejected", None])
def test_the_mapped_document_must_be_accepted(mapped, status):
    _doc(mapped, status=status)
    with pytest.raises(SystemExit, match="not 'accepted'"):
        require_gated_preregistration("fixture", mapped, allow_dirty=False)


@pytest.mark.parametrize("seeds_line", [
    "",                                                    # no line at all
    "**Stage-1 seeds:** _not yet derived — after the pilot_\n",   # ADR-012's placeholder
    "Stage-1 seeds: 12\n",                                 # right words, wrong format
])
def test_an_accepted_mapped_document_without_a_seed_count_keeps_the_window_shut(mapped, seeds_line):
    """ADR-012 fixed the seed-count *rule* and left the *number* to the validation pilot, so
    acceptance alone must not open 482-821 (clause 8)."""
    _doc(mapped, seeds_line=seeds_line)
    with pytest.raises(SystemExit, match="no stage-1 seed count"):
        require_gated_preregistration("fixture", mapped, allow_dirty=False)


def test_the_guard_refuses_in_adr_016s_order(mapped, monkeypatch, tmp_path):
    """Mode before tree, tree before path, path before content."""
    other = _doc(tmp_path / "elsewhere" / "ADR-999-x.md", status="proposed")
    monkeypatch.setattr(preregistration, "git_dirty", lambda *a, **k: True)
    with pytest.raises(SystemExit, match="not a gated mode"):
        require_gated_preregistration("pilot", other, allow_dirty=True)
    with pytest.raises(SystemExit, match="uncommitted"):
        require_gated_preregistration("fixture", other, allow_dirty=False)
    monkeypatch.setattr(preregistration, "git_dirty", lambda *a, **k: False)
    with pytest.raises(SystemExit, match="not the document mapped"):
        require_gated_preregistration("fixture", other, allow_dirty=False)


def test_a_map_entry_is_read_against_the_repo_root_not_the_working_directory(monkeypatch, tmp_path):
    """Real entries are repo-relative. ADR-016 is accepted and carries no seed line, so reaching the
    seed-count refusal shows the path matched."""
    monkeypatch.setattr(preregistration, "git_dirty", lambda *a, **k: False)
    monkeypatch.setattr(preregistration, "GATED_MODES",
                        {"fixture": Path("decisions") / ADR_016.name})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="no stage-1 seed count"):
        require_gated_preregistration("fixture", ADR_016, allow_dirty=False)


def test_gate_is_closed_whatever_document_is_named(tmp_path, monkeypatch):
    """ADR-016 clause 3. Under the old guard, ADR-012 and the temporary file both opened 482-821."""
    monkeypatch.setattr(preregistration, "git_dirty", lambda *a, **k: False)
    for doc in (None, ADR_012, _doc(tmp_path / "ADR-999-x.md")):
        with pytest.raises(SystemExit, match=r"ADR-016.*gates/GATE-DGF1-1\.md"):
            require_gated_preregistration("gate", doc, allow_dirty=False)


def test_adr_012_passes_every_content_check_so_its_refusal_is_the_closure():
    """Keeps the refusal above from passing vacuously: ADR-012 is accepted, and its seed line
    matches the guard's own pattern."""
    text = ADR_012.read_text(encoding="utf-8")
    assert preregistration._STATUS.search(text).group(1).lower() == "accepted"
    assert int(preregistration._SEEDS.search(text).group(1)) == 8


def test_the_real_map_is_empty():
    """After ADR-016 no mode opens the test window. A later gated batch's implementation adds its
    entry, and updates this test, in the same change."""
    assert preregistration.GATED_MODES == {}
    assert set(preregistration.CLOSED_MODES) == {"gate"}


def test_the_real_map_never_names_gate_adr_012_or_adr_015():
    """ADR-016 clause 5 binds every future entry, not only today's empty map."""
    assert "gate" not in preregistration.GATED_MODES
    forbidden = {ADR_012.resolve(), ADR_015.resolve()}
    for mode, doc in preregistration.GATED_MODES.items():
        resolved = (preregistration.REPO_ROOT / doc).resolve()
        assert resolved not in forbidden, mode
        assert not resolved.name.startswith(("ADR-012", "ADR-015")), mode


# -- the gated floor model (ADR-012 clause 2) --------------------------------------------------
def test_xgboost_is_the_gated_model_with_the_adr_parameters():
    from adapters.dgf1.baselines_tabular import FLOOR_MODELS, GATED_MODEL, resolve_params

    assert GATED_MODEL == "xgboost"
    p = resolve_params("xgboost")
    assert p["tree_method"] == "hist" and p["n_estimators"] == 300 and p["learning_rate"] == 0.1
    assert p["max_depth"] == 6 and p["subsample"] == 1.0 and p["colsample_bytree"] == 1.0
    assert p["n_jobs"] == 8, "n_jobs must be a fixed integer: hist determinism depends on threads"
    assert "early_stopping_rounds" not in FLOOR_MODELS["xgboost"]


def test_the_validation_grid_is_nine_configurations_on_the_two_tuned_axes():
    from adapters.dgf1.baselines_tabular import FLOOR_GRID

    assert set(FLOOR_GRID) == {"max_depth", "subsample"}
    assert len(FLOOR_GRID["max_depth"]) * len(FLOOR_GRID["subsample"]) == 9


def test_tuned_axes_override_but_unknown_parameters_are_rejected():
    from adapters.dgf1.baselines_tabular import resolve_params

    assert resolve_params("xgboost", {"max_depth": 4, "subsample": 0.8})["max_depth"] == 4
    with pytest.raises(ValueError, match="unknown parameter"):
        resolve_params("xgboost", {"learning_rate_typo": 0.5})


def test_xgboost_is_reproducible_and_weights_the_positive_class():
    data = _synthetic()
    d = floor_design(data, SPLIT_VAL, "parity")
    a, _ = fit_predict("xgboost", d, seed=0)
    b, _ = fit_predict("xgboost", d, seed=0)
    assert np.array_equal(a, b)
    # scale_pos_weight is computed from the training rows, so a fit on a rebalanced training set
    # must differ from one on the original
    assert (d.y_train == 1).sum() > 0 and (d.y_train == 0).sum() > 0


def test_a_row_states_the_configuration_that_produced_it(tmp_path):
    """ADR-012 clause 7: the gate file must say which configuration each arm's tuning selected.
    The tuned axes were always inside the hashed config, so they were recoverable only by
    rebuilding hashes — the detour ADR-008 was forced into when Gate-3's rows carried no arm key.
    A row must say what it is on its own."""
    registry = tmp_path / "registry.csv"
    _, m = run_floor("xgboost", "parity", SPLIT_VAL, 0, _synthetic(), dgf1_base_config(),
                     registry_path=registry, overrides={"max_depth": 4, "subsample": 0.8})
    assert m["max_depth"] == 4 and m["subsample"] == 0.8

    logged = json.loads(list(csv.DictReader(registry.open(encoding="utf-8", newline="")))[0]["metrics_json"])
    assert logged["max_depth"] == 4 and logged["subsample"] == 0.8, "row cannot state its own config"
    assert logged["arm"] == "xgboost-parity" and logged["baseline"] == "xgboost"


def test_run_floor_persists_the_scored_vector_with_its_ids(tmp_path):
    """ADR-012 clause 10: path + content hash in the row, ids in the file."""
    from gbe.eval import load_scores

    registry, scores = tmp_path / "registry.csv", tmp_path / "scores"
    run_id, m = run_floor("xgboost", "parity", SPLIT_VAL, 0, _synthetic(), dgf1_base_config(),
                          registry_path=registry, scores_dir=scores)
    assert m["scores_sha256"] and len(m["scores_sha256"]) == 64
    saved = load_scores(scores / f"{run_id}.npz")
    assert saved["node_ids"].size == m["n_score"] == saved["proba"].size
    logged = json.loads(list(csv.DictReader(registry.open(encoding="utf-8", newline="")))[0]["metrics_json"])
    assert logged["scores_sha256"] == m["scores_sha256"]
