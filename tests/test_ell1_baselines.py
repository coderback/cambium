"""Guards for the ELL-1 tabular-floor baselines — fast, on a synthetic separable graph.

Checks the *protocol*, not the numbers: only ≤34 nodes train, only 35–49 nodes test,
unknowns excluded, illicit is the positive class, and one RunSession run writes one
well-formed registry row.
"""

from __future__ import annotations

import csv
import json

import torch
from torch_geometric.data import Data

from gbe.eval import TemporalSplit
from adapters.ell1.baselines_tabular import (
    evaluate,
    fit_predict,
    prepare_labelled_split,
    run_baseline,
)
from adapters.ell1.datasource_elliptic import ILLICIT, LICIT, UNKNOWN

SPLIT = TemporalSplit(train_max=34, test_min=35, test_max=49)


def _synthetic_data() -> Data:
    """Linearly separable: illicit at +2, licit at -2, in both train and test windows.

    Includes one unknown node in each window that must be dropped by the supervised split.
    """
    rows = []  # (feat_center, label, time_step)
    for t in (5, 12, 20, 30):          # train window
        rows += [(2.0, ILLICIT, t), (-2.0, LICIT, t)]
    for t in (36, 40, 45):             # test window
        rows += [(2.0, ILLICIT, t), (-2.0, LICIT, t)]
    rows += [(0.0, UNKNOWN, 10), (0.0, UNKNOWN, 42)]  # must be excluded

    x = torch.tensor([[c, c] for c, _, _ in rows], dtype=torch.float)
    y = torch.tensor([lbl for _, lbl, _ in rows], dtype=torch.long)
    ts = torch.tensor([t for _, _, t in rows], dtype=torch.long)
    data = Data(x=x, y=y, time_step=ts)
    data.labelled_mask = y != UNKNOWN
    return data


def test_split_respects_window_and_drops_unknown():
    X_train, y_train, X_test, y_test, meta = prepare_labelled_split(_synthetic_data(), SPLIT)
    # 4 train steps × 2 labelled = 8; 3 test steps × 2 = 6; both unknowns dropped.
    assert meta == {"n_train": 8, "n_test": 6, "n_test_illicit": 3}
    assert UNKNOWN not in set(y_train.tolist()) and UNKNOWN not in set(y_test.tolist())


def test_separable_floor_scores_perfectly_with_illicit_positive():
    X_train, y_train, X_test, y_test, _ = prepare_labelled_split(_synthetic_data(), SPLIT)
    for baseline in ("rf", "lr"):
        proba, pred = fit_predict(baseline, X_train, y_train, X_test, seed=0)
        m = evaluate(y_test, proba, pred)
        assert set(m) == {"illicit_f1", "illicit_recall", "illicit_precision", "illicit_auc"}
        assert m["illicit_f1"] == 1.0 and m["illicit_recall"] == 1.0 and m["illicit_auc"] == 1.0


def test_run_baseline_writes_one_registry_row(tmp_path):
    data = _synthetic_data()
    X_train, y_train, X_test, y_test, meta = prepare_labelled_split(data, SPLIT)
    reg = tmp_path / "registry.csv"

    run_id, metrics = run_baseline(
        "rf", 1, X_train, y_train, X_test, y_test, meta,
        base_cfg_values={"model": "ell1", "phase": "P0", "data_snapshot_id": "synthetic"},
        test_window="35-49", registry_path=reg,
    )

    with reg.open(newline="", encoding="utf-8") as fh:
        written = list(csv.DictReader(fh))
    assert len(written) == 1
    row = written[0]
    assert row["run_id"] == run_id and row["seed"] == "1"
    logged = json.loads(row["metrics_json"])
    assert logged["baseline"] == "rf" and logged["test_window"] == "35-49"
    assert {"illicit_f1", "illicit_recall", "illicit_auc", "n_test_illicit"} <= set(logged)
