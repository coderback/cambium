"""ELL-1 tabular-floor baselines: Random Forest + Logistic Regression, no graph.

Doc `docs/01-elliptic-embedding-model-BUILD.md` §5/§8: the *number to beat*. These
classifiers see only the 165 node features (ADR-001; `time_step` excluded) — no edges,
no message passing — so any later GNN gain over this floor is attributable to structure.

Protocol (doc §2.2 option a, §2.3): supervised on **labelled** nodes only
(`illicit=1`/`licit=0`, `unknown` dropped), train on steps ≤34, evaluate on steps 35–49.
Illicit is the positive class; we report F1 / recall / AUC on it, never accuracy. The
LR scaler is fit on **train only** — a scaler fit over all nodes leaks test-period stats.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gbe.eval import TemporalSplit, split_masks
from gbe.run import RunSession
from gbe.run.config import resolve_config
from adapters.ell1.datasource_elliptic import ILLICIT
from adapters.ell1.train_gnn import PROVENANCE_KEYS

# Reasonable, not toy (doc §10: a weak floor is the #1 desk-reject cause).
RF_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "n_jobs": -1,
}
LR_PARAMS: dict[str, Any] = {
    "max_iter": 2000,
    "class_weight": "balanced",
}

BASELINES = ("rf", "lr")


def prepare_labelled_split(data, split: TemporalSplit):
    """Split labelled nodes by the temporal cutoff → numpy train/test arrays + support.

    Returns ``(X_train, y_train, X_test, y_test, meta)``; ``meta`` carries support counts.
    Unknown-class nodes are excluded from both sides (supervised gate, doc §2.2a).
    """
    train_mask, test_mask = split_masks(data.time_step, split)
    labelled = data.labelled_mask
    tr = (train_mask & labelled).numpy()
    te = (test_mask & labelled).numpy()

    X = data.x.numpy()
    y = data.y.numpy()
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]
    meta = {
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "n_test_illicit": int((y_test == ILLICIT).sum()),
    }
    return X_train, y_train, X_test, y_test, meta


def fit_predict(baseline: str, X_train, y_train, X_test, seed: int):
    """Fit a baseline and return ``(illicit_proba, hard_pred)`` on ``X_test``.

    RF trains on raw features (tree-based, scale-invariant); LR trains on a train-fit
    ``StandardScaler`` (and the same scaler is applied to the test features).
    """
    if baseline == "rf":
        clf = RandomForestClassifier(random_state=seed, **RF_PARAMS)
        clf.fit(X_train, y_train)
        X_eval = X_test
    elif baseline == "lr":
        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(random_state=seed, **LR_PARAMS)
        clf.fit(scaler.transform(X_train), y_train)
        X_eval = scaler.transform(X_test)
    else:
        raise ValueError(f"unknown baseline {baseline!r}; expected one of {BASELINES}")

    illicit_col = list(clf.classes_).index(ILLICIT)
    proba = clf.predict_proba(X_eval)[:, illicit_col]
    pred = clf.predict(X_eval)
    return proba, pred


def evaluate(y_test, proba, pred) -> dict[str, float]:
    """Illicit-class F1 / recall / precision + ROC-AUC. Never accuracy (doc §5)."""
    return {
        "illicit_f1": float(f1_score(y_test, pred, pos_label=ILLICIT, zero_division=0)),
        "illicit_recall": float(recall_score(y_test, pred, pos_label=ILLICIT, zero_division=0)),
        "illicit_precision": float(precision_score(y_test, pred, pos_label=ILLICIT, zero_division=0)),
        "illicit_auc": float(roc_auc_score(y_test, proba)),
    }


def run_baseline(
    baseline: str,
    seed: int,
    X_train,
    y_train,
    X_test,
    y_test,
    meta: dict[str, Any],
    base_cfg_values: dict[str, Any],
    test_window: str,
    registry_path=None,
) -> tuple[str, dict[str, Any]]:
    """One baseline at one seed, through :class:`RunSession` → exactly one registry row.

    ``baseline`` is folded into the hashed config (provenance) and into the logged metrics
    (so the gate assembler can group the seeds). Returns ``(run_id, metrics)``.
    """
    cfg = resolve_config({**base_cfg_values, "baseline": baseline, "seed": seed})
    with RunSession(cfg, notes=f"ell1 tabular floor: {baseline}", registry_path=registry_path) as run:
        proba, pred = fit_predict(baseline, X_train, y_train, X_test, run.seed)
        metrics = evaluate(y_test, proba, pred)
        metrics.update({"baseline": baseline, "test_window": test_window, **meta})
        # Same provenance echo as the GNN path: a row should identify its own experiment
        # without the reader recomputing config hashes (see train_gnn.PROVENANCE_KEYS).
        metrics.update(
            {k: base_cfg_values[k] for k in PROVENANCE_KEYS if k in base_cfg_values}
        )
        run.log_metrics(metrics)
    return run.run_id, metrics
