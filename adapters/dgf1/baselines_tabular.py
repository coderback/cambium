"""DGF-1 tabular floor — the comparator Gate 1 is measured against (ADR-007, ADR-011 clause 4).

**The gated floor is the PARITY floor.** It sees exactly the node-level information the GNN's
inputs contain — the raw 17 features **and** the statistics derived from the same graph view (the
11-wide edge-type histogram, whose row sum is degree, and the two recency features) — computed by
the same function, :func:`adapters.dgf1.features.node_inputs`. Neighbourhoods keep growing after
users appear and degree tracks the label, so without parity only the GNN could read that
post-appearance activity; with it, Gate 1 tests message passing *beyond* local counts (ADR-011
clause 4, clause-5 test 8). The raw-17 floor is **reported, not gated**.

Protocol, mirroring the GNN's exactly (ADR-011 clauses 2–4):

* **train** on labelled users existing by ``train_max`` (369), features from the **training view**
  (the graph as of 369);
* **score** the labelled users who first appear inside the window, features from the **scoring
  view** (the graph as of the window's last step);
* background users (``y`` in {2, 3}) are never trained on and never scored.

Trees take the inputs **untransformed** (scale-invariant). Logistic regression takes them through
the **GNN's own fitted transform** (:func:`adapters.dgf1.features.fit_input_transform`) — same
statistics, same zero-variance rule — so the two paths cannot differ in preprocessing either.

Metrics route through the core :func:`gbe.eval.classification_metrics` — never re-stated here (the
ELL-1 floor re-stated them and so never gained ``auprc``; that must not be inherited, see its
docstring). Every AUPRC travels with its window's prevalence and positive count (ADR-007).

**Model choice is open.** doc-02 §5/§8 names **XGBoost** as the gated comparator; it is not
installed, and installing it mid-EXTRACT is an environment change the researcher decides (ADR-008
clause 4 pins numpy/scikit-learn, which a careless install can move). Until then this module offers
the ELL-1 floor's two models. **Scale warning:** ELL-1's RF settings (300 full-depth trees) were
sized for ~30k rows; at ~860k they may not fit in this machine's 15 GB. Floor hyperparameters at
DGraph scale are a Gate-1 pre-registration question, not settled here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Data

from gbe.eval import TemporalSplit, classification_metrics
from gbe.run import RunSession
from gbe.run.config import resolve_config
from adapters.dgf1.datasource_dgraph import (
    FRAUD,
    graph_view,
    train_seed_mask,
    window_target_mask,
)
from adapters.dgf1.features import (
    INPUT_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    fit_input_transform,
    node_inputs,
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "parity": INPUT_FEATURE_NAMES,   # gated comparator (ADR-011 clause 4)
    "raw17": RAW_FEATURE_NAMES,      # reported, not gated
}

# ELL-1's floor settings, carried over as the starting point — NOT validated at DGraph scale.
FLOOR_MODELS: dict[str, dict[str, Any]] = {
    "rf": {"n_estimators": 300, "class_weight": "balanced", "n_jobs": -1},
    "lr": {"max_iter": 2000, "class_weight": "balanced"},
}
# Which models need the GNN's standardisation (trees are scale-invariant).
STANDARDISED: dict[str, bool] = {"rf": False, "lr": True}

# Config keys echoed from the hashed config into the row so a batch reads back without
# recomputing hashes (the ELL-1 lesson, GATE-ELL1-3). Explicit, so a typo cannot invent a field.
PROVENANCE_KEYS: tuple[str, ...] = ("arm", "experiment", "features", "window")


@dataclass(frozen=True)
class FloorDesign:
    """Train and score matrices for one (window, feature set), plus the window's support."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_score: np.ndarray
    y_score: np.ndarray
    feature_names: tuple[str, ...]
    meta: dict[str, Any]


def floor_design(
    data: Data, split: TemporalSplit, feature_set: str, standardise: bool = False
) -> FloorDesign:
    """Build the floor's matrices from the same inputs, views and node sets as the GNN.

    ``split`` is a DGF-1 window (``dgf1_splits()["val"]`` or ``["test"]``): training rows use the
    view as of ``split.train_max``, scoring rows the view as of ``split.test_max``.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; expected one of {list(FEATURE_SETS)}")
    if split.test_max is None:
        raise ValueError("a DGF-1 window must have an explicit last step (test_max)")
    names = FEATURE_SETS[feature_set]
    cols = [INPUT_FEATURE_NAMES.index(n) for n in names]

    train_in = node_inputs(data, graph_view(data, split.train_max))
    score_in = node_inputs(data, graph_view(data, split.test_max))
    if standardise:
        transform = fit_input_transform(data, split)   # fitted on the training view, <= train_max
        train_in, score_in = transform.transform(train_in), transform.transform(score_in)

    seeds = train_seed_mask(data, split)
    targets = window_target_mask(data, split)
    y_score = data.y[targets]
    n_pos = int((y_score == FRAUD).sum())
    meta = {
        "n_train": int(seeds.sum()),
        "n_score": int(targets.sum()),
        "n_score_fraud": n_pos,
        "prevalence": n_pos / max(int(targets.sum()), 1),   # AUPRC's chance level (ADR-007)
        "window": f"{split.test_min}-{split.test_max}",
    }
    return FloorDesign(
        X_train=train_in[seeds][:, cols].numpy(),
        y_train=data.y[seeds].numpy(),
        X_score=score_in[targets][:, cols].numpy(),
        y_score=y_score.numpy(),
        feature_names=names,
        meta=meta,
    )


def fit_predict(model: str, design: FloorDesign, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit one floor model and return ``(fraud_proba, hard_pred)`` on the scoring rows."""
    if model not in FLOOR_MODELS:
        raise ValueError(
            f"unknown floor model {model!r}; available: {list(FLOOR_MODELS)}. doc-02 names XGBoost "
            "as the gated comparator — it is not installed; adding it is the researcher's decision."
        )
    params = FLOOR_MODELS[model]
    clf = (RandomForestClassifier if model == "rf" else LogisticRegression)(
        random_state=seed, **params
    )
    clf.fit(design.X_train, design.y_train)
    fraud_col = list(clf.classes_).index(FRAUD)
    return clf.predict_proba(design.X_score)[:, fraud_col], clf.predict(design.X_score)


def evaluate(design: FloorDesign, proba: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Core metrics on the fraud class (``fraud_auc`` = ROC-AUC, ``fraud_auprc`` = average
    precision), always alongside the window's prevalence and positive count (ADR-007)."""
    m = classification_metrics(design.y_score, proba, pred, pos_label=FRAUD)
    out = {f"fraud_{k}": v for k, v in m.items()}
    out.update(design.meta)
    return out


def run_floor(
    model: str,
    feature_set: str,
    split: TemporalSplit,
    seed: int,
    data: Data,
    base_cfg: dict[str, Any],
    registry_path=None,
) -> tuple[str, dict[str, Any]]:
    """One floor model × feature set × window × seed, through :class:`RunSession` -> one row.

    Deliberately takes no view of *which* window is allowed: the runner script refuses the test
    window without an accepted pre-registration ADR, so the guard sits where a human invokes it.
    """
    window = f"{split.test_min}-{split.test_max}"
    cfg_values = {
        **base_cfg,
        "baseline": model,
        "floor_params": FLOOR_MODELS.get(model, {}),
        "features": feature_set,
        "window": window,
        "arm": f"{model}-{feature_set}",
        "seed": seed,
    }
    cfg = resolve_config(cfg_values)
    with RunSession(cfg, notes=f"dgf1 floor: {model}/{feature_set} on {window}",
                    registry_path=registry_path) as run:
        design = floor_design(data, split, feature_set, standardise=STANDARDISED.get(model, False))
        proba, pred = fit_predict(model, design, run.seed)
        logged = evaluate(design, proba, pred)
        logged["baseline"] = model
        logged.update({k: cfg_values[k] for k in PROVENANCE_KEYS if k in cfg_values})
        run.log_metrics(logged)
    return run.run_id, logged
