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
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Data

from gbe.eval import TemporalSplit, classification_metrics, save_scores
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

# `xgboost` is the **gated** floor model (doc-02 §5/§8; parameters fixed by ADR-012 clause 2).
# `rf`/`lr` are ELL-1's settings, kept for reported rows — NOT validated at DGraph scale.
#
# `n_jobs` is a fixed integer, never -1: XGBoost's `hist` is deterministic given identical data
# order *and* thread count, so a machine-dependent thread count would make reproducibility
# machine-dependent (ADR-005). `max_depth`/`subsample` carry ADR-012's defaults and are the two
# axes its validation grid tunes. `scale_pos_weight` is computed per fit (it depends on the
# training rows), and there is no early stopping, so nothing selects on the scored window.
FLOOR_MODELS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "tree_method": "hist", "n_estimators": 300, "learning_rate": 0.1,
        "max_depth": 6, "subsample": 1.0, "colsample_bytree": 1.0,
        "min_child_weight": 1, "reg_lambda": 1.0, "n_jobs": 8,
    },
    "rf": {"n_estimators": 300, "class_weight": "balanced", "n_jobs": -1},
    "lr": {"max_iter": 2000, "class_weight": "balanced"},
}
GATED_MODEL = "xgboost"

# ADR-012 clause 2: nine configurations, one seed each, scored on validation, selected on AUPRC —
# the same budget as the GNN's retune, so neither arm is tuned harder than the other.
FLOOR_GRID: dict[str, list[Any]] = {"max_depth": [4, 6, 8], "subsample": [1.0, 0.8, 0.6]}

# Which models need the GNN's standardisation (trees are scale-invariant).
STANDARDISED: dict[str, bool] = {"xgboost": False, "rf": False, "lr": True}

# Config keys echoed from the hashed config into the row so a batch reads back without
# recomputing hashes (the ELL-1 lesson, GATE-ELL1-3). Explicit, so a typo cannot invent a field.
PROVENANCE_KEYS: tuple[str, ...] = ("arm", "experiment", "features", "window")

# The tuned axes are echoed too, flattened, because a *row* must say which configuration produced
# it: ADR-012 clause 7 requires the gate file to state what each arm's tuning selected. They were
# always in the hashed config, so they were recoverable by rebuilding hashes — which is exactly the
# detour ADR-008 had to take when Gate-3's rows carried no `ablation` key, and the reason
# PROVENANCE_KEYS exists at all.
TUNED_AXES: tuple[str, ...] = ("max_depth", "subsample")


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


def resolve_params(model: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """This model's fixed parameters, with the tuned axes overridden (ADR-012 clause 2)."""
    if model not in FLOOR_MODELS:
        raise ValueError(f"unknown floor model {model!r}; available: {list(FLOOR_MODELS)}")
    params = {**FLOOR_MODELS[model], **(overrides or {})}
    unknown = set(params) - set(FLOOR_MODELS[model])
    if unknown:
        raise ValueError(f"override introduces unknown parameter(s) {sorted(unknown)} for {model!r}")
    return params


def fit_predict(
    model: str, design: FloorDesign, seed: int, overrides: dict[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one floor model and return ``(fraud_proba, hard_pred)`` on the scoring rows."""
    params = resolve_params(model, overrides)
    if model == "xgboost":
        from xgboost import XGBClassifier  # adapter-only import

        n_pos = int((design.y_train == FRAUD).sum())
        n_neg = int(design.y_train.size - n_pos)
        # The imbalance handling doc-02 §2.2.5 requires; depends on the training rows, so it is
        # computed here rather than pinned in FLOOR_MODELS.
        clf = XGBClassifier(random_state=seed, scale_pos_weight=n_neg / max(n_pos, 1), **params)
    else:
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
    overrides: dict[str, Any] | None = None,
    scores_dir: str | Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """One floor model × feature set × window × seed, through :class:`RunSession` -> one row.

    Deliberately takes no view of *which* window is allowed: the runner script refuses the test
    window without an accepted pre-registration ADR, so the guard sits where a human invokes it.
    """
    window = f"{split.test_min}-{split.test_max}"
    cfg_values = {
        **base_cfg,
        "baseline": model,
        "floor_params": resolve_params(model, overrides),
        "features": feature_set,
        "window": window,
        "arm": f"{model}-{feature_set}",
        "seed": seed,
    }
    cfg = resolve_config(cfg_values)
    with RunSession(cfg, notes=f"dgf1 floor: {model}/{feature_set} on {window}",
                    registry_path=registry_path) as run:
        design = floor_design(data, split, feature_set, standardise=STANDARDISED.get(model, False))
        proba, pred = fit_predict(model, design, run.seed, overrides)
        logged = evaluate(design, proba, pred)
        logged["baseline"] = model
        logged.update({k: cfg_values[k] for k in PROVENANCE_KEYS if k in cfg_values})
        # Flattened tuned axes, so the row states its own configuration (ADR-012 clause 7).
        params = cfg_values.get("floor_params", {})
        logged.update({k: params[k] for k in TUNED_AXES if k in params})

        # ADR-012 clause 10: the scored vector survives the run, with the ids it belongs to, so a
        # paired bootstrap (or any later question) never needs the arm re-run.
        if scores_dir is not None:
            target_ids = window_target_mask(data, split).nonzero().view(-1).numpy()
            path = Path(scores_dir) / f"{run.run_id}.npz"
            digest = save_scores(path, node_ids=target_ids, y_true=design.y_score, proba=proba)
            logged["scores_path"] = str(path)
            logged["scores_sha256"] = digest
        run.log_metrics(logged)
    return run.run_id, logged
