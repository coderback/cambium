"""Paired bootstrap over scored instances, plus Boyd's logit interval (ADR-012 clause 10).

**Why paired.** Gate criteria in this program model *seed* variance only (ADR-006). The other term
— which users happen to constitute the scored window — is real and, at DGF-1's support, of the same
order: simulated at 183,469 users with 2,717 positives, the paired bootstrap SE of the AUPRC
difference is ≈0.005 against a seeds-only SE of ≈0.007. Resampling users **once per replicate and
scoring both arms on the identical resample** cancels the shared "which users are hard" component,
which is what makes the difference's interval ~2× tighter than each arm's own.

**Why stratified.** AUPRC's chance level *is* the prevalence, so a resample that changes the
positive count changes the quantity being estimated. Positives and negatives are therefore resampled
separately, preserving both counts (Boyd, Eng & Page 2013).

**Reported, never gated** (ADR-012 clause 10, option (a)): the gated criterion stays seed-based;
this interval is reported beside it, and a disagreement between the two is stated in the gate file.

Domain-agnostic: labels, probabilities, a positive class.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

Z_95 = 1.959964  # two-sided normal quantile; pass another z for a different level


def resample_index(
    rng: np.random.Generator, pos: np.ndarray, neg: np.ndarray, *, stratified: bool = True
) -> np.ndarray | None:
    """One bootstrap resample of row indices; ``None`` if it is unusable (one class only).

    Split out so the **prevalence-preserving** property can be tested directly rather than
    inferred from a metric: stratified resampling draws the positives and the negatives
    separately, so every replicate carries *exactly* the original positive count. That is what
    keeps AUPRC's chance level — which *is* the prevalence — fixed across replicates.
    """
    if stratified:
        return np.concatenate([rng.choice(pos, pos.size, replace=True),
                               rng.choice(neg, neg.size, replace=True)])
    n = pos.size + neg.size
    idx = rng.integers(0, n, n)
    all_idx = np.concatenate([pos, neg])
    picked = all_idx[idx]
    is_pos = np.isin(picked, pos)
    if is_pos.all() or not is_pos.any():
        return None
    return picked


def paired_bootstrap_difference(
    y_true: np.ndarray,
    proba_floor: np.ndarray,
    proba_model: np.ndarray,
    *,
    pos_label: int = 1,
    n_resamples: int = 1000,
    seed: int = 0,
    stratified: bool = True,
) -> dict[str, dict[str, float]]:
    """Bootstrap CI of ``model − floor`` for ROC-AUC and AUPRC, both arms on one resample.

    The sign convention is fixed by the argument order: a **positive** difference means the model is
    ahead of the floor. Returns, per metric, ``{"mean", "se", "ci_lo", "ci_hi", "n_resamples"}``
    (percentile interval, 2.5/97.5).

    ``n_resamples`` and ``seed`` default to ADR-012's pinned values, so an interval recomputed from
    the stored score files reproduces exactly.
    """
    y = np.asarray(y_true)
    a = np.asarray(proba_floor, dtype=float)
    b = np.asarray(proba_model, dtype=float)
    if not (y.shape == a.shape == b.shape) or y.ndim != 1:
        raise ValueError(f"y_true, proba_floor, proba_model must be 1-D and equal length; "
                         f"got {y.shape}, {a.shape}, {b.shape}")

    pos = np.flatnonzero(y == pos_label)
    neg = np.flatnonzero(y != pos_label)
    if pos.size < 2 or neg.size < 2:
        raise ValueError(f"need at least 2 positives and 2 negatives; got {pos.size}/{neg.size}")

    rng = np.random.default_rng(seed)
    diffs: dict[str, list[float]] = {"auc": [], "auprc": []}
    for _ in range(n_resamples):
        idx = resample_index(rng, pos, neg, stratified=stratified)
        if idx is None:
            continue
        yi = y[idx]
        diffs["auc"].append(roc_auc_score(yi, b[idx]) - roc_auc_score(yi, a[idx]))
        diffs["auprc"].append(
            average_precision_score(yi, b[idx], pos_label=pos_label)
            - average_precision_score(yi, a[idx], pos_label=pos_label)
        )

    out: dict[str, dict[str, float]] = {}
    for metric, values in diffs.items():
        v = np.asarray(values, dtype=float)
        out[metric] = {
            "mean": float(v.mean()),
            "se": float(v.std(ddof=1)),
            "ci_lo": float(np.quantile(v, 0.025)),
            "ci_hi": float(np.quantile(v, 0.975)),
            "n_resamples": int(v.size),
        }
    return out


def straddles_zero(interval: dict[str, Any]) -> bool:
    """True if a difference CI includes 0 — the case ADR-012 requires the gate file to state."""
    return interval["ci_lo"] <= 0.0 <= interval["ci_hi"]


def logit_interval(ap: float, n_pos: int, z: float = Z_95) -> tuple[float, float]:
    """Boyd, Eng & Page (2013) logit CI for average precision, from the **positive** count.

    Their finding is that what governs an AUPRC interval is the number of positives, not the size of
    the dataset, and that the logit interval behaves better than a bootstrap at high skew — which is
    why it is reported per arm alongside the paired bootstrap of the difference.
    """
    if not 0.0 < ap < 1.0:
        raise ValueError(f"average precision must lie strictly in (0, 1); got {ap}")
    if n_pos < 1:
        raise ValueError(f"n_pos must be >= 1; got {n_pos}")
    eta = math.log(ap / (1.0 - ap))
    se_eta = math.sqrt(1.0 / (n_pos * ap * (1.0 - ap)))
    lo, hi = eta - z * se_eta, eta + z * se_eta
    return 1.0 / (1.0 + math.exp(-lo)), 1.0 / (1.0 + math.exp(-hi))
