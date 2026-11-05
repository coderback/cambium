"""Guards for the shared classification metrics — the two ADR-007 pins on ``auprc``.

ADR-007 gates DGF-1 on ROC-AUC *and* AUPRC. Both pins below exist because AUPRC has two
properties that are easy to get quietly wrong, and a wrong gate metric is not visible in the
output — it just moves the bar.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

from gbe.eval import classification_metrics


def test_auprc_is_average_precision_not_the_trapezoidal_pr_auc():
    """AUPRC must be `average_precision_score`, never `auc(recall, precision)` (ADR-007).

    Linear interpolation between PR points is optimistically biased, so the trapezoidal form
    reports a *different* number on identical scores. Pinning the definition is what stops a
    "mechanical" gate criterion from depending on which call site evaluated it — the same class
    of pin as ADR-006's ``ddof=1``.
    """
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    proba = np.array([0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4])
    pred = (proba >= 0.5).astype(int)

    got = classification_metrics(y_true, proba, pred, pos_label=1)["auprc"]

    expected = average_precision_score(y_true, proba, pos_label=1)
    precision, recall, _ = precision_recall_curve(y_true, proba, pos_label=1)
    trapezoidal = auc(recall, precision)

    assert got == expected
    # The guard only has teeth if the two definitions actually differ on this fixture.
    assert not np.isclose(expected, trapezoidal, atol=1e-3), (
        "fixture no longer separates the two definitions — the test cannot fail as written"
    )
    assert not np.isclose(got, trapezoidal, atol=1e-3)


def test_auprc_chance_level_is_prevalence_not_half():
    """A random ranker scores ~prevalence on AUPRC and ~0.5 on ROC-AUC (ADR-007).

    This is the whole reason ADR-007 gates on both. It also pins the interpretive rule: an AUPRC
    is meaningless without the prevalence of the window it scores, so a reader who assumes 0.5 is
    "chance" would read a *perfectly uninformative* model as a catastrophic one.
    """
    rng = np.random.default_rng(0)  # fixed: gate-grade code is deterministic (ADR-005)
    n, prevalence = 20_000, 0.05
    y_true = (rng.random(n) < prevalence).astype(int)
    proba = rng.random(n)  # scores carry no information about y_true
    pred = (proba >= 0.5).astype(int)

    metrics = classification_metrics(y_true, proba, pred, pos_label=1)
    observed_prevalence = float(y_true.mean())

    assert abs(metrics["auprc"] - observed_prevalence) < 0.01
    assert abs(metrics["auc"] - 0.5) < 0.02
    # The contrast is the point: the same uninformative scores look like 0.5 to ROC-AUC.
    assert metrics["auprc"] < 0.1 < metrics["auc"]


def test_both_threshold_free_metrics_are_returned():
    """The key set is part of the contract — ADR-007 requires both, unconditionally."""
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.9])
    pred = np.array([0, 0, 1, 1])

    metrics = classification_metrics(y_true, proba, pred, pos_label=1)

    assert set(metrics) == {"f1", "recall", "precision", "auc", "auprc"}
    assert all(isinstance(v, float) for v in metrics.values())
    # Perfect separation: both threshold-free metrics saturate.
    assert metrics["auc"] == 1.0 and metrics["auprc"] == 1.0


def test_pos_label_selects_the_positive_class_for_auprc():
    """``pos_label`` reaches AUPRC, so it is not silently scored against the wrong class."""
    y_true = np.array([0, 0, 1, 1])
    proba_for_1 = np.array([0.1, 0.2, 0.8, 0.9])  # ranks class 1 perfectly
    pred = np.array([0, 0, 1, 1])

    as_one = classification_metrics(y_true, proba_for_1, pred, pos_label=1)["auprc"]
    # Scoring the *other* class with the same score vector inverts the ranking, so a perfect
    # AUPRC must not survive the swap.
    as_zero = classification_metrics(y_true, proba_for_1, pred, pos_label=0)["auprc"]

    assert as_one == 1.0
    assert as_zero < as_one
    assert as_zero == average_precision_score(y_true, proba_for_1, pos_label=0)


def test_roc_auc_and_auprc_agree_on_the_positive_class_for_zero_one_labels():
    """The documented asymmetry: ``roc_auc_score`` infers the positive class as the greater
    label while AUPRC is told explicitly. For 0/1 labels — every current adapter — they agree,
    and this pins that they do."""
    y_true = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.05, 0.2, 0.3, 0.7, 0.95])
    pred = (proba >= 0.5).astype(int)

    metrics = classification_metrics(y_true, proba, pred, pos_label=1)

    assert metrics["auc"] == roc_auc_score(y_true, proba)
    assert metrics["auprc"] == average_precision_score(y_true, proba, pos_label=1)
    assert metrics["auc"] == 1.0 and metrics["auprc"] == 1.0
