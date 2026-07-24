"""Classification metrics for the temporal-holdout harness (doc-00 §4/§7).

The harness owns the metric definitions so every model — and every baseline it is compared
against — scores the *same way*. These are the exact sklearn calls the ELL-1 tabular floor
uses (`adapters/ell1/baselines_tabular.evaluate`), so a GNN number is directly comparable to
the RF/LR floor: same positive class, same F1/recall/precision, same ROC-AUC, never accuracy
(useless under ~2% prevalence, doc-01 §5).

Domain-agnostic: the positive class is a parameter, keys are generic; the adapter renames
them (e.g. ``f1`` -> ``illicit_f1``) for its registry rows.
"""

from __future__ import annotations

from typing import Sequence

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(
    y_true: Sequence[int],
    proba: Sequence[float],
    pred: Sequence[int],
    pos_label: int,
) -> dict[str, float]:
    """Positive-class F1 / recall / precision + ROC-AUC.

    Args:
        y_true: ground-truth labels.
        proba: predicted probability of the positive class (for ROC-AUC).
        pred: hard predicted labels (for F1 / recall / precision).
        pos_label: the positive class value (ELL-1: illicit).

    Returns:
        ``{"f1", "recall", "precision", "auc"}`` as plain floats.
    """
    return {
        "f1": float(f1_score(y_true, pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, pred, pos_label=pos_label, zero_division=0)),
        "precision": float(precision_score(y_true, pred, pos_label=pos_label, zero_division=0)),
        "auc": float(roc_auc_score(y_true, proba)),
    }
