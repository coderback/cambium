"""Classification metrics for the temporal-holdout harness (doc-00 §4/§7).

The harness owns the metric definitions so every model — and every baseline it is compared
against — scores the *same way*: same positive class, same F1/recall/precision, same ROC-AUC,
same AUPRC, never accuracy (useless under ~2% prevalence, doc-01 §5).

**Two threshold-free metrics, and both are required (ADR-007).** ``auc`` is ROC-AUC — the
comparability metric, because it is what public leaderboards report (DGraph, doc-02 §2.1).
``auprc`` is average precision — the *sensitivity* metric, because ROC and PR space are not
order-equivalent under class skew (Davis & Goadrich 2006) and ROC-AUC alone hides precision
collapse at low prevalence. Measured on ELL-1 at 6.5% prevalence: two arms matched on recall to
within 0.0013 differed by **0.379** in precision, which ROC-AUC compressed into **0.016**. Neither
metric is sufficient alone, which is why both are returned unconditionally rather than by flag.

Two properties of ``auprc`` that callers must not forget:

* **Its chance level is the prevalence of the scored window, not 0.5.** A random ranker scores
  ~prevalence. So an AUPRC is uninterpretable without the prevalence and positive count of the
  window it scores, and AUPRC is never compared across datasets or across windows of differing
  prevalence (ADR-007).
* **It is `average_precision_score`, never `sklearn.metrics.auc(recall, precision)`.** The latter
  linearly interpolates between PR points, which is optimistically biased in PR space; the two
  disagree on identical scores. Pinned here so a "mechanical" gate criterion cannot depend on
  which call site evaluated it — the same class of pin as ADR-006's ``ddof=1``.

**A positive-class asymmetry worth knowing (pre-existing, surfaced by ADR-007).** ``roc_auc_score``
takes no ``pos_label`` and infers the positive class as the *greater* label; ``average_precision_score``
takes ``pos_label`` explicitly and is passed it here. The two agree whenever the positive class is
the greater label — true for every current adapter (ELL-1: illicit=1, licit=0) — and would
silently disagree for an adapter that encoded its positive class as the smaller value. Any such
adapter must binarise before calling this function.

Domain-agnostic: the positive class is a parameter, keys are generic; the adapter renames
them (e.g. ``f1`` -> ``illicit_f1``) for its registry rows.

**Note on the ELL-1 tabular floor.** ``adapters/ell1/baselines_tabular.evaluate`` *re-states* these
sklearn calls rather than calling this function, and as of ADR-007 it does **not** emit ``auprc``.
That divergence is harmless for ELL-1 (closed; its dated gates are ROC-AUC/F1 and stand as
recorded) but it must not be inherited: DGF-1's tabular floor has to route through this function,
or Gate 1's AUPRC clause has no comparator to be measured against — and AUPRC cannot be
back-filled onto a registry row, which stores summary metrics rather than per-node scores.
"""

from __future__ import annotations

from typing import Sequence

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: Sequence[int],
    proba: Sequence[float],
    pred: Sequence[int],
    pos_label: int,
) -> dict[str, float]:
    """Positive-class F1 / recall / precision + ROC-AUC + AUPRC.

    Args:
        y_true: ground-truth labels.
        proba: predicted probability of the positive class (for ROC-AUC and AUPRC).
        pred: hard predicted labels (for F1 / recall / precision).
        pos_label: the positive class value (ELL-1: illicit).

    Returns:
        ``{"f1", "recall", "precision", "auc", "auprc"}`` as plain floats, where ``auc`` is
        ROC-AUC and ``auprc`` is average precision (chance level = prevalence, not 0.5).
    """
    return {
        "f1": float(f1_score(y_true, pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, pred, pos_label=pos_label, zero_division=0)),
        "precision": float(precision_score(y_true, pred, pos_label=pos_label, zero_division=0)),
        "auc": float(roc_auc_score(y_true, proba)),
        "auprc": float(average_precision_score(y_true, proba, pos_label=pos_label)),
    }
