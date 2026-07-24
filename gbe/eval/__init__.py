"""gbe.eval — temporal-holdout harness, metrics, ablations, gate runner."""

from gbe.eval.metrics import classification_metrics
from gbe.eval.temporal import (
    TemporalSplit,
    assert_no_temporal_leakage,
    induced_train_subgraph,
    split_masks,
)

__all__ = [
    "TemporalSplit",
    "split_masks",
    "induced_train_subgraph",
    "assert_no_temporal_leakage",
    "classification_metrics",
]
