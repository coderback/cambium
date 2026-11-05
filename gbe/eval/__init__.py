"""gbe.eval — temporal-holdout harness, metrics, ablations, uncertainty, gate runner."""

from gbe.eval.ablations import (
    configuration_model_edges,
    random_graph_edges,
    remove_edges,
    scramble_edges,
)
from gbe.eval.bootstrap import logit_interval, paired_bootstrap_difference, straddles_zero
from gbe.eval.metrics import classification_metrics
from gbe.eval.scores import assert_aligned, content_hash, load_scores, save_scores
from gbe.eval.temporal import (
    TemporalSplit,
    assert_no_temporal_leakage,
    edges_as_of,
    split_masks,
)

__all__ = [
    "TemporalSplit",
    "split_masks",
    "edges_as_of",
    "assert_no_temporal_leakage",
    "classification_metrics",
    "scramble_edges",
    "random_graph_edges",
    "configuration_model_edges",
    "remove_edges",
    "save_scores",
    "load_scores",
    "content_hash",
    "assert_aligned",
    "paired_bootstrap_difference",
    "straddles_zero",
    "logit_interval",
]
