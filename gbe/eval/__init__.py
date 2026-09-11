"""gbe.eval — temporal-holdout harness, metrics, ablations, gate runner."""

from gbe.eval.ablations import (
    configuration_model_edges,
    random_graph_edges,
    remove_edges,
    scramble_edges,
)
from gbe.eval.metrics import classification_metrics
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
]
