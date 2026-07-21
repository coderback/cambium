"""Temporal-holdout split machinery — domain-agnostic (no dataset values baked in).

The single honest way to claim *prediction* rather than *memorisation*: train on the
graph as of <= a cutoff, test on targets that appear after it (doc-00 §7). This module
is pure machinery — a :class:`TemporalSplit` carries the cutoff, and the functions turn a
per-node time tensor into train/test masks, a strict-inductive train subgraph, and a
leakage assertion. The concrete cutoff values live in each adapter's config, never here
(gbe/CLAUDE.md rule 3): EDR-1/EDL-1 will split on filing *dates*, not integer steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.utils import subgraph


@dataclass(frozen=True)
class TemporalSplit:
    """An immutable temporal holdout: train on ``time <= train_max``, test after.

    Args:
        train_max: last time value included in training (inclusive).
        test_min: first time value included in the test window (inclusive).
        test_max: last test time value (inclusive); ``None`` means "to the end".
        time_attr: name of the per-node time attribute on the graph (provenance only;
            the functions below take the time tensor explicitly).
    """

    train_max: int
    test_min: int
    test_max: int | None = None
    time_attr: str = "time_step"

    def __post_init__(self) -> None:
        if self.test_min <= self.train_max:
            raise ValueError(
                f"test_min ({self.test_min}) must be > train_max ({self.train_max}); "
                "an overlapping split is a temporal leak by construction."
            )
        if self.test_max is not None and self.test_max < self.test_min:
            raise ValueError(
                f"test_max ({self.test_max}) must be >= test_min ({self.test_min})."
            )


def split_masks(time_step: Tensor, split: TemporalSplit) -> tuple[Tensor, Tensor]:
    """Boolean (train_mask, test_mask) over nodes from a per-node time tensor.

    Train is ``time <= train_max``; test is ``test_min <= time <= test_max`` (open-ended
    when ``test_max is None``). The two masks are disjoint by construction of the split.
    """
    train_mask = time_step <= split.train_max
    test_mask = time_step >= split.test_min
    if split.test_max is not None:
        test_mask = test_mask & (time_step <= split.test_max)
    return train_mask, test_mask


def induced_train_subgraph(
    edge_index: Tensor,
    time_step: Tensor,
    split: TemporalSplit,
) -> tuple[Tensor, Tensor]:
    """Strict-inductive train subgraph: edges with *both* endpoints in the train set.

    This is the enforcement of the strict inductive protocol (doc-01 §5): the encoder
    only ever sees train-period nodes, so no message-passing edge may reach a test node.
    Any edge crossing the cutoff is dropped.

    Returns:
        ``(edge_index_train, train_node_mask)``. Node ids are *not* relabelled, so the
        returned edge_index still indexes into the original node tensors.
    """
    train_node_mask = time_step <= split.train_max
    edge_index_train, _ = subgraph(
        train_node_mask,
        edge_index,
        relabel_nodes=False,
        num_nodes=time_step.numel(),
    )
    return edge_index_train, train_node_mask


def assert_no_temporal_leakage(
    edge_index: Tensor,
    time_step: Tensor,
    split: TemporalSplit,
) -> None:
    """Raise if any edge in ``edge_index`` touches a post-cutoff node.

    Call this on the edge set actually used for training. It has teeth: pass the raw
    (un-induced) train edges and it raises on any cross-cutoff edge; pass the output of
    :func:`induced_train_subgraph` and it must be silent. Reusable by every DataSource's
    leakage test (doc-00 §7 — leakage checks are harness-level, not adapter code).
    """
    endpoint_time = time_step[edge_index]  # shape [2, E]
    crossing = endpoint_time > split.train_max
    if bool(crossing.any()):
        n_bad = int(crossing.any(dim=0).sum())
        raise AssertionError(
            f"temporal leakage: {n_bad} training edge(s) touch a node with "
            f"time > train_max ({split.train_max}). The training tensors reach the "
            "held-out future window."
        )
