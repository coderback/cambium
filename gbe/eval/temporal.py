"""Temporal-holdout split machinery — domain-agnostic (no dataset values baked in).

The single honest way to claim *prediction* rather than *memorisation*: train on the
graph as of <= a cutoff, test on targets that appear after it (doc-00 §7). This module
is pure machinery — a :class:`TemporalSplit` carries the cutoff, and the functions turn
per-node and per-edge times into train/test masks, the graph as of a time, and a leakage
assertion. The concrete cutoff values live in each adapter's config, never here
(gbe/CLAUDE.md rule 3): EDR-1/EDL-1 will split on filing *dates*, not integer steps.

**Edges carry their own dates, and they are a required input (ADR-011).** "The graph as of
``T``" means *every edge dated <= T* — not every edge between nodes that existed by ``T``.
The two coincide only when every edge lies within one time value (Elliptic). On a graph of
persistent entities they do not: two users who both appeared before the cutoff can gain an
edge long after it (DGraph: 399,366 such edges at the training cutoff). The previous
``induced_train_subgraph`` reasoned only about node times and was retired for exactly that
reason. There is deliberately **no default** for ``edge_time``: inferring an edge's date from
its endpoints' times *is* the node-induced reading. A graph without native edge dates derives
them — and asserts the derivation is valid — in its own adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


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


def _check_edge_time(edge_index: Tensor, edge_time: Tensor) -> None:
    if edge_time.dim() != 1 or edge_time.numel() != edge_index.size(1):
        raise ValueError(
            f"edge_time must be 1-D with one date per edge: got shape {tuple(edge_time.shape)} "
            f"for {edge_index.size(1)} edges."
        )


def edges_as_of(edge_index: Tensor, *, edge_time: Tensor, t_max: int) -> Tensor:
    """The graph as of ``t_max``: exactly the edges dated ``<= t_max``, in their original order.

    One function serves both uses in ADR-011: the **training graph** (``t_max = train_max``,
    clause 3) and the **scoring view** of a window (``t_max`` = the window's last step,
    clause 4). Node ids are not relabelled, so the result still indexes the original node
    tensors.

    Edge order is preserved (a boolean mask), which is what keeps the refactor bit-for-bit for
    ELL-1: on a graph whose edges lie within one time value, this returns the same edges in the
    same order as the retired node-induced path (measured: ``torch.equal`` on Elliptic at both
    cutoffs, ADR-011 clause 6), so neighbour sampling — and therefore every number — is unchanged.

    Args:
        edge_index: ``[2, E]`` edge tensor.
        edge_time: ``[E]`` per-edge dates. **Keyword-only and required** — see the module
            docstring for why there is no default.
        t_max: last date included (inclusive).
    """
    _check_edge_time(edge_index, edge_time)
    return edge_index[:, edge_time <= t_max]


def assert_no_temporal_leakage(
    edge_index: Tensor,
    time_step: Tensor,
    split: TemporalSplit,
    *,
    edge_time: Tensor,
) -> None:
    """Raise if any training edge touches a post-cutoff node **or is itself dated post-cutoff**.

    Call this on the edge set actually used for training. Two checks, both needed:

    * **endpoint times** — no edge may reach a node with time ``> train_max``;
    * **edge dates** — no edge may be dated ``> train_max``. This is the check the endpoint
      test cannot do: an edge between two pre-cutoff nodes can still be dated after the cutoff,
      and on a persistent-entity graph that is the common case, not the corner (ADR-011).

    It has teeth: the node-induced graph of a dated-edge fixture fails it, and the output of
    :func:`edges_as_of` at ``split.train_max`` passes it. Reusable by every DataSource's leakage
    test (doc-00 §7 — leakage checks are harness-level, not adapter code).

    Args:
        edge_index: ``[2, E]`` training edges.
        time_step: ``[N]`` per-node times.
        split: the temporal split whose ``train_max`` is the cutoff.
        edge_time: ``[E]`` per-edge dates. **Keyword-only and required.**
    """
    _check_edge_time(edge_index, edge_time)

    endpoint_time = time_step[edge_index]  # shape [2, E]
    crossing = endpoint_time > split.train_max
    if bool(crossing.any()):
        n_bad = int(crossing.any(dim=0).sum())
        raise AssertionError(
            f"temporal leakage: {n_bad} training edge(s) touch a node with "
            f"time > train_max ({split.train_max}). The training tensors reach the "
            "held-out future window."
        )

    late = edge_time > split.train_max
    if bool(late.any()):
        raise AssertionError(
            f"temporal leakage: {int(late.sum())} training edge(s) are dated after "
            f"train_max ({split.train_max}) although both their endpoints are pre-cutoff. "
            "A node-time-only filter admits these; filter with edges_as_of (ADR-011)."
        )
