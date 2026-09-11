"""DGF-1 DataSource: the DGraph-Fin snapshot -> one PyG temporal graph plus dated views.

Doc `docs/02-dgraph-fin-embedding-model-BUILD.md` §2.2, with the split and protocol fixed by
ADR-011. What lives here rather than in the core (ADR-011 clause 6):

* **node time = the earliest incident edge time** (clause 1). DGraph dates edges, not users; the
  minimum is prefix-determined, so window membership at any boundary depends only on edges dated at
  or before it (ADR-010's constraint).
* **the labelled mask** — ``y`` carries four values: 0 normal, 1 fraud, and **two** background
  values, 2 and 3 (ADR-010). Background users stay in the graph for message passing and are masked
  from the loss.
* **graph views** — the graph as of a date (the core's :func:`gbe.eval.edges_as_of`), with reverse
  edges (doc-02 §2.2.4) and the node statistics derived from exactly that edge set (§2.2.4b,
  ADR-011 clause 3): the 11-wide edge-type histogram and two recency features, measured from the
  view's own cutoff, never from step 821.

What it deliberately does **not** do: scale or transform features. Scaling is fitted on
training-window users only and belongs to the training step (ADR-011 clause 3, clause-5 test 7).
The raw ``x`` is returned untouched — note that **~50% of its values are exactly -1** on the
snapshot (measured 2026-09-11), most likely a missing-value sentinel; that is the transform's
problem, and a fit over all users would leak test-window statistics into it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import Data

from gbe.eval import TemporalSplit, edges_as_of

# Pinned from doc-02 §2.1 as amended by ADR-010; verified by scripts/verify_dgraph_snapshot.py.
EXPECTED_NODES = 3_700_550
EXPECTED_EDGES = 4_300_999
EXPECTED_FEATURES = 17
EDGE_TYPE_MIN, EDGE_TYPE_MAX = 1, 11          # 11 contiguous types, no type 0 (ADR-010)
NUM_EDGE_TYPES = EDGE_TYPE_MAX - EDGE_TYPE_MIN + 1

NORMAL, FRAUD = 0, 1
BACKGROUND = (2, 3)

RECENCY_FEATURES = ("steps_since_last_edge", "steps_since_first_edge")
VIEW_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"edge_type_{t}" for t in range(EDGE_TYPE_MIN, EDGE_TYPE_MAX + 1)
) + RECENCY_FEATURES


def node_time_earliest_edge(edge_index: Tensor, edge_time: Tensor, num_nodes: int) -> Tensor:
    """Earliest incident-edge time per node, both endpoints credited; ``-1`` for a node with none.

    Both endpoints count because being named as someone's emergency contact is evidence that the
    node exists (ADR-011 clause 1). A min-reduction, so it is order-independent.
    """
    sentinel = torch.iinfo(torch.long).max
    out = torch.full((num_nodes,), sentinel, dtype=torch.long)
    for endpoint in (edge_index[0], edge_index[1]):
        out.scatter_reduce_(0, endpoint, edge_time.long(), reduce="amin")
    out[out == sentinel] = -1
    return out


def labelled_mask(y: Tensor) -> Tensor:
    """``(y == 0) | (y == 1)`` — never ``y != <one background value>``: there are two (ADR-010)."""
    return (y == NORMAL) | (y == FRAUD)


def prepare_dgraph(
    x: Tensor,
    edge_index: Tensor,
    edge_time: Tensor,
    edge_type: Tensor,
    y: Tensor,
    *,
    reverse_edges: bool = True,
    official_masks: dict[str, Tensor] | None = None,
    strict: bool = False,
) -> Data:
    """Assemble the DGF-1 graph from raw tensors (the testable core of :func:`load_dgraph`).

    ``edge_index`` is kept **forward-only** (as shipped); reverse edges are added per view in
    :func:`graph_view`, so each contact is counted once per endpoint in the histogram.

    Raises (doc-first escalation, never absorbed):
      * an edge type outside 1..11 — ADR-010 measured exactly 11; a 0 or 12 means the data or the
        doc changed;
      * a node with no incident edge — it has no time and cannot be placed in any window
        (ADR-010 measured zero; ADR-011 clause 1 requires asserting it);
      * with ``strict``, any departure from the pinned snapshot shape.
    """
    n = x.size(0)
    if edge_time.dim() != 1 or edge_type.dim() != 1 or edge_time.numel() != edge_index.size(1) \
            or edge_type.numel() != edge_index.size(1):
        raise ValueError("edge_time and edge_type must be 1-D with one value per edge.")

    if edge_index.numel() and (int(edge_type.min()) < EDGE_TYPE_MIN or int(edge_type.max()) > EDGE_TYPE_MAX):
        raise AssertionError(
            f"edge_type outside {EDGE_TYPE_MIN}..{EDGE_TYPE_MAX} (got {int(edge_type.min())}.."
            f"{int(edge_type.max())}). ADR-010 fixed 11 types with no type 0 — reconcile the doc "
            "before trusting any run; do not widen the histogram to make this pass."
        )

    node_time = node_time_earliest_edge(edge_index, edge_time, n)
    isolated = int((node_time < 0).sum())
    if isolated:
        raise AssertionError(
            f"{isolated} node(s) have no incident edge, so no time: they cannot be placed in any "
            "window (ADR-011 clause 1). ADR-010 measured zero on the snapshot."
        )

    if strict:
        _assert_canonical(x, edge_index)

    data = Data(
        x=x, edge_index=edge_index, edge_time=edge_time.long(), edge_type=edge_type.long(), y=y,
    )
    data.node_time = node_time
    data.labelled_mask = labelled_mask(y)
    data.reverse_edges = bool(reverse_edges)
    for name, mask in (official_masks or {}).items():
        data[name] = mask  # the official RANDOM split: reported track only, never gated (ADR-010)
    return data


def load_dgraph(
    root: str | Path = "data/dgraph", reverse_edges: bool = True, strict: bool = True
) -> Data:
    """Load the DGraph-Fin snapshot (PyG ``DGraphFin``) into the DGF-1 graph.

    PyG does not download it (registration-gated); see ``scripts/verify_dgraph_snapshot.py``.
    """
    from torch_geometric.datasets import DGraphFin  # adapter-only import (gbe/ never sees it)

    raw = DGraphFin(root=str(root))[0]
    return prepare_dgraph(
        raw.x, raw.edge_index, raw.edge_time, raw.edge_type, raw.y,
        reverse_edges=reverse_edges,
        official_masks={
            "official_train_mask": raw.train_mask,
            "official_val_mask": raw.val_mask,
            "official_test_mask": raw.test_mask,
        },
        strict=strict,
    )


def _assert_canonical(x: Tensor, edge_index: Tensor) -> None:
    for what, got, want in (
        ("node count", x.size(0), EXPECTED_NODES),
        ("feature count", x.size(1), EXPECTED_FEATURES),
        ("edge count", edge_index.size(1), EXPECTED_EDGES),
    ):
        if got != want:
            raise AssertionError(
                f"doc-vs-data discrepancy on {what}: loaded {got:,}, doc-02 §2.1 pins {want:,}. "
                "Reconcile the doc first (ADR-001 precedent); do not edit the constant."
            )


# ------------------------------------------------------------------------------ graph views


@dataclass(frozen=True)
class GraphView:
    """The graph as it stood at ``t_max``, as one model input.

    ``edge_index``/``edge_time``/``edge_type`` are the message-passing edges dated ``<= t_max``
    (reverse edges included when the graph enables them, each keeping its forward edge's date and
    type). ``node_features`` are the view-derived statistics, ``[N, len(VIEW_FEATURE_NAMES)]``,
    computed from the forward edges of this view only. ``in_view`` marks users who exist by
    ``t_max``; users outside it have no edge in the view (so are never reached by message passing)
    and all-zero view features.
    """

    t_max: int
    edge_index: Tensor
    edge_time: Tensor
    edge_type: Tensor
    node_features: Tensor
    in_view: Tensor
    feature_names: tuple[str, ...] = VIEW_FEATURE_NAMES


def view_node_features(
    edge_index: Tensor,
    edge_time: Tensor,
    edge_type: Tensor,
    node_time: Tensor,
    t_max: int,
) -> tuple[Tensor, Tensor]:
    """The 11-wide edge-type histogram + two recency features, from **forward** edges dated <= t_max.

    * ``edge_type_k`` — incident edges of type ``k``, both endpoints credited, so a row sums to the
      user's degree in the view (ADR-011 clause 4).
    * ``steps_since_last_edge`` — ``t_max`` minus the user's most recent incident edge in the view.
    * ``steps_since_first_edge`` — ``t_max`` minus ``node_time``: the user's age at the view.

    Both recency features are measured from **this view's cutoff** — never step 821, which would
    bake the dataset's end date into every training feature (ADR-011 clause 3). Users not yet in
    the view get zeros.

    Returns ``(features [N, 13] float, in_view [N] bool)``. Raises if handed an edge dated after
    ``t_max``, or if the edge set and ``node_time`` disagree about who is in the view (they cannot,
    for a node time derived by :func:`node_time_earliest_edge` — the prefix property).
    """
    if edge_time.numel() and int(edge_time.max()) > t_max:
        raise AssertionError(f"view features were handed an edge dated after t_max={t_max}.")
    n = node_time.numel()

    hist = torch.zeros(n * NUM_EDGE_TYPES, dtype=torch.long)
    col = edge_type.long() - EDGE_TYPE_MIN
    ones = torch.ones(edge_index.size(1), dtype=torch.long)
    for endpoint in (edge_index[0], edge_index[1]):
        hist.index_add_(0, endpoint * NUM_EDGE_TYPES + col, ones)

    last = torch.full((n,), -1, dtype=torch.long)
    for endpoint in (edge_index[0], edge_index[1]):
        last.scatter_reduce_(0, endpoint, edge_time.long(), reduce="amax")

    in_view = (node_time >= 0) & (node_time <= t_max)
    if not torch.equal(in_view, last >= 0):
        raise AssertionError(
            "node_time and the view's edges disagree about who exists by t_max — node_time was "
            "not derived from these edges (ADR-011 clause 1 prefix property)."
        )

    zero = torch.zeros(n, dtype=torch.long)
    since_last = torch.where(in_view, t_max - last, zero)
    since_first = torch.where(in_view, t_max - node_time, zero)
    feats = torch.cat(
        [hist.view(n, NUM_EDGE_TYPES), since_last[:, None], since_first[:, None]], dim=1
    ).float()
    return feats, in_view


def graph_view(data: Data, t_max: int) -> GraphView:
    """The graph as of ``t_max`` — the training graph (``t_max = train_max``, ADR-011 clause 3) or
    a window's scoring view (``t_max`` = the window's last step, clause 4)."""
    keep = data.edge_time <= t_max  # the same predicate edges_as_of applies, for the attributes
    fwd = edges_as_of(data.edge_index, edge_time=data.edge_time, t_max=t_max)
    et, ety = data.edge_time[keep], data.edge_type[keep]
    feats, in_view = view_node_features(fwd, et, ety, data.node_time, t_max)

    if data.reverse_edges:
        ei = torch.cat([fwd, fwd.flip(0)], dim=1)
        et, ety = torch.cat([et, et]), torch.cat([ety, ety])
    else:
        ei = fwd
    return GraphView(t_max=int(t_max), edge_index=ei, edge_time=et, edge_type=ety,
                     node_features=feats, in_view=in_view)


# ------------------------------------------------------------------------------ node sets


def train_seed_mask(data: Data, split: TemporalSplit) -> Tensor:
    """Labelled users who exist by ``train_max`` — the only nodes a loss may touch (clause-5 test 2)."""
    return data.labelled_mask & (data.node_time <= split.train_max)


def window_target_mask(data: Data, split: TemporalSplit) -> Tensor:
    """Scored targets: labelled users whose node time falls inside the window (ADR-011 clause 2).

    A labelled user who appeared before the window is never a target, even though users persist.
    """
    hi = split.test_max if split.test_max is not None else int(data.node_time.max())
    return data.labelled_mask & (data.node_time >= split.test_min) & (data.node_time <= hi)
