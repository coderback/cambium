"""Prototype: what matched-time's arms receive, so the test can read it at the model's door (ADR-023 clause 2).

The test this exists for reads the tensor actually passed to the model, row by row, and compares it with
an oracle built only from independently filtered edges. Reading anything earlier would miss a transform
refit on the scoring view, and reading the sampler's own edge dates would miss a misdated reverse edge.

Runs on a synthetic graph only. It reads no dataset, uses no real label and computes no model metric.
This is not row 6b's builder, which goes in ``adapters/dgf1/matched.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from adapters.dgf1.baselines_tabular import FloorDesign  # noqa: E402
from adapters.dgf1.datasource_dgraph import (  # noqa: E402
    EDGE_TYPE_MIN,
    NUM_EDGE_TYPES,
    GraphView,
    node_time_earliest_edge,
    view_node_features,
)
from adapters.dgf1.features import INPUT_FEATURE_NAMES  # noqa: E402
from adapters.dgf1.sampling import assert_temporal_batch, first_appearance_loader  # noqa: E402
from gbe.features.scaling import Standardizer  # noqa: E402


@dataclass(frozen=True)
class Graph:
    """A graph with dated forward edges. Reverse edges exist only inside ``eligible_edges``."""

    x: Tensor
    fwd: Tensor
    edge_time: Tensor
    edge_type: Tensor
    node_time: Tensor
    y: Tensor

    def as_of(self, t_max: int) -> tuple[Tensor, Tensor, Tensor]:
        keep = self.edge_time <= t_max
        return self.fwd[:, keep], self.edge_time[keep], self.edge_type[keep]


class AsOfStatistics:
    """Each (node, time) pair's view statistics, read from sorted dated forward edges by binary search."""

    def __init__(self, fwd: Tensor, edge_time: Tensor, edge_type: Tensor, node_time: Tensor):
        self.span = int(edge_time.max()) + 1
        nodes = torch.cat([fwd[0], fwd[1]])
        times = torch.cat([edge_time, edge_time])
        cols = torch.cat([edge_type, edge_type]) - EDGE_TYPE_MIN
        key = nodes * self.span + times
        order = torch.argsort(key)
        m = key.numel()
        cum = torch.zeros((m + 1, NUM_EDGE_TYPES), dtype=torch.int32)
        cum[torch.arange(1, m + 1), cols[order]] = 1
        self.cum = cum.cumsum(0, dtype=torch.int32)
        self.key = key[order]
        self.times = times[order]
        self.node_time = node_time

    def __call__(self, nodes: Tensor, at: Tensor) -> Tensor:
        lo = torch.searchsorted(self.key, nodes * self.span)
        hi = torch.searchsorted(self.key, nodes * self.span + at, right=True)
        if not bool((hi > lo).all()):
            raise AssertionError("a node in a subgraph has no edge dated at or before its seed's time")
        hist = (self.cum[hi] - self.cum[lo]).float()
        since_last = (at - self.times[hi - 1]).float()
        since_first = (at - self.node_time[nodes]).float()
        return torch.cat([hist, since_last[:, None], since_first[:, None]], dim=1)


def statistics_table(g: Graph, window_end: int) -> AsOfStatistics:
    """The table stops at the window's end, so the pilot is no unrecorded look at later edges (clause 2)."""
    fwd, et, ety = g.as_of(window_end)
    return AsOfStatistics(fwd, et, ety, g.node_time)


def eligible_edges(g: Graph, window_end: int) -> GraphView:
    """Message-passing edges as of the window's end. A reverse edge keeps its forward edge's date."""
    fwd, et, ety = g.as_of(window_end)
    both, times, types = torch.cat([fwd, fwd.flip(0)], dim=1), torch.cat([et, et]), torch.cat([ety, ety])
    empty = torch.empty(0)
    return GraphView(t_max=window_end, edge_index=both, edge_time=times, edge_type=types,
                     node_features=empty, in_view=empty)


def fit_transform(g: Graph, train_max: int) -> Standardizer:
    """ADR-011 clause 3's transform: users up to ``train_max``, from the training view as of ``train_max``."""
    fwd, et, ety = g.as_of(train_max)
    feats, _ = view_node_features(fwd, et, ety, g.node_time, train_max)
    return Standardizer.fit(torch.cat([g.x, feats], dim=1), g.node_time <= train_max)


def check_seed_recency(batch, raw: Tensor) -> None:
    """The rows about to go through the transform: a seed's two recency values are 0 at its own time."""
    if bool(raw[: batch.batch_size, -2:].any()):
        raise AssertionError("a seed's recency is not 0 at its own node time")


def row_times(batch) -> Tensor:
    """Each row's seed's time."""
    return batch.seed_time[batch.batch]


def raw_inputs(batch, stats: AsOfStatistics, at: Tensor) -> Tensor:
    """Raw features beside each node's statistics as of ``at``, before the frozen transform."""
    return torch.cat([batch.x, stats(batch.n_id, at)], dim=1)


def gnn_pass(model, g: Graph, seeds: Tensor, window_end: int, transform: Standardizer, num_neighbors, batch_size: int,
             shuffle: bool, loss=None) -> None:
    """Every batch of one pass. ``model`` is called at the door, with the same arguments the real loop uses."""
    view = eligible_edges(g, window_end)
    stats = statistics_table(g, window_end)
    for batch in first_appearance_loader(g.x, view, g.node_time, seeds, num_neighbors, batch_size, shuffle=shuffle):
        assert_temporal_batch(batch)
        at = row_times(batch)
        raw = raw_inputs(batch, stats, at)
        check_seed_recency(batch, raw)
        _, logits = model(transform.transform(raw), batch.edge_index)
        if loss is not None:
            own = batch.n_id[: batch.batch_size]
            loss(logits[: batch.batch_size], g.y[own])


def train_pass(model, g: Graph, seeds: Tensor, train_max: int, transform: Standardizer, fan_out, batch_size: int,
               loss=None) -> None:
    gnn_pass(model, g, seeds, train_max, transform, fan_out, batch_size, shuffle=True, loss=loss)


def score_pass(model, g: Graph, targets: Tensor, window_end: int, transform: Standardizer, layers: int, batch_size: int) -> None:
    gnn_pass(model, g, targets, window_end, transform, [-1] * layers, batch_size, shuffle=False)


def own_time_rows(g: Graph, users: Tensor, window_end: int) -> Tensor:
    """Each user's own row: raw features beside statistics as of the user's own time, untransformed."""
    return torch.cat([g.x[users], statistics_table(g, window_end)(users, g.node_time[users])], dim=1)


def floor_pass(fit_predict, g: Graph, seeds: Tensor, targets: Tensor, train_max: int, window_end: int, seed: int):
    """The floor's door is the design handed to ``fit_predict`` (``adapters/dgf1/baselines_tabular.py``)."""
    X_train = own_time_rows(g, seeds, train_max)
    X_score = own_time_rows(g, targets, window_end)
    design = FloorDesign(X_train=X_train.numpy(), y_train=g.y[seeds].numpy(), X_score=X_score.numpy(),
                         y_score=g.y[targets].numpy(), feature_names=INPUT_FEATURE_NAMES, meta={})
    return fit_predict("xgboost", design, seed)


# ------------------------------------------------------------------ the exploratory rows (ADR-023 clause 4)


def window_only_forward(g: Graph, first: int, last: int) -> tuple[Tensor, Tensor, Tensor]:
    """ADR-011's window-only view: forward edges with both endpoints first appearing inside the window."""
    inside = (g.node_time >= first) & (g.node_time <= last)
    keep = (g.edge_time <= last) & inside[g.fwd[0]] & inside[g.fwd[1]]
    return g.fwd[:, keep], g.edge_time[keep], g.edge_type[keep]


def window_only_statistics(g: Graph, first: int, last: int) -> Tensor:
    """Statistics from the view's own forward edges, counted from the window's end. Age runs from a user's first
    edge in this view, and a user the view isolates gets zeros, as ``view_node_features`` requires."""
    fwd, et, ety = window_only_forward(g, first, last)
    feats, _ = view_node_features(fwd, et, ety, node_time_earliest_edge(fwd, et, g.x.size(0)), last)
    return feats


def window_only_pass(model, g: Graph, targets: Tensor, first: int, last: int, transform: Standardizer, layers: int,
                     batch_size: int) -> None:
    fwd, _, _ = window_only_forward(g, first, last)
    stats = window_only_statistics(g, first, last)
    graph = Data(x=g.x, edge_index=torch.cat([fwd, fwd.flip(0)], dim=1), num_nodes=g.x.size(0))
    loader = NeighborLoader(graph, num_neighbors=[-1] * layers, input_nodes=targets, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    for batch in loader:
        model(transform.transform(torch.cat([batch.x, stats[batch.n_id]], dim=1)), batch.edge_index)


def _refuse(view: str) -> None:
    raise ValueError(f"the exploratory path refuses view {view!r}: its models give no gated test-window score")


def exploratory_gnn(model, view: str, g: Graph, targets: Tensor, first: int, last: int, transform: Standardizer,
                    layers: int, batch_size: int) -> None:
    """A gated model, its frozen transform, and the view's inputs."""
    if view == "first_appearance":
        score_pass(model, g, targets, last, transform, layers, batch_size)
    elif view == "window_only":
        window_only_pass(model, g, targets, first, last, transform, layers, batch_size)
    else:
        _refuse(view)


def gated_rows(g: Graph, users: Tensor, t: int) -> Tensor:
    """A user's row under the gated view as of ``t``: the rows the floor's gated model was trained on."""
    fwd, et, ety = g.as_of(t)
    feats, _ = view_node_features(fwd, et, ety, g.node_time, t)
    return torch.cat([g.x[users], feats[users]], dim=1)


def exploratory_floor(fit_predict, view: str, g: Graph, seeds: Tensor, targets: Tensor, train_max: int, first: int,
                      last: int, seed: int):
    """A floor trained on the gated view, scored under the view. Its rows stay untransformed."""
    if view == "first_appearance":
        X_score = own_time_rows(g, targets, last)
    elif view == "window_only":
        X_score = torch.cat([g.x[targets], window_only_statistics(g, first, last)[targets]], dim=1)
    else:
        _refuse(view)
    design = FloorDesign(X_train=gated_rows(g, seeds, train_max).numpy(), y_train=g.y[seeds].numpy(),
                         X_score=X_score.numpy(), y_score=g.y[targets].numpy(), feature_names=INPUT_FEATURE_NAMES, meta={})
    return fit_predict("xgboost", design, seed)
