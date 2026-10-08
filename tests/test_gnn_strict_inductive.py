"""Strict-inductive protocol guards for the ELL-1 GNN trainer (doc-01 §5).

The load-bearing property: the encoder never sees a post-cutoff node in training, and eval
never runs a full-graph forward. These tests assert both on a synthetic temporal graph with a
deliberate cross-cutoff edge, plus a tiny end-to-end smoke (Stage-A verification) proving the
train->eval pipeline runs on CPU and yields in-range illicit-class metrics.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from gbe.eval import (
    TemporalSplit,
    assert_no_temporal_leakage,
    edges_as_of,
    split_masks,
)
from gbe.run.seeding import seed_everything
from adapters.ell1.datasource_elliptic import ILLICIT, LICIT, UNKNOWN
from adapters.ell1.train_gnn import GNNHParams, evaluate, standardize_fit_on_train, train_model, build_model
from adapters.ell1.train_gnn import resolve_device

SPLIT = TemporalSplit(train_max=34, test_min=35, test_max=49)


def _synthetic_temporal_graph() -> Data:
    """Illicit at +2 / licit at -2, in both windows; within-step edges + one crossing edge."""
    rows = []  # (center, label, step)
    for t in (10, 20, 30):          # train window
        rows += [(2.0, ILLICIT, t), (-2.0, LICIT, t), (0.0, UNKNOWN, t)]
    for t in (36, 40, 45):          # test window
        rows += [(2.0, ILLICIT, t), (-2.0, LICIT, t), (0.0, UNKNOWN, t)]

    x = torch.tensor([[c, c, c] for c, _, _ in rows], dtype=torch.float)
    y = torch.tensor([lbl for _, lbl, _ in rows], dtype=torch.long)
    ts = torch.tensor([t for _, _, t in rows], dtype=torch.long)

    # within-step edges (illicit<->licit at each step), plus ONE crossing edge node0(step10)->node9(step36)
    within = []
    for base in range(0, len(rows), 3):  # each step contributes a (illicit, licit) pair
        within += [(base, base + 1), (base + 1, base)]
    crossing = [(0, 9)]  # train node -> test node: must be dropped from training
    ei = torch.tensor(within + crossing, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=ei, y=y, time_step=ts)
    data.labelled_mask = y != UNKNOWN
    return data


def _edge_time(data: Data) -> torch.Tensor:
    """Fixture edge dates (ADR-011 makes them a required input). Each edge is dated at its later
    endpoint: an edge cannot exist before both its endpoints do. So the within-step edges carry
    their step and the one crossing edge 0->9 is dated 36 — after the cutoff."""
    return torch.maximum(data.time_step[data.edge_index[0]], data.time_step[data.edge_index[1]])


def _train_graph(data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    et = _edge_time(data)
    edge_train = edges_as_of(data.edge_index, edge_time=et, t_max=SPLIT.train_max)
    return edge_train, et[et <= SPLIT.train_max]


def test_train_subgraph_has_no_post_cutoff_edge():
    data = _synthetic_temporal_graph()
    edge_train, et_train = _train_graph(data)
    train_mask, _ = split_masks(data.time_step, SPLIT)
    # the crossing edge (0->9) must be gone; the guard must be silent on what remains
    assert_no_temporal_leakage(edge_train, data.time_step, SPLIT, edge_time=et_train)
    assert bool((data.time_step[edge_train] <= 34).all())
    assert bool(train_mask[edge_train.unique()].all())


def test_eval_subgraph_touches_only_test_nodes():
    """The eval forward runs over the test-induced subgraph — endpoints all >= test_min."""
    data = _synthetic_temporal_graph()
    _, test_mask = split_masks(data.time_step, SPLIT)
    # same construction evaluate() uses, but without relabelling so we can read times
    sub_edge, _ = subgraph(test_mask, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
    assert sub_edge.numel() > 0
    assert bool((data.time_step[sub_edge] >= SPLIT.test_min).all()), "a train node leaked into eval"


def test_end_to_end_smoke_cpu():
    seed_everything(0)
    data = _synthetic_temporal_graph()
    dev = resolve_device("cpu")
    hp = GNNHParams(backbone="graphsage", num_layers=2, hidden_dim=16,
                    fan_out=(5, 5), epochs=5, batch_size=8)

    x = standardize_fit_on_train(data.x, data.time_step, SPLIT.train_max)
    edge_train, _ = _train_graph(data)
    train_mask, _ = split_masks(data.time_step, SPLIT)
    seeds = train_mask & data.labelled_mask

    model = build_model(x.size(1), hp, dev)
    train_model(model, x, edge_train, data.y, seeds, hp, dev)
    metrics, meta = evaluate(model, x, data.edge_index, data.time_step, data.y, SPLIT, dev)

    assert set(metrics) == {"f1", "recall", "precision", "auc", "auprc"}  # auprc: ADR-007
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert meta["n_test"] == 6 and meta["n_test_illicit"] == 3  # 3 test steps * (1 illicit,1 licit)
