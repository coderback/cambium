"""The shared node-classification training loop (EXTRACT, doc-00 §9).

Extracted from `adapters/ell1/train_gnn.py` when DGF-1 became the second model to need it — the
build-path rule is that the second use case proves the abstraction (doc-00 §1), not the first.
What lives here is what both models do **identically**: assemble encoder → backbone → head from a
set of hyperparameters, and train it with neighbour sampling over a graph the caller has already
made safe.

What deliberately stays in the adapters, because the two models differ:

* **which edges the training graph contains** — ELL-1 dates edges by their shared time step,
  DGF-1 by `edge_time` (ADR-011); both hand the result in as ``edge_index_train``;
* **feature preparation** — ELL-1 standardises 165 raw columns, DGF-1 standardises 30 columns of
  raw + view-derived statistics;
* **scoring** — ELL-1 forwards over a window-induced subgraph, DGF-1 scores three different views
  through a loader (ADR-012 clause 3).

The loop takes the graph and the seed mask as given and never inspects a cutoff, so it cannot
smuggle a leakage decision into the core (gbe/CLAUDE.md rule 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from gbe.features.encoder import TabularMLPEncoder
from gbe.gnn.backbone import BACKBONES
from gbe.gnn.heads import NodeClassificationHead
from gbe.gnn.model import NodeClassifier


@dataclass
class GNNHParams:
    """The Phase-1 hyperparameters (doc-00 §6.4 search space + fixed training budget).

    Field order and :meth:`as_dict` are load-bearing: they land in every run's hashed config, so
    changing either changes config hashes across the registry.
    """

    backbone: str = "graphsage"
    num_layers: int = 2
    hidden_dim: int = 128
    aggr: str = "mean"          # graphsage only; ignored by gcn
    dropout: float = 0.2
    lr: float = 1e-3
    fan_out: tuple[int, ...] = (15, 10)
    encoder_layers: int = 2
    norm: str = "layer"
    epochs: int = 40
    batch_size: int = 1024
    weight_decay: float = 5e-4

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["fan_out"] = list(self.fan_out)
        return d


def balanced_class_weights(
    y: Tensor, seed_mask: Tensor, device: torch.device, classes: Sequence[int] = (0, 1)
) -> Tensor:
    """Inverse-frequency class weights from the seed nodes, in ``classes`` order.

    sklearn's "balanced" scheme: ``n / (k * count_c)``, so the weights average to 1. Counts are
    clamped at 1 so a class absent from the seeds cannot divide by zero. The positive class is a
    parameter, never assumed, because an adapter may encode it either way.
    """
    labels = y[seed_mask]
    n = labels.numel()
    counts = torch.tensor(
        [int((labels == c).sum()) for c in classes], dtype=torch.float
    ).clamp_min(1.0)
    return (n / (len(classes) * counts)).to(device)


def build_model(in_dim: int, hp: GNNHParams, device: torch.device) -> NodeClassifier:
    """Assemble encoder -> backbone -> node-classification head from hyperparameters."""
    encoder = TabularMLPEncoder(
        in_dim=in_dim,
        hidden_dim=hp.hidden_dim,
        out_dim=hp.hidden_dim,
        num_layers=hp.encoder_layers,
        dropout=hp.dropout,
    )
    backbone_cls = BACKBONES[hp.backbone]
    kwargs: dict[str, Any] = dict(
        in_dim=hp.hidden_dim,
        hidden_dim=hp.hidden_dim,
        num_layers=hp.num_layers,
        dropout=hp.dropout,
        norm=hp.norm,
    )
    if hp.backbone == "graphsage":
        kwargs["aggr"] = hp.aggr
    backbone = backbone_cls(**kwargs)
    head = NodeClassificationHead(in_dim=backbone.out_dim, num_classes=2)
    return NodeClassifier(encoder, backbone, head).to(device)


def fan_out_for(hp: GNNHParams) -> list[int]:
    """One fan-out per message-passing layer; repeat the last value if the config is shorter.

    doc-00 §6.4 couples layers {2,3} with length-2 fan-outs, so a 3-layer config needs the third
    hop filled in.
    """
    fo = list(hp.fan_out)
    while len(fo) < hp.num_layers:
        fo.append(fo[-1])
    return fo[: hp.num_layers]


def train_model(
    model: NodeClassifier,
    x: Tensor,
    edge_index_train: Tensor,
    y: Tensor,
    train_seed_mask: Tensor,
    hp: GNNHParams,
    device: torch.device,
    on_epoch: Callable[[int, NodeClassifier], None] | None = None,
    classes: Sequence[int] = (0, 1),
) -> NodeClassifier:
    """Train with neighbour sampling over the graph the caller supplies.

    ``edge_index_train`` must already be the training graph — the adapter decides what that means
    (ELL-1: edges dated <= the cutoff by shared step; DGF-1: edges dated <= the cutoff by
    ``edge_time``, ADR-011 clause 3). Seeds are the labelled training nodes, so every sampled
    neighbourhood stays inside that graph.

    ``on_epoch(epoch, model)`` is an optional hook (the ELL-1 sweep uses it to report and prune).
    """
    data = Data(x=x, edge_index=edge_index_train, y=y, num_nodes=x.size(0))
    loader = NeighborLoader(
        data,
        num_neighbors=fan_out_for(hp),
        input_nodes=train_seed_mask,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=0,  # in-process -> deterministic under the run seed
    )
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    loss_fn = nn.CrossEntropyLoss(
        weight=balanced_class_weights(y, train_seed_mask, device, classes)
    )

    for epoch in range(hp.epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            _, logits = model(batch.x, batch.edge_index)
            seeds = batch.batch_size  # first `batch_size` nodes are the seeds
            loss = loss_fn(logits[:seeds], batch.y[:seeds])
            loss.backward()
            opt.step()
        if on_epoch is not None:
            on_epoch(epoch, model)
    return model
