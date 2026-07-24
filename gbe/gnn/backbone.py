"""GNN backbones — the shared message-passing core (doc-00 §3.2).

Two backbones, frozen zoo (doc-00 §3.2): :class:`GraphSAGE` (homogeneous / inductive, the
ELL-1/DGF-1 backbone) and :class:`GCN` (the standard-GNN comparator). Both share one stack
structure and differ only in the convolution used, so a model swaps backbone by name via
:data:`BACKBONES` without touching the trainer.

Normalisation is LayerNorm or GraphNorm only — never the batch-statistic normaliser banned
by gbe/CLAUDE.md rule 2 (its statistics, computed over a batch that under a full-graph pass
includes test-period nodes, are the Elliptic 0.807->0.12 leakage vector). Depth defaults to
2 (doc-00 §3.2); an optional Jumping-Knowledge concat is wired but off by default (the
Phase-3 ablation row).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv, GraphNorm, SAGEConv


def _make_norm(kind: str, dim: int) -> nn.Module:
    """LayerNorm / GraphNorm only. Any other kind (notably the banned batch normaliser)
    is rejected here so it can never enter the stack."""
    kind = kind.lower()
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "graph":
        return GraphNorm(dim)
    raise ValueError(
        f"norm must be 'layer' or 'graph' (gbe/CLAUDE.md rule 2), got {kind!r}"
    )


class GNNBackbone(nn.Module):
    """A stack of message-passing layers with norm + ReLU + dropout between them.

    Subclasses supply a ``conv_builder(in_dim, out_dim) -> MessagePassing``. The readout is
    the final node representation (doc-00 §3.3): no head-specific assumptions, so it can feed
    a classifier, a vector index, or a future soft-token projector alike.

    Args:
        in_dim: input feature dimension (the FeatureEncoder's ``out_dim``).
        hidden_dim: hidden / output width per layer.
        num_layers: number of message-passing layers (2 default; 3 with a multi-hop story).
        dropout: dropout probability between layers.
        norm: 'layer' or 'graph'.
        jk: if True, concatenate every layer's output (Jumping-Knowledge); ``out_dim`` then
            becomes ``num_layers * hidden_dim``. Default False (Phase-3 ablation).
    """

    def __init__(
        self,
        conv_builder: Callable[[int, int], nn.Module],
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        norm: str = "layer",
        jk: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = num_layers
        self.dropout = dropout
        self.jk = jk
        self.out_dim = num_layers * hidden_dim if jk else hidden_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = in_dim
        for _ in range(num_layers):
            self.convs.append(conv_builder(d, hidden_dim))
            self.norms.append(_make_norm(norm, hidden_dim))
            d = hidden_dim

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        outs: list[Tensor] = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            # Dropout on every layer's activation, including the last, so the readout is
            # regularised the same way whether or not a head follows.
            x = torch.dropout(x, p=self.dropout, train=self.training)
            outs.append(x)
        if self.jk:
            return torch.cat(outs, dim=-1)
        return x


class GraphSAGE(GNNBackbone):
    """Inductive GraphSAGE (Hamilton et al. 2017). Aggregator is a hyperparameter."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        aggr: str = "mean",
        norm: str = "layer",
        jk: bool = False,
    ) -> None:
        super().__init__(
            conv_builder=lambda i, o: SAGEConv(i, o, aggr=aggr),
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            jk=jk,
        )


class GCN(GNNBackbone):
    """The standard GCN comparator (Kipf & Welling 2017). No aggregator choice."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        norm: str = "layer",
        jk: bool = False,
    ) -> None:
        super().__init__(
            conv_builder=lambda i, o: GCNConv(i, o),
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            jk=jk,
        )


# Name -> class. A model selects its backbone by name (doc-00 §3.2 frozen zoo); GATv2 would
# enter here only as a named ablation, new architectures only via an ADR.
BACKBONES: dict[str, type[GNNBackbone]] = {
    "graphsage": GraphSAGE,
    "gcn": GCN,
}
