"""FeatureEncoder — node content -> the tensor that seeds the GNN.

Domain-agnostic (gbe/CLAUDE.md rule 1): the *interface* is shared by all four models;
only the implementation differs by content type (doc-00 §3.1). ELL-1/DGF-1 have tabular
features -> a small MLP projector (this file); EDR-1/EDL-1 will add a frozen text-embedder
implementation behind the same interface. There is no pretrained model here — the tabular
path is trained from scratch with the GNN (doc-00 §6 Stage 1).

Normalisation is **LayerNorm**, never BatchNorm: a BatchNorm layer here would mix statistics
across the batch (and, under a full-graph pass, across test-period nodes) — the leakage
vector banned throughout the encoder/GNN stack (doc-00 §3.2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class FeatureEncoder(nn.Module, ABC):
    """content tensor ``[N, in_dim]`` -> feature tensor ``[N, out_dim]``.

    Subclasses set ``out_dim`` (the GNN input dimension) and implement :meth:`forward`.
    """

    out_dim: int

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - interface
        ...


class TabularMLPEncoder(FeatureEncoder):
    """A 2-3 layer MLP projecting raw tabular features into the GNN input dim.

    ``Linear -> LayerNorm -> ReLU -> Dropout`` per hidden layer, then a final ``Linear`` to
    ``out_dim``. This is the *entire* feature path for the tabular models (doc-01 §3): no
    pretrained weights, no text encoder.

    Args:
        in_dim: raw feature dimension (Elliptic: 165).
        hidden_dim: width of the hidden layer(s).
        out_dim: output dimension = the GNN input dimension.
        num_layers: total Linear layers (>=1). 2-3 is the doc default; 1 is a plain projection.
        dropout: dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.out_dim = out_dim

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
