"""gbe.gnn — GNN backbone + Head registry. LayerNorm/GraphNorm only (see gbe/CLAUDE.md)."""

from gbe.gnn.backbone import BACKBONES, GCN, GNNBackbone, GraphSAGE
from gbe.gnn.heads import HEADS, NodeClassificationHead
from gbe.gnn.model import NodeClassifier

__all__ = [
    "GNNBackbone",
    "GraphSAGE",
    "GCN",
    "BACKBONES",
    "NodeClassificationHead",
    "HEADS",
    "NodeClassifier",
]
