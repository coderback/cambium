"""gbe.gnn — GNN backbone + Head registry + the shared trainer. LayerNorm/GraphNorm only
(see gbe/CLAUDE.md)."""

from gbe.gnn.backbone import BACKBONES, GCN, GNNBackbone, GraphSAGE
from gbe.gnn.heads import HEADS, NodeClassificationHead
from gbe.gnn.model import NodeClassifier
from gbe.gnn.train import (
    GNNHParams,
    balanced_class_weights,
    build_model,
    fan_out_for,
    train_model,
)

__all__ = [
    "GNNBackbone",
    "GraphSAGE",
    "GCN",
    "BACKBONES",
    "NodeClassificationHead",
    "HEADS",
    "NodeClassifier",
    "GNNHParams",
    "balanced_class_weights",
    "build_model",
    "fan_out_for",
    "train_model",
]
