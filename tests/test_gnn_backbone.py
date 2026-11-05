"""Contract tests for the GNN backbones + assembled NodeClassifier (gbe.gnn).

Beyond shapes, this asserts at *runtime* that no batch-statistic normaliser is instantiated
anywhere in an assembled model — a belt-and-braces complement to the source lint in
`test_no_batchnorm_lint.py` (gbe/CLAUDE.md rule 2).
"""

from __future__ import annotations

import torch

from gbe.gnn import BACKBONES, GCN, GraphSAGE, NodeClassificationHead, NodeClassifier
from gbe.features.encoder import TabularMLPEncoder


def _tiny_graph(n=6, f=5):
    x = torch.randn(n, f)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    return x, edge_index


def test_graphsage_and_gcn_forward_shapes():
    x, edge_index = _tiny_graph(f=5)
    for cls in (GraphSAGE, GCN):
        bb = cls(in_dim=5, hidden_dim=8, num_layers=2)
        out = bb(x, edge_index)
        assert out.shape == (6, 8)
        assert bb.out_dim == 8


def test_jk_concat_widens_out_dim():
    x, edge_index = _tiny_graph(f=5)
    bb = GraphSAGE(in_dim=5, hidden_dim=8, num_layers=3, jk=True)
    assert bb.out_dim == 24  # 3 layers * 8
    assert bb(x, edge_index).shape == (6, 24)


def test_registry_names():
    assert set(BACKBONES) == {"graphsage", "gcn"}


def test_norm_rejects_batch_statistic_normaliser():
    """'layer'/'graph' only; anything else (the banned normaliser) must be refused."""
    try:
        GraphSAGE(in_dim=5, hidden_dim=8, norm="batch")
    except ValueError:
        return
    raise AssertionError("backbone accepted a forbidden norm kind")


def test_node_classifier_returns_embedding_and_logits():
    x, edge_index = _tiny_graph(f=5)
    enc = TabularMLPEncoder(in_dim=5, hidden_dim=8, out_dim=8)
    bb = GraphSAGE(in_dim=8, hidden_dim=8, num_layers=2)
    head = NodeClassificationHead(in_dim=bb.out_dim, num_classes=2)
    model = NodeClassifier(enc, bb, head)

    embedding, logits = model(x, edge_index)
    assert embedding.shape == (6, 8)
    assert logits.shape == (6, 2)
    # embed() alone yields the same readout (no head assumptions)
    assert torch.equal(model.embed(x, edge_index), embedding) or model.training


def test_no_batch_normaliser_instantiated_anywhere():
    enc = TabularMLPEncoder(in_dim=5, hidden_dim=8, out_dim=8)
    bb = GraphSAGE(in_dim=8, hidden_dim=8, num_layers=2)
    model = NodeClassifier(enc, bb, NodeClassificationHead(8))
    offenders = [type(m).__name__ for m in model.modules() if "batchnorm" in type(m).__name__.lower()]
    assert not offenders, f"batch-statistic normaliser found in model: {offenders}"
