"""ELL-1 DataSource: the three Elliptic Bitcoin CSVs -> one PyG temporal graph.

Doc `docs/01-elliptic-embedding-model-BUILD.md` §2. Loads all nodes (unknown-class
nodes are kept for message passing and masked in the loss, doc §2.2.2), remaps the
string transaction ids to contiguous node indices, and builds a
``torch_geometric.data.Data`` with a per-node ``time_step`` (1..49) so the temporal
split in :mod:`gbe.eval` can hold out steps 35-49.

Edges are symmetrised by default (mean aggregation on a sparse directed graph starves
in-degree-0 nodes, doc §3); the directed graph is available behind ``symmetrise=False``
as an ablation, and the choice is recorded on the returned object.

Features are returned **raw** — standardisation is fit-on-train and belongs to the
training step, not the loader, so a full-data statistic never leaks the test window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# Canonical Elliptic dimensions (doc §2.1, ADR-001). Verified by ``load_elliptic(strict=True)``.
# Raw CSV = 167 cols = txId + time_step + 165 features. The paper's "166" counts time_step
# among the local features; we carry time_step on data.time_step (split only), never in x.
EXPECTED_NODES = 203_769
EXPECTED_STEPS = 49
EXPECTED_FEATURES = 165

# Class label encoding. The CSV uses '1'=illicit, '2'=licit, 'unknown'=unlabelled.
ILLICIT, LICIT, UNKNOWN = 1, 0, -1

FEATURES_CSV = "elliptic_txs_features.csv"
CLASSES_CSV = "elliptic_txs_classes.csv"
EDGELIST_CSV = "elliptic_txs_edgelist.csv"


def _load_features(path: Path) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """Return (features_df, x[N,F] float tensor, time_step[N] long tensor).

    The features file has no header: column 0 is the transaction id, column 1 is the
    time step (1..49), and the remaining columns are the node features.
    """
    df = pd.read_csv(path, header=None)
    time_step = torch.tensor(df.iloc[:, 1].to_numpy(), dtype=torch.long)
    x = torch.tensor(df.iloc[:, 2:].to_numpy(), dtype=torch.float)
    return df, x, time_step


def _load_labels(path: Path, txid_to_idx: dict, n_nodes: int) -> torch.Tensor:
    """Return y[N] long tensor with ILLICIT / LICIT / UNKNOWN encoding."""
    df = pd.read_csv(path)  # header: txId, class
    y = torch.full((n_nodes,), UNKNOWN, dtype=torch.long)
    mapping = {"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}
    for txid, cls in zip(df.iloc[:, 0], df.iloc[:, 1]):
        idx = txid_to_idx.get(txid)
        if idx is not None:
            y[idx] = mapping.get(str(cls), UNKNOWN)
    return y


def _load_edges(path: Path, txid_to_idx: dict) -> torch.Tensor:
    """Return edge_index[2,E] long tensor, remapping string txIds to node indices."""
    df = pd.read_csv(path)  # header: txId1, txId2
    src = df.iloc[:, 0].map(txid_to_idx)
    dst = df.iloc[:, 1].map(txid_to_idx)
    if src.isna().any() or dst.isna().any():
        raise ValueError("edgelist references a txId absent from the features file.")
    edges = np.stack([src.to_numpy(), dst.to_numpy()])
    return torch.from_numpy(edges).long()


def load_elliptic(
    data_dir: str | Path = "data/elliptic",
    symmetrise: bool = True,
    strict: bool = True,
) -> Data:
    """Load the Elliptic dataset into a PyG :class:`Data` object.

    Args:
        data_dir: directory holding the three ``elliptic_txs_*.csv`` files.
        symmetrise: if True (default) make edges undirected; if False keep them directed
            (the ablation). The choice is recorded on ``data.symmetrised``.
        strict: if True, assert the canonical 203,769 nodes / 49 steps / 166 features.
            Pass False for tiny fixtures or Elliptic++.

    Returns:
        ``Data(x, edge_index, y, time_step, labelled_mask, symmetrised)``. All nodes are
        present (unknowns kept for message passing); ``labelled_mask = y != UNKNOWN``.
    """
    data_dir = Path(data_dir)
    df_feat, x, time_step = _load_features(data_dir / FEATURES_CSV)

    txids = df_feat.iloc[:, 0].tolist()
    txid_to_idx = {txid: i for i, txid in enumerate(txids)}
    n_nodes = len(txids)

    y = _load_labels(data_dir / CLASSES_CSV, txid_to_idx, n_nodes)
    edge_index = _load_edges(data_dir / EDGELIST_CSV, txid_to_idx)

    if symmetrise:
        edge_index = to_undirected(edge_index, num_nodes=n_nodes)

    if strict:
        _assert_canonical(n_nodes, time_step, x)

    data = Data(x=x, edge_index=edge_index, y=y, time_step=time_step)
    data.labelled_mask = y != UNKNOWN
    data.symmetrised = bool(symmetrise)
    return data


def _assert_canonical(n_nodes: int, time_step: torch.Tensor, x: torch.Tensor) -> None:
    """Assert the pinned Elliptic dimensions; escalate a feature-count mismatch to the doc."""
    assert n_nodes == EXPECTED_NODES, f"expected {EXPECTED_NODES} nodes, got {n_nodes}"

    n_steps = int(time_step.unique().numel())
    assert n_steps == EXPECTED_STEPS, f"expected {EXPECTED_STEPS} time steps, got {n_steps}"
    assert int(time_step.max()) == EXPECTED_STEPS, (
        f"expected max time step {EXPECTED_STEPS}, got {int(time_step.max())}"
    )

    n_features = x.shape[1]
    if n_features != EXPECTED_FEATURES:
        # Doc-first escalation (CLAUDE.md): the doc pins 166; some mirrors of the CSV
        # yield a different feature count. Do NOT silently accept it — surface the
        # discrepancy so the doc is reconciled (or amended via ADR) before proceeding.
        raise AssertionError(
            f"feature-count mismatch: loaded {n_features}, doc-01 §2.1 pins "
            f"{EXPECTED_FEATURES}. This is a doc-vs-data discrepancy — reconcile the "
            "build doc (doc-first) before trusting any run, do not edit this constant "
            "to make it pass."
        )


def derive_edge_time(edge_index: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
    """Elliptic's edge dates: the time step both endpoints share (ADR-011 clause 6).

    Elliptic ships no edge timestamps, but every edge joins two transactions of the same step
    (Weber et al. 2019; measured 468,710 / 468,710), so that step *is* the edge's date. The core
    requires edge dates and never infers them (`gbe.eval.temporal`), so the inference lives here,
    where the reason it is valid is known — and it is **asserted, not assumed**: an edge joining
    two different steps raises, because on such an edge "the later endpoint's step" would be the
    node-induced reading ADR-011 retired.

    Recompute it from whichever ``edge_index`` is actually in use rather than storing it on the
    loaded graph: the Phase-3 ablation arms rewire the edges (within a step), so a date attached
    at load time would describe edges that no longer exist.
    """
    src_t = time_step[edge_index[0]]
    dst_t = time_step[edge_index[1]]
    crossing = src_t != dst_t
    if bool(crossing.any()):
        raise AssertionError(
            f"{int(crossing.sum())} edge(s) join two different time steps. Elliptic's edges are "
            "within-step by construction, so this edge set is not Elliptic's — its dates cannot "
            "be derived from node steps (ADR-011 clause 6)."
        )
    return src_t.clone()
