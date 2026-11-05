"""Temporal neighbour sampling for DGF-1's **first-appearance** sensitivity row (ADR-011 clause 4).

That row scores each user on the graph as it stood at **the user's own node time**: every sampled
edge, at every hop, must be dated at or before the seed user's ``node_time``. PyG's ``NeighborLoader``
does this natively with ``time_attr="edge_time"`` and a per-seed ``input_time`` — the pyg-lib kernel
keeps edges with ``edge_time <= seed_time`` (inclusive) and applies the seed's time at **every** hop,
not the intermediate node's (read in the pyg-lib source by the ADR-011 literature review; available
in the installed pyg-lib 0.8.0 / PyG 2.8.0.post1). Temporal sampling forces disjoint subgraphs, so
each sampled node belongs to exactly one seed (``batch.batch``).

**The row is produced only if this sampler is deterministic under ADR-005's strict setting**
(ADR-011 clause 4, clause-5 test 10): the fixture tests pin the property, and
``scripts/verify_dgf1_temporal_sampler.py`` checks it on the real graph. If it fails, the row is not
produced and that is recorded — there is no non-deterministic fallback.

Two modes, both checked: ``num_neighbors=[-1, ...]`` takes **every** qualifying neighbour (exact, no
randomness at all), and a finite fan-out samples uniformly among qualifying neighbours (random,
so it must reproduce from the seed). Which mode the row uses follows how the gated view is scored,
which the GNN trainer session decides; this module does not pre-empt it.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from adapters.dgf1.datasource_dgraph import GraphView


def first_appearance_loader(
    x: Tensor,
    view: GraphView,
    node_time: Tensor,
    input_nodes: Tensor,
    num_neighbors: Sequence[int],
    batch_size: int,
    shuffle: bool = False,
) -> NeighborLoader:
    """A loader in which each seed ``v`` sees only edges dated ``<= node_time[v]``, at every hop.

    Args:
        x: ``[N, F]`` node inputs (whatever the model consumes; sampling does not read them).
        view: the graph view whose edges are eligible — pass the scoring view of the window
            (e.g. as of 481 for val); the temporal constraint then narrows each seed to its own time.
        node_time: ``[N]`` per-node times; the seeds' entries become their ``input_time``.
        input_nodes: 1-D index tensor of the users to score.
        num_neighbors: per-hop fan-out; ``-1`` means every qualifying neighbour.
    """
    graph = Data(x=x, edge_index=view.edge_index, edge_time=view.edge_time, num_nodes=x.size(0))
    return NeighborLoader(
        graph,
        num_neighbors=list(num_neighbors),
        input_nodes=input_nodes,
        input_time=node_time[input_nodes],
        time_attr="edge_time",
        temporal_strategy="uniform",
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # in-process: sampling order and RNG stay under the run's seed
    )


def assert_temporal_batch(batch: Data) -> None:
    """Raise unless every sampled edge is dated at or before its seed's time (clause-5 test 10).

    Cheap enough to run on every batch at scoring time as well as in tests — the guard should sit
    where the numbers are produced, not only in the suite.
    """
    if batch.edge_index.numel() == 0:
        return
    seed_of_dst = batch.batch[batch.edge_index[1]]
    if not torch.equal(seed_of_dst, batch.batch[batch.edge_index[0]]):
        raise AssertionError("a sampled edge joins two different seeds' subgraphs (disjointness broken)")
    late = batch.edge_time > batch.seed_time[seed_of_dst]
    if bool(late.any()):
        raise AssertionError(
            f"{int(late.sum())} sampled edge(s) are dated after their seed user's node time — the "
            "first-appearance view would read the future (ADR-011 clause 4)."
        )


def batch_signature(batch: Data) -> tuple[Tensor, ...]:
    """Everything that defines a sampled batch, for exact run-to-run comparison."""
    return (batch.n_id, batch.e_id, batch.edge_index, batch.batch, batch.seed_time)
