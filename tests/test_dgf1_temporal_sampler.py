"""Temporal-sampler guards for the first-appearance row — ADR-011 clause 4, clause-5 test 10.

Two properties decide whether that row may be produced at all: **every sampled edge, at every hop,
is dated at or before its seed user's node time**, and **two runs at one seed give identical batches**
under ADR-005's strict determinism (``seed_everything`` pins ``use_deterministic_algorithms``). Both
are pinned here on the synthetic DGraph-shaped fixture, each with a control showing the check could
fail. The real-graph check is ``scripts/verify_dgf1_temporal_sampler.py``.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from gbe.run.seeding import seed_everything
from adapters.dgf1.datasource_dgraph import graph_view, window_target_mask
from adapters.dgf1.features import node_inputs
from adapters.dgf1.sampling import assert_temporal_batch, batch_signature, first_appearance_loader
from tests.test_dgf1_floor import SPLIT_TEST, _synthetic

EXACT = [-1, -1, -1]
FANOUT = [3, 2, 2]


def _loader(data, fanout, seed, batch_size=32):
    seed_everything(seed)
    view = graph_view(data, SPLIT_TEST.test_max)
    targets = window_target_mask(data, SPLIT_TEST).nonzero().view(-1)
    return first_appearance_loader(node_inputs(data, view), view, data.node_time, targets,
                                   fanout, batch_size)


def _signatures(loader):
    return [batch_signature(b) for b in loader]


def _identical(a, b) -> bool:
    return len(a) == len(b) and all(
        all(torch.equal(x, y) for x, y in zip(sa, sb)) for sa, sb in zip(a, b)
    )


# -- the temporal constraint -----------------------------------------------------------------
@pytest.mark.parametrize("fanout", [EXACT, FANOUT], ids=["exact", "fanout"])
def test_every_sampled_edge_at_every_hop_is_dated_by_its_seeds_time(fanout):
    data = _synthetic()
    n_edges = 0
    for batch in _loader(data, fanout, seed=0):
        assert_temporal_batch(batch)
        n_edges += batch.edge_index.size(1)
    assert n_edges > 0, "nothing was sampled — the check would pass vacuously"


def test_seed_time_is_each_scored_users_node_time():
    data = _synthetic()
    for batch in _loader(data, EXACT, seed=0):
        assert torch.equal(batch.seed_time, data.node_time[batch.n_id[: batch.batch_size]])


def test_the_constraint_has_teeth_a_non_temporal_loader_reads_the_future():
    """Control: on the same view and seeds, an ordinary loader samples edges dated after the seed
    user's node time, and the guard rejects them. So the temporal loader's pass is not vacuous."""
    data = _synthetic()
    view = graph_view(data, SPLIT_TEST.test_max)
    targets = window_target_mask(data, SPLIT_TEST).nonzero().view(-1)
    g = Data(x=node_inputs(data, view), edge_index=view.edge_index, edge_time=view.edge_time,
             num_nodes=data.num_nodes)
    batch = next(iter(NeighborLoader(g, num_neighbors=[-1, -1], input_nodes=targets,
                                     batch_size=len(targets), disjoint=True)))
    batch.seed_time = data.node_time[batch.n_id[: batch.batch_size]]
    with pytest.raises(AssertionError, match="dated after their seed"):
        assert_temporal_batch(batch)


# -- determinism under ADR-005 ---------------------------------------------------------------
def test_same_seed_gives_identical_batches_under_strict_determinism():
    data = _synthetic()
    first = _signatures(_loader(data, FANOUT, seed=0))
    assert torch.are_deterministic_algorithms_enabled()
    second = _signatures(_loader(data, FANOUT, seed=0))
    assert _identical(first, second)


def test_exact_mode_has_no_randomness_at_all():
    data = _synthetic()
    assert _identical(_signatures(_loader(data, EXACT, seed=0)),
                      _signatures(_loader(data, EXACT, seed=1)))


def test_a_truncating_fanout_actually_responds_to_the_seed():
    """Control for the determinism test: if the seed were inert, 'identical at one seed' would
    prove nothing about reproducibility from a recorded seed."""
    data = _synthetic()
    assert not _identical(_signatures(_loader(data, [1, 1, 1], seed=0)),
                          _signatures(_loader(data, [1, 1, 1], seed=1)))
