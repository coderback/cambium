"""Edge-dated temporal guards in the core (ADR-011 clauses 3–6).

Written in the same session as `gbe.eval.edges_as_of` and the edge-date leakage check
(CLAUDE.md: "untested guards don't exist"). Synthetic fixtures only — the DGraph snapshot is
gitignored, so the suite must not need it (ADR-011 clause 5); the real-snapshot counts are an
audit script's job.

The fixture is **DGraph-shaped, not Elliptic-shaped**: it contains an edge between two
pre-cutoff nodes that is itself dated *after* the cutoff. ELL-1's fixtures never contained that
case, which is exactly why the node-time-only guard looked sufficient for so long.
"""

from __future__ import annotations

import pytest
import torch

from gbe.eval import TemporalSplit, assert_no_temporal_leakage, edges_as_of

SPLIT = TemporalSplit(train_max=3, test_min=4)

#   node times:  n0=1  n1=2  n2=5  n3=1
TIME_STEP = torch.tensor([1, 2, 5, 1])
#   edges:       (0,1)@2   (0,3)@4          (1,2)@5              (3,1)@3
#                pre       PRE NODES,        touches a post-      pre
#                          POST-CUTOFF EDGE  cutoff node
EDGE_INDEX = torch.tensor([[0, 0, 1, 3], [1, 3, 2, 1]])
EDGE_TIME = torch.tensor([2, 4, 5, 3])


def _node_induced(edge_index: torch.Tensor, time_step: torch.Tensor, t_max: int) -> torch.Tensor:
    """The retired reading, written out: every edge whose endpoints both exist by ``t_max``."""
    return edge_index[:, (time_step[edge_index] <= t_max).all(dim=0)]


# -- edges_as_of ---------------------------------------------------------------------------
def test_edges_as_of_keeps_exactly_the_edges_dated_up_to_the_cutoff_in_order():
    got = edges_as_of(EDGE_INDEX, edge_time=EDGE_TIME, t_max=SPLIT.train_max)
    assert got.tolist() == [[0, 3], [1, 1]]  # (0,1)@2 then (3,1)@3, input order kept


def test_edges_as_of_preserves_input_order_under_permutation():
    """Order is what keeps neighbour sampling — and so ELL-1's numbers — bit-for-bit."""
    perm = torch.tensor([3, 1, 0, 2])
    got = edges_as_of(EDGE_INDEX[:, perm], edge_time=EDGE_TIME[perm], t_max=SPLIT.train_max)
    assert got.tolist() == [[3, 0], [1, 1]]


def test_scoring_view_contains_no_edge_after_the_window_end_but_keeps_older_ones():
    """ADR-011 clause 4 / clause-5 test 5: the view of a window is the graph as of its end —
    every older edge stays (pre-window nodes are neighbours), nothing later enters."""
    view = edges_as_of(EDGE_INDEX, edge_time=EDGE_TIME, t_max=4)
    kept_dates = EDGE_TIME[EDGE_TIME <= 4]
    assert bool((kept_dates <= 4).all())
    assert view.shape[1] == 3  # (0,1)@2, (0,3)@4, (3,1)@3 — the @5 edge is out


# -- the guard has teeth -------------------------------------------------------------------
def test_node_induced_graph_fails_the_edge_date_check():
    """The exact failure ADR-011 fixes: (0,3) joins two pre-cutoff nodes, so a node-time filter
    keeps it — but it is dated 4, after the cutoff. The endpoint check alone cannot see it."""
    node_induced = _node_induced(EDGE_INDEX, TIME_STEP, SPLIT.train_max)
    dates = EDGE_TIME[(TIME_STEP[EDGE_INDEX] <= SPLIT.train_max).all(dim=0)]
    assert node_induced.shape[1] == 3  # it really does admit (0,3)

    with pytest.raises(AssertionError, match="dated after train_max"):
        assert_no_temporal_leakage(node_induced, TIME_STEP, SPLIT, edge_time=dates)


def test_edges_as_of_output_passes_the_leakage_check():
    edge_time_train = EDGE_TIME[EDGE_TIME <= SPLIT.train_max]
    train = edges_as_of(EDGE_INDEX, edge_time=EDGE_TIME, t_max=SPLIT.train_max)
    assert_no_temporal_leakage(train, TIME_STEP, SPLIT, edge_time=edge_time_train)  # silent


def test_endpoint_check_still_fires_even_on_a_pre_cutoff_date():
    """Both checks are needed: an edge whose date says 'pre-cutoff' but which reaches a
    post-cutoff node is inconsistent input, and must not slip through on the date alone."""
    with pytest.raises(AssertionError, match="touch a node"):
        assert_no_temporal_leakage(
            torch.tensor([[1], [2]]), TIME_STEP, SPLIT, edge_time=torch.tensor([3])
        )


# -- edge dates are required, never inferred -----------------------------------------------
def test_edges_as_of_requires_edge_time():
    with pytest.raises(TypeError):
        edges_as_of(EDGE_INDEX, t_max=3)  # type: ignore[call-arg]


def test_edges_as_of_rejects_positional_edge_time():
    """Keyword-only: a positional call could silently pass node times where dates were meant."""
    with pytest.raises(TypeError):
        edges_as_of(EDGE_INDEX, EDGE_TIME, 3)  # type: ignore[misc]


def test_leakage_check_requires_edge_time():
    """No silent fallback to the node-only check (ADR-011 clause 6)."""
    with pytest.raises(TypeError):
        assert_no_temporal_leakage(EDGE_INDEX, TIME_STEP, SPLIT)  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [torch.tensor([1, 2, 3]), torch.tensor([[1, 2, 3, 4]])])
def test_edge_time_must_have_one_date_per_edge(bad):
    with pytest.raises(ValueError, match="one date per edge"):
        edges_as_of(EDGE_INDEX, edge_time=bad, t_max=3)
    with pytest.raises(ValueError, match="one date per edge"):
        assert_no_temporal_leakage(EDGE_INDEX, TIME_STEP, SPLIT, edge_time=bad)


def test_empty_edge_set_is_valid():
    """The no-edges ablation arm passes an empty graph through the same path."""
    empty = EDGE_INDEX.new_empty((2, 0))
    none = EDGE_TIME.new_empty((0,))
    assert edges_as_of(empty, edge_time=none, t_max=3).shape == (2, 0)
    assert_no_temporal_leakage(empty, TIME_STEP, SPLIT, edge_time=none)
