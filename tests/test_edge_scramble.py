"""Integrity guards for the edge-scramble ablation (doc-00 §8, doc-01 §4 Phase 3).

Written in the same session as the ablation (CLAUDE.md: "untested guards don't exist").

The load-bearing property is the *temporal* one: a scramble must not manufacture an edge that
crosses the train/test cutoff. If it did, the ablation would either leak the future into
training or have its edges silently dropped by the induced-subgraph step — and in both cases
the scrambled run would stop being comparable to the real-graph run it is the control for.
"""

from __future__ import annotations

import pytest
import torch

from gbe.eval import (
    TemporalSplit,
    assert_no_temporal_leakage,
    configuration_model_edges,
    edges_as_of,
    random_graph_edges,
    remove_edges,
    scramble_edges,
)

SPLIT = TemporalSplit(train_max=34, test_min=35, test_max=49)


def _train_edges(edge_index, time_step):
    """The train graph (ADR-011): edges dated <= train_max. Every fixture edge lies within one
    step (asserted by the tests that call this), so each edge is dated by its shared step."""
    return edges_as_of(edge_index, edge_time=time_step[edge_index[0]], t_max=SPLIT.train_max)


def _within_step_graph(n_per_step: int = 40, steps=(10, 30, 34, 35, 40, 49)):
    """A graph whose every edge is within a single time step — Elliptic's property (doc §2.2.3)."""
    time_step = torch.repeat_interleave(torch.tensor(steps), n_per_step)
    src, dst = [], []
    for s, step in enumerate(steps):
        base = s * n_per_step
        for i in range(n_per_step - 1):
            src += [base + i, base + i + 1]      # reciprocal pair -> symmetrised
            dst += [base + i + 1, base + i]
    return torch.tensor([src, dst]), time_step


def test_scramble_preserves_degree_sequence():
    """Topology is intact: only the identity of the node at each position changes."""
    edge_index, time_step = _within_step_graph()
    scrambled = scramble_edges(edge_index, time_step, seed=0)

    n = time_step.numel()
    before = torch.bincount(edge_index[0], minlength=n).sort().values
    after = torch.bincount(scrambled[0], minlength=n).sort().values
    assert torch.equal(before, after), "scramble changed the degree sequence"
    assert scrambled.shape == edge_index.shape


def test_scramble_never_crosses_the_temporal_cutoff():
    """The guard with teeth: no scrambled edge may join a train node to a test node."""
    edge_index, time_step = _within_step_graph()
    scrambled = scramble_edges(edge_index, time_step, seed=0)

    endpoint_time = time_step[scrambled]
    assert torch.equal(endpoint_time[0], endpoint_time[1]), "an edge now spans two time steps"

    # and the real consequence: the train graph keeps exactly as many edges as the real graph's
    edge_train = _train_edges(scrambled, time_step)
    assert_no_temporal_leakage(edge_train, time_step, SPLIT, edge_time=time_step[edge_train[0]])
    train_edges_before = _train_edges(edge_index, time_step).shape[1]
    assert edge_train.shape[1] == train_edges_before, (
        "scrambling changed how many training edges survive induction — the ablation would "
        "no longer be comparable to the real-graph run"
    )


def test_scramble_actually_rewires():
    """A no-op scramble would make the ablation vacuously 'pass'."""
    edge_index, time_step = _within_step_graph()
    scrambled = scramble_edges(edge_index, time_step, seed=0)
    assert not torch.equal(scrambled, edge_index), "scramble left the edge set unchanged"


def test_scramble_preserves_symmetry():
    """A symmetrised input must stay symmetrised (ELL-1 symmetrises by default, doc §3)."""
    edge_index, time_step = _within_step_graph()
    scrambled = scramble_edges(edge_index, time_step, seed=0)
    forward = {(int(u), int(v)) for u, v in zip(*scrambled)}
    assert all((v, u) in forward for u, v in forward), "reciprocal pairs were broken"


def test_scramble_is_reproducible_from_the_seed():
    """A registry row pins a seed; the scramble it names must be recoverable from it."""
    edge_index, time_step = _within_step_graph()
    a = scramble_edges(edge_index, time_step, seed=7)
    b = scramble_edges(edge_index, time_step, seed=7)
    c = scramble_edges(edge_index, time_step, seed=8)
    assert torch.equal(a, b), "same seed produced a different scramble"
    assert not torch.equal(a, c), "different seeds produced the same scramble"


# -- the other Gate-3 arms (ADR-006 clauses 2, 3, 5) -----------------------------------

@pytest.mark.parametrize("rewire", [random_graph_edges, configuration_model_edges])
def test_every_rewiring_arm_stays_within_a_time_step(rewire):
    """ADR-006 requires this of *every* ablation, not just edge-scramble.

    An arm that forges an edge across the 34/35 cutoff either leaks the future into training or
    has that edge silently dropped by induction — in both cases it stops being comparable to the
    real-graph arm it is the control for, and nothing would report the problem.
    """
    edge_index, time_step = _within_step_graph()
    rewired = rewire(edge_index, time_step, seed=0)

    endpoint_time = time_step[rewired]
    assert torch.equal(endpoint_time[0], endpoint_time[1]), (
        f"{rewire.__name__} produced an edge spanning two time steps"
    )
    edge_train = _train_edges(rewired, time_step)
    assert_no_temporal_leakage(edge_train, time_step, SPLIT, edge_time=time_step[edge_train[0]])


def test_random_graph_destroys_the_degree_sequence():
    """The ER control's defining property — otherwise it is just another scramble."""
    edge_index, time_step = _within_step_graph()
    random = random_graph_edges(edge_index, time_step, seed=0)

    n = time_step.numel()
    before = torch.bincount(edge_index[0], minlength=n).sort().values
    after = torch.bincount(random[0], minlength=n).sort().values
    assert not torch.equal(before, after), "ER control preserved the degree sequence"
    # edge count is matched up to collision removal (self-loops / duplicate pairs)
    assert after.sum() <= before.sum()
    assert after.sum() > 0.8 * before.sum(), "too many edges lost to collisions"


def test_configuration_model_preserves_every_node_degree_exactly():
    """The whole point of clause 5: keep *how many*, randomise *who*."""
    edge_index, time_step = _within_step_graph()
    config = configuration_model_edges(edge_index, time_step, seed=0)

    n = time_step.numel()
    before = torch.bincount(edge_index[0], minlength=n)
    after = torch.bincount(config[0], minlength=n)
    assert torch.equal(before, after), (
        "configuration model changed a node's degree — it would then confound 'who you "
        "connect to' with 'how many connections you have', which is the one thing it exists "
        "to separate"
    )


def test_configuration_model_actually_rewires_and_stays_simple():
    """Degree-preserving must not mean unchanged; and no self-loops or duplicate edges."""
    edge_index, time_step = _within_step_graph()
    config = configuration_model_edges(edge_index, time_step, seed=0)

    # Compare edge *sets*. (Sorting the rows of a [2, E] tensor would be vacuous here: this
    # rewire preserves every degree, so each row's sorted endpoint multiset is unchanged by
    # construction and such an assertion could never fail.)
    before = {(int(u), int(v)) for u, v in zip(*edge_index)}
    after = {(int(u), int(v)) for u, v in zip(*config)}
    assert before != after, "configuration model left the wiring unchanged"

    assert not bool((config[0] == config[1]).any()), "self-loop created"
    assert len(after) == config.shape[1], "duplicate edge created (multigraph)"


def test_remove_edges_disables_message_passing():
    """GNN-removed: an empty edge set, same model and parameter count."""
    edge_index, _ = _within_step_graph()
    empty = remove_edges(edge_index)
    assert empty.shape == (2, 0)
    assert empty.dtype == edge_index.dtype


def test_rewiring_arms_are_reproducible_from_the_seed():
    edge_index, time_step = _within_step_graph()
    for rewire in (random_graph_edges, configuration_model_edges):
        a = rewire(edge_index, time_step, seed=5)
        b = rewire(edge_index, time_step, seed=5)
        assert torch.equal(a, b), f"{rewire.__name__} is not reproducible from its seed"
