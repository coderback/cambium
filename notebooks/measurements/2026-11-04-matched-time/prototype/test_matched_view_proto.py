"""The inputs and edges at each arm's door equal an oracle built only from independently filtered edges."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_view_proto as P  # noqa: E402

from adapters.dgf1.datasource_dgraph import node_time_earliest_edge, view_node_features  # noqa: E402
from gbe.features.scaling import Standardizer  # noqa: E402

TRAIN_MAX, VAL_END, TEST_END = 12, 20, 30
LAYERS = 3
# Hand-made users, kept apart from the random ones so the fixture's four traps cannot be overwritten.
A, B, C, D = 40, 41, 42, 43


def synthetic(seed: int = 0) -> P.Graph:
    g = torch.Generator().manual_seed(seed)
    n, e = 56, 170
    src, dst = torch.randint(0, 40, (e,), generator=g), torch.randint(0, 40, (e,), generator=g)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    when = torch.randint(1, TEST_END + 1, (src.numel(),), generator=g)
    kind = torch.randint(1, 12, (src.numel(),), generator=g)
    # A appears at 1 with B. B gains edges later, so B's statistics at 1 differ from B's at any later
    # step. D reaches A only at 28, so the reverse edge A -> D must stay out of A's early subgraph.
    hand = torch.tensor([[A, B, B, D, C], [B, C, D, A, A]])
    hand_t = torch.tensor([1, 5, 9, 28, 11])
    hand_k = torch.tensor([1, 2, 3, 4, 5])
    # Users who first appear in the later windows, each with an edge after appearing. The random users
    # above all appear early, so without these the later windows would hold nobody to score.
    late = [44 + k for k in range(12)]
    late_t = [13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29]
    join = torch.tensor([late + late, list(range(12)) + list(range(20, 32))])
    join_t = torch.tensor(late_t + [min(t + 1, TEST_END) for t in late_t])
    join_k = torch.tensor([1 + k % 11 for k in range(24)])
    fwd = torch.cat([torch.stack([src, dst]), hand, join], dim=1)
    when, kind = torch.cat([when, hand_t, join_t]), torch.cat([kind, hand_k, join_k])
    node_time = node_time_earliest_edge(fwd, when, n)
    x = torch.randn((n, 17), generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    return P.Graph(x=x, fwd=fwd, edge_time=when, edge_type=kind, node_time=node_time, y=y)


G = synthetic()
SEEDS = ((G.node_time >= 0) & (G.node_time <= TRAIN_MAX)).nonzero().view(-1)
TARGETS = ((G.node_time > TRAIN_MAX) & (G.node_time <= VAL_END)).nonzero().view(-1)
TEST_TARGETS = ((G.node_time > VAL_END) & (G.node_time <= TEST_END)).nonzero().view(-1)


# ------------------------------------------------------------------ the oracle: independent of the builder


def oracle_statistics(g: P.Graph, t: int) -> torch.Tensor:
    keep = g.edge_time <= t
    feats, _ = view_node_features(g.fwd[:, keep], g.edge_time[keep], g.edge_type[keep], g.node_time, t)
    return feats


def oracle_transform(g: P.Graph) -> Standardizer:
    return Standardizer.fit(torch.cat([g.x, oracle_statistics(g, TRAIN_MAX)], dim=1), g.node_time <= TRAIN_MAX)


def own_times(g: P.Graph, batch) -> torch.Tensor:
    """Each row's seed's own time, from node_time and the row's seed: not from anything the sampler carried."""
    return g.node_time[batch.n_id[: batch.batch_size]][batch.batch]


def expected_door(g: P.Graph, batch, frozen: Standardizer | None) -> torch.Tensor:
    rows = [torch.cat([g.x[n], oracle_statistics(g, t)[n]]) for n, t in zip(batch.n_id.tolist(), own_times(g, batch).tolist())]
    raw = torch.stack(rows)
    return frozen.transform(raw) if frozen is not None else raw


def exact_neighbourhood(g: P.Graph, u: int, t: int, layers: int) -> Counter:
    """Every (src, dst) edge an exact sampler takes for seed ``u``, from the forward edges dated at most ``t``."""
    into = defaultdict(list)
    for s, d, when in zip(g.fwd[0].tolist(), g.fwd[1].tolist(), g.edge_time.tolist()):
        if when <= t:
            into[d].append(s)
            into[s].append(d)
    seen, frontier, edges = {u}, [u], Counter()
    for _ in range(layers):
        following = []
        for w in frontier:
            for v in into[w]:
                edges[(v, w)] += 1
                if v not in seen:
                    seen.add(v)
                    following.append(v)
        frontier = following
    return edges


def edges_by_seed(batch, edge_index: torch.Tensor) -> dict[int, Counter]:
    out: dict[int, Counter] = defaultdict(Counter)
    seeds = batch.n_id[: batch.batch_size].tolist()
    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        out[seeds[int(batch.batch[d])]][(int(batch.n_id[s]), int(batch.n_id[d]))] += 1
    return out


class Door:
    """Stands where the model stands, and keeps exactly what it was handed."""

    def __init__(self):
        self.seen = []

    def __call__(self, x, edge_index):
        self.seen.append((x.clone(), edge_index.clone()))
        return x, x[:, :2]


@pytest.fixture
def door(monkeypatch):
    """A door, plus the batches its calls came from, recorded around the real loader."""
    gate, batches, real = Door(), [], P.first_appearance_loader

    def recording(*a, **k):
        for b in real(*a, **k):
            batches.append(b)
            yield b

    monkeypatch.setattr(P, "first_appearance_loader", recording)
    return gate, batches


def trained(door, fan_out=(2, 2, 2)):
    torch.manual_seed(0)
    P.train_pass(door[0], G, SEEDS, TRAIN_MAX, P.fit_transform(G, TRAIN_MAX), list(fan_out), 4)
    assert len(door[0].seen) == len(door[1]) > 0
    return list(zip(door[0].seen, door[1]))


def scored(door, targets=TARGETS, end=VAL_END):
    P.score_pass(door[0], G, targets, end, P.fit_transform(G, TRAIN_MAX), LAYERS, 4)
    assert len(door[0].seen) == len(door[1]) > 0
    return list(zip(door[0].seen, door[1]))


# ------------------------------------------------------------------ the fixture can tell these apart


def test_the_fixture_s_later_activity_changes_a_neighbour_s_statistics():
    assert not torch.equal(oracle_statistics(G, 1)[B], oracle_statistics(G, TRAIN_MAX)[B])
    assert int(G.node_time[A]) == 1 and int(G.node_time[B]) == 1


def test_the_fixture_has_a_reverse_edge_that_a_wrong_date_would_let_in():
    assert (D, A) not in exact_neighbourhood(G, A, 1, 1) and (A, D) not in exact_neighbourhood(G, A, 1, 1)
    assert (D, A) in exact_neighbourhood(G, A, 28, 1)


def test_the_fixture_s_transform_moves_if_refit_on_a_scoring_view():
    frozen, refit = oracle_transform(G), P.fit_transform(G, VAL_END)
    assert not torch.equal(frozen.mean, refit.mean)


def test_the_fixture_has_a_neighbour_whose_own_time_differs_from_its_seed_s():
    pairs = exact_neighbourhood(G, int(C), int(G.node_time[C]), 1)
    assert any(int(G.node_time[v]) != int(G.node_time[C]) for v, _ in pairs)


def test_every_window_has_seeds_or_the_tests_would_pass_on_nothing():
    assert SEEDS.numel() > 10 and TARGETS.numel() > 2 and TEST_TARGETS.numel() > 0


# ------------------------------------------------------------------ the GNN's door


@pytest.mark.parametrize("fan_out", [(2, 2, 2), (-1, -1, -1)])
def test_the_training_door_equals_the_oracle_row_by_row(door, fan_out):
    frozen = oracle_transform(G)
    for (inputs, _), batch in trained(door, fan_out):
        assert inputs.size(1) == 30
        assert torch.equal(inputs, expected_door(G, batch, frozen))


def test_the_scoring_door_equals_the_oracle_row_by_row(door):
    frozen = oracle_transform(G)
    for (inputs, _), batch in scored(door):
        assert torch.equal(inputs, expected_door(G, batch, frozen))


def test_the_test_window_s_door_equals_the_oracle_with_the_same_frozen_transform(door):
    frozen = oracle_transform(G)
    for (inputs, _), batch in scored(door, TEST_TARGETS, TEST_END):
        assert torch.equal(inputs, expected_door(G, batch, frozen))


def test_a_neighbour_row_at_the_door_is_as_of_its_seed_s_time_not_its_own(door):
    """Seed C's neighbour A appeared earlier than C, so reading A at A's own time would give other values."""
    frozen = oracle_transform(G)
    P.train_pass(door[0], G, torch.tensor([C]), TRAIN_MAX, P.fit_transform(G, TRAIN_MAX), [-1] * LAYERS, 4)
    (inputs, _), batch = door[0].seen[0], door[1][0]
    row = batch.n_id.tolist().index(A)
    assert torch.equal(inputs[row], frozen.transform(torch.cat([G.x[A], oracle_statistics(G, int(G.node_time[C]))[A]])[None])[0])
    assert not torch.equal(inputs[row], frozen.transform(torch.cat([G.x[A], oracle_statistics(G, 1)[A]])[None])[0])


# ------------------------------------------------------------------ the edges at the door


def test_exact_edges_at_the_door_equal_the_filter_in_both_directions(door):
    """Equality catches an edge let in by a wrong date, which assert_temporal_batch cannot see, and a lost edge."""
    checked = 0
    for (_, edge_index), batch in scored(door):
        found = edges_by_seed(batch, edge_index)
        for u in batch.n_id[: batch.batch_size].tolist():
            assert found[u] == exact_neighbourhood(G, u, int(G.node_time[u]), LAYERS)
            checked += 1
    assert checked == TARGETS.numel()


def test_exact_edges_in_training_equal_the_filter_too(door):
    for (_, edge_index), batch in trained(door, (-1, -1, -1)):
        found = edges_by_seed(batch, edge_index)
        for u in batch.n_id[: batch.batch_size].tolist():
            assert found[u] == exact_neighbourhood(G, u, int(G.node_time[u]), LAYERS)


def test_sampled_edges_at_a_fan_out_are_a_subset_of_the_filter(door):
    """Equality cannot hold under a fan-out, so a sampled batch is held to a subset, and it does sample less."""
    fewer = 0
    for (_, edge_index), batch in trained(door, (1, 1, 1)):
        found = edges_by_seed(batch, edge_index)
        for u in batch.n_id[: batch.batch_size].tolist():
            exact = exact_neighbourhood(G, u, int(G.node_time[u]), LAYERS)
            assert not found[u] - exact
            fewer += sum(found[u].values()) < sum(exact.values())
    assert fewer > 0


# ------------------------------------------------------------------ the floor's door


class FloorSpy:
    def __call__(self, model, design, seed):
        self.model, self.design, self.seed = model, design, seed
        return "fitted"


def expected_floor_rows(g, users):
    return torch.stack([torch.cat([g.x[u], oracle_statistics(g, int(g.node_time[u]))[u]]) for u in users.tolist()])


def test_the_floor_s_training_rows_are_own_time_and_untransformed():
    spy = FloorSpy()
    assert P.floor_pass(spy, G, SEEDS, TARGETS, TRAIN_MAX, VAL_END, 0) == "fitted"
    assert torch.equal(torch.from_numpy(spy.design.X_train), expected_floor_rows(G, SEEDS))
    assert spy.design.X_train.shape[1] == 30 and torch.equal(torch.from_numpy(spy.design.y_train), G.y[SEEDS])


def test_the_floor_s_scoring_rows_are_own_time_and_untransformed():
    spy = FloorSpy()
    P.floor_pass(spy, G, SEEDS, TARGETS, TRAIN_MAX, VAL_END, 0)
    assert torch.equal(torch.from_numpy(spy.design.X_score), expected_floor_rows(G, TARGETS))
    assert torch.equal(torch.from_numpy(spy.design.y_score), G.y[TARGETS])


# ------------------------------------------------------------------ what each pass must do


def test_training_batches_are_shuffled_across_times_and_scoring_keeps_its_order(door):
    seeds = SEEDS[torch.argsort(G.node_time[SEEDS])]
    torch.manual_seed(0)
    P.train_pass(door[0], G, seeds, TRAIN_MAX, P.fit_transform(G, TRAIN_MAX), [2, 2, 2], 4)
    order = torch.cat([b.n_id[: b.batch_size] for b in door[1]])
    assert not torch.equal(order, seeds) and sorted(order.tolist()) == sorted(seeds.tolist())
    assert any(torch.unique(b.seed_time).numel() > 1 for b in door[1])
    door[1].clear()
    P.score_pass(door[0], G, seeds, TRAIN_MAX, P.fit_transform(G, TRAIN_MAX), LAYERS, 4)
    assert torch.equal(torch.cat([b.n_id[: b.batch_size] for b in door[1]]), seeds)


def test_every_batch_passes_the_temporal_check_and_the_recency_check(door, monkeypatch):
    temporal, recency, real_t, real_r = [], [], P.assert_temporal_batch, P.check_seed_recency
    monkeypatch.setattr(P, "assert_temporal_batch", lambda b: (temporal.append(1), real_t(b))[1])
    monkeypatch.setattr(P, "check_seed_recency", lambda b, s: (recency.append(1), real_r(b, s))[1])
    n = len(trained(door))
    assert len(temporal) == n and len(recency) == n


def test_the_recency_check_refuses_a_seed_whose_recency_is_not_zero(door):
    batch = trained(door)[0][1]

    class Wrong:
        def __call__(self, nodes, at):
            return torch.ones((nodes.numel(), 13))

    with pytest.raises(AssertionError, match="recency"):
        P.check_seed_recency(batch, Wrong())


def test_the_statistics_table_and_the_edges_stop_at_the_window_s_end():
    assert int(P.statistics_table(G, VAL_END).times.max()) <= VAL_END
    assert int(P.eligible_edges(G, VAL_END).edge_time.max()) <= VAL_END
    assert int(G.edge_time.max()) > VAL_END


def test_a_reverse_edge_keeps_its_forward_edge_s_date():
    view = P.eligible_edges(G, TEST_END)
    half = view.edge_time.numel() // 2
    assert torch.equal(view.edge_index[:, half:], view.edge_index[:, :half].flip(0))
    assert torch.equal(view.edge_time[half:], view.edge_time[:half])
