"""The inputs and edges at each arm's door equal an oracle built only from independently filtered edges."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_view_proto as P  # noqa: E402

from adapters.dgf1.datasource_dgraph import node_time_earliest_edge, view_node_features  # noqa: E402
from adapters.dgf1.train_gnn import window_only_edges  # noqa: E402
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
    # Edges between late users, so the window-only view has edges. The last is dated after both users' window.
    among = torch.tensor([[44, 45, 46, 50, 51, 44], [45, 46, 47, 51, 52, 47]])
    among_t = torch.tensor([15, 16, 18, 24, 26, 27])
    among_k = torch.tensor([1, 2, 3, 4, 5, 6])
    fwd = torch.cat([torch.stack([src, dst]), hand, join, among], dim=1)
    when, kind = torch.cat([when, hand_t, join_t, among_t]), torch.cat([kind, hand_k, join_k, among_k])
    node_time = node_time_earliest_edge(fwd, when, n)
    x = torch.randn((n, 17), generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    return P.Graph(x=x, fwd=fwd, edge_time=when, edge_type=kind, node_time=node_time, y=y)


G = synthetic()
SEEDS = ((G.node_time >= 0) & (G.node_time <= TRAIN_MAX)).nonzero().view(-1)
TARGETS = ((G.node_time > TRAIN_MAX) & (G.node_time <= VAL_END)).nonzero().view(-1)
TEST_TARGETS = ((G.node_time > VAL_END) & (G.node_time <= TEST_END)).nonzero().view(-1)
VAL_FIRST = TRAIN_MAX + 1


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


def test_the_loss_reads_each_batch_s_own_seeds_labels(door):
    """The temporal loader builds its graph with no labels, so the loop aligns them itself: this is that step."""
    seen = []
    torch.manual_seed(0)
    P.train_pass(door[0], G, SEEDS, TRAIN_MAX, P.fit_transform(G, TRAIN_MAX), [2, 2, 2], 4,
                 loss=lambda logits, y: seen.append((logits, y)))
    assert len(seen) == len(door[1]) > 0
    for (logits, y), batch in zip(seen, door[1]):
        assert logits.size(0) == batch.batch_size and torch.equal(y, G.y[batch.n_id[: batch.batch_size]])


def test_every_batch_passes_the_temporal_check_and_the_recency_check(door, monkeypatch):
    temporal, recency, real_t, real_r = [], [], P.assert_temporal_batch, P.check_seed_recency
    monkeypatch.setattr(P, "assert_temporal_batch", lambda b: (temporal.append(1), real_t(b))[1])
    monkeypatch.setattr(P, "check_seed_recency", lambda b, s: (recency.append(1), real_r(b, s))[1])
    n = len(trained(door))
    assert len(temporal) == n and len(recency) == n


@pytest.mark.parametrize("column", [-2, -1])
def test_the_recency_check_refuses_any_seed_whose_recency_is_not_zero(door, column):
    batch = trained(door)[0][1]
    raw = torch.zeros((batch.n_id.numel(), 30))
    P.check_seed_recency(batch, raw)
    raw[batch.batch_size - 1, column] = 1.0
    with pytest.raises(AssertionError, match="recency"):
        P.check_seed_recency(batch, raw)


def test_the_recency_check_leaves_neighbours_alone(door):
    """Neighbours' recency still varies (clause 2)."""
    batch = next(b for _, b in trained(door) if b.n_id.numel() > b.batch_size)
    raw = torch.zeros((batch.n_id.numel(), 30))
    raw[batch.batch_size:, -2:] = 1.0
    P.check_seed_recency(batch, raw)


def test_the_run_time_check_reads_the_rows_on_their_way_to_the_model(door, monkeypatch):
    """Inputs read at the window's end, not the seed's time: the check sees it, on the tensor itself."""
    monkeypatch.setattr(P, "row_times", lambda batch: torch.full_like(batch.n_id, TRAIN_MAX))
    with pytest.raises(AssertionError, match="recency"):
        trained(door)


def test_the_statistics_table_and_the_edges_stop_at_the_window_s_end():
    assert int(P.statistics_table(G, VAL_END).times.max()) <= VAL_END
    assert int(P.eligible_edges(G, VAL_END).edge_time.max()) <= VAL_END
    assert int(G.edge_time.max()) > VAL_END


def test_a_reverse_edge_keeps_its_forward_edge_s_date():
    view = P.eligible_edges(G, TEST_END)
    half = view.edge_time.numel() // 2
    assert torch.equal(view.edge_index[:, half:], view.edge_index[:, :half].flip(0))
    assert torch.equal(view.edge_time[half:], view.edge_time[:half])


# ------------------------------------------------------------------ the exploratory rows: each view, each arm


def oracle_window_forward(g: P.Graph, first: int, last: int):
    """The window-only view's forward edges, filtered edge by edge and independently of the builder."""
    keep = torch.tensor([bool(first <= g.node_time[s] <= last and first <= g.node_time[d] <= last and w <= last)
                         for s, d, w in zip(g.fwd[0], g.fwd[1], g.edge_time)])
    return g.fwd[:, keep], g.edge_time[keep], g.edge_type[keep]


def oracle_window_statistics(g: P.Graph, first: int, last: int) -> torch.Tensor:
    fwd, et, ety = oracle_window_forward(g, first, last)
    return view_node_features(fwd, et, ety, node_time_earliest_edge(fwd, et, g.x.size(0)), last)[0]


def exact_over(edge_index: torch.Tensor, u: int, layers: int) -> Counter:
    """Every (src, dst) edge an exact sampler takes for seed ``u``, from edges that already run both ways."""
    into = defaultdict(list)
    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        into[d].append(s)
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


@pytest.fixture
def window_door(monkeypatch):
    gate, batches, real = Door(), [], P.NeighborLoader

    def recording(*a, **k):
        for b in real(*a, **k):
            batches.append(b)
            yield b

    monkeypatch.setattr(P, "NeighborLoader", recording)
    return gate, batches


def test_the_statistics_refuse_a_node_with_no_edge_at_or_before_its_time():
    """Clause 8 relies on this raise: a reverse edge dated early reaches a node before its first edge."""
    u = int(TARGETS[0])
    table = P.statistics_table(G, VAL_END)
    table(torch.tensor([u]), G.node_time[u][None])
    with pytest.raises(AssertionError, match="no edge dated at or before"):
        table(torch.tensor([u]), G.node_time[u][None] - 1)


def test_the_fixture_has_a_user_whose_first_edge_in_the_window_only_view_is_later_than_their_first_edge():
    """Otherwise a window-only age read from the whole graph would pass the door tests."""
    fwd, et, _ = oracle_window_forward(G, VAL_FIRST, VAL_END)
    in_view = torch.zeros(G.x.size(0), dtype=torch.bool)
    in_view[fwd.flatten()] = True
    first_in_view = node_time_earliest_edge(fwd, et, G.x.size(0))
    assert bool((first_in_view[in_view] > G.node_time[in_view]).any())


def test_the_fixture_s_window_only_view_has_edges_and_isolates_a_user():
    stats = oracle_window_statistics(G, VAL_FIRST, VAL_END)
    assert oracle_window_forward(G, VAL_FIRST, VAL_END)[0].size(1) > 0
    assert bool(stats[TARGETS].sum(1).eq(0).any()) and bool(stats[TARGETS].sum(1).gt(0).any())


def test_the_fixture_has_a_window_only_edge_dated_after_the_window():
    inside = (G.node_time >= VAL_FIRST) & (G.node_time <= VAL_END)
    assert bool(((G.edge_time > VAL_END) & inside[G.fwd[0]] & inside[G.fwd[1]]).any())


def test_the_window_only_gnn_door_equals_the_oracle_row_by_row(window_door):
    frozen, stats = oracle_transform(G), oracle_window_statistics(G, VAL_FIRST, VAL_END)
    P.exploratory_gnn(window_door[0], "window_only", G, TARGETS, VAL_FIRST, VAL_END, P.fit_transform(G, TRAIN_MAX), LAYERS, 4)
    assert len(window_door[0].seen) == len(window_door[1]) > 0
    for (inputs, _), batch in zip(window_door[0].seen, window_door[1]):
        rows = torch.stack([torch.cat([G.x[n], stats[n]]) for n in batch.n_id.tolist()])
        assert torch.equal(inputs, frozen.transform(rows))


def test_the_window_only_gnn_s_edges_equal_the_tested_building_block_in_both_directions(window_door):
    """One seed a batch, so each batch can be compared with that seed's exact neighbourhood."""
    edges = window_only_edges(SimpleNamespace(node_time=G.node_time), P.eligible_edges(G, VAL_END),
                              SimpleNamespace(test_min=VAL_FIRST, test_max=VAL_END))
    P.exploratory_gnn(window_door[0], "window_only", G, TARGETS, VAL_FIRST, VAL_END, P.fit_transform(G, TRAIN_MAX), LAYERS, 1)
    non_empty = 0
    for (_, edge_index), batch in zip(window_door[0].seen, window_door[1]):
        found = Counter((int(batch.n_id[s]), int(batch.n_id[d])) for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()))
        assert found == exact_over(edges, int(batch.n_id[0]), LAYERS)
        non_empty += bool(found)
    assert non_empty > 0


def test_the_first_appearance_gnn_door_equals_the_oracle_row_by_row(door):
    frozen = oracle_transform(G)
    P.exploratory_gnn(door[0], "first_appearance", G, TARGETS, VAL_FIRST, VAL_END, P.fit_transform(G, TRAIN_MAX), LAYERS, 4)
    assert len(door[0].seen) == len(door[1]) > 0
    for (inputs, _), batch in zip(door[0].seen, door[1]):
        assert torch.equal(inputs, expected_door(G, batch, frozen))


@pytest.mark.parametrize("view", ["first_appearance", "window_only"])
def test_each_exploratory_floor_trains_on_the_gated_view_untransformed(view):
    spy = FloorSpy()
    P.exploratory_floor(spy, view, G, SEEDS, TARGETS, TRAIN_MAX, VAL_FIRST, VAL_END, 0)
    expected = torch.cat([G.x[SEEDS], oracle_statistics(G, TRAIN_MAX)[SEEDS]], dim=1)
    assert torch.equal(torch.from_numpy(spy.design.X_train), expected)
    assert torch.equal(torch.from_numpy(spy.design.y_train), G.y[SEEDS])
    assert torch.equal(torch.from_numpy(spy.design.y_score), G.y[TARGETS])


def test_the_first_appearance_floor_scores_on_own_time_rows():
    spy = FloorSpy()
    P.exploratory_floor(spy, "first_appearance", G, SEEDS, TARGETS, TRAIN_MAX, VAL_FIRST, VAL_END, 0)
    assert torch.equal(torch.from_numpy(spy.design.X_score), expected_floor_rows(G, TARGETS))


def test_the_window_only_floor_scores_on_the_view_s_own_statistics():
    spy = FloorSpy()
    P.exploratory_floor(spy, "window_only", G, SEEDS, TARGETS, TRAIN_MAX, VAL_FIRST, VAL_END, 0)
    expected = torch.cat([G.x[TARGETS], oracle_window_statistics(G, VAL_FIRST, VAL_END)[TARGETS]], dim=1)
    assert torch.equal(torch.from_numpy(spy.design.X_score), expected)


@pytest.mark.parametrize("view", ["gated", "matched", ""])
def test_the_exploratory_path_refuses_any_other_view(view):
    gate, spy = Door(), FloorSpy()
    with pytest.raises(ValueError, match="refuses view"):
        P.exploratory_gnn(gate, view, G, TEST_TARGETS, VAL_END + 1, TEST_END, P.fit_transform(G, TRAIN_MAX), LAYERS, 4)
    with pytest.raises(ValueError, match="refuses view"):
        P.exploratory_floor(spy, view, G, SEEDS, TEST_TARGETS, TRAIN_MAX, VAL_END + 1, TEST_END, 0)
    assert gate.seen == [] and not hasattr(spy, "design")
