"""Leakage and integrity guards for the DGF-1 DataSource (ADR-010, ADR-011 clause 5).

Written in the same session as the DataSource (CLAUDE.md: "untested guards don't exist").
Synthetic fixtures only — the snapshot is gitignored, so the suite must not need it; the
real-snapshot counts are `scripts/audit_dgf1_datasource.py`'s job (ADR-011 clause 5).

Covers clause-5 tests 1, 2, 4, 5 and 6. Tests 7 (fit-on-train transform) and 8 (parity floor)
arrive with the training step and the floor; test 10 with the temporal sampler.

The fixture is DGraph-shaped: it has an edge between two users who both exist before the cutoff
that is itself dated **after** it (e2 below) — the case ELL-1's fixtures never contained.
"""

from __future__ import annotations

import pytest
import torch
import yaml

from gbe.eval import TemporalSplit, assert_no_temporal_leakage, edges_as_of
from adapters.dgf1.datasource_dgraph import (
    NUM_EDGE_TYPES,
    RECENCY_FEATURES,
    VIEW_FEATURE_NAMES,
    graph_view,
    labelled_mask,
    node_time_earliest_edge,
    prepare_dgraph,
    train_seed_mask,
    view_node_features,
    window_target_mask,
)
from adapters.dgf1.eval import CONFIG_PATH, dgf1_splits

T = 3  # fixture train_max
SPLIT_VAL = TemporalSplit(train_max=T, test_min=4, test_max=5)
SPLIT_TEST = TemporalSplit(train_max=T, test_min=6, test_max=8)

#          e0     e1     e2          e3      e4     e5     e6     e7
SRC = [0, 1, 0, 3, 4, 5, 6, 7]
DST = [1, 2, 2, 0, 1, 4, 3, 6]
TIME = [1, 2, 5, 3, 4, 6, 7, 8]      # e2: users 0 (t1) and 2 (t2) exist by 3; the edge is dated 5
TYPE = [1, 3, 1, 11, 2, 2, 5, 1]
Y = [0, 1, 2, 3, 0, 1, 0, 1]          # users 2 and 3 are background — the TWO background values
EXPECTED_NODE_TIME = [1, 1, 2, 3, 4, 6, 7, 8]


def _data(reverse_edges=True, **overrides):
    kw = dict(
        x=torch.arange(8 * 17, dtype=torch.float).view(8, 17),
        edge_index=torch.tensor([SRC, DST]),
        edge_time=torch.tensor(TIME),
        edge_type=torch.tensor(TYPE),
        y=torch.tensor(Y),
    )
    kw.update(overrides)
    return prepare_dgraph(reverse_edges=reverse_edges, **kw)


# -- clause 1: node time, labels, escalations -----------------------------------------------
def test_node_time_is_the_earliest_incident_edge_crediting_both_endpoints():
    assert _data().node_time.tolist() == EXPECTED_NODE_TIME


def test_labelled_mask_excludes_both_background_values():
    """ADR-010: y carries 2 AND 3 as background; `y != 2` alone would label user 3."""
    assert labelled_mask(torch.tensor(Y)).tolist() == [True, True, False, False, True, True, True, True]


def test_a_node_without_edges_is_rejected_not_placed():
    with pytest.raises(AssertionError, match="no incident edge"):
        _data(x=torch.zeros(9, 17), y=torch.tensor(Y + [0]))


@pytest.mark.parametrize("bad_type", [0, 12])
def test_an_edge_type_outside_1_to_11_escalates(bad_type):
    """ADR-010: 11 types, no type 0. A 12-wide histogram would ship a permanently-zero column."""
    types = torch.tensor(TYPE)
    types[0] = bad_type
    with pytest.raises(AssertionError, match="edge_type outside"):
        _data(edge_type=types)


# -- clause-5 test 1: training edge set is exactly edges_as_of(…, T), reverse-doubled ------
def test_training_view_equals_edges_as_of_then_reverse_doubled():
    data = _data()
    view = graph_view(data, T)
    fwd = edges_as_of(data.edge_index, edge_time=data.edge_time, t_max=T)
    assert torch.equal(view.edge_index, torch.cat([fwd, fwd.flip(0)], dim=1))
    assert fwd.tolist() == [[0, 1, 3], [1, 2, 0]]            # e0, e1, e3 — e2 is gone
    assert not any((u, v) in {(0, 2), (2, 0)} for u, v in view.edge_index.t().tolist())


def test_reverse_edges_keep_their_forward_date_and_type():
    view = graph_view(_data(), T)
    half = view.edge_index.size(1) // 2
    assert torch.equal(view.edge_time[:half], view.edge_time[half:])
    assert torch.equal(view.edge_type[:half], view.edge_type[half:])


def test_reverse_edges_off_is_forward_only():
    data = _data(reverse_edges=False)
    view = graph_view(data, T)
    assert torch.equal(view.edge_index, edges_as_of(data.edge_index, edge_time=data.edge_time, t_max=T))


def test_training_view_passes_the_core_leakage_check():
    data = _data()
    view = graph_view(data, T)
    assert_no_temporal_leakage(view.edge_index, data.node_time, SPLIT_VAL, edge_time=view.edge_time)


def test_node_induced_graph_would_have_leaked_here():
    """The teeth, on this fixture: filtering by node time keeps e2, which is dated after T."""
    data = _data()
    both_exist = (data.node_time[data.edge_index] <= T).all(dim=0)
    assert bool((both_exist & (data.edge_time > T)).any()), "fixture lost its DGraph-shaped edge"
    with pytest.raises(AssertionError, match="dated after"):
        assert_no_temporal_leakage(
            data.edge_index[:, both_exist], data.node_time, SPLIT_VAL,
            edge_time=data.edge_time[both_exist],
        )


# -- clause-5 test 2: seeds ------------------------------------------------------------------
def test_train_seeds_are_labelled_users_existing_by_train_max():
    seeds = train_seed_mask(_data(), SPLIT_VAL)
    assert seeds.nonzero().view(-1).tolist() == [0, 1]    # users 2, 3 exist by 3 but are background


def test_window_targets_are_labelled_users_first_appearing_in_the_window():
    data = _data()
    assert window_target_mask(data, SPLIT_VAL).nonzero().view(-1).tolist() == [4]
    assert window_target_mask(data, SPLIT_TEST).nonzero().view(-1).tolist() == [5, 6, 7]
    assert not bool((window_target_mask(data, SPLIT_TEST) & train_seed_mask(data, SPLIT_TEST)).any())


# -- clause-5 test 4: prefix-determinism -----------------------------------------------------
def test_window_membership_at_every_boundary_depends_only_on_edges_up_to_it():
    data = _data()
    for b in range(0, 10):
        keep = data.edge_time <= b
        t_b = node_time_earliest_edge(data.edge_index[:, keep], data.edge_time[keep], 8)
        from_prefix = (t_b >= 0) & (t_b <= b)
        assert torch.equal(from_prefix, data.node_time <= b), f"boundary {b}"


# -- clause-5 test 5: scoring views ----------------------------------------------------------
def test_scoring_view_has_nothing_after_its_window_but_keeps_older_edges():
    view = graph_view(_data(), 5)                           # the fixture's val window ends at 5
    assert int(view.edge_time.max()) <= 5
    fwd = view.edge_index[:, : view.edge_index.size(1) // 2].t().tolist()
    assert [0, 2] in fwd, "the scoring view dropped e2, an older-user edge it should keep"


# -- clause-5 test 6: view features come from the <= T edge set alone -----------------------
def test_view_features_equal_features_recomputed_from_the_prefix_alone():
    data = _data()
    keep = data.edge_time <= T
    ei, et, ety = data.edge_index[:, keep], data.edge_time[keep], data.edge_type[keep]
    alone, _ = view_node_features(ei, et, ety, node_time_earliest_edge(ei, et, 8), T)
    assert torch.equal(graph_view(data, T).node_features, alone)


def test_later_edges_cannot_change_earlier_view_features():
    data = _data()
    before = graph_view(data, T).node_features.clone()
    data.edge_type[data.edge_time > T] = 7                  # rewrite every post-T edge's type
    assert torch.equal(graph_view(data, T).node_features, before)


def test_histogram_is_11_wide_and_rows_sum_to_degree_in_the_view():
    data = _data()
    view = graph_view(data, T)
    hist = view.node_features[:, :NUM_EDGE_TYPES]
    assert hist.shape == (8, 11)
    assert hist[0].tolist() == [1] + [0] * 9 + [1]         # user 0 at T=3: e0 type 1, e3 type 11
    fwd = view.edge_index[:, : view.edge_index.size(1) // 2]
    degree = torch.bincount(fwd.reshape(-1), minlength=8).float()
    assert torch.equal(hist.sum(dim=1), degree)


def test_recency_is_measured_from_the_views_own_cutoff():
    data = _data()
    at_t = graph_view(data, T).node_features[:, NUM_EDGE_TYPES:]
    at_end = graph_view(data, 8).node_features[:, NUM_EDGE_TYPES:]
    # user 0: last edge in the <=3 view is e3@3, first appearance 1
    assert at_t[0].tolist() == [0.0, 2.0]
    # user 2: last edge e1@2 -> 1 step ago; age 3 - 2 = 1
    assert at_t[2].tolist() == [1.0, 1.0]
    # user 4 does not exist by 3: no view edges, zero features
    assert at_t[4].tolist() == [0.0, 0.0]
    # at the dataset end (8) user 0's last edge is e2@5 -> 3 steps; age 7
    assert at_end[0].tolist() == [3.0, 7.0]


# -- config ----------------------------------------------------------------------------------
def test_windows_come_from_the_config_and_share_one_train_max():
    s = dgf1_splits()
    assert (s["val"].train_max, s["val"].test_min, s["val"].test_max) == (369, 370, 481)
    assert (s["test"].train_max, s["test"].test_min, s["test"].test_max) == (369, 482, 821)


def test_non_contiguous_windows_are_rejected(tmp_path):
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["split"]["test_min"] = 490                          # a gap between val and test
    bad = tmp_path / "config.yaml"
    bad.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        dgf1_splits(bad)


def test_config_names_exactly_the_implemented_view_features():
    """The feature list is in the config hash; it must describe what the code computes."""
    vf = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["view_features"]
    assert vf["edge_type_histogram"] == NUM_EDGE_TYPES
    assert tuple(vf["recency"]) == RECENCY_FEATURES
    assert len(VIEW_FEATURE_NAMES) == NUM_EDGE_TYPES + len(RECENCY_FEATURES)
