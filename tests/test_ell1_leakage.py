"""Leakage + strict-inductive guards for the ELL-1 temporal split and Elliptic loader.

Written in the same session as the pipeline (CLAUDE.md: "untested guards don't exist").
Runs on synthetic fixtures — the real Elliptic data is gitignored and absent — so it
exercises the split/leakage *logic*, and the loader parsing on tiny in-memory CSVs.
"""

from __future__ import annotations

import torch

import pytest

from gbe.eval import (
    TemporalSplit,
    assert_no_temporal_leakage,
    edges_as_of,
    split_masks,
)
from adapters.ell1.datasource_elliptic import (
    ILLICIT,
    LICIT,
    UNKNOWN,
    derive_edge_time,
    load_elliptic,
)
from adapters.ell1.eval import ell1_eval_split

# ELL-1's split, per doc-01 §2.3.
SPLIT = TemporalSplit(train_max=34, test_min=35, test_max=49)


# -- the split values come from the adapter config, not the core -----------------------
def test_ell1_eval_split_from_config():
    split = ell1_eval_split()
    assert (split.train_max, split.test_min, split.test_max) == (34, 35, 49)
    assert split.time_attr == "time_step"


def test_overlapping_split_is_rejected():
    """An overlapping split is a temporal leak by construction — must not be constructible."""
    try:
        TemporalSplit(train_max=34, test_min=34)
    except ValueError:
        return
    raise AssertionError("TemporalSplit accepted an overlapping train/test window")


# -- split correctness -----------------------------------------------------------------
def test_split_masks_disjoint_and_correct():
    time_step = torch.arange(1, 50)  # steps 1..49, one node each
    train_mask, test_mask = split_masks(time_step, SPLIT)

    assert train_mask.sum() == 34 and test_mask.sum() == 15
    assert not bool((train_mask & test_mask).any()), "train and test overlap"
    assert time_step[train_mask].max() <= 34
    assert time_step[test_mask].min() >= 35


# -- no training tensor touches steps 35-49 --------------------------------------------
def test_train_tensors_never_reach_test_window():
    time_step = torch.tensor([30, 34, 35, 40])
    x = torch.randn(4, 3)
    y = torch.tensor([ILLICIT, LICIT, ILLICIT, LICIT])
    train_mask, _ = split_masks(time_step, SPLIT)

    # Everything gathered by the train mask is strictly pre-cutoff.
    assert bool((time_step[train_mask] <= 34).all())
    assert x[train_mask].shape[0] == 2 and y[train_mask].shape[0] == 2


def test_train_graph_only_references_train_nodes():
    time_step = torch.tensor([30, 34, 35, 40])
    # (0,1) both train; (1,2) crosses cutoff; (2,3) both test. Dated at the later endpoint.
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_time = torch.tensor([34, 35, 40])
    edge_train = edges_as_of(edge_index, edge_time=edge_time, t_max=SPLIT.train_max)
    train_node_mask, _ = split_masks(time_step, SPLIT)

    endpoints = edge_train.unique()
    assert bool(train_node_mask[endpoints].all()), "train graph reached a test node"
    assert edge_train.shape[1] == 1, "only the (0,1) train-train edge should survive"
    assert_no_temporal_leakage(edge_train, time_step, SPLIT, edge_time=torch.tensor([34]))


# -- the guard has teeth ---------------------------------------------------------------
def test_leakage_assertion_fires_on_crossing_edge():
    time_step = torch.tensor([30, 34, 35, 40])
    raw_train_edges = torch.tensor([[0, 1], [1, 2]])  # (1,2) crosses the cutoff
    with pytest.raises(AssertionError, match="touch a node"):
        assert_no_temporal_leakage(
            raw_train_edges, time_step, SPLIT, edge_time=torch.tensor([34, 35])
        )


# -- ELL-1's edge dates (ADR-011 clause 5 test 9) ---------------------------------------
def test_derive_edge_time_is_the_shared_step():
    """Elliptic has no edge timestamps; each edge is dated by the one step both endpoints share."""
    time_step = torch.tensor([3, 3, 7, 7])
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]])
    assert derive_edge_time(edge_index, time_step).tolist() == [3, 3, 7]


def test_derive_edge_time_rejects_an_edge_spanning_two_steps():
    """If an edge ever spans two steps, "the later endpoint's step" would be the node-induced
    reading ADR-011 retired — so the derivation must refuse rather than guess."""
    time_step = torch.tensor([3, 3, 7])
    edge_index = torch.tensor([[0, 1], [1, 2]])  # (1,2) joins step 3 to step 7
    with pytest.raises(AssertionError, match="two different time steps"):
        derive_edge_time(edge_index, time_step)


def test_derived_dates_reproduce_the_node_induced_train_graph_exactly():
    """The bit-for-bit argument for the EXTRACT refactor, on a fixture: for within-step edges, the
    edge-date filter returns the same edges *in the same order* as the retired node-induced mask,
    so neighbour sampling — and every ADR-008 reference number — is unchanged."""
    gen = torch.Generator().manual_seed(0)
    time_step = torch.randint(1, 50, (400,), generator=gen)
    src, dst = [], []
    for step in time_step.unique():
        idx = (time_step == step).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        pairs = idx[torch.randint(0, idx.numel(), (3 * idx.numel(), 2), generator=gen)]
        src.append(pairs[:, 0])
        dst.append(pairs[:, 1])
    edge_index = torch.stack([torch.cat(src), torch.cat(dst)])

    for cut in (29, 34):
        node_induced = edge_index[:, (time_step[edge_index] <= cut).all(dim=0)]
        via_dates = edges_as_of(
            edge_index, edge_time=derive_edge_time(edge_index, time_step), t_max=cut
        )
        assert torch.equal(node_induced, via_dates)


# -- loader parsing (synthetic CSVs, no real data) -------------------------------------
def _write_fixture(dir_path):
    (dir_path / "elliptic_txs_features.csv").write_text(
        "\n".join(
            [
                "100,1,0.1,0.2,0.3",
                "200,34,0.4,0.5,0.6",
                "300,35,0.7,0.8,0.9",
                "400,49,1.0,1.1,1.2",
            ]
        ),
        encoding="utf-8",
    )
    (dir_path / "elliptic_txs_classes.csv").write_text(
        "txId,class\n100,1\n200,2\n300,unknown\n400,1\n", encoding="utf-8"
    )
    (dir_path / "elliptic_txs_edgelist.csv").write_text(
        "txId1,txId2\n100,200\n300,400\n", encoding="utf-8"
    )


def test_loader_parses_and_maps_labels(tmp_path):
    _write_fixture(tmp_path)
    data = load_elliptic(tmp_path, symmetrise=True, strict=False)

    assert data.num_nodes == 4
    assert data.x.shape == (4, 3)
    assert data.time_step.tolist() == [1, 34, 35, 49]
    # Label mapping: illicit=1, licit=0, unknown=-1; unknowns kept, not filtered.
    assert data.y.tolist() == [ILLICIT, LICIT, UNKNOWN, ILLICIT]
    assert data.labelled_mask.tolist() == [True, True, False, True]
    assert data.symmetrised is True


def test_loader_symmetrise_flag_controls_direction(tmp_path):
    _write_fixture(tmp_path)
    directed = load_elliptic(tmp_path, symmetrise=False, strict=False)
    undirected = load_elliptic(tmp_path, symmetrise=True, strict=False)

    assert directed.symmetrised is False
    assert directed.edge_index.shape[1] == 2
    # Two non-reciprocal directed edges -> four directed entries once symmetrised.
    assert undirected.edge_index.shape[1] == 4
