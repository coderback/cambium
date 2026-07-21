"""Leakage + strict-inductive guards for the ELL-1 temporal split and Elliptic loader.

Written in the same session as the pipeline (CLAUDE.md: "untested guards don't exist").
Runs on synthetic fixtures — the real Elliptic data is gitignored and absent — so it
exercises the split/leakage *logic*, and the loader parsing on tiny in-memory CSVs.
"""

from __future__ import annotations

import torch

from gbe.eval import (
    TemporalSplit,
    assert_no_temporal_leakage,
    induced_train_subgraph,
    split_masks,
)
from adapters.ell1.datasource_elliptic import ILLICIT, LICIT, UNKNOWN, load_elliptic
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


def test_induced_train_subgraph_only_references_train_nodes():
    time_step = torch.tensor([30, 34, 35, 40])
    # (0,1) both train; (1,2) crosses cutoff; (2,3) both test.
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_train, train_node_mask = induced_train_subgraph(edge_index, time_step, SPLIT)

    endpoints = edge_train.unique()
    assert bool(train_node_mask[endpoints].all()), "induced subgraph reached a test node"
    assert edge_train.shape[1] == 1, "only the (0,1) train-train edge should survive"
    assert_no_temporal_leakage(edge_train, time_step, SPLIT)  # must be silent


# -- the guard has teeth ---------------------------------------------------------------
def test_leakage_assertion_fires_on_crossing_edge():
    time_step = torch.tensor([30, 34, 35, 40])
    raw_train_edges = torch.tensor([[0, 1], [1, 2]])  # (1,2) crosses the cutoff
    try:
        assert_no_temporal_leakage(raw_train_edges, time_step, SPLIT)
    except AssertionError:
        return
    raise AssertionError("assert_no_temporal_leakage missed a cross-cutoff edge")


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
