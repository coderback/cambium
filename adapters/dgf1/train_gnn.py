"""DGF-1 training and scoring (doc-02 §3/§4 Phase 1; ADR-011 clauses 2-4; ADR-012 clause 3).

* **Train** on the graph as of ``train_max`` (369): the view's edges are already edge-date filtered
  (`gbe.eval.edges_as_of`, via `graph_view`), the inputs are raw-17 + view statistics standardised
  by a transform fitted on that same view, and the seeds are labelled users existing by the cutoff.
* **Score** a window under the **gated view only**: the graph as of the window's last step
  (ADR-011 clause 2), with inputs built from that same graph and the training transform frozen.

The two reported sensitivity views ADR-011 clause 4 pre-registered (window-only, first appearance)
are **not scored here**. Their first implementation built the GNN's inputs from the window-end
graph under every view, breaking ADR-011 clause 3, so ADR-013 withdrew their numbers and removed
them from this code path. The building blocks stay, tested and wired into no row:
`window_only_edges` below, and `adapters.dgf1.sampling`. Whether and how they are used again is the
matched-time pre-registration's decision (ADR-013 clause 5).

Scoring uses **exact neighbourhoods** (``num_neighbors = [-1] * layers``), so no score depends on an
evaluation seed (ADR-012 clause 3; measured feasible at ~8k nodes/batch). Seeds arrive in
``input_nodes`` order (``shuffle=False``), which is what lets the scored vector be aligned to its
node ids.

The trainer is **window-agnostic**: it takes a `TemporalSplit`. Nothing here may open the test
window — that is the runner's pre-registration guard (ADR-012 clause 8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from gbe.eval import TemporalSplit, classification_metrics, save_scores
from gbe.features.scaling import Standardizer
from gbe.gnn import GNNHParams, build_model, train_model
from gbe.gnn.model import NodeClassifier
from gbe.run import RunSession, resolve_device
from gbe.run.config import resolve_config
from adapters.dgf1.datasource_dgraph import (
    FRAUD,
    NORMAL,
    GraphView,
    graph_view,
    train_seed_mask,
    window_target_mask,
)
from adapters.dgf1.features import fit_input_transform, node_inputs

PROVENANCE_KEYS: tuple[str, ...] = ("arm", "experiment", "features", "window")

# The retuned knobs, echoed flat into every row so a row states the configuration that produced it
# (ADR-012 clause 7). They were always inside the hashed `gnn` config, so the first retune batch's
# rows were recoverable only by rebuilding hashes — the same gap the floor had, and the detour
# ADR-008 was forced into. `epochs` is included because clause 3's runtime fallback can change it.
TUNED_KNOBS: tuple[str, ...] = ("lr", "batch_size", "epochs")
SCORING_BATCH_SIZE = 1024   # deliberately *not* hp.batch_size: that is ADR-012's tuned knob, and
                            # a retune must not silently change what scoring costs.


def train_dgf1(
    data: Data,
    split: TemporalSplit,
    hp: GNNHParams,
    device: torch.device,
    transform: Standardizer | None = None,
) -> tuple[NodeClassifier, Standardizer, GraphView]:
    """Train on the graph as of ``split.train_max``; return ``(model, transform, train_view)``.

    The fitted transform is returned so scoring reuses it **frozen**: refitting on a scoring view
    would leak that window's statistics into the inputs (ADR-011 clause 3).
    """
    train_view = graph_view(data, split.train_max)
    transform = transform or fit_input_transform(data, split)
    x = transform.transform(node_inputs(data, train_view))
    seeds = train_seed_mask(data, split)
    model = build_model(x.size(1), hp, device)
    train_model(model, x, train_view.edge_index, data.y, seeds, hp, device,
                classes=(NORMAL, FRAUD))
    return model, transform, train_view


def window_only_edges(data: Data, view: GraphView, split: TemporalSplit) -> Tensor:
    """ELL-1's eval rule applied to DGraph: both endpoints must first appear inside the window.

    A tested building block wired into no row (ADR-013 clause 3). It keeps reverse edges, so any
    view statistic built from its output would double-count — which is why ADR-013 clause 5 item 7
    requires forward edges only.
    """
    hi = split.test_max if split.test_max is not None else int(data.node_time.max())
    inside = (data.node_time >= split.test_min) & (data.node_time <= hi)
    return view.edge_index[:, inside[view.edge_index[0]] & inside[view.edge_index[1]]]


@torch.no_grad()
def score_view(
    model: NodeClassifier,
    data: Data,
    split: TemporalSplit,
    transform: Standardizer,
    hp: GNNHParams,
    device: torch.device,
    view_kind: str = "gated",
    batch_size: int = SCORING_BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """Score the window's targets under the gated view -> ``{node_ids, y_true, proba, pred}``.

    ``view_kind`` survives only so that a caller asking for anything else is refused loudly rather
    than handed gated numbers under another name (ADR-013 clauses 3 and 6).
    """
    if view_kind != "gated":
        raise ValueError(
            f"unknown view {view_kind!r}: only 'gated' is scored. ADR-013 removed the reported "
            "sensitivity views from this path; ADR-013 clause 5 governs any replacement."
        )
    if split.test_max is None:
        raise ValueError("a DGF-1 window must have an explicit last step (test_max)")

    view = graph_view(data, split.test_max)
    x = transform.transform(node_inputs(data, view))
    targets = window_target_mask(data, split).nonzero().view(-1)

    graph = Data(x=x, y=data.y, num_nodes=data.num_nodes, edge_index=view.edge_index)
    loader = NeighborLoader(
        graph, num_neighbors=[-1] * hp.num_layers, input_nodes=targets,
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    model.eval()
    proba, pred = [], []
    for batch in loader:
        batch = batch.to(device)
        _, logits = model(batch.x, batch.edge_index)
        seeds = logits[: batch.batch_size]
        proba.append(torch.softmax(seeds, dim=-1)[:, FRAUD].cpu())
        pred.append(seeds.argmax(dim=-1).cpu())

    return {
        "node_ids": targets.numpy(),
        "y_true": data.y[targets].numpy(),
        "proba": torch.cat(proba).numpy(),
        "pred": torch.cat(pred).numpy(),
    }


def evaluate_view(scored: dict[str, np.ndarray]) -> dict[str, Any]:
    """Core metrics on the fraud class, with the window's prevalence and positive count (ADR-007)."""
    m = classification_metrics(scored["y_true"], scored["proba"], scored["pred"], pos_label=FRAUD)
    n = int(scored["y_true"].size)
    n_pos = int((scored["y_true"] == FRAUD).sum())
    return {**{f"fraud_{k}": v for k, v in m.items()},
            "n_score": n, "n_score_fraud": n_pos, "prevalence": n_pos / max(n, 1)}


def run_config_values(
    seed: int,
    split: TemporalSplit,
    hp: GNNHParams,
    base_cfg: dict[str, Any],
    feature_set: str = "parity",
) -> dict[str, Any]:
    """The config a DGF-1 run hashes into its row. Factored out so the repro check can confirm,
    before spending a training run, that it would hash the same configuration as its reference."""
    return {**base_cfg, "seed": seed, "gnn": hp.as_dict(),
            "window": f"{split.test_min}-{split.test_max}",
            "arm": f"dgf1-{feature_set}", "features": feature_set}


def run_dgf1(
    seed: int,
    data: Data,
    split: TemporalSplit,
    hp: GNNHParams,
    base_cfg: dict[str, Any],
    registry_path=None,
    device: str | None = None,
    scores_dir: str | Path | None = None,
    feature_set: str = "parity",
) -> tuple[str, dict[str, Any]]:
    """One DGF-1 run: train once, score the gated view, write exactly one registry row.

    Metrics are logged bare (``fraud_auc``, ``fraud_auprc``, …) and scores are persisted with their
    ids (ADR-012 clause 10). Rows before ADR-013 also carry ``window_only_*`` and
    ``first_appearance_*`` keys; those are withdrawn, and this function no longer writes them.
    """
    dev = resolve_device(device)
    cfg_values = run_config_values(seed, split, hp, base_cfg, feature_set)
    window = cfg_values["window"]
    cfg = resolve_config(cfg_values)

    with RunSession(cfg, notes=f"dgf1 gnn on {window}", registry_path=registry_path) as run:
        model, transform, _ = train_dgf1(data, split, hp, dev)

        gated = score_view(model, data, split, transform, hp, dev, "gated")
        logged: dict[str, Any] = evaluate_view(gated)

        if scores_dir is not None:
            path = Path(scores_dir) / f"{run.run_id}.npz"
            logged["scores_sha256"] = save_scores(
                path, node_ids=gated["node_ids"], y_true=gated["y_true"], proba=gated["proba"]
            )
            logged["scores_path"] = str(path)

        logged["backbone"] = hp.backbone
        logged["n_train"] = int(train_seed_mask(data, split).sum())
        logged.update({k: cfg_values[k] for k in PROVENANCE_KEYS if k in cfg_values})
        hp_values = hp.as_dict()
        logged.update({k: hp_values[k] for k in TUNED_KNOBS})
        run.log_metrics(logged)
    return run.run_id, logged
