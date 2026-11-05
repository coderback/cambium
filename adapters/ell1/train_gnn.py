"""ELL-1 strict-inductive GNN training + evaluation (doc-01 §3, §5).

Wires the domain-agnostic core (`gbe.features` + `gbe.gnn` + `gbe.eval` + `gbe.run`) to the
Elliptic graph. **The training loop itself now lives in `gbe.gnn.train`** — it was extracted when
DGF-1 became its second use case (doc-00 §1/§9, the EXTRACT milestone). `GNNHParams`,
`build_model`, `train_model` and `resolve_device` are re-exported here unchanged, so every script
and test that imports them from this module keeps working, and ELL-1 runs the identical code in the
identical order (ADR-008's checker is the bar).

What stays here is what is Elliptic-specific:

The strict inductive protocol (doc-01 §5, the 0.807->0.12 defense):
  * **Train** message passing runs only over the train graph — edges dated <= train_max
    (`gbe.eval.edges_as_of`, ADR-011). Elliptic's edges carry no date of their own, but every
    one lies within a single step, so `derive_edge_time` dates each by that shared step (and
    asserts it); the train graph is then exactly the edges with both endpoints in steps
    <= train_max. Loss is on labelled train nodes; unknown train nodes stay in for message
    passing (all <= cutoff, no leak).
  * **Eval** runs one forward over the subgraph induced by *test* nodes only (both endpoints
    in the 35-49 window) — never a full-graph pass, so no test node ever aggregates a train
    node and no batch statistic spans the cutoff.

Features are standardised **fit-on-train** (statistics from steps <= train_max only), matching
the baselines' discipline so a full-data statistic never leaks the test window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from sklearn.preprocessing import QuantileTransformer
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from gbe.eval import (
    TemporalSplit,
    classification_metrics,
    edges_as_of,
    split_masks,
)
from gbe.features.scaling import Standardizer
from gbe.gnn.train import GNNHParams, balanced_class_weights, build_model, train_model
from gbe.run import RunSession, resolve_device
from gbe.run.config import resolve_config
from adapters.ell1.datasource_elliptic import ILLICIT, LICIT, UNKNOWN, derive_edge_time

# Re-exported from the core so existing call sites (`from adapters.ell1.train_gnn import ...`)
# keep working after the extraction.
__all__ = [
    "GNNHParams", "build_model", "train_model", "resolve_device", "class_weights",
    "standardize_fit_on_train", "rank_gauss_fit_on_train", "FEATURE_TRANSFORMS",
    "frozen_hparams", "predict_window", "evaluate", "run_gnn", "PROVENANCE_KEYS",
]

# Config keys that identify *which experiment a row belongs to*, echoed from the hashed config
# into the row so a batch can be read back without recomputing config hashes to find out.
PROVENANCE_KEYS: tuple[str, ...] = ("ablation", "arm", "experiment", "features")


def class_weights(y: Tensor, seed_mask: Tensor, device: torch.device) -> Tensor:
    """Balanced 2-class weights ``[w_licit, w_illicit]`` from the labelled train nodes.

    ELL-1's positive class is ``ILLICIT`` and its negative ``LICIT`` (1 and 0), which is the order
    the core's default already uses; naming them here keeps the adapter's meaning explicit.
    """
    return balanced_class_weights(y, seed_mask, device, classes=(LICIT, ILLICIT))


def standardize_fit_on_train(x: Tensor, time_step: Tensor, train_max: int) -> Tensor:
    """Standardise features using mean/std from steps <= train_max only (fit-on-train).

    A scaler fit over all nodes would leak test-period statistics (doc-01 §2.2.5). Returns a
    new standardised tensor; the input is left untouched.

    Delegates to the core `gbe.features.Standardizer` (extracted when DGF-1 became the second
    use). Same operations in the same order, so the same floats: verified ``torch.equal`` on real
    Elliptic at both cutoffs, and the core's zero-variance rule never fires here (no Elliptic
    column has training std below 1e-6). ADR-008's checker is the bar.
    """
    return Standardizer.fit(x, time_step <= train_max).transform(x)


def rank_gauss_fit_on_train(x: Tensor, time_step: Tensor, train_max: int) -> Tensor:
    """Rank-transform features to a Gaussian, quantiles fit on steps <= train_max only.

    Same fit-on-train discipline as :func:`standardize_fit_on_train`, but rank-based: it is
    immune to the heavy tails in Elliptic's aggregated features (a third of the 165 still
    reach |z| > 50 after standardisation), which a tree baseline ignores by construction and
    an MLP does not. The diagnostic arm for handicap #3 — not the frozen ADR-003 path.
    """
    train_mask = (time_step <= train_max).numpy()
    arr = x.numpy()
    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, int(train_mask.sum())),
        subsample=None,
        random_state=0,
    )
    qt.fit(arr[train_mask])
    return torch.from_numpy(qt.transform(arr)).float()


FEATURE_TRANSFORMS: dict[str, Callable[[Tensor, Tensor, int], Tensor]] = {
    "standardize": standardize_fit_on_train,
    "rank_gauss": rank_gauss_fit_on_train,
}


def frozen_hparams(config_path=None) -> tuple[GNNHParams, dict[str, Any], str]:
    """Read the frozen ADR-003 ``gnn:`` block from the adapter config.

    One source of truth for the frozen config: the Gate-1 runner and the inner-window
    diagnostics both parse it here, so they cannot drift apart.

    Returns ``(hparams, base_config_values, device_string)``.
    """
    import yaml

    from adapters.ell1.eval import CONFIG_PATH

    path = Path(config_path) if config_path is not None else CONFIG_PATH
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    g = cfg["gnn"]
    hp = GNNHParams(
        backbone=g["backbone"], num_layers=g["num_layers"], hidden_dim=g["hidden_dim"],
        aggr=g["aggr"], dropout=g["dropout"], lr=float(g["lr"]), fan_out=tuple(g["fan_out"]),
        encoder_layers=g["encoder_layers"], norm=g["norm"], epochs=g["epochs"],
        batch_size=g["batch_size"], weight_decay=float(g["weight_decay"]),
    )
    base_cfg = {
        "model": cfg["model"],
        "phase": "P1",  # these are Phase-1 runs (config.yaml's default P0 was for Gate 0)
        "data_snapshot_id": cfg["data_snapshot_id"],
        "split": cfg["split"],
        "symmetrise": cfg["symmetrise"],
        "features": 165,  # ADR-001
    }
    return hp, base_cfg, str(cfg.get("device", "auto"))


@torch.no_grad()
def predict_window(
    model,
    x: Tensor,
    edge_index: Tensor,
    time_step: Tensor,
    y: Tensor,
    split: TemporalSplit,
    device: torch.device,
):
    """Strict-inductive forward over one window's induced subgraph -> labelled-node scores.

    Builds the subgraph of nodes in ``[test_min, test_max]`` (both endpoints in the window),
    forwards it once, and returns ``(y_true, proba_illicit, pred, time_step, meta)`` restricted
    to the labelled nodes. ``pred`` is the plain argmax (the 0.5 operating point).

    This is the single scoring path: :func:`evaluate` and the inner-window diagnostics both go
    through it, so a diagnostic number is directly comparable to a gate number.
    """
    _, test_mask = split_masks(time_step, split)
    sub_edge, _ = subgraph(
        test_mask, edge_index, relabel_nodes=True, num_nodes=x.size(0)
    )
    x_test = x[test_mask].to(device)
    y_test = y[test_mask]

    model.eval()
    _, logits = model(x_test, sub_edge.to(device))
    proba_illicit = torch.softmax(logits, dim=-1)[:, ILLICIT].cpu()
    pred = logits.argmax(dim=-1).cpu()

    labelled = y_test != UNKNOWN
    meta = {
        "n_test": int(labelled.sum()),
        "n_test_illicit": int((y_test[labelled] == ILLICIT).sum()),
    }
    return (
        y_test[labelled].numpy(),
        proba_illicit[labelled].numpy(),
        pred[labelled].numpy(),
        time_step[test_mask][labelled].numpy(),
        meta,
    )


def evaluate(
    model,
    x: Tensor,
    edge_index: Tensor,
    time_step: Tensor,
    y: Tensor,
    split: TemporalSplit,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, int]]:
    """Strict-inductive eval on the test window: forward over the test-induced subgraph only.

    Scores the labelled window nodes at the argmax operating point. Returns ``(metrics, meta)``
    with illicit-class F1/recall/precision/AUC and the support counts for the gate file.
    """
    y_l, proba_l, pred_l, _, meta = predict_window(
        model, x, edge_index, time_step, y, split, device
    )
    metrics = classification_metrics(y_l, proba_l, pred_l, pos_label=ILLICIT)
    return metrics, meta


def run_gnn(
    backbone: str,
    seed: int,
    data: Data,
    split: TemporalSplit,
    hp: GNNHParams,
    base_cfg_values: dict[str, Any],
    registry_path=None,
    device: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """One backbone at one seed, end-to-end, through :class:`RunSession` -> one registry row.

    ``backbone`` overrides ``hp.backbone`` and is folded into the hashed config and the logged
    metrics (so the gate assembler can group seeds). Returns ``(run_id, metrics)``.
    """
    hp = GNNHParams(**{**hp.as_dict(), "backbone": backbone, "fan_out": tuple(hp.fan_out)})
    dev = resolve_device(device)
    cfg = resolve_config(
        {**base_cfg_values, "backbone": backbone, "seed": seed, "gnn": hp.as_dict()}
    )
    with RunSession(cfg, notes=f"ell1 gnn: {backbone}", registry_path=registry_path) as run:
        # RunSession.__enter__ has already seeded python/numpy/torch(/cuda).
        x = standardize_fit_on_train(data.x, data.time_step, split.train_max)
        # ADR-011: the training graph is the edges dated <= train_max. The dates are derived from
        # the edge set actually in use (ablation arms rewire it) and asserted within-step, which
        # makes this the same edges in the same order as the retired node-induced path — so the
        # frozen ADR-008 reference numbers cannot move.
        edge_time = derive_edge_time(data.edge_index, data.time_step)
        edge_train = edges_as_of(data.edge_index, edge_time=edge_time, t_max=split.train_max)
        train_node_mask, _ = split_masks(data.time_step, split)
        train_seed_mask = train_node_mask & data.labelled_mask

        model = build_model(x.size(1), hp, dev)
        train_model(model, x, edge_train, data.y, train_seed_mask, hp, dev,
                    classes=(LICIT, ILLICIT))
        metrics, meta = evaluate(model, x, data.edge_index, data.time_step, data.y, split, dev)

        logged = {f"illicit_{k}": v for k, v in metrics.items()}
        logged.update(
            {
                "backbone": backbone,
                "test_window": f"{split.test_min}-{split.test_max}",
                "n_train": int(train_seed_mask.sum()),
                **meta,
            }
        )
        # Experiment-identifying config keys, echoed into the row. They are already in the
        # hashed config, so they were always *recoverable* by recomputing hashes — but a row
        # should say what it is without that detour, or reading a batch back means
        # reconstructing provenance instead of reading it. The list is explicit rather than
        # "echo everything" so a config typo cannot silently invent a metrics field.
        logged.update(
            {k: base_cfg_values[k] for k in PROVENANCE_KEYS if k in base_cfg_values}
        )
        run.log_metrics(logged)
    return run.run_id, logged
