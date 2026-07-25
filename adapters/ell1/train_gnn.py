"""ELL-1 strict-inductive GNN training + evaluation (doc-01 §3, §5).

Wires the domain-agnostic core (`gbe.features` + `gbe.gnn` + `gbe.eval` + `gbe.run`) to the
Elliptic graph. Kept in the adapter for now; the genuinely-shared loop is extracted to
`gbe/` when DGF-1 gives it a second use case (doc-00 §1 build-path discipline).

The strict inductive protocol (doc-01 §5, the 0.807->0.12 defense):
  * **Train** message passing runs only over the induced train subgraph — edges with *both*
    endpoints in steps <= train_max (`gbe.eval.induced_train_subgraph`). Loss is on labelled
    train nodes; unknown train nodes stay in for message passing (all <= cutoff, no leak).
  * **Eval** runs one forward over the subgraph induced by *test* nodes only (both endpoints
    in the 35-49 window) — never a full-graph pass, so no test node ever aggregates a train
    node and no batch statistic spans the cutoff.

Features are standardised **fit-on-train** (statistics from steps <= train_max only), matching
the baselines' discipline so a full-data statistic never leaks the test window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from sklearn.preprocessing import QuantileTransformer
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph

from gbe.eval import (
    TemporalSplit,
    classification_metrics,
    induced_train_subgraph,
    split_masks,
)
from gbe.features.encoder import TabularMLPEncoder
from gbe.gnn import BACKBONES, NodeClassificationHead, NodeClassifier
from gbe.run import RunSession
from gbe.run.config import resolve_config
from adapters.ell1.datasource_elliptic import ILLICIT, LICIT, UNKNOWN


@dataclass
class GNNHParams:
    """The Phase-1 hyperparameters (doc-00 §6.4 search space + fixed training budget)."""

    backbone: str = "graphsage"
    num_layers: int = 2
    hidden_dim: int = 128
    aggr: str = "mean"          # graphsage only; ignored by gcn
    dropout: float = 0.2
    lr: float = 1e-3
    fan_out: tuple[int, ...] = (15, 10)
    encoder_layers: int = 2
    norm: str = "layer"
    epochs: int = 40
    batch_size: int = 1024
    weight_decay: float = 5e-4

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["fan_out"] = list(self.fan_out)
        return d


def resolve_device(device: str | None) -> torch.device:
    """'cuda'/'cpu'/None -> a torch.device; None (or 'auto') picks cuda when present."""
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def standardize_fit_on_train(x: Tensor, time_step: Tensor, train_max: int) -> Tensor:
    """Standardise features using mean/std from steps <= train_max only (fit-on-train).

    A scaler fit over all nodes would leak test-period statistics (doc-01 §2.2.5). Returns a
    new standardised tensor; the input is left untouched.
    """
    train_mask = time_step <= train_max
    mu = x[train_mask].mean(dim=0, keepdim=True)
    sigma = x[train_mask].std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sigma


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


def class_weights(y: Tensor, seed_mask: Tensor, device: torch.device) -> Tensor:
    """Balanced 2-class weights [w_licit, w_illicit] from the labelled train nodes."""
    labels = y[seed_mask]
    n = labels.numel()
    counts = torch.tensor(
        [int((labels == LICIT).sum()), int((labels == ILLICIT).sum())],
        dtype=torch.float,
    ).clamp_min(1.0)
    # inverse-frequency, normalised so weights average to 1 (sklearn 'balanced' scheme)
    w = n / (2.0 * counts)
    return w.to(device)


def build_model(in_dim: int, hp: GNNHParams, device: torch.device) -> NodeClassifier:
    """Assemble encoder -> backbone -> node-classification head from hyperparameters."""
    encoder = TabularMLPEncoder(
        in_dim=in_dim,
        hidden_dim=hp.hidden_dim,
        out_dim=hp.hidden_dim,
        num_layers=hp.encoder_layers,
        dropout=hp.dropout,
    )
    backbone_cls = BACKBONES[hp.backbone]
    kwargs: dict[str, Any] = dict(
        in_dim=hp.hidden_dim,
        hidden_dim=hp.hidden_dim,
        num_layers=hp.num_layers,
        dropout=hp.dropout,
        norm=hp.norm,
    )
    if hp.backbone == "graphsage":
        kwargs["aggr"] = hp.aggr
    backbone = backbone_cls(**kwargs)
    head = NodeClassificationHead(in_dim=backbone.out_dim, num_classes=2)
    return NodeClassifier(encoder, backbone, head).to(device)


def _fan_out_for(hp: GNNHParams) -> list[int]:
    """NeighborLoader needs one fan-out per message-passing layer; extend by repeating the
    last value when a 3-layer config is paired with a length-2 fan-out (doc §6.4 couples
    layers {2,3} with length-2 fan-outs)."""
    fo = list(hp.fan_out)
    while len(fo) < hp.num_layers:
        fo.append(fo[-1])
    return fo[: hp.num_layers]


def train_model(
    model: NodeClassifier,
    x: Tensor,
    edge_index_train: Tensor,
    y: Tensor,
    train_seed_mask: Tensor,
    hp: GNNHParams,
    device: torch.device,
    on_epoch: Callable[[int, NodeClassifier], None] | None = None,
) -> NodeClassifier:
    """Train under the strict inductive protocol via neighbour sampling on the train subgraph.

    ``edge_index_train`` must already be the induced train subgraph (train-train edges only);
    seeds are the labelled train nodes, so every sampled neighbourhood stays <= train_max.
    ``on_epoch(epoch, model)`` is an optional hook (HPO uses it to report/prune per epoch).
    """
    data = Data(x=x, edge_index=edge_index_train, y=y, num_nodes=x.size(0))
    loader = NeighborLoader(
        data,
        num_neighbors=_fan_out_for(hp),
        input_nodes=train_seed_mask,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=0,  # in-process -> deterministic under the run seed
    )
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, train_seed_mask, device))

    for epoch in range(hp.epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            _, logits = model(batch.x, batch.edge_index)
            seeds = batch.batch_size  # first `batch_size` nodes are the seeds
            loss = loss_fn(logits[:seeds], batch.y[:seeds])
            loss.backward()
            opt.step()
        if on_epoch is not None:
            on_epoch(epoch, model)
    return model


@torch.no_grad()
def predict_window(
    model: NodeClassifier,
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
    model: NodeClassifier,
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
        edge_train, train_node_mask = induced_train_subgraph(
            data.edge_index, data.time_step, split
        )
        train_seed_mask = train_node_mask & data.labelled_mask

        model = build_model(x.size(1), hp, dev)
        train_model(model, x, edge_train, data.y, train_seed_mask, hp, dev)
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
        # Ablation arm, when the caller is running one. It is already in the hashed config, so
        # it was always *recoverable* by recomputing hashes — but a row should say what it is
        # without that detour, or assembling a gate file means reconstructing provenance
        # instead of reading it.
        if "ablation" in base_cfg_values:
            logged["ablation"] = str(base_cfg_values["ablation"])
        run.log_metrics(logged)
    return run.run_id, logged
