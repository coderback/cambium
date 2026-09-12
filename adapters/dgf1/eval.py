"""DGF-1's temporal windows — concrete instances of the core :class:`TemporalSplit` (ADR-011).

The machinery is domain-agnostic (``gbe.eval``); the 369 / 481 / 821 values are model-specific and
read from ``adapters/dgf1/config.yaml`` so they land in the config hash. Both windows share one
``train_max`` because every run trains on the same set (ADR-011 clause 2): ``val`` scores the
retune and the seed pilot, ``test`` is scored by gate runs only.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from gbe.eval import TemporalSplit
from gbe.gnn import GNNHParams

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def dgf1_base_config(config_path: str | Path = CONFIG_PATH) -> dict:
    """The provenance block every DGF-1 run hashes into its registry row (model, phase, snapshot,
    split, reverse edges, view features) — read from the adapter config, one source of truth."""
    with Path(config_path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return {k: cfg[k] for k in
            ("model", "phase", "data_snapshot_id", "split", "reverse_edges", "view_features")}


def dgf1_hparams(config_path: str | Path = CONFIG_PATH) -> tuple[GNNHParams, str]:
    """The inherited ADR-003 ``gnn:`` block as hyperparameters, plus the configured device.

    One source of truth for DGF-1's frozen region, so the runner and any diagnostic cannot drift
    apart — ELL-1 keeps its own copy via `frozen_hparams`, and neither adapter reads the other's.
    """
    with Path(config_path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    g = cfg["gnn"]
    hp = GNNHParams(
        backbone=g["backbone"], num_layers=g["num_layers"], hidden_dim=g["hidden_dim"],
        aggr=g["aggr"], dropout=g["dropout"], lr=float(g["lr"]), fan_out=tuple(g["fan_out"]),
        encoder_layers=g["encoder_layers"], norm=g["norm"], epochs=g["epochs"],
        batch_size=g["batch_size"], weight_decay=float(g["weight_decay"]),
    )
    return hp, str(cfg.get("device", "auto"))


def dgf1_retune_grid(config_path: str | Path = CONFIG_PATH) -> list[tuple[float, int]]:
    """ADR-012 clause 3's grid as ``(lr, batch_size)`` pairs — the only two knobs DGF-1 retunes."""
    with Path(config_path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    base_lr = float(cfg["gnn"]["lr"])
    grid = cfg["retune_grid"]
    return [(base_lr * float(m), int(b))
            for m in grid["lr_multipliers"] for b in grid["batch_sizes"]]


def dgf1_splits(config_path: str | Path = CONFIG_PATH) -> dict[str, TemporalSplit]:
    """``{"val": ..., "test": ...}`` from the adapter config's ``split`` block.

    Raises if the windows are not contiguous (train | val | test with no gap and no overlap): a gap
    would silently drop users from every window, an overlap would score users twice.
    """
    with Path(config_path).open("r", encoding="utf-8") as fh:
        s = yaml.safe_load(fh)["split"]
    if not (s["val_min"] == s["train_max"] + 1 and s["test_min"] == s["val_max"] + 1):
        raise ValueError(
            f"DGF-1 windows must be contiguous: train <= {s['train_max']}, val "
            f"{s['val_min']}-{s['val_max']}, test {s['test_min']}-{s['test_max']} (ADR-011 clause 2)."
        )
    attr = str(s.get("time_attr", "node_time"))
    return {
        "val": TemporalSplit(
            train_max=int(s["train_max"]), test_min=int(s["val_min"]),
            test_max=int(s["val_max"]), time_attr=attr,
        ),
        "test": TemporalSplit(
            train_max=int(s["train_max"]), test_min=int(s["test_min"]),
            test_max=int(s["test_max"]), time_attr=attr,
        ),
    }
