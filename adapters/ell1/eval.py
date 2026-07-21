"""ELL-1's temporal split — the concrete instance of the core :class:`TemporalSplit`.

The machinery is domain-agnostic (``gbe.eval``); the 34 / 35 / 49 cutoff values are
model-specific and read from ``adapters/ell1/config.yaml`` so they land in the config
hash and thus in ``experiments/registry.csv`` (see the config file's header).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from gbe.eval import TemporalSplit

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def ell1_eval_split(config_path: str | Path = CONFIG_PATH) -> TemporalSplit:
    """Build ELL-1's :class:`TemporalSplit` from the adapter config's ``split`` block."""
    with Path(config_path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    split = cfg["split"]
    return TemporalSplit(
        train_max=int(split["train_max"]),
        test_min=int(split["test_min"]),
        test_max=int(split["test_max"]) if split.get("test_max") is not None else None,
        time_attr=str(split.get("time_attr", "time_step")),
    )
