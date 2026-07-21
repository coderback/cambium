"""Spot-resumable checkpointing.

RunPod spot instances are reclaimed mid-run, so resume must be *exact*: not just
the model and optimizer, but the RNG states of python/numpy/torch(/cuda), so the
continued run draws the same random numbers an uninterrupted run would have. That
is what makes a resumed run bit-for-bit identical to one that was never killed.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config_hash: str,
    seed: int,
    step: int,
) -> Path:
    """Bundle model + optimizer + provenance + RNG state to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config_hash": config_hash,
        "seed": seed,
        "step": step,
        "rng_state": _capture_rng_state(),
    }
    torch.save(bundle, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint in place.

    Loads model (and optimizer, if given) state and, by default, the RNG states so
    training continues exactly. Returns the bundle metadata (``step``, ``seed``,
    ``config_hash``) for the caller to resume from.
    """
    path = Path(path)
    # weights_only=False: the bundle intentionally carries python/numpy RNG objects.
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(bundle["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(bundle["optimizer_state"])
    if restore_rng:
        _restore_rng_state(bundle.get("rng_state", {}))
    return {
        "step": bundle["step"],
        "seed": bundle["seed"],
        "config_hash": bundle["config_hash"],
    }
