"""Append-only writer for ``experiments/registry.csv``.

One row per run. The registry is the single source of truth for every reported
number in the program (CLAUDE.md), so this module exposes **exactly one** mutating
operation: :func:`append_run`. There is deliberately no update, delete, or
overwrite API — rows, once written, are never touched again.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

# Exact column order — do not reorder without an ADR (papers read this file).
REGISTRY_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "run_id",
    "model",
    "phase",
    "config_hash",
    "git_commit",
    "git_dirty",
    "seed",
    "data_snapshot_id",
    "metrics_json",
    "wall_clock_s",
    "notes",
)


def default_registry_path(repo_root: Path | None = None) -> Path:
    """Path to the canonical registry (``experiments/registry.csv``)."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    return root / "experiments" / "registry.csv"


def _serialise(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def append_run(
    row: Mapping[str, Any],
    path: str | Path | None = None,
) -> Path:
    """Append a single run row to the registry, writing the header only if new.

    This is the *only* public operation on the registry. Existing rows are never
    read back for modification; the file is opened strictly in append mode.

    Args:
        row: mapping keyed by (a subset of) ``REGISTRY_COLUMNS``. ``metrics_json``
            may be passed as a dict/list and will be JSON-serialised. Unknown keys
            are rejected so a typo can never silently create a malformed row.
        path: registry path; defaults to ``experiments/registry.csv``.

    Returns:
        The path written to.
    """
    unknown = set(row) - set(REGISTRY_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown registry columns: {sorted(unknown)}")

    path = Path(path) if path is not None else default_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {col: _serialise(row.get(col)) for col in REGISTRY_COLUMNS}

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
        fh.flush()
        os.fsync(fh.fileno())

    return path
