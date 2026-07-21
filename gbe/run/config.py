"""Experiment config: YAML load, stable content-hash, git provenance, seed.

The config hash is the reproducibility anchor: two runs with the same resolved
config produce the same hash regardless of key ordering or file formatting.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SEED = 0


def _resolve(raw: Any) -> Any:
    """Normalise a loaded config into a canonical, JSON-serialisable form.

    Sorting is handled at hash time (``sort_keys=True``); here we only coerce
    mappings to plain dicts so YAML-specific node types do not leak in.
    """
    if isinstance(raw, dict):
        return {str(k): _resolve(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        return [_resolve(v) for v in raw]
    return raw


def compute_config_hash(resolved: dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON of the resolved config (order-independent)."""
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_commit(repo_root: Path | None = None) -> str:
    """Current git commit hash, or ``"unknown"`` if not a git repo / git absent."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# Run *outputs* that must not count as a dirty working tree: the append-only registry
# is written by every run, so a tracked registry.csv would mark every run after the first
# in a batch as dirty for a non-reason. dirty must mean uncommitted *code/config* (ADR-002).
IGNORED_DIRTY_PATHS: frozenset[str] = frozenset({"experiments/registry.csv"})


def _dirty_paths(porcelain: str, ignore: frozenset[str]) -> list[str]:
    """Paths from ``git status --porcelain`` output, minus ignored run-output artifacts."""
    paths: list[str] = []
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()  # porcelain: 2 status chars + space, then the path
        if " -> " in path:  # rename/copy: "old -> new"
            path = path.split(" -> ", 1)[1]
        path = path.strip('"')
        if path not in ignore:
            paths.append(path)
    return paths


def git_dirty(
    repo_root: Path | None = None,
    ignore: frozenset[str] = IGNORED_DIRTY_PATHS,
) -> bool:
    """True if the working tree has uncommitted code/config (or git state is unknown).

    Run-output artifacts (``ignore``, default the append-only registry) are excluded so a
    run's ``dirty`` flag reflects whether its *inputs* were committed, not that it appended
    its own row to a tracked registry (ADR-002).
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(_dirty_paths(out.stdout, ignore))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


@dataclass(frozen=True)
class ResolvedConfig:
    """An immutable, hashed view of an experiment config plus its provenance."""

    values: dict[str, Any]
    config_hash: str
    git_commit: str
    git_dirty: bool
    seed: int
    source_path: str | None = None
    _frozen_json: str = field(default="", repr=False, compare=False)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    @property
    def model(self) -> str:
        return str(self.values.get("model", "unknown"))

    @property
    def phase(self) -> str:
        return str(self.values.get("phase", "unknown"))

    @property
    def data_snapshot_id(self) -> str:
        return str(self.values.get("data_snapshot_id", "unknown"))


def resolve_config(
    values: dict[str, Any],
    repo_root: Path | None = None,
    source_path: str | None = None,
) -> ResolvedConfig:
    """Build a :class:`ResolvedConfig` from an in-memory config dict."""
    resolved = _resolve(values)
    config_hash = compute_config_hash(resolved)
    seed = int(resolved.get("seed", DEFAULT_SEED))
    return ResolvedConfig(
        values=resolved,
        config_hash=config_hash,
        git_commit=git_commit(repo_root),
        git_dirty=git_dirty(repo_root),
        seed=seed,
        source_path=source_path,
    )


def load_config(path: str | Path, repo_root: Path | None = None) -> ResolvedConfig:
    """Load a YAML experiment config and resolve it (hash + git provenance + seed)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a mapping, got {type(raw).__name__}")
    return resolve_config(raw, repo_root=repo_root, source_path=str(path))
