"""Guard: the registry is append-only, config hashing is stable, seeds distinguish rows."""

from __future__ import annotations

import csv

import gbe.run.registry as registry
from gbe.run.config import resolve_config
from gbe.run.registry import REGISTRY_COLUMNS, append_run


def _row(seed: int, config_hash: str = "abc123") -> dict:
    return {
        "timestamp_utc": "2026-07-21T00:00:00+00:00",
        "run_id": f"toy-{seed}",
        "model": "toy",
        "phase": "P0",
        "config_hash": config_hash,
        "git_commit": "deadbeef",
        "git_dirty": False,
        "seed": seed,
        "data_snapshot_id": "snap-0",
        "metrics_json": {"auc": 0.5},
        "wall_clock_s": 1.0,
        "notes": "",
    }


def test_no_mutating_api_exists():
    """The registry module must expose no update/delete/overwrite operation."""
    public = {name for name in dir(registry) if not name.startswith("_")}
    for forbidden in ("update_run", "delete_run", "overwrite_run", "update", "delete", "remove"):
        assert forbidden not in public, f"registry unexpectedly exposes {forbidden!r}"


def test_append_does_not_rewrite_prior_rows(tmp_path):
    """Appending a second row leaves the first row byte-identical."""
    path = tmp_path / "registry.csv"

    append_run(_row(0), path=path)
    after_first = path.read_bytes()

    append_run(_row(1), path=path)
    after_second = path.read_bytes()

    # The full first-write content must be a prefix of the file after the 2nd write.
    assert after_second.startswith(after_first)

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert list(rows[0].keys()) == list(REGISTRY_COLUMNS)
    assert rows[0]["run_id"] == "toy-0"
    assert rows[1]["run_id"] == "toy-1"


def test_identical_configs_same_hash():
    """Same resolved config (any key order) -> identical config hash."""
    a = resolve_config({"model": "toy", "seed": 0, "lr": 0.1, "layers": 2})
    b = resolve_config({"layers": 2, "lr": 0.1, "seed": 0, "model": "toy"})
    assert a.config_hash == b.config_hash


def test_different_seed_distinct_rows(tmp_path):
    """Two runs differing only by seed produce two distinct registry rows."""
    path = tmp_path / "registry.csv"
    append_run(_row(0), path=path)
    append_run(_row(1), path=path)

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    seeds = {r["seed"] for r in rows}
    assert seeds == {"0", "1"}
    assert len(rows) == 2


def test_unknown_column_rejected(tmp_path):
    """A typo'd column must raise rather than silently write a malformed row."""
    path = tmp_path / "registry.csv"
    bad = _row(0)
    bad["walltime"] = 1.0  # not a real column
    try:
        append_run(bad, path=path)
    except KeyError:
        return
    raise AssertionError("append_run accepted an unknown column")
