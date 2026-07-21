"""RunSession — the context manager that ties config, seeding, and the registry
together so that *every* run produces exactly one well-formed registry row.

    with RunSession(cfg, notes="smoke test") as run:
        seed = run.seed
        ...  # train
        run.log_metrics({"auc": 0.83, "f1": 0.61})

On exit (even on exception) the session writes one row: provenance from the
resolved config, wall-clock from a monotonic timer, and whatever metrics were
logged. One run in, one row out — no manual registry calls at the call site.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gbe.run.config import ResolvedConfig
from gbe.run.registry import append_run
from gbe.run.seeding import seed_everything


def _new_run_id(model: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model}-{stamp}-{uuid.uuid4().hex[:8]}"


class RunSession:
    """Context manager: seed on enter, write one registry row on exit."""

    def __init__(
        self,
        config: ResolvedConfig,
        notes: str = "",
        registry_path: str | Path | None = None,
        write_on_error: bool = True,
    ) -> None:
        self.config = config
        self.notes = notes
        self.registry_path = registry_path
        self.write_on_error = write_on_error

        self.run_id: str = _new_run_id(config.model)
        self.seed: int = config.seed
        self._metrics: dict[str, Any] = {}
        self._start_perf: float | None = None
        self._start_iso: str | None = None
        self.wall_clock_s: float | None = None
        self.written: bool = False

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> "RunSession":
        self.seed = seed_everything(self.config.seed)
        self._start_perf = time.perf_counter()
        self._start_iso = datetime.now(timezone.utc).isoformat()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.wall_clock_s = round(time.perf_counter() - (self._start_perf or 0.0), 6)
        if exc_type is not None and not self.write_on_error:
            return False  # propagate without recording a row
        self._write_row(errored=exc_type is not None)
        return False  # never suppress exceptions

    # -- public --------------------------------------------------------------
    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Merge a dict of metrics into what will be recorded at exit."""
        self._metrics.update(metrics)

    # -- internal ------------------------------------------------------------
    def _write_row(self, errored: bool) -> Path:
        notes = self.notes
        if errored:
            notes = (notes + " | ERRORED").strip(" |")
        path = append_run(
            {
                "timestamp_utc": self._start_iso,
                "run_id": self.run_id,
                "model": self.config.model,
                "phase": self.config.phase,
                "config_hash": self.config.config_hash,
                "git_commit": self.config.git_commit,
                "git_dirty": self.config.git_dirty,
                "seed": self.seed,
                "data_snapshot_id": self.config.data_snapshot_id,
                "metrics_json": self._metrics,
                "wall_clock_s": self.wall_clock_s,
                "notes": notes,
            },
            path=self.registry_path,
        )
        self.written = True
        return path
