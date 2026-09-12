"""Pre-registration guards shared by the DGF-1 runners (ADR-012 clause 8).

Two refusals, encoded rather than remembered, and deliberately in one place so the floor runner and
the GNN runner cannot drift apart:

* **the test window needs an accepted pre-registration that carries its seed count** — ADR-012
  fixes the *rule* for that count and leaves the *number* to a validation pilot, so an accepted ADR
  whose count has not been derived yet must still keep 482-821 shut;
* **a dirty tree is refused** — rows must be reproducible from a commit, and dirty rows have forced
  full re-runs twice (ADR-002, and the Gate-0/Gate-1 batches).

The seed count is **returned**, so a caller runs the number the document fixed rather than one
typed on the command line.
"""

from __future__ import annotations

import re
from pathlib import Path

from gbe.run.config import git_dirty

_STATUS = re.compile(r"^\*\*Status:\*\*\s*(\w+)", re.MULTILINE)
# ADR-012 clause 8 fixes this line's exact shape, so the guard is not left guessing a format.
_SEEDS = re.compile(r"^\*\*Stage-1 seeds:\*\*\s*(\d+)", re.MULTILINE)


def require_accepted_preregistration(path: str | Path | None) -> int:
    """Exit unless ``path`` is an ADR that is **accepted** *and* carries a stage-1 seed count.

    Returns the count, so the caller can run exactly the pre-registered number of seeds.
    """
    if path is None:
        raise SystemExit(
            "refusing --window test: no --preregistration given. DGF-1's Gate-1 pre-registration ADR "
            "must be accepted before any run touches the test window (CLAUDE.md)."
        )
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"refusing --window test: pre-registration {p} does not exist.")
    text = p.read_text(encoding="utf-8")

    status = _STATUS.search(text)
    if not status or status.group(1).lower() != "accepted":
        raise SystemExit(
            f"refusing --window test: {p.name} status is "
            f"{status.group(1) if status else 'missing'!r}, not 'accepted'."
        )

    seeds = _SEEDS.search(text)
    if not seeds:
        raise SystemExit(
            f"refusing --window test: {p.name} carries no stage-1 seed count. ADR-012 clause 5 "
            "derives it from the validation pilot; record it as '**Stage-1 seeds:** <integer>' "
            "before any test-window run."
        )
    return int(seeds.group(1))


def require_clean_tree(allow_dirty: bool = False) -> None:
    """Exit on uncommitted code/config, unless explicitly overridden for a throwaway smoke run."""
    if git_dirty() and not allow_dirty:
        raise SystemExit(
            "refusing to run: uncommitted code/config, so these rows would log git_dirty=true "
            "and could not be cited. Commit first."
        )
