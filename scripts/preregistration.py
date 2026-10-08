"""The pre-registration guard shared by the DGF-1 runners (ADR-012 clause 8, ADR-016).

It lives in one place so the floor runner and the GNN runner cannot drift apart. Two refusals,
encoded rather than remembered:

* **the test window opens only through a mapped mode** (ADR-016). ``GATED_MODES`` maps each gated
  mode to the one document that may open it, and a mode name means one batch in every runner. The
  map is empty: Gate 1's ``gate`` mode is closed, and a later gated batch adds its own entry when its
  pre-registration is implemented. The document must also be accepted *and* carry its seed count —
  ADR-012 fixed the *rule* for that count and left the *number* to a validation pilot, so an accepted
  document whose count is not derived yet keeps 482-821 shut;
* **a dirty tree is refused** — rows must be reproducible from a commit, and dirty rows have forced
  full re-runs twice (ADR-002, and the Gate-0/Gate-1 batches). Outside a gated mode, ``--allow-dirty``
  overrides it for smoke runs; inside one, nothing does.

The seed count is **returned**, so a caller runs the number the document fixed rather than one
typed on the command line.
"""

from __future__ import annotations

import re
from pathlib import Path

from gbe.run.config import git_dirty

REPO_ROOT = Path(__file__).resolve().parents[1]

_STATUS = re.compile(r"^\*\*Status:\*\*\s*(\w+)", re.MULTILINE)
# ADR-012 clause 8 fixes this line's exact shape, so the guard is not left guessing a format.
_SEEDS = re.compile(r"^\*\*Stage-1 seeds:\*\*\s*(\d+)", re.MULTILINE)

# ADR-016 clause 1: each gated mode -> the one document, relative to the repo root, that may open the
# test window for it. Empty since ADR-016. A later gated batch (Gate 3's ablations, ADR-013 clause 5's
# matched-time batch) adds its own entry, mapped to its own accepted pre-registration, as part of
# implementing that ADR. Never ADR-012 or ADR-015, and never the key `gate` (ADR-016 clause 5).
GATED_MODES: dict[str, Path] = {}

# ADR-016 clause 3: modes closed for good, each with the reason its refusal gives.
CLOSED_MODES: dict[str, str] = {
    "gate": "closed by ADR-016 (decisions/ADR-016-close-gate1-test-window.md). Gate 1's one "
            "pre-registered test batch ran at 09c6e72 and its verdict is signed "
            "(gates/GATE-DGF1-1.md), so --preregistration is not consulted, ADR-012 included.",
}


def require_gated_preregistration(mode: str, path: str | Path | None, allow_dirty: bool) -> int:
    """Exit unless ``mode`` may open the test window with the document at ``path``; return its count.

    Refuses, in ADR-016's order: a closed or unmapped mode; a dirty tree, or ``allow_dirty`` set; a
    path whose resolved form is not the mapped document's; a status other than accepted; a document
    without its ``**Stage-1 seeds:**`` line.
    """
    if mode in CLOSED_MODES:
        raise SystemExit(f"refusing --mode {mode}: {CLOSED_MODES[mode]}")
    if mode not in GATED_MODES:
        raise SystemExit(
            f"refusing --mode {mode}: it is not a gated mode. The test window opens only through "
            "a mode mapped in scripts/preregistration.py to its own accepted pre-registration "
            "(ADR-016)."
        )

    # A path match is only as good as the file at that path, so the tree is checked here rather
    # than left to each runner's main (ADR-016 clause 2).
    if allow_dirty:
        raise SystemExit(f"refusing --mode {mode} with --allow-dirty: a gated batch runs from a "
                         "committed tree only (ADR-016 clause 2).")
    if git_dirty(REPO_ROOT):
        raise SystemExit(f"refusing --mode {mode}: uncommitted code/config, so the pre-registration "
                         "read here might not be the committed one (ADR-016 clause 2). Commit first.")

    mapped = GATED_MODES[mode]
    if path is None:
        raise SystemExit(f"refusing --mode {mode}: no --preregistration given; it must be {mapped}.")
    p = Path(path)
    if p.resolve() != (REPO_ROOT / mapped).resolve():
        raise SystemExit(f"refusing --mode {mode}: {p} is not the document mapped to this mode "
                         f"({mapped}).")
    if not p.is_file():
        raise SystemExit(f"refusing --mode {mode}: pre-registration {p} does not exist.")
    text = p.read_text(encoding="utf-8")

    status = _STATUS.search(text)
    if not status or status.group(1).lower() != "accepted":
        raise SystemExit(
            f"refusing --mode {mode}: {p.name} status is "
            f"{status.group(1) if status else 'missing'!r}, not 'accepted'."
        )

    seeds = _SEEDS.search(text)
    if not seeds:
        raise SystemExit(
            f"refusing --mode {mode}: {p.name} carries no stage-1 seed count. Record it as "
            "'**Stage-1 seeds:** <integer>' (ADR-012 clause 8's format) before any test-window run."
        )
    return int(seeds.group(1))


def require_clean_tree(allow_dirty: bool = False) -> None:
    """Exit on uncommitted code/config, unless explicitly overridden for a throwaway smoke run."""
    if git_dirty() and not allow_dirty:
        raise SystemExit(
            "refusing to run: uncommitted code/config, so these rows would log git_dirty=true "
            "and could not be cited. Commit first."
        )
