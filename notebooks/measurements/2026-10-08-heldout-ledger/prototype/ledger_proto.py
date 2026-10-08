"""Prototype: the held-out ledger (plan row 5), written before its ADR specifies it (ADR-017 item 3).

One append-only CSV records every look at a held-out set. Three writers share one function:
- the pre-registration guard, when a gated or reported-only batch opens a held-out set;
- every committed script that reads a held-out set outside the guard;
- a person, by hand, for a look a script cannot record (a subagent's, or an incidental one).

A look is recorded *before* the held-out data is read, so a crash mid-look still leaves its row.

The researcher's decision of 2026-10-08: a labels or scores look needs an accepted document that
names it in advance. So ``record_look`` refuses those tiers unless ``authority`` is an accepted
document. A look nobody authorised (an agent's slip, an incidental sighting) is still recorded, as
``incidental`` with an authority beginning "none", and only the hand path may do that.
"""
from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = ("timestamp_utc", "held_out_set", "tier", "what", "by", "via", "authority", "git_commit")
TIERS = ("structure", "labels", "scores")
HELD_OUT_SETS = ("dgf1-temporal-test-482-821", "dgraph-official-test")
_STATUS = re.compile(r"^\*\*Status:\*\*\s*(\w+)", re.MULTILINE)


class LedgerError(ValueError):
    pass


def check_authority(tier: str, authority: str, repo_root: Path, incidental: bool) -> None:
    """Labels and scores need an accepted document; an incidental look says it had none."""
    if incidental:
        if not authority.startswith("none"):
            raise LedgerError("an incidental look records its authority as 'none: <why>'")
        return
    if tier == "structure":
        return
    doc = repo_root / authority
    if not doc.is_file():
        raise LedgerError(f"a {tier} look needs an accepted document; {authority!r} is not a file")
    status = _STATUS.search(doc.read_text(encoding="utf-8"))
    if not status or status.group(1).lower() != "accepted":
        raise LedgerError(f"a {tier} look needs an accepted document; {authority} is not accepted")


def record_look(path: Path, *, held_out_set: str, tier: str, what: str, by: str, via: str,
                authority: str, git_commit: str, repo_root: Path, incidental: bool = False,
                now: datetime | None = None) -> dict:
    """Append one row and flush it to disk before returning. Never rewrites an existing row."""
    if held_out_set not in HELD_OUT_SETS:
        raise LedgerError(f"unknown held-out set {held_out_set!r}")
    if tier not in TIERS:
        raise LedgerError(f"tier must be one of {TIERS}, not {tier!r}")
    for name, value in (("what", what), ("by", by), ("via", via), ("authority", authority)):
        if not value.strip():
            raise LedgerError(f"{name} is empty")
    if re.search(r"\d\.\d|\d{3,}", what):
        # The ledger names what was seen, never the value: it is committed to a public repo.
        raise LedgerError("'what' must describe the look, not carry a figure")
    check_authority(tier, authority, repo_root, incidental)
    row = {"timestamp_utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
           "held_out_set": held_out_set, "tier": tier, "what": what, "by": by, "via": via,
           "authority": authority, "git_commit": git_commit}
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    return row


def read_ledger(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_append_only(committed: str, working: str) -> bool:
    """The working ledger keeps every committed byte, in order, and only adds after them."""
    return working.startswith(committed)


def disclosure(rows: list[dict], held_out_set: str, before: str) -> list[str]:
    """The gate file's 'seen before this batch' list, generated from the ledger."""
    seen = [r for r in rows if r["held_out_set"] == held_out_set and r["timestamp_utc"] < before]
    seen.sort(key=lambda r: r["timestamp_utc"])
    return [f"{r['timestamp_utc']} · {r['tier']} · {r['what']} ({r['by']}, via {r['via']}; "
            f"authority: {r['authority']})" for r in seen]


# ---- the guard, reduced to what the ledger needs ------------------------------------------------

@dataclass
class GatedMode:
    document: str
    held_out_set: str


def open_held_out(mode: str, modes: dict[str, GatedMode], ledger: Path, git_commit: str,
                  repo_root: Path, by: str = "researcher") -> GatedMode:
    """The guard's last step: after every refusal check has passed, record the look, then return."""
    if mode not in modes:
        raise SystemExit(f"refusing --mode {mode}: not a mapped mode")
    m = modes[mode]
    record_look(ledger, held_out_set=m.held_out_set, tier="scores",
                what=f"gated batch, mode {mode}: model scores on the set", by=by,
                via="scripts/preregistration.py", authority=m.document, git_commit=git_commit,
                repo_root=repo_root)
    return m


# ---- the lint: every held-out reader outside the guard records its look -------------------------

LOADS = re.compile(r"\b(load_dgraph|prepare_dgraph)\b|\bDGraphFin\(")
NAMES_HELD_OUT = re.compile(r"""splits\[["']test["']\]|\btest_mask\b|_window\(\s*["']test["']""")
READS_GATE_SCORES = re.compile(r"\bload_scores\(")
RECORDS = re.compile(r"\brecord_look\(|\brequire_gated_preregistration\(")
INCIDENTAL = re.compile(r"\bincidental\s*=\s*True\b")
HAND_PATH = "scripts/heldout_ledger.py"


def unrecorded_readers(sources: dict[str, str]) -> list[str]:
    """Scripts that can read a held-out set but neither record a look nor pass the guard."""
    out = []
    for name, src in sources.items():
        reads = (LOADS.search(src) and NAMES_HELD_OUT.search(src)) or READS_GATE_SCORES.search(src)
        if reads and not RECORDS.search(src):
            out.append(name)
    return sorted(out)


def scripts_claiming_incidental(sources: dict[str, str]) -> list[str]:
    """Only the hand path may record an incidental look; a script's look is never incidental."""
    return sorted(n for n, src in sources.items() if n != HAND_PATH and INCIDENTAL.search(src))


def csv_text(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
