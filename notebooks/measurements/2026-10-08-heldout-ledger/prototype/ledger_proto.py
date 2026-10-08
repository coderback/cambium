"""Prototype v2: the held-out ledger at the data's chokepoint (ADR-021 round 1; ADR-017 item 3).

The researcher's decision of 2026-10-08, after round 1: a look is recorded where the data is handed
out, not by each caller. The DGF-1 adapter stops returning the held-out parts by default:
- ``splits()`` returns the validation window only;
- ``load()`` drops the official test mask.

Each held-out part comes only from an ``open_*`` accessor. The accessor records the look (with its
tier and authority) and only then returns the part. The guard, ADR-015's runner, assemblers and
any future script all go through it. A lint over committed code is the backstop for anything that
reaches the raw data another way.

A labels or scores look needs an accepted document that names the look first. That means it is
accepted, its acceptance date is on or before the look, and it names the script making the look.
"""
from __future__ import annotations

import ast
import csv
import io
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

COLUMNS = ("timestamp_utc", "held_out_set", "tier", "what", "by", "via", "authority", "git_commit")
TIERS = ("structure", "labels", "scores")
# Each held-out set, with the sets it shares units with (ADR-015:234: the test window shares users
# with the official test mask). Disclosure and the post-look rule cover the overlapping sets too.
HELD_OUT_SETS: dict[str, frozenset[str]] = {
    "dgf1-temporal-test-482-821": frozenset({"dgraph-official-test"}),
    "dgraph-official-test": frozenset({"dgf1-temporal-test-482-821"}),
}
LEDGER_REL = "experiments/heldout_ledger.csv"
_STATUS = re.compile(r"^\*\*Status:\*\*\s*(\w+)", re.MULTILINE)
_ACCEPTED_ON = re.compile(r"^\*\*Date:\*\*.*?accepted\W*(\d{4}-\d{2}-\d{2})", re.MULTILINE)
# A number on its own (12, 1.48, 12%, 2,717) is a figure; a digit inside an identifier is not.
_FIGURE = re.compile(r"(?<![\w-])\d+(?:[.,]\d+)*%?(?![\w-])")


class LedgerError(ValueError):
    pass


# ---- the authority ------------------------------------------------------------------------------

def check_authority(tier: str, authority: str, via: str, when: datetime, repo_root: Path,
                    incidental: bool) -> None:
    """Labels and scores need an accepted document, accepted by then, that names ``via``."""
    if incidental:
        if not authority.startswith("none"):
            raise LedgerError("an incidental look records its authority as 'none: <why>'")
        return
    if tier == "structure":
        return
    doc = repo_root / authority
    if not doc.is_file():
        raise LedgerError(f"a {tier} look needs an accepted document; {authority!r} is not a file")
    text = doc.read_text(encoding="utf-8")
    status = _STATUS.search(text)
    if not status or status.group(1).lower() != "accepted":
        raise LedgerError(f"a {tier} look needs an accepted document; {authority} is not accepted")
    on = _ACCEPTED_ON.search(text)
    if not on or date.fromisoformat(on.group(1)) > when.date():
        raise LedgerError(f"{authority} was not accepted before this look")
    if via not in text:
        raise LedgerError(f"{authority} does not name {via}, so it did not name this look first")


# ---- the ledger ---------------------------------------------------------------------------------

def record_look(path: Path, *, held_out_set: str, tier: str, what: str, by: str, via: str,
                authority: str, git_commit: str, repo_root: Path, incidental: bool = False,
                now: datetime | None = None) -> dict:
    """Append one row and flush it to disk before returning. Never rewrites an existing row."""
    when = now or datetime.now(timezone.utc)
    if held_out_set not in HELD_OUT_SETS:
        raise LedgerError(f"unknown held-out set {held_out_set!r}")
    if tier not in TIERS:
        raise LedgerError(f"tier must be one of {TIERS}, not {tier!r}")
    for name, value in (("what", what), ("by", by), ("via", via), ("authority", authority)):
        if not value.strip():
            raise LedgerError(f"{name} is empty")
    if _FIGURE.search(what):
        # The ledger names what was seen, never the value: it is committed to a public repo.
        raise LedgerError("'what' must describe the look, not carry a figure")
    check_authority(tier, authority, via, when, repo_root, incidental)
    row = {"timestamp_utc": when.isoformat(timespec="seconds"), "held_out_set": held_out_set,
           "tier": tier, "what": what, "by": by, "via": via, "authority": authority,
           "git_commit": git_commit}
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


def ledger_change_is_append(repo_root: Path, rel: str = LEDGER_REL) -> bool:
    """What git_dirty may ignore: an append to HEAD's ledger, and nothing else."""
    head = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=repo_root, capture_output=True,
                          text=True, encoding="utf-8")
    committed = head.stdout if head.returncode == 0 else ""
    p = repo_root / rel
    working = p.read_text(encoding="utf-8") if p.exists() else ""
    return is_append_only(committed, working)


def disclosure(rows: list[dict], held_out_set: str, before: str) -> list[str]:
    """What was seen before a batch: earlier rows for its set and every set that overlaps it."""
    sets = {held_out_set} | HELD_OUT_SETS[held_out_set]
    seen = [r for r in rows if r["held_out_set"] in sets and r["timestamp_utc"] < before]
    seen.sort(key=lambda r: r["timestamp_utc"])
    return [f"{r['timestamp_utc']} · {r['held_out_set']} · {r['tier']} · {r['what']} "
            f"({r['by']}, via {r['via']}; authority: {r['authority']})" for r in seen]


# ---- the chokepoint: the adapter hands out held-out parts only through these --------------------

@dataclass(frozen=True)
class Window:
    lo: int
    hi: int


def splits(config: dict) -> dict[str, Window]:
    """The windows anyone may have: validation only. The test window comes from open_test_window."""
    return {"val": Window(config["val_min"], config["val_max"])}


def load(raw: dict) -> dict:
    """The graph without the official test mask; open_official_test_mask hands that out."""
    return {k: v for k, v in raw.items() if k != "test_mask"}


def open_test_window(config: dict, *, tier: str, authority: str, by: str, via: str, ledger: Path,
                     repo_root: Path, git_commit: str, what: str, now: datetime | None = None) -> Window:
    record_look(ledger, held_out_set="dgf1-temporal-test-482-821", tier=tier, what=what, by=by,
                via=via, authority=authority, git_commit=git_commit, repo_root=repo_root, now=now)
    return Window(config["test_min"], config["test_max"])


def open_official_test_mask(load_raw: Callable[[], dict], *, tier: str, authority: str, by: str,
                            via: str, ledger: Path, repo_root: Path, git_commit: str, what: str,
                            now: datetime | None = None):
    record_look(ledger, held_out_set="dgraph-official-test", tier=tier, what=what, by=by, via=via,
                authority=authority, git_commit=git_commit, repo_root=repo_root, now=now)
    return load_raw()["test_mask"]


# ---- the backstop lint over committed code --------------------------------------------------------

ACCESSOR_MODULES = ("adapters/dgf1/datasource_dgraph.py", "adapters/dgf1/eval.py")
HAND_PATH = "scripts/heldout_ledger.py"
SCOPE = ("scripts/", "adapters/dgf1/", "notebooks/")
RAW_CALLS = {"DGraphFin": "PyG loader", "TemporalSplit": "hand-built window"}
OPENERS = {"open_test_window", "open_official_test_mask"}


def _name(node: ast.AST) -> str:
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


def backstop(sources: dict[str, str]) -> list[str]:
    """Code that could reach held-out data without an accessor, or claims an incidental look.

    Parsed, not grepped: a pattern inside a string (a test fixture, this lint's own tables) is not a
    route to the data, and a call is.
    """
    out = []
    for name, src in sources.items():
        if (name in ACCESSOR_MODULES or not name.startswith(SCOPE)
                or Path(name).name.startswith("test_")      # tests use synthetic fixtures
                or "/prototype/" in name):                  # ADR evidence, not a route to data
            continue
        found, opens, reads_scores = set(), False, False
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call):
                fn = _name(node.func)
                if fn in RAW_CALLS:
                    found.add(RAW_CALLS[fn])
                opens |= fn in OPENERS
                reads_scores |= fn == "load_scores"
                if name != HAND_PATH and any(k.arg == "incidental" for k in node.keywords):
                    found.add("claims an incidental look")
            elif isinstance(node, ast.Attribute) and node.attr.endswith("test_mask"):
                found.add("official test mask")
            elif (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                  and isinstance(node.slice.value, str) and node.slice.value.endswith("test_mask")):
                found.add("official test mask")
        if reads_scores and not opens:
            found.add("score files without an accessor")
        out += [f"{name}: {f}" for f in sorted(found)]
    return sorted(out)


def csv_text(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
