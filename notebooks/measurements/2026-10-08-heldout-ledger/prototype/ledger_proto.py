"""Prototype v4: the held-out ledger at the data's chokepoint (ADR-021 rounds 1-3; ADR-017 item 3).

What the adapter hands out by default carries no held-out label and nothing derived from one:
- ``splits()`` returns the validation window only;
- ``load()`` drops all three official masks (together they mark exactly the labelled users);
- ``load()`` hides the label of every user who appears after the validation window.

Held-out parts come only from accessors, which record the look first:
- ``open_test_window`` returns the window, with the true labels only at the labels or scores tier;
- ``open_official_track`` returns all labels and the three official masks, at the labels or scores
  tier only, for the official track (ADR-015).

``integrity_ok`` checks whole-snapshot totals against the published ones and returns pass or fail
only. It is not a look (ADR-021 clause 2).

A labels or scores look needs an accepted document, in HEAD and accepted on an earlier day, that
carries a structured line naming the set, the tier and the script. The script is the running
``__main__``, never a caller's argument. Each look carries a ``batch`` label, so two batches under
one document stay apart.

v4 changes, after round 3:
- the default load drops all three official masks;
- the official-track accessor replaces the official-test-mask accessor;
- the integrity check;
- the structured authority line, and ``via`` from ``__main__``;
- the batch column.
"""
from __future__ import annotations

import ast
import csv
import io
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

COLUMNS = ("timestamp_utc", "held_out_set", "tier", "what", "by", "via", "authority", "batch",
           "git_commit")
TIERS = ("structure", "labels", "scores")
# Each held-out set, with the sets it shares units with (ADR-015:234). Disclosure and the post-look
# rule cover the overlapping sets too.
HELD_OUT_SETS: dict[str, frozenset[str]] = {
    "dgf1-temporal-test-482-821": frozenset({"dgraph-official-test"}),
    "dgraph-official-test": frozenset({"dgf1-temporal-test-482-821"}),
}
LEDGER_REL = "experiments/heldout_ledger.csv"
_STATUS = re.compile(r"^\*\*Status:\*\*\s*(\w+)", re.MULTILINE)
_ACCEPTED_ON = re.compile(r"^\*\*Date:\*\*.*?accepted\W*(\d{4}-\d{2}-\d{2})", re.MULTILINE)
# The authorising line: **Held-out look:** <set> · <tier> · <script>
_LOOK_LINE = re.compile(r"^\*\*Held-out look:\*\*\s*(\S+)\s*·\s*(\w+)\s*·\s*(\S+)\s*$", re.MULTILINE)
# A number on its own (12, 1.48, 12%, 2,717) is a figure; a digit inside an identifier is not.
_FIGURE = re.compile(r"(?<![\w-])\d+(?:[.,]\d+)*%?(?![\w-])")
# Backfill rows record looks made before ADR-021; the implementation sets this to its acceptance.
BACKFILL_BEFORE = datetime(2026, 10, 9, tzinfo=timezone.utc)
HIDDEN = -1    # the label every user after the validation window carries until a window is opened
OFFICIAL_MASKS = ("train_mask", "val_mask", "test_mask")


class LedgerError(ValueError):
    pass


# ---- the running script ---------------------------------------------------------------------------

def current_script(repo_root: Path) -> str:
    """The repo-relative path of the running ``__main__``; a look is made by a committed script."""
    f = getattr(sys.modules.get("__main__"), "__file__", None)
    if not f:
        raise LedgerError("a look must be made by a script with a file, not an interactive session")
    p = Path(f).resolve()
    try:
        rel = p.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        raise LedgerError(f"{p} is not inside the repository") from None
    if not rel.endswith(".py"):
        raise LedgerError(f"{rel} is not a Python script")
    return rel


# ---- the authority ------------------------------------------------------------------------------

def check_authority(held_out_set: str, tier: str, authority: str, via: str, when: datetime,
                    repo_root: Path, incidental: bool) -> None:
    """Labels and scores need a document accepted, in HEAD, on an earlier day, naming this look."""
    if incidental:
        if not authority.startswith("none"):
            raise LedgerError("an incidental look records its authority as 'none: <why>'")
        return
    if tier == "structure":
        return
    head = subprocess.run(["git", "show", f"HEAD:{authority}"], cwd=repo_root, capture_output=True,
                          text=True, encoding="utf-8")
    if head.returncode != 0:
        raise LedgerError(f"a {tier} look needs an accepted document; {authority!r} is not in HEAD")
    text = head.stdout
    status = _STATUS.search(text)
    if not status or status.group(1).lower() != "accepted":
        raise LedgerError(f"a {tier} look needs an accepted document; {authority} is not accepted")
    on = _ACCEPTED_ON.search(text)
    if not on or date.fromisoformat(on.group(1)) >= when.date():
        raise LedgerError(f"{authority} was not accepted on an earlier day than this look")
    if via != current_script(repo_root):
        raise LedgerError(f"via {via!r} is not the running script")
    named = {(s, t, v) for s, t, v in _LOOK_LINE.findall(text)}
    allowed = {tier} | ({"scores"} if tier == "labels" else set())   # a scores look covers its labels
    if not any((held_out_set, t, via) in named for t in allowed):
        raise LedgerError(f"{authority} has no 'Held-out look:' line for {held_out_set} · {tier} · {via}")


# ---- the ledger ---------------------------------------------------------------------------------

def record_look(path: Path, *, held_out_set: str, tier: str, what: str, by: str, via: str,
                authority: str, batch: str, git_commit: str, repo_root: Path,
                incidental: bool = False, backfill: bool = False,
                now: datetime | None = None) -> dict:
    """Append one row and flush it to disk before returning. Never rewrites an existing row."""
    when = now or datetime.now(timezone.utc)
    if backfill and when >= BACKFILL_BEFORE:
        raise LedgerError("a backfill row records a look from before ADR-021, not after")
    if held_out_set not in HELD_OUT_SETS:
        raise LedgerError(f"unknown held-out set {held_out_set!r}")
    if tier not in TIERS:
        raise LedgerError(f"tier must be one of {TIERS}, not {tier!r}")
    for name, value in (("what", what), ("by", by), ("via", via), ("authority", authority),
                        ("batch", batch)):
        if not value.strip():
            raise LedgerError(f"{name} is empty")
    if _FIGURE.search(what):
        # The ledger names what was seen, never the value: it is committed to a public repo.
        raise LedgerError("'what' must describe the look, not carry a figure")
    if not backfill:      # a backfill row keeps the authority its look ran under; nothing else is skipped
        check_authority(held_out_set, tier, authority, via, when, repo_root, incidental)
    row = {"timestamp_utc": when.isoformat(timespec="seconds"), "held_out_set": held_out_set,
           "tier": tier, "what": what, "by": by, "via": via, "authority": authority,
           "batch": batch, "git_commit": git_commit}
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


def disclosure(rows: list[dict], held_out_set: str, batch: str) -> list[str]:
    """What was seen before a batch: earlier rows for its set and every set that overlaps it."""
    starts = [r["timestamp_utc"] for r in rows if r["batch"] == batch]
    if not starts:
        raise LedgerError(f"batch {batch!r} has no opener row in the ledger")
    before = min(starts)
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


def load(raw: dict, config: dict) -> dict:
    """The graph without any official mask, and with every later user's label hidden.

    The official masks go because together they mark exactly the labelled users: with node times,
    they would say which hidden users are labelled.
    """
    data = {k: v for k, v in raw.items() if k not in OFFICIAL_MASKS}
    data["y"] = [HIDDEN if t > config["val_max"] else y for y, t in zip(raw["y"], raw["node_time"])]
    return data


def integrity_ok(load_raw: Callable[[], dict], expected: dict[int, int]) -> bool:
    """Whole-snapshot label totals against the published ones: pass or fail, nothing else."""
    y = load_raw()["y"]
    return all(sum(1 for v in y if v == k) == n for k, n in expected.items())


def _open(ledger: Path, held_out_set: str, *, tier: str, authority: str, by: str, what: str,
          batch: str, repo_root: Path, git_commit: str, now: datetime | None) -> None:
    record_look(ledger, held_out_set=held_out_set, tier=tier, what=what, by=by,
                via=current_script(repo_root), authority=authority, batch=batch,
                git_commit=git_commit, repo_root=repo_root, now=now)


def open_test_window(config: dict, load_raw: Callable[[], dict], *, tier: str, authority: str,
                     by: str, what: str, batch: str, ledger: Path, repo_root: Path, git_commit: str,
                     now: datetime | None = None) -> tuple[Window, list | None]:
    """Record the look, then return the window, and its true labels only for a labels or scores look."""
    _open(ledger, "dgf1-temporal-test-482-821", tier=tier, authority=authority, by=by, what=what,
          batch=batch, repo_root=repo_root, git_commit=git_commit, now=now)
    window = Window(config["test_min"], config["test_max"])
    if tier == "structure":
        return window, None
    return window, list(load_raw()["y"])


def open_official_track(load_raw: Callable[[], dict], *, tier: str, authority: str, by: str,
                        what: str, batch: str, ledger: Path, repo_root: Path, git_commit: str,
                        now: datetime | None = None) -> dict:
    """Record the look, then return every label and the three official masks (ADR-015's track)."""
    if tier == "structure":
        raise LedgerError("the official masks are label-derived; open them at the labels or scores tier")
    _open(ledger, "dgraph-official-test", tier=tier, authority=authority, by=by, what=what,
          batch=batch, repo_root=repo_root, git_commit=git_commit, now=now)
    raw = load_raw()
    return {"y": list(raw["y"]), **{m: raw[m] for m in OFFICIAL_MASKS}}


# ---- the backstop lint over committed code --------------------------------------------------------

ACCESSOR_MODULES = ("adapters/dgf1/datasource_dgraph.py", "adapters/dgf1/eval.py",
                    "adapters/dgf1/heldout.py")
HAND_PATH = "scripts/heldout_ledger.py"
SCOPE = ("scripts/", "adapters/dgf1/", "notebooks/")
RAW_CALLS = {"DGraphFin": "PyG loader", "TemporalSplit": "hand-built window"}
OPENERS = {"open_test_window", "open_official_track"}


def _name(node: ast.AST) -> str:
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


def _mask_name(s: str) -> bool:
    return s.endswith("_mask") and s not in ("labelled_mask", "train_seed_mask", "window_target_mask")


def backstop(sources: dict[str, str]) -> list[str]:
    """Code that could reach held-out data without an accessor, or claims an incidental look.

    Parsed, not grepped: a pattern inside a string (a test fixture, this lint's own tables) is not a
    route to the data, and a call is.
    """
    out = []
    for name, src in sources.items():
        if (name in ACCESSOR_MODULES or not name.startswith(SCOPE)
                or Path(name).name.startswith("test_")      # tests use synthetic fixtures
                or re.match(r"notebooks/measurements/[^/]+/prototype/", name)):   # ADR evidence
            continue
        found, opens, reads_scores = set(), False, False
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call):
                fn = _name(node.func)
                if fn in RAW_CALLS:
                    found.add(RAW_CALLS[fn])
                opens |= fn in OPENERS
                reads_scores |= fn == "load_scores"
                if name != HAND_PATH and any(k.arg in ("incidental", "backfill") for k in node.keywords):
                    found.add("claims an incidental or backfill look")
            elif isinstance(node, ast.Attribute) and _mask_name(node.attr):
                found.add("an official mask")
            elif (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                  and isinstance(node.slice.value, str) and _mask_name(node.slice.value)):
                found.add("an official mask")
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
