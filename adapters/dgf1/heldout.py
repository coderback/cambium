"""The held-out ledger: every look at a held-out set is recorded before the data is read (ADR-021).

A labels or scores look needs an accepted document, in HEAD, whose latest change waited a night, with
a line naming the set, the tier, the script and the batch:

    **Held-out look:** <set> · <tier> · <script> · <batch>

The script is the running ``__main__`` and the time is the clock's, so a caller supplies neither.
The ledger's path and the repository root are module constants: no argument, environment variable or
command-line option redirects the ledger. Only ``scripts/heldout_ledger.py`` passes ``incidental``,
``backfill`` or ``now``.

The ledger names what was seen, never its value, because it is committed to a public repository.

The default paths (``load_dgraph``, ``dgf1_splits``) carry no held-out label and nothing derived from
one. The held-out parts come only from the accessors here, and each records the look before it reads
anything:

- ``open_test_window`` returns the test window, with the true labels at the labels or scores tier;
- ``open_official_track`` returns every label and the three official masks, for ADR-015's track;
- ``integrity_ok`` compares whole-snapshot totals with the published ones. It returns pass or fail
  only, and it is not a look (ADR-021 clause 2).
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch import Tensor

from adapters.dgf1.datasource_dgraph import BACKGROUND, FRAUD, NORMAL, _load_raw
from adapters.dgf1.eval import dgf1_window_bounds
from gbe.eval import TemporalSplit
from gbe.run.config import git_commit

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "experiments" / "heldout_ledger.csv"

COLUMNS = ("timestamp_utc", "held_out_set", "tier", "what", "by", "via", "authority", "batch", "git_commit")
TIERS = ("structure", "labels", "scores")
TEMPORAL_TEST, OFFICIAL_TEST = "dgf1-temporal-test-482-821", "dgraph-official-test"
# Each held-out set, with the sets it shares users with (ADR-015:234). Disclosure and the post-look
# rule cover the overlapping sets too (ADR-021 clause 1).
HELD_OUT_SETS: dict[str, frozenset[str]] = {
    TEMPORAL_TEST: frozenset({OFFICIAL_TEST}),
    OFFICIAL_TEST: frozenset({TEMPORAL_TEST}),
}
# The dataset's published whole-snapshot class totals (ADR-010:24).
PUBLISHED_TOTALS = {"normal": 1_210_092, "fraud": 15_509, "background": 2_474_949}
# The researcher's calendar: the wait is "not the same calendar day" there (ADR-021 clause 3).
LOCAL_TZ = ZoneInfo("Europe/London")
WAIT = timedelta(hours=12)
# A backfill row records a look from before ADR-021's acceptance.
BACKFILL_BEFORE = datetime(2026, 10, 8, 19, 37, tzinfo=LOCAL_TZ)

_STATUS = re.compile(r"^\*\*Status:\*\*\s*(\w+)", re.MULTILINE)
_STATUS_LINE = re.compile(r"^\*\*Status:\*\*.*$", re.MULTILINE)
_DATE_LINE = re.compile(r"^\*\*Date:\*\*.*$", re.MULTILINE)
_ACCEPTED_ON = re.compile(r"accepted\W*(\d{4}-\d{2}-\d{2})(?:\s*\((\d{2}:\d{2})\))?")
# On the Status line, every dated change counts: an amendment or an erratum restarts the wait.
_DATED = re.compile(r"(\d{4}-\d{2}-\d{2})(?:\s*\((\d{2}:\d{2})\))?")
_LOOK_LINE = re.compile(r"^\*\*Held-out look:\*\*\s*(\S+)\s*·\s*(\w+)\s*·\s*(\S+)\s*·\s*(\S+)\s*$", re.MULTILINE)
# A number on its own (12, 1.48, 12%, 2,717) is a figure; a digit inside a name is not.
_FIGURE = re.compile(r"(?<![\w-])\d+(?:[.,]\d+)*%?(?![\w-])")


class LedgerError(ValueError):
    """A look that may not be recorded, and so may not be made."""


def _require_known(held_out_set: str) -> None:
    if held_out_set not in HELD_OUT_SETS:
        raise LedgerError(f"unknown held-out set {held_out_set!r}")


def _clock() -> datetime:
    """The only source of a look's time, apart from the hand path's earlier one; tests replace it."""
    return datetime.now(timezone.utc)


def current_script() -> str:
    """The repo-relative path of the running ``__main__``; a look is made by a committed script."""
    f = getattr(sys.modules.get("__main__"), "__file__", None)
    if not f:
        raise LedgerError("a look must be made by a script with a file, not an interactive session")
    p = Path(f).resolve()
    try:
        rel = p.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        raise LedgerError(f"{p} is not inside the repository") from None
    if not rel.endswith(".py"):
        raise LedgerError(f"{rel} is not a Python script")
    return rel


def effective_acceptance(text: str) -> datetime | None:
    """The latest acceptance, amendment or erratum on the Status and Date lines, in the local zone.

    A date without a recorded time counts as the end of that day, so the wait is never shortened.
    """
    found = []
    for line_re, item_re in ((_DATE_LINE, _ACCEPTED_ON), (_STATUS_LINE, _DATED)):
        line = line_re.search(text)
        for d, t in item_re.findall(line.group(0) if line else ""):
            found.append(datetime.fromisoformat(f"{d}T{t or '23:59'}").replace(tzinfo=LOCAL_TZ))
    return max(found) if found else None


def check_authority(held_out_set: str, tier: str, authority: str, via: str, batch: str, when: datetime,
                    incidental: bool = False) -> None:
    """Labels and scores need a document accepted, in HEAD, a night earlier, naming this look."""
    if incidental:
        if not authority.startswith("none"):
            raise LedgerError("an incidental look records its authority as 'none: <why>'")
        return
    if tier == "structure":
        return
    head = subprocess.run(["git", "show", f"HEAD:{authority}"], cwd=REPO_ROOT, capture_output=True,
                          text=True, encoding="utf-8")
    if head.returncode != 0:
        raise LedgerError(f"a {tier} look needs an accepted document; {authority!r} is not in HEAD")
    text = head.stdout
    status = _STATUS.search(text)
    if not status or status.group(1).lower() != "accepted":
        raise LedgerError(f"a {tier} look needs an accepted document; {authority} is not accepted")
    on = effective_acceptance(text)
    local = when.astimezone(LOCAL_TZ)
    if on is None or on.date() >= local.date() or local - on < WAIT:
        raise LedgerError(f"{authority} was not accepted a night before this look")
    named = set(_LOOK_LINE.findall(text))
    allowed = {tier} | ({"scores"} if tier == "labels" else set())   # a scores look covers its labels
    if not any((held_out_set, t, via, batch) in named for t in allowed):
        raise LedgerError(f"{authority} has no 'Held-out look:' line for "
                          f"{held_out_set} · {tier} · {via} · {batch}")


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    with LEDGER.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def record_look(held_out_set: str, tier: str, what: str, by: str, authority: str, batch: str, *,
                incidental: bool = False, backfill: bool = False, now: datetime | None = None,
                via: str | None = None, commit: str | None = None) -> dict:
    """Append one row and sync it to disk before returning. Never rewrites an existing row.

    Refuses, and writes nothing, on anything but a look it may record. ``incidental``, ``now``,
    ``via`` and ``commit`` belong to the hand path; the last two only to a backfill row, which names
    its source and the commit its look ran at.
    """
    clock = _clock()
    if now is not None and now > clock:
        raise LedgerError("a look's time may not be later than the clock")
    when = now or clock
    if backfill and when >= BACKFILL_BEFORE:
        raise LedgerError("a backfill row records a look from before ADR-021, not after")
    if not backfill and (via is not None or commit is not None):
        raise LedgerError("only a backfill row names its own via or commit")
    if backfill and not via:
        raise LedgerError("a backfill row names its source in via")
    via = via if backfill else current_script()
    _require_known(held_out_set)
    if tier not in TIERS:
        raise LedgerError(f"tier must be one of {TIERS}, not {tier!r}")
    for name, value in (("what", what), ("by", by), ("authority", authority), ("batch", batch)):
        if not value.strip():
            raise LedgerError(f"{name} is empty")
    if _FIGURE.search(what):
        raise LedgerError("'what' must describe the look, not carry a figure")
    if not backfill:      # a backfill row keeps the authority its look ran under; nothing else is skipped
        check_authority(held_out_set, tier, authority, via, batch, when, incidental)
    if any(r["batch"] == batch and r["authority"] != authority for r in read_ledger()):
        raise LedgerError(f"batch {batch!r} already belongs to another authority")
    row = {"timestamp_utc": when.astimezone(timezone.utc).isoformat(timespec="seconds"),
           "held_out_set": held_out_set, "tier": tier, "what": what, "by": by, "via": via,
           "authority": authority, "batch": batch, "git_commit": commit or git_commit(REPO_ROOT)}
    new_file = not LEDGER.exists() or LEDGER.stat().st_size == 0
    with LEDGER.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        if new_file:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    return row


def seen_so_far(rows: list[dict], held_out_set: str) -> list[str]:
    """Every recorded look at a set and the sets that overlap it, in time order.

    A pre-registration or reported-only document discloses these before its own batch has opened
    (ADR-021 clause 4), so this needs no batch.
    """
    _require_known(held_out_set)
    sets = {held_out_set} | HELD_OUT_SETS[held_out_set]
    seen = [r for r in rows if r["held_out_set"] in sets]
    seen.sort(key=lambda r: r["timestamp_utc"])
    return [f"{r['timestamp_utc']} · {r['held_out_set']} · {r['tier']} · {r['what']} "
            f"({r['by']}, via {r['via']}; authority: {r['authority']})" for r in seen]


def disclosure(rows: list[dict], held_out_set: str, batch: str) -> list[str]:
    """What was seen before a batch: earlier rows for its set and every set that overlaps it."""
    starts = [r["timestamp_utc"] for r in rows if r["batch"] == batch]
    if not starts:
        raise LedgerError(f"batch {batch!r} has no opener row in the ledger")
    before = min(starts)
    return seen_so_far([r for r in rows if r["timestamp_utc"] < before], held_out_set)


def open_test_window(*, tier: str, what: str, by: str, authority: str, batch: str
                     ) -> tuple[TemporalSplit, Tensor | None]:
    """Record the look, then return the test window, plus the true labels unless it is a structure look.

    ``adapters.dgf1.datasource_dgraph.with_labels`` puts the labels back for scoring.
    """
    record_look(TEMPORAL_TEST, tier, what, by, authority, batch)
    s = dgf1_window_bounds()
    window = TemporalSplit(train_max=int(s["train_max"]), test_min=int(s["test_min"]),
                           test_max=int(s["test_max"]), time_attr=str(s.get("time_attr", "node_time")))
    return window, (None if tier == "structure" else _load_raw().y)


def open_official_track(*, tier: str, what: str, by: str, authority: str, batch: str) -> dict[str, Tensor]:
    """Record the look, then return every label and the three official masks (ADR-015's track)."""
    if tier == "structure":
        raise LedgerError("the official masks are label-derived; open them at the labels or scores tier")
    record_look(OFFICIAL_TEST, tier, what, by, authority, batch)
    raw = _load_raw()
    return {"y": raw.y, "official_train_mask": raw.train_mask, "official_val_mask": raw.val_mask,
            "official_test_mask": raw.test_mask}


def integrity_ok() -> bool:
    """Whole-snapshot label totals against the published ones: pass or fail, and nothing else."""
    y = _load_raw().y
    return (int((y == NORMAL).sum()) == PUBLISHED_TOTALS["normal"]
            and int((y == FRAUD).sum()) == PUBLISHED_TOTALS["fraud"]
            and int(torch.isin(y, torch.tensor(BACKGROUND)).sum()) == PUBLISHED_TOTALS["background"])
