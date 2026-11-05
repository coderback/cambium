"""Prototype: the go line, the pilot's attempts, the test batch's outcomes and what a decline prints
(ADR-023 clauses 3, 5 and 7).

- **The go line.** The runner opens the test window only if the ADR, in HEAD, carries exactly one line
  ``**Go/no-go:** go · <date> · <run ids>``, and the rule, recomputed from the clean attempt those run ids
  name, says go. Prose that quotes the line must not count, because the ADR's look lines stay valid after a
  no-go, and a look cannot be undone. The check comes before the opener, so a refused run writes no ledger row.
- **Attempts.** The registry is append-only, so a retried pilot's rows sit beside the failed one's. Each
  attempt has its own tags. Its status is read once, from its first ten pilot rows and first two
  reproduction rows in registry order, so a row added later cannot change it. The first clean attempt is
  final: the rule reads it and no other, and no pilot starts after it.
- **The budget.** A void attempt's rows are not "the pilot's rows" in ADR-022:69-70, so running out of
  budget can still decline. Ten clean rows are, so a reproduction is then owed before any decline, and it
  cannot be skipped to reach one.
- **The batch.** A run that errored with no metric runs again on its commit. A completed row never does.
  Anything else ends the batch as ``refused: broken batch``.

Reads no data and computes no model metric: its inputs are values already in registry rows.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from gate_stats import pilot_arm  # noqa: E402
from go_no_go_proto import METRICS, PILOT_SEEDS, BrokenPilot, RuleInputs, decide, rule_inputs  # noqa: E402

MODEL, VALIDATION = "dgf1", "370-481"
DGF1_ARM, FLOOR_ARM = "dgf1-parity-matched", "xgboost-parity-matched"
ARMS = (DGF1_ARM, FLOOR_ARM)
REPRODUCED_KEYS = (*METRICS, "scores_sha256")
PILOT_ROWS = len(ARMS) * len(PILOT_SEEDS)
REPRO_ROWS = len(ARMS)
BATCH_TAG, BATCH_REPRO_TAG, EXPLORATORY_REPRO_TAG = "matched_time", "matched_batch_repro", "matched_exploratory_repro"
CODE_PATHS = ("adapters/", "gbe/", "scripts/", "configs/")

_GO_LINE_START = re.compile(r"^\*\*Go/no-go:\*\*.*$", re.MULTILINE)
_GO = re.compile(r"\*\*Go/no-go:\*\* go · \d{4}-\d{2}-\d{2} · \S")
_TAG = re.compile(r"^matched_pilot_(\d+)$")


class NoGoLine(ValueError):
    """The test window stays shut."""


class ReproductionFailed(BrokenPilot):
    """Seed 0's second run ran and did not match: no rule, and the attempt is void."""


class ReproductionOwed(Exception):
    """Ten clean rows exist and seed 0's second run does not. Nothing may end the attempt first."""


class AttemptRefused(ValueError):
    """The runner will not write this row."""


class BrokenBatch(ValueError):
    """The test batch cannot be read. The report says ``refused: broken batch``."""


# ------------------------------------------------------------------ the go line


def go_recorded(adr_text: str) -> bool:
    """Exactly one line opens with the go/no-go field, and it says go with a date and run ids."""
    lines = _GO_LINE_START.findall(adr_text)
    return len(lines) == 1 and bool(_GO.match(lines[0]))


def text_at_head(repo: Path, relative_path: str) -> str:
    """The committed text. A go line only in the working tree must not open anything (ADR-021 I3's rule)."""
    shown = subprocess.run(["git", "show", f"HEAD:{relative_path}"], cwd=repo, capture_output=True, encoding="utf-8")
    if shown.returncode != 0:
        raise NoGoLine(f"{relative_path} is not in HEAD")
    return shown.stdout


def go_earned(adr_text: str, rows: Sequence[Mapping[str, str]], delta_gated: Mapping[str, float]) -> bool:
    """The line is there, it names the first clean attempt's rows, and the rule on those rows says go."""
    k = first_clean(rows)
    if not go_recorded(adr_text) or k is None:
        return False
    named = set(_GO_LINE_START.findall(adr_text)[0].split(" · ", 2)[2].split())
    used = {r["run_id"] for r in (*pilot_rows(rows, k), *repro_rows(rows, k))}
    return named == used and bool(decide(clean_attempt(rows, k), delta_gated)["go"])


def open_for_batch(adr_text: str, rows: Sequence[Mapping[str, str]], delta_gated: Mapping[str, float], head: str,
                   opener: Callable[[], Any]) -> Any:
    """Refuse before the opener runs, so a refused run records no look. Every test-window path comes through here."""
    if not go_earned(adr_text, rows, delta_gated):
        raise NoGoLine("no earned '**Go/no-go:** go' line opens a line of this ADR; the test window stays shut")
    check_batch_reproduced(rows, first_clean(rows), head)
    return opener()


# ------------------------------------------------------------------ one attempt's rows


def pilot_tag(k: int) -> str:
    return f"matched_pilot_{k}"


def repro_tag(k: int) -> str:
    return f"matched_repro_{k}"


def _metrics(row: Mapping[str, str]) -> dict:
    return json.loads(row["metrics_json"])


def _clean_row(row: Mapping[str, str]) -> bool:
    return row["git_dirty"] == "false" and _metrics(row).get("deterministic") is True and "ERRORED" not in row["notes"]


def tagged(rows: Sequence[Mapping[str, str]], tag: str) -> list[Mapping[str, str]]:
    return [r for r in rows if _metrics(r).get("experiment") == tag]


def pilot_rows(rows: Sequence[Mapping[str, str]], k: int) -> list[Mapping[str, str]]:
    """The attempt's pilot rows. Only the first ten count, so a row added later cannot change its status."""
    return tagged(rows, pilot_tag(k))[:PILOT_ROWS]


def repro_rows(rows: Sequence[Mapping[str, str]], k: int) -> list[Mapping[str, str]]:
    return tagged(rows, repro_tag(k))[:REPRO_ROWS]


def attempts_in(rows: Sequence[Mapping[str, str]]) -> list[int]:
    found = {int(m.group(1)) for r in rows if (m := _TAG.match(str(_metrics(r).get("experiment"))))}
    return sorted(found)


def attempt_values(rows: Sequence[Mapping[str, str]], k: int, arm: str) -> dict[str, dict[int, float]]:
    """One arm of one attempt, by metric. A reproduction row and another attempt's rows are never read."""
    own = pilot_rows(rows, k)
    try:
        return {m: pilot_arm(own, model=MODEL, arm=arm, metric=m, window=VALIDATION) for m in METRICS}
    except (ValueError, KeyError) as why:
        raise BrokenPilot(f"attempt {k}, arm {arm}: {why}") from why


def _one_commit(rows: Sequence[Mapping[str, str]], k: int) -> None:
    """Both arms and the reproduction ran on one commit. ``pilot_arm`` only checks one arm at a time."""
    commits = {r["git_commit"] for r in (*pilot_rows(rows, k), *repro_rows(rows, k))}
    if len(commits) != 1:
        raise BrokenPilot(f"attempt {k} spans commits {sorted(commits)}")


def _matches(first: Mapping, second: Mapping[str, str], what: str) -> str | None:
    """Why a reproduction row is no match for the row it repeats, or None."""
    if not _clean_row(second):
        return f"{what}: {second['run_id']} is not a clean, deterministic row"
    for key in REPRODUCED_KEYS:
        if first.get(key) is None or first.get(key) != _metrics(second).get(key):
            return f"{what}: {key} did not reproduce"
    return None


def _check_reproduced(rows: Sequence[Mapping[str, str]], k: int) -> None:
    pilot, repro = pilot_rows(rows, k), repro_rows(rows, k)
    if any(r["seed"] != "0" for r in repro):
        raise ReproductionFailed(f"attempt {k}: a reproduction row is not seed 0")
    owed = []
    for arm in ARMS:
        first = [_metrics(r) for r in pilot if _metrics(r).get("arm") == arm and r["seed"] == "0"]
        second = [r for r in repro if _metrics(r).get("arm") == arm]
        if len(first) != 1 or len(second) > 1:
            raise ReproductionFailed(f"attempt {k}, arm {arm}: seed 0 needs one pilot row and one reproduction row")
        if not second:
            owed.append(arm)
            continue
        why = _matches(first[0], second[0], f"attempt {k}, arm {arm}")
        if why:
            raise ReproductionFailed(why)
    if owed:
        raise ReproductionOwed(f"attempt {k}: all ten rows are clean, and seed 0's reproduction is owed on {owed}")


def clean_attempt(rows: Sequence[Mapping[str, str]], k: int) -> RuleInputs:
    """What the rule may see of attempt ``k``, or an exception that says why it may not see anything yet."""
    dgf1, floor = attempt_values(rows, k, DGF1_ARM), attempt_values(rows, k, FLOOR_ARM)
    inputs = rule_inputs(dgf1, floor)
    _one_commit(rows, k)
    _check_reproduced(rows, k)
    return inputs


def status(rows: Sequence[Mapping[str, str]], k: int) -> str:
    """``clean``, ``owed`` (ten clean rows, reproduction still to run) or ``void``."""
    try:
        clean_attempt(rows, k)
    except ReproductionOwed:
        return "owed"
    except BrokenPilot:
        return "void"
    return "clean"


def first_clean(rows: Sequence[Mapping[str, str]]) -> int | None:
    """The attempt the rule reads. It takes no attempt number, so no caller can choose a later one."""
    return next((k for k in attempts_in(rows) if status(rows, k) == "clean"), None)


# ------------------------------------------------------------------ what the runner will write


def next_attempt(rows: Sequence[Mapping[str, str]], head: str, root_causes: Mapping[int, str],
                 environmental: Iterable[int] = ()) -> int:
    """The attempt a new pilot runs as, or a refusal. ADR-013:280's "never retry" holds across attempts."""
    found = attempts_in(rows)
    live = [k for k in found if status(rows, k) != "void"]
    if live:
        raise AttemptRefused(f"attempt {live[0]} is {status(rows, live[0])}, so no further pilot may start")
    if not found:
        return 1
    missing = [k for k in found if k not in root_causes]
    if missing:
        raise AttemptRefused(f"attempts {missing} have no recorded root cause")
    last = found[-1]
    if head in {r["git_commit"] for r in pilot_rows(rows, last)} and last not in set(environmental):
        raise AttemptRefused(f"attempt {last} ran on {head}: never a retry, unless its cause is environmental")
    return last + 1


def may_write_repro(rows: Sequence[Mapping[str, str]], k: int, head: str) -> None:
    """A reproduction runs only while owed, and only on the attempt's own commit."""
    if status(rows, k) != "owed":
        raise AttemptRefused(f"attempt {k} is {status(rows, k)}, and owes no reproduction")
    if head != pilot_rows(rows, k)[0]["git_commit"]:
        raise AttemptRefused(f"attempt {k}'s reproduction must run on its own commit, not {head}")


# ------------------------------------------------------------------ the budget, and what a decline prints


def budget_can_decline(rows: Sequence[Mapping[str, str]]) -> bool:
    """Only when every attempt is void. A clean attempt locks the budget, and so does a reproduction owed."""
    return all(status(rows, k) == "void" for k in attempts_in(rows))


def _gaps(rows: Sequence[Mapping[str, str]], k: int) -> str:
    """Each metric's gap over whatever rows the attempt has. These are only ever printed, never ruled on."""
    parts = []
    for m in METRICS:
        means = []
        for arm in ARMS:
            values = [_metrics(r)[m] for r in pilot_rows(rows, k) if _metrics(r).get("arm") == arm and m in _metrics(r)]
            means.append(sum(values) / len(values) if values else None)
        parts.append(f"{m} gap {means[0] - means[1]:+.4f}" if None not in means else f"{m} gap n/a")
    return ", ".join(parts)


def attempt_lines(rows: Sequence[Mapping[str, str]], delta_gated: Mapping[str, float],
                  root_causes: Mapping[int, str]) -> list[str]:
    """Every attempt, for a decline and for the report: its gaps, then the rule's result or why it was not computed.

    A reproduction still owed raises: it runs first, so no line can say "failed" of a run that never happened.
    """
    lines = []
    for k in attempts_in(rows):
        try:
            result = decide(clean_attempt(rows, k), delta_gated)
            lines.append(f"attempt {k}: {_gaps(rows, k)}; rule: {'go' if result['go'] else 'no-go'}")
        except ReproductionFailed as why:
            lines.append(f"attempt {k} (void): {_gaps(rows, k)}; rule not computed: reproduction failed; "
                         f"{why}; root cause: {root_causes.get(k, 'not found')}")
        except BrokenPilot as why:
            lines.append(f"attempt {k} (void): {_gaps(rows, k)}; rule not computed: broken pilot; "
                         f"{why}; root cause: {root_causes.get(k, 'not found')}")
    return lines or ["no pilot rows exist; rule not computed"]


# ------------------------------------------------------------------ the test batch


def check_batch_reproduced(rows: Sequence[Mapping[str, str]], k: int, head: str) -> None:
    """Before the first look: seed 0 of each arm, on validation, at the batch's commit, repeats the clean attempt."""
    pilot, own = pilot_rows(rows, k), tagged(rows, BATCH_REPRO_TAG)
    for arm in ARMS:
        first = [_metrics(r) for r in pilot if _metrics(r).get("arm") == arm and r["seed"] == "0"]
        second = [r for r in own if _metrics(r).get("arm") == arm and r["seed"] == "0" and r["git_commit"] == head]
        if len(second) != 1:
            raise BrokenBatch(f"arm {arm}: the batch's commit {head} needs one reproduction row, not {len(second)}")
        why = _matches(first[0], second[0], f"arm {arm}")
        if why:
            raise BrokenBatch(why)


def between_looks_ok(changed_paths: Iterable[str]) -> bool:
    """Commits between the looks may hold the ledger, the header and the lab, and no code."""
    return not any(p.startswith(CODE_PATHS) for p in changed_paths)


def _errored(row: Mapping[str, str]) -> bool:
    return "ERRORED" in row["notes"]


def batch_values(rows: Sequence[Mapping[str, str]], arm: str, metric: str, n: int) -> dict[int, float]:
    """One arm's test rows at ``n`` seeds. An errored row with no metric is never read, and it needs its re-run."""
    values: dict[int, float] = {}
    for r in tagged(rows, BATCH_TAG):
        m = _metrics(r)
        if m.get("arm") != arm:
            continue
        if _errored(r):
            if any(key in m for key in METRICS):
                raise BrokenBatch(f"run {r['run_id']} errored after a metric was written")
            continue
        if not _clean_row(r):
            raise BrokenBatch(f"run {r['run_id']} is not a clean, deterministic row")
        if int(r["seed"]) in values:
            raise BrokenBatch(f"arm {arm} has seed {r['seed']} twice; a completed row is never re-run")
        if metric not in m or not math.isfinite(float(m[metric])):
            raise BrokenBatch(f"run {r['run_id']} holds no finite {metric}")
        values[int(r["seed"])] = float(m[metric])
    if set(values) != set(range(n)):
        raise BrokenBatch(f"arm {arm} holds seeds {sorted(values)}, not seeds 0 to {n - 1}")
    return values


def may_rerun(rows: Sequence[Mapping[str, str]], arm: str, seed: int, head: str) -> None:
    """Only a run that errored with no metric, with no completed row, on the commit it errored on."""
    own = [r for r in tagged(rows, BATCH_TAG) if _metrics(r).get("arm") == arm and r["seed"] == str(seed)]
    errored = [r for r in own if _errored(r)]
    if not errored or len(errored) != len(own):
        raise BrokenBatch(f"arm {arm} seed {seed}: a re-run needs an errored row and no completed one")
    if any(key in _metrics(r) for r in errored for key in METRICS):
        raise BrokenBatch(f"arm {arm} seed {seed}: the errored row holds a metric")
    if any(r["git_commit"] != head for r in errored):
        raise BrokenBatch(f"arm {arm} seed {seed}: a code change ends the batch; the re-run must be on the same commit")


def batch_refusal(rows: Sequence[Mapping[str, str]], n: int) -> str | None:
    """What the report prints in place of a reading, or None when both arms hold ``n`` clean rows on each metric."""
    try:
        for arm in ARMS:
            for metric in METRICS:
                batch_values(rows, arm, metric, n)
    except BrokenBatch as why:
        return f"refused: broken batch: {why}"
    return None


def exploratory_rows_run(rows: Sequence[Mapping[str, str]], gated_pilot_hashes: Mapping[str, str]) -> bool:
    """A mismatch drops the exploratory rows, with a dated reason, and leaves the pre-registered batch running."""
    own = tagged(rows, EXPLORATORY_REPRO_TAG)
    for arm, expected in gated_pilot_hashes.items():
        found = [r for r in own if _metrics(r).get("arm") == arm and r["seed"] == "0"]
        if len(found) != 1 or not _clean_row(found[0]) or _metrics(found[0]).get("scores_sha256") != expected:
            return False
    return True
