"""Prototype: the go line, the pilot's attempts, the test batch's outcomes and what a decline prints
(ADR-023 clauses 3, 5 and 7).

- **The go line.** The runner opens the test window only if the ADR, in HEAD, carries exactly one line
  ``**Go/no-go:** go · <date> · <run ids>``, carries no ``**Closed:**`` line, and the rule, recomputed from the
  clean attempt those run ids name, says go. Prose that quotes the line must not count, because the ADR's look
  lines stay valid after a no-go, and a look cannot be undone. The check comes before the opener, so a refused
  run writes no ledger row.
- **Attempts.** The registry is append-only, so a retried pilot's rows sit beside the failed one's. Each
  attempt has its own tags. Its status is read once, from its first ten pilot rows and the reproduction rows
  that fix its outcome, in registry order, and a later attempt's first row closes it, so a row added later
  cannot change it. The first clean attempt is final: the rule reads it and no other, and no pilot starts after it.
- **A reproduction stopped short.** A run that stops with no metric written did not fail, so it stays owed and
  may run once more on its commit, after a recorded cause. A second stop voids the attempt, which prints as
  "did not complete". Only a finished, clean row that differs is a failed reproduction.
- **The budget.** A void attempt's rows are not "the pilot's rows" in ADR-022:69-70, so running out of
  budget can still decline. Ten clean rows are, so a reproduction is then owed before any decline, and it
  cannot be skipped to reach one.
- **The batch.** A run that errored with no metric runs again on the same code. A completed row never does. A
  seed still missing is owed, not broken. Anything else ends the batch as ``refused: broken batch``, and a look
  already read prints with it, labelled, so breaking a batch hides nothing.

Reads no data and computes no model metric: its inputs are values already in registry rows.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from gate_stats import judge, pilot_arm  # noqa: E402
from go_no_go_proto import FINAL_N, METRICS, PILOT_SEEDS, BrokenPilot, RuleInputs, below_margin, decide, reading, rule_inputs  # noqa: E402

MODEL, VALIDATION = "dgf1", "370-481"
DGF1_ARM, FLOOR_ARM = "dgf1-parity-matched", "xgboost-parity-matched"
ARMS = (DGF1_ARM, FLOOR_ARM)
EXPLORATORY_ARMS = ("dgf1-parity-exploratory", "xgboost-parity-exploratory")
REPRODUCED_KEYS = (*METRICS, "scores_sha256")
PILOT_ROWS = len(ARMS) * len(PILOT_SEEDS)
REPRO_ROWS = len(ARMS)
MAX_STOPS = 1
BATCH_TAG, BATCH_REPRO_TAG, EXPLORATORY_REPRO_TAG = "matched_time", "matched_batch_repro", "matched_exploratory_repro"
EXPLORATORY_TAG = "matched_exploratory"
ADR_PATH = "decisions/ADR-023-matched-time-sensitivity-preregistration.md"
ASSEMBLER = "scripts/assemble_dgf1_matched.py"
BETWEEN_LOOKS_FILES = frozenset({"experiments/heldout_ledger.csv", "experiments/registry.csv", "gates/REPORT-DGF1-matched-time.md"})
BETWEEN_LOOKS_DIRS = ("notebooks/lab/",)

_GO_LINE_START = re.compile(r"^\*\*Go/no-go:\*\*.*$", re.MULTILINE)
_GO = re.compile(r"\*\*Go/no-go:\*\* go · \d{4}-\d{2}-\d{2} · \S")
_CLOSED = re.compile(r"^\*\*Closed:\*\*", re.MULTILINE)
_SIGNED = re.compile(r"^\*\*Signed:\*\*", re.MULTILINE)
_ATTEMPT_CAUSE = re.compile(r"^\*\*Root cause, attempt (\d+):\*\* (environmental · )?(\S.*)$", re.MULTILINE)
_RUN_CAUSE = re.compile(r"^\*\*Root cause, run (\S+):\*\* (\S.*)$", re.MULTILINE)
_REAL_DATA = re.compile(r"^\*\*Real-data check:\*\* passed · \d{4}-\d{2}-\d{2} · (\S+)\s*$", re.MULTILINE)
_TAG = re.compile(r"^matched_pilot_(\d+)$")


class NoGoLine(ValueError):
    """The test window stays shut."""


class ReproductionFailed(BrokenPilot):
    """Seed 0's second run finished clean and did not match: no rule, and the attempt is void."""


class ReproductionStopped(BrokenPilot):
    """Seed 0's second run stopped short twice with no metric written. It did not fail, and the attempt is void."""


class ReproductionOwed(Exception):
    """Ten clean rows exist and seed 0's second run does not. Nothing may end the attempt first."""


class AttemptRefused(ValueError):
    """The runner will not write this row."""


class BrokenBatch(ValueError):
    """The test batch cannot be read. The report says ``refused: broken batch``."""


class SeedsMissing(BrokenBatch):
    """A seed or metric has not run yet. It is owed, and it ends the batch only with a dated reason."""


class ReportSigned(ValueError):
    """Re-assembling a signed report would be a new look and would erase its signature."""


@dataclass(frozen=True)
class Design:
    """What the first look's reading needs, all fixed before that look."""

    n1: int
    margins: Mapping[str, float]
    pilot_spreads: Mapping[str, tuple[float, float]]


# ------------------------------------------------------------------ the header's lines, read from HEAD


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


def root_causes_in(adr_text: str) -> dict[int, str]:
    """``**Root cause, attempt <k>:** <cause>``, anchored, so prose that quotes the field is not a record."""
    return {int(m.group(1)): m.group(3).strip() for m in _ATTEMPT_CAUSE.finditer(adr_text)}


def environmental_in(adr_text: str) -> set[int]:
    """Attempts whose recorded cause opens ``environmental · ``."""
    return {int(m.group(1)) for m in _ATTEMPT_CAUSE.finditer(adr_text) if m.group(2)}


def run_causes_in(adr_text: str) -> dict[str, str]:
    """``**Root cause, run <run id>:** <cause>``, for a test-batch run or a batch reproduction row."""
    return {m.group(1): m.group(2).strip() for m in _RUN_CAUSE.finditer(adr_text)}


def go_earned(adr_text: str, rows: Sequence[Mapping[str, str]], delta_gated: Mapping[str, float]) -> bool:
    """The line is there, it names the first clean attempt's rows, and the rule on those rows says go."""
    k = first_clean(rows)
    if not go_recorded(adr_text) or k is None:
        return False
    named = set(_GO_LINE_START.findall(adr_text)[0].split(" · ", 2)[2].split())
    used = {r["run_id"] for r in (*pilot_rows(rows, k), *repro_rows(rows, k))}
    return named == used and bool(decide(clean_attempt(rows, k), delta_gated)["go"])


def open_for_batch(adr_text: str, rows: Sequence[Mapping[str, str]], delta_gated: Mapping[str, float],
                   moved: Callable[[str], Iterable[str]], opener: Callable[[], Any]) -> Any:
    """Refuse before the opener runs, so a refused run records no look. Every test-window path comes through here."""
    if _CLOSED.search(adr_text):
        raise NoGoLine("a '**Closed:**' line ended this batch; the test window stays shut")
    if not go_earned(adr_text, rows, delta_gated):
        raise NoGoLine("no earned '**Go/no-go:** go' line opens a line of this ADR; the test window stays shut")
    check_batch_reproduced(rows, first_clean(rows), moved, run_causes_in(adr_text))
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


def _errored(row: Mapping[str, str]) -> bool:
    return "ERRORED" in row["notes"]


def _stopped_short(row: Mapping[str, str]) -> bool:
    """An errored row with no metric written. A row that holds a metric is not this: its values have been seen."""
    return _errored(row) and not any(key in _metrics(row) for key in METRICS)


def tagged(rows: Sequence[Mapping[str, str]], tag: str) -> list[Mapping[str, str]]:
    return [r for r in rows if _metrics(r).get("experiment") == tag]


def _until_closed(rows: Sequence[Mapping[str, str]], k: int) -> list[Mapping[str, str]]:
    """The registry as attempt ``k`` can read it: a later attempt's first row closes it, so no row after counts."""
    open_rows = []
    for r in rows:
        later = _TAG.match(str(_metrics(r).get("experiment")))
        if later and int(later.group(1)) > k:
            break
        open_rows.append(r)
    return open_rows


def pilot_rows(rows: Sequence[Mapping[str, str]], k: int) -> list[Mapping[str, str]]:
    """The attempt's pilot rows. Only the first ten count, so a row added later cannot change its status."""
    return tagged(_until_closed(rows, k), pilot_tag(k))[:PILOT_ROWS]


def _repro_scan(rows: Sequence[Mapping[str, str]], k: int) -> tuple[list[Mapping[str, str]], list[Mapping[str, str]]]:
    """(finished, stopped short), in registry order, up to the rows that finish it, so a later row changes nothing."""
    finished, stopped = [], []
    for r in tagged(_until_closed(rows, k), repro_tag(k)):
        if _stopped_short(r):
            stopped.append(r)
        else:
            finished.append(r)
            if len(finished) == REPRO_ROWS:
                break
    return finished, stopped


def repro_rows(rows: Sequence[Mapping[str, str]], k: int) -> list[Mapping[str, str]]:
    """The finished reproduction rows, which are the ones a go line names."""
    return _repro_scan(rows, k)[0]


def attempts_in(rows: Sequence[Mapping[str, str]]) -> list[int]:
    found = {int(m.group(1)) for r in rows if (m := _TAG.match(str(_metrics(r).get("experiment"))))}
    return sorted(found)


def attempt_values(rows: Sequence[Mapping[str, str]], k: int, arm: str) -> dict[str, dict[int, float]]:
    """One arm of one attempt, by metric. A reproduction row and another attempt's rows are never read."""
    own = pilot_rows(rows, k)
    try:
        return {m: pilot_arm(own, model=MODEL, arm=arm, metric=m, window=VALIDATION) for m in METRICS}
    except (ValueError, KeyError, TypeError) as why:
        raise BrokenPilot(f"attempt {k}, arm {arm}: {why}") from why


def _one_commit(rows: Sequence[Mapping[str, str]], k: int) -> None:
    """Both arms and the reproduction ran on one commit. ``pilot_arm`` only checks one arm at a time."""
    finished, stopped = _repro_scan(rows, k)
    commits = {r["git_commit"] for r in (*pilot_rows(rows, k), *finished, *stopped)}
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
    pilot = pilot_rows(rows, k)
    repro, stopped = _repro_scan(rows, k)
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
        if not _clean_row(second[0]):
            raise BrokenPilot(f"attempt {k}, arm {arm}: {second[0]['run_id']} is not a clean, deterministic row")
        why = _matches(first[0], second[0], f"attempt {k}, arm {arm}")
        if why:
            raise ReproductionFailed(why)
    if len(stopped) > MAX_STOPS:
        raise ReproductionStopped(f"attempt {k}: seed 0's reproduction stopped short {len(stopped)} times, with {owed} still to run")
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


def _mismatched(rows: Sequence[Mapping[str, str]], k: int) -> bool:
    """A reproduction row that is no stop and no match: investigated, and never retried, whatever its cause is called."""
    pilot = pilot_rows(rows, k)
    for second in repro_rows(rows, k):
        first = [_metrics(r) for r in pilot if _metrics(r).get("arm") == _metrics(second).get("arm") and r["seed"] == "0"]
        if first and _matches(first[0], second, ""):
            return True
    return False


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
    excused = set(environmental)
    for k in found:
        if head in {r["git_commit"] for r in pilot_rows(rows, k)} and (k not in excused or _mismatched(rows, k)):
            raise AttemptRefused(f"attempt {k} ran on {head}: never a retry, unless its cause is environmental and no reproduction differed")
    return found[-1] + 1


def may_start_pilot(adr_text: str, rows: Sequence[Mapping[str, str]], head: str, moved: Callable[[str], Iterable[str]]) -> int:
    """The attempt this command runs as, read from the ADR at HEAD. It never resumes an attempt, so a short one stays short."""
    checked = _REAL_DATA.findall(adr_text)
    if len(checked) != 1:
        raise AttemptRefused("no single '**Real-data check:** passed' line: the check comes before any pilot")
    if not between_looks_ok(moved(checked[0])):
        raise AttemptRefused(f"the code has moved since the real-data check at {checked[0]}")
    return next_attempt(rows, head, root_causes_in(adr_text), environmental_in(adr_text))


def may_write_repro(rows: Sequence[Mapping[str, str]], k: int, arm: str, head: str, *, root_causes: Mapping[int, str],
                    tree_dirty: bool) -> None:
    """A reproduction runs only while owed, on the attempt's own committed code, and only on an arm still to finish."""
    if tree_dirty:
        raise AttemptRefused("a reproduction refuses uncommitted code, with no override")
    if status(rows, k) != "owed":
        raise AttemptRefused(f"attempt {k} is {status(rows, k)}, and owes no reproduction")
    if head != pilot_rows(rows, k)[0]["git_commit"]:
        raise AttemptRefused(f"attempt {k}'s reproduction must run on its own commit, not {head}")
    if any(_metrics(r).get("arm") == arm for r in repro_rows(rows, k)):
        raise AttemptRefused(f"attempt {k}, arm {arm} has finished its reproduction, and a finished row never runs again")
    if _repro_scan(rows, k)[1] and k not in root_causes:
        raise AttemptRefused(f"attempt {k}: a stopped reproduction runs again only after a recorded root cause")


# ------------------------------------------------------------------ the budget, and what a decline prints


def budget_can_decline(rows: Sequence[Mapping[str, str]]) -> bool:
    """Only when every attempt is void. A clean attempt locks the budget, and so does a reproduction owed."""
    return all(status(rows, k) == "void" for k in attempts_in(rows))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _gaps(rows: Sequence[Mapping[str, str]], k: int) -> str:
    """Each metric's gap over whatever rows the attempt has. These are only ever printed, never ruled on."""
    parts = []
    for m in METRICS:
        means = []
        for arm in ARMS:
            values = [_metrics(r)[m] for r in pilot_rows(rows, k) if _metrics(r).get("arm") == arm and _number(_metrics(r).get(m))]
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
            stops = "".join(f"; stopped and run again: {r['run_id']}" for r in _repro_scan(rows, k)[1])
            lines.append(f"attempt {k}: {_gaps(rows, k)}; rule: {'go' if result['go'] else 'no-go'}{stops}")
        except ReproductionStopped as why:
            lines.append(f"attempt {k} (void): {_gaps(rows, k)}; rule not computed: reproduction did not complete; "
                         f"{why}; root cause: {root_causes.get(k, 'not found')}")
        except ReproductionFailed as why:
            lines.append(f"attempt {k} (void): {_gaps(rows, k)}; rule not computed: reproduction failed; "
                         f"{why}; root cause: {root_causes.get(k, 'not found')}")
        except BrokenPilot as why:
            lines.append(f"attempt {k} (void): {_gaps(rows, k)}; rule not computed: broken pilot; "
                         f"{why}; root cause: {root_causes.get(k, 'not found')}")
    return lines or ["no pilot rows exist; rule not computed"]


# ------------------------------------------------------------------ the test batch


def between_looks_ok(changed_paths: Iterable[str]) -> bool:
    """Only the ledger, the registry, the lab and the report may differ. This ADR's header never shows up as moved."""
    return all(p in BETWEEN_LOOKS_FILES or p.startswith(BETWEEN_LOOKS_DIRS) for p in changed_paths)


def _below_header(text: str) -> str:
    return text.partition("\n## ")[2]


def moved_since(repo: Path, commit: str) -> list[str]:
    """Paths that differ between ``commit`` and HEAD. The ADR counts only if its text below the header differs."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, encoding="utf-8", check=True).stdout

    moved = []
    for path in run("diff", "--name-only", commit, "HEAD").splitlines():
        if path == ADR_PATH and _below_header(run("show", f"{commit}:{path}")) == _below_header(run("show", f"HEAD:{path}")):
            continue
        moved.append(path)
    return moved


def check_batch_reproduced(rows: Sequence[Mapping[str, str]], k: int, moved: Callable[[str], Iterable[str]],
                           causes: Mapping[str, str]) -> None:
    """Seed 0 of each arm, on validation, repeats the clean attempt, on code nothing has moved since."""
    pilot, own = pilot_rows(rows, k), tagged(rows, BATCH_REPRO_TAG)
    for arm in ARMS:
        first = [_metrics(r) for r in pilot if _metrics(r).get("arm") == arm and r["seed"] == "0"]
        done = [r for r in own if _metrics(r).get("arm") == arm and r["seed"] == "0" and not _stopped_short(r)]
        if not done:
            raise BrokenBatch(f"arm {arm}: the batch needs a reproduction row, and has none")
        if len({r["git_commit"] for r in done}) != len(done):
            raise BrokenBatch(f"arm {arm}: two reproduction rows on one commit; a mismatch is never re-run until it matches")
        for earlier in done[:-1]:
            if _matches(first[0], earlier, "") and earlier["run_id"] not in causes:
                raise BrokenBatch(f"arm {arm}: {earlier['run_id']} did not match, and has no recorded root cause")
        why = _matches(first[0], done[-1], f"arm {arm}")
        if why:
            raise BrokenBatch(why)
        if not between_looks_ok(moved(done[-1]["git_commit"])):
            raise BrokenBatch(f"arm {arm}: the code has moved since its reproduction at {done[-1]['git_commit']}")


def batch_values(rows: Sequence[Mapping[str, str]], arm: str, metric: str, n: int,
                 moved: Callable[[str], Iterable[str]]) -> dict[int, float]:
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
        if not between_looks_ok(moved(r["git_commit"])):
            raise BrokenBatch(f"run {r['run_id']} ran on code that has moved since")
        if int(r["seed"]) in values:
            raise BrokenBatch(f"arm {arm} has seed {r['seed']} twice; a completed row is never re-run")
        if not _number(m.get(metric)):
            raise BrokenBatch(f"run {r['run_id']} holds no finite {metric}")
        values[int(r["seed"])] = float(m[metric])
    if set(values) - set(range(n)):
        raise BrokenBatch(f"arm {arm} holds seeds {sorted(values)}, not seeds 0 to {n - 1}")
    if set(range(n)) - set(values):
        raise SeedsMissing(f"arm {arm} holds seeds {sorted(values)}, not seeds 0 to {n - 1}")
    return values


def may_rerun(rows: Sequence[Mapping[str, str]], arm: str, seed: int, moved: Callable[[str], Iterable[str]],
              causes: Mapping[str, str]) -> None:
    """Only a run that errored with no metric, with no completed row and a recorded cause, on code that has not moved."""
    own = [r for r in tagged(rows, BATCH_TAG) if _metrics(r).get("arm") == arm and r["seed"] == str(seed)]
    errored = [r for r in own if _errored(r)]
    if not errored or len(errored) != len(own):
        raise BrokenBatch(f"arm {arm} seed {seed}: a re-run needs an errored row and no completed one")
    if any(key in _metrics(r) for r in errored for key in METRICS):
        raise BrokenBatch(f"arm {arm} seed {seed}: the errored row holds a metric")
    if any(not between_looks_ok(moved(r["git_commit"])) for r in errored):
        raise BrokenBatch(f"arm {arm} seed {seed}: a code change ends the batch; the re-run must be on the same code")
    if any(r["run_id"] not in causes for r in errored):
        raise BrokenBatch(f"arm {arm} seed {seed}: a re-run needs a recorded root cause")


def look_read(ledger_rows: Iterable[Mapping[str, str]]) -> bool:
    """A scores row by the assembler. The runner's own opener row is for scores nobody has read yet."""
    return any(r["batch"] == "matched-time" and r["tier"] == "scores" and r["by"] == ASSEMBLER for r in ledger_rows)


def _first_values(rows: Sequence[Mapping[str, str]], arm: str, metric: str, n1: int) -> list[float]:
    """The first look's own rows: each seed's first completed row, and never a second look's seeds."""
    values: dict[int, float] = {}
    for r in tagged(rows, BATCH_TAG):
        m = _metrics(r)
        if m.get("arm") == arm and not _errored(r) and int(r["seed"]) not in values:
            values[int(r["seed"])] = float(m[metric])
    return [values[s] for s in range(n1)]


def first_look_line(rows: Sequence[Mapping[str, str]], design: Design) -> str:
    """The look already read, as it read: its gap and its row of the table, labelled so it supports no claim."""
    parts = []
    for metric in METRICS:
        dgf1, floor = (_first_values(rows, arm, metric, design.n1) for arm in ARMS)
        settings = {"n1": design.n1, "n2": FINAL_N, "stage": 1, "pilot_s": design.pilot_spreads[metric]}
        gain = judge(dgf1, floor, **settings)
        below = below_margin(dgf1, floor, design.margins[metric], **settings)
        gap = sum(dgf1) / len(dgf1) - sum(floor) / len(floor)
        parts.append(f"{metric} gap {gap:+.4f}, {reading(gain, below)}")
    return f"first look, n={design.n1}, stage 1 only, stage 2 broken, supports no claim: " + "; ".join(parts)


def batch_report(rows: Sequence[Mapping[str, str]], n: int, ledger_rows: Iterable[Mapping[str, str]], design: Design,
                 moved: Callable[[str], Iterable[str]], *, ended: str | None = None) -> str | None:
    """What prints in place of a reading, or None when both arms hold ``n`` clean rows on each metric.

    A seed still missing is owed, and it raises, until the researcher ends the batch with a dated reason. Whatever
    ends the batch, a look already read prints with the refusal, so breaking a batch never hides what was seen.
    """
    try:
        for arm in ARMS:
            for metric in METRICS:
                batch_values(rows, arm, metric, n, moved)
    except SeedsMissing as why:
        if ended is None:
            raise
        reason = f"{ended}; {why}"
    except BrokenBatch as why:
        reason = str(why)
    else:
        return None
    text = f"refused: broken batch: {reason}"
    return f"{text}\n{first_look_line(rows, design)}" if look_read(ledger_rows) else text


def may_assemble(report_text: str | None) -> None:
    """Before any row is read (as ADR-013:266-268): a signed report is never assembled again."""
    if report_text is not None and _SIGNED.search(report_text):
        raise ReportSigned("this report is signed: re-assembling it would be a new look and would erase the signature")


def exploratory_rows_run(rows: Sequence[Mapping[str, str]], gated_pilot_hashes: Mapping[str, str]) -> bool:
    """A mismatch drops the exploratory rows, with a dated reason, and leaves the pre-registered batch running."""
    own = tagged(rows, EXPLORATORY_REPRO_TAG)
    for arm, expected in gated_pilot_hashes.items():
        found = [r for r in own if _metrics(r).get("arm") == arm and r["seed"] == "0"]
        if len(found) != 1 or not _clean_row(found[0]) or _metrics(found[0]).get("scores_sha256") != expected:
            return False
    return True


def exploratory_view_reads(rows: Sequence[Mapping[str, str]], view: str, n: int) -> bool:
    """A view prints only with ``n`` clean rows per arm. Anything else drops that view and raises nothing."""
    for arm in EXPLORATORY_ARMS:
        seeds: set[int] = set()
        for r in tagged(rows, EXPLORATORY_TAG):
            m = _metrics(r)
            if m.get("arm") != arm or m.get("view") != view:
                continue
            if not _clean_row(r) or int(r["seed"]) in seeds or not all(_number(m.get(key)) for key in METRICS):
                return False
            seeds.add(int(r["seed"]))
        if seeds != set(range(n)):
            return False
    return True
