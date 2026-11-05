"""Prototype: the go line, the pilot's attempts and what a decline prints (ADR-023 clauses 3, 5 and 7).

- **The go line.** The runner opens the test window only if the ADR, in HEAD, carries exactly one line
  ``**Go/no-go:** go · <date> · <run ids>``. Prose that quotes the line must not count, because the
  ADR's look lines stay valid after a no-go, and a look cannot be undone. The check comes before the
  opener, so a refused run writes no ledger row.
- **Attempts.** The registry is append-only, so a retried pilot's rows sit beside the failed one's. Each
  attempt has its own tags, and everything here reads one attempt, named by the caller. A void attempt's
  rows are never read by the rule.
- **The budget.** A void attempt's rows are not "the pilot's rows" in ADR-022:69-70, so running out of
  budget can still decline. What the decline prints says the rule was not computed.

Reads no data and computes no model metric: its inputs are values already in registry rows.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from gate_stats import pilot_arm  # noqa: E402
from go_no_go_proto import METRICS, BrokenPilot, RuleInputs, decide, rule_inputs  # noqa: E402

MODEL, VALIDATION = "dgf1", "370-481"
DGF1_ARM, FLOOR_ARM = "dgf1-parity-matched", "xgboost-parity-matched"
ARMS = (DGF1_ARM, FLOOR_ARM)
REPRODUCED_KEYS = (*METRICS, "scores_sha256")

_GO_LINE_START = re.compile(r"^\*\*Go/no-go:\*\*.*$", re.MULTILINE)
_GO = re.compile(r"\*\*Go/no-go:\*\* go · \d{4}-\d{2}-\d{2} · \S")
_TAG = re.compile(r"^matched_pilot_(\d+)$")


class NoGoLine(ValueError):
    """The test window stays shut."""


class ReproductionFailed(BrokenPilot):
    """Seed 0's second run did not match, or never ran: no rule, and the attempt is void."""


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


def open_after_go_line(adr_text: str, opener: Callable[[], Any]) -> Any:
    """Refuse before the opener runs, so a refused run records no look."""
    if not go_recorded(adr_text):
        raise NoGoLine("no '**Go/no-go:** go' line opens a line of this ADR; the test window stays shut")
    return opener()


# ------------------------------------------------------------------ one attempt's rows


def pilot_tag(k: int) -> str:
    return f"matched_pilot_{k}"


def repro_tag(k: int) -> str:
    return f"matched_repro_{k}"


def _metrics(row: Mapping[str, str]) -> dict:
    return json.loads(row["metrics_json"])


def tagged(rows: Sequence[Mapping[str, str]], tag: str) -> list[Mapping[str, str]]:
    return [r for r in rows if _metrics(r).get("experiment") == tag]


def attempts_in(rows: Sequence[Mapping[str, str]]) -> list[int]:
    found = {int(m.group(1)) for r in rows if (m := _TAG.match(str(_metrics(r).get("experiment"))))}
    return sorted(found)


def attempt_values(rows: Sequence[Mapping[str, str]], k: int, arm: str) -> dict[str, dict[int, float]]:
    """One arm of one attempt, by metric. A reproduction row and another attempt's rows are never read."""
    own = tagged(rows, pilot_tag(k))
    try:
        return {m: pilot_arm(own, model=MODEL, arm=arm, metric=m, window=VALIDATION) for m in METRICS}
    except (ValueError, KeyError) as why:
        raise BrokenPilot(f"attempt {k}, arm {arm}: {why}") from why


def _one_commit(rows: Sequence[Mapping[str, str]], k: int) -> None:
    """Both arms and the reproduction ran on one commit. ``pilot_arm`` only checks one arm at a time."""
    commits = {r["git_commit"] for r in (*tagged(rows, pilot_tag(k)), *tagged(rows, repro_tag(k)))}
    if len(commits) != 1:
        raise BrokenPilot(f"attempt {k} spans commits {sorted(commits)}")


def _check_reproduced(rows: Sequence[Mapping[str, str]], k: int) -> None:
    pilot, repro = tagged(rows, pilot_tag(k)), tagged(rows, repro_tag(k))
    for arm in ARMS:
        first = [_metrics(r) for r in pilot if _metrics(r).get("arm") == arm and r["seed"] == "0"]
        second = [_metrics(r) for r in repro if _metrics(r).get("arm") == arm and r["seed"] == "0"]
        if len(first) != 1 or len(second) != 1:
            raise ReproductionFailed(f"attempt {k}, arm {arm}: seed 0 needs one pilot row and one reproduction row")
        for key in REPRODUCED_KEYS:
            if first[0].get(key) is None or first[0].get(key) != second[0].get(key):
                raise ReproductionFailed(f"attempt {k}, arm {arm}: {key} did not reproduce")


def clean_attempt(rows: Sequence[Mapping[str, str]], k: int) -> RuleInputs:
    """What the rule may see of attempt ``k``, or an exception that makes the attempt void."""
    dgf1, floor = attempt_values(rows, k, DGF1_ARM), attempt_values(rows, k, FLOOR_ARM)
    inputs = rule_inputs(dgf1, floor)
    _one_commit(rows, k)
    _check_reproduced(rows, k)
    return inputs


def budget_can_decline(rows: Sequence[Mapping[str, str]]) -> bool:
    """Only a clean attempt's rows are "the pilot's rows" (ADR-022:69-70), and then the rule decides."""
    for k in attempts_in(rows):
        try:
            clean_attempt(rows, k)
            return False
        except BrokenPilot:
            continue
    return True


# ------------------------------------------------------------------ what a decline prints


def _gaps(rows: Sequence[Mapping[str, str]], k: int) -> str:
    """Each metric's gap over whatever rows the attempt has. These are only ever printed, never ruled on."""
    parts = []
    for m in METRICS:
        means = []
        for arm in ARMS:
            values = [_metrics(r)[m] for r in tagged(rows, pilot_tag(k)) if _metrics(r).get("arm") == arm and m in _metrics(r)]
            means.append(sum(values) / len(values) if values else None)
        parts.append(f"{m} gap {means[0] - means[1]:+.4f}" if None not in means else f"{m} gap n/a")
    return ", ".join(parts)


def decline_lines(rows: Sequence[Mapping[str, str]], delta_gated: Mapping[str, float],
                  root_causes: Mapping[int, str]) -> list[str]:
    """Every attempt: its gaps, then the rule's result, or why the rule was not computed."""
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
