"""The go line opens the window only as itself, attempts stay apart, and the budget has an exit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_guards_proto as M  # noqa: E402

GO = "**Go/no-go:** go · 2026-11-20 · dgf1-a dgf1-b"
NO_GO = "**Go/no-go:** no-go · 2026-11-20 · dgf1-a dgf1-b"
ADR_023 = "decisions/ADR-023-matched-time-sensitivity-preregistration.md"
DELTA = {"fraud_auc": 0.04, "fraud_auprc": 0.008}


# ------------------------------------------------------------------ the go line


@pytest.mark.parametrize("text", [GO, f"# ADR\n\n{GO}\nmore\n", f"{GO}\r\nmore\r\n"])
def test_the_go_line_opens(text):
    assert M.go_recorded(text) is True


@pytest.mark.parametrize("text", [
    "", NO_GO, f"text\n{NO_GO}\n",
    f"recorded as a line `{GO}`\n",                     # quoted in prose
    f"`{GO}`\n",                                        # quoted, at a line's start
    f"  {GO}\n",                                        # indented, so a quotation
    f"> {GO}\n",
    f"prose before {GO}\n",
    "**Go/no-go:** go\n",                               # no date or run ids
    "**Go/no-go:** go · 2026-11-20 · \n",
    "**Go/no-go:** go · soon · dgf1-a\n",
    "**Go/no-go:** go-ahead · 2026-11-20 · dgf1-a\n",
    "**Go/no-go:** Go · 2026-11-20 · dgf1-a\n",
    "**Go/no-go:** <go or no-go> · <date> · <run ids>\n",
    f"{GO}\n{NO_GO}\n",                                 # both: a record that contradicts itself
    f"{NO_GO}\n{GO}\n",
    f"{GO}\n{GO}\n",
])
def test_anything_else_stays_shut(text):
    assert M.go_recorded(text) is False


def test_the_adr_s_own_committed_text_does_not_open_the_window():
    """Read-only, from this repository. The ADR mentions the line in prose, and no go is recorded yet."""
    repo = Path(__file__).resolve().parents[4]
    assert M.go_recorded(M.text_at_head(repo, ADR_023)) is False


def test_the_check_comes_before_the_opener():
    opened = []
    with pytest.raises(M.NoGoLine):
        M.open_after_go_line(NO_GO, lambda: opened.append(1))
    assert opened == []
    assert M.open_after_go_line(GO, lambda: "window") == "window"


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "core.autocrlf=false", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "adr.md").write_text("# ADR\n", encoding="utf-8")
    git("add", "adr.md")
    git("commit", "-q", "-m", "x")
    return tmp_path, git


def test_a_go_line_only_in_the_working_tree_stays_shut(repo):
    root, git = repo
    (root / "adr.md").write_text(f"# ADR\n{GO}\n", encoding="utf-8")
    assert M.go_recorded(M.text_at_head(root, "adr.md")) is False
    git("add", "adr.md")
    git("commit", "-q", "-m", "y")
    assert M.go_recorded(M.text_at_head(root, "adr.md")) is True


def test_a_document_not_in_head_stays_shut(repo):
    with pytest.raises(M.NoGoLine, match="not in HEAD"):
        M.text_at_head(repo[0], "other.md")


# ------------------------------------------------------------------ rows


def row(tag, arm, seed, auc=0.7, auprc=0.03, sha="abc", commit="c1", dirty="false", notes="", **extra):
    m = {"experiment": tag, "arm": arm, "window": M.VALIDATION, "deterministic": True,
         "fraud_auc": auc, "fraud_auprc": auprc, "scores_sha256": sha, **extra}
    return {"run_id": f"{tag}-{arm}-{seed}-{commit}", "model": M.MODEL, "seed": str(seed), "git_commit": commit,
            "git_dirty": dirty, "notes": notes, "metrics_json": json.dumps(m)}


def attempt(k=1, commit="c1", dgf1_auc=0.70, floor_auc=0.72, repro_sha="s0"):
    """A whole clean attempt: five seeds per arm, and seed 0's reproduction on each."""
    rows = []
    for arm, base in ((M.DGF1_ARM, dgf1_auc), (M.FLOOR_ARM, floor_auc)):
        for s in range(5):
            rows.append(row(M.pilot_tag(k), arm, s, auc=base + 0.001 * s, auprc=0.03 + 0.0001 * s,
                            sha=f"s{s}", commit=commit))
        rows.append(row(M.repro_tag(k), arm, 0, auc=base, auprc=0.03, sha=repro_sha, commit=commit))
    return rows


def test_a_clean_attempt_reaches_the_rule():
    inputs = M.clean_attempt(attempt(), 1)
    assert inputs.best_auc == pytest.approx(0.722)


# ------------------------------------------------------------------ attempts stay apart


def test_a_reproduction_row_is_not_read_as_a_pilot_row():
    """It shares its arm and window with a pilot row, so without the filter seed 0 would count twice."""
    values = M.attempt_values(attempt(), 1, M.DGF1_ARM)
    assert sorted(values["fraud_auc"]) == [0, 1, 2, 3, 4]


def test_a_later_attempt_reads_past_a_failed_one():
    """The registry is append-only: attempt 1 has an errored row, a dirty row and another commit."""
    failed = attempt(1, commit="old")[:9] + [row(M.pilot_tag(1), M.FLOOR_ARM, 4, commit="old", notes="ERRORED"),
                                             row(M.pilot_tag(1), M.FLOOR_ARM, 4, commit="other", dirty="true")]
    rows = failed + attempt(2, commit="new", floor_auc=0.73)
    assert M.clean_attempt(rows, 2).best_auc == pytest.approx(0.732)


def test_a_void_attempt_is_never_read_by_the_rule():
    rows = attempt(1, repro_sha="different") + attempt(2, commit="c2")
    with pytest.raises(M.ReproductionFailed):
        M.clean_attempt(rows, 1)
    M.clean_attempt(rows, 2)


@pytest.mark.parametrize("spoil", [
    lambda rows: rows + [row(M.pilot_tag(1), M.DGF1_ARM, 2)],                      # a seed twice
    lambda rows: [r for r in rows if not (r["seed"] == "3" and M.pilot_tag(1) in r["run_id"])],   # a seed missing
    lambda rows: rows + [row(M.pilot_tag(1), M.DGF1_ARM, 5)],                      # an extra seed
    lambda rows: rows + [row(M.pilot_tag(1), M.DGF1_ARM, 2, notes="ERRORED")],
    lambda rows: rows + [row(M.pilot_tag(1), M.FLOOR_ARM, 2, dirty="true")],
])
def test_a_spoiled_attempt_is_void(spoil):
    with pytest.raises(M.BrokenPilot):
        M.clean_attempt(spoil(attempt()), 1)


def test_a_missing_metric_makes_an_attempt_void():
    rows = attempt()
    m = json.loads(rows[2]["metrics_json"])
    del m["fraud_auprc"]
    rows[2] = {**rows[2], "metrics_json": json.dumps(m)}
    with pytest.raises(M.BrokenPilot):
        M.clean_attempt(rows, 1)


def test_the_arms_must_share_a_commit():
    """`pilot_arm` collects commits per call, so two arms at two commits would otherwise pass."""
    rows = [r if M.FLOOR_ARM not in r["run_id"] else {**r, "git_commit": "c2"} for r in attempt()]
    with pytest.raises(M.BrokenPilot, match="spans commits"):
        M.clean_attempt(rows, 1)


def test_the_reproduction_must_share_the_pilot_s_commit():
    rows = [r if M.repro_tag(1) not in r["run_id"] else {**r, "git_commit": "c2"} for r in attempt()]
    with pytest.raises(M.BrokenPilot, match="spans commits"):
        M.clean_attempt(rows, 1)


# ------------------------------------------------------------------ reproduction


@pytest.mark.parametrize("key,value", [("scores_sha256", "different"), ("fraud_auc", 0.1), ("fraud_auprc", 0.1)])
@pytest.mark.parametrize("arm", M.ARMS)
def test_a_mismatch_on_any_field_of_either_arm_fails_reproduction(key, value, arm):
    rows = attempt()
    for i, r in enumerate(rows):
        m = json.loads(r["metrics_json"])
        if m["experiment"] == M.repro_tag(1) and m["arm"] == arm:
            rows[i] = {**r, "metrics_json": json.dumps({**m, key: value})}
    with pytest.raises(M.ReproductionFailed, match=key):
        M.clean_attempt(rows, 1)


@pytest.mark.parametrize("arm", M.ARMS)
def test_a_reproduction_that_never_ran_fails(arm):
    rows = [r for r in attempt() if not (M.repro_tag(1) in r["run_id"] and arm in r["run_id"])]
    with pytest.raises(M.ReproductionFailed, match="seed 0"):
        M.clean_attempt(rows, 1)


def test_two_missing_hashes_do_not_count_as_a_match():
    rows = attempt()
    rows = [{**r, "metrics_json": json.dumps({k: v for k, v in json.loads(r["metrics_json"]).items() if k != "scores_sha256"})}
            for r in rows]
    with pytest.raises(M.ReproductionFailed, match="scores_sha256"):
        M.clean_attempt(rows, 1)


# ------------------------------------------------------------------ the budget


def test_the_budget_can_decline_before_any_rows_exist():
    assert M.budget_can_decline([]) is True


def test_the_budget_cannot_decline_once_a_clean_attempt_exists():
    assert M.budget_can_decline(attempt()) is False


def test_the_budget_can_decline_after_a_failed_reproduction():
    """The day-13 case: every row exists and reproduction failed. Without this nothing could ever end it."""
    assert M.budget_can_decline(attempt(repro_sha="different")) is True


def test_the_budget_can_decline_with_an_unfinished_attempt():
    assert M.budget_can_decline(attempt()[:4]) is True


def test_a_clean_attempt_after_a_void_one_locks_the_budget():
    assert M.budget_can_decline(attempt(1, repro_sha="different") + attempt(2, commit="c2")) is False


# ------------------------------------------------------------------ what a decline prints


def test_a_void_attempt_prints_its_gaps_and_that_no_rule_ran():
    line = M.decline_lines(attempt(repro_sha="different"), DELTA, {1: "a shuffled loader"})[0]
    assert "attempt 1 (void)" in line and "fraud_auc gap -0.0200" in line and "fraud_auprc gap +0.0000" in line
    assert "rule not computed: reproduction failed" in line and "a shuffled loader" in line
    assert "rule:" not in line


def test_a_broken_attempt_says_so_differently():
    rows = attempt() + [row(M.pilot_tag(1), M.DGF1_ARM, 2, notes="ERRORED")]
    line = M.decline_lines(rows, DELTA, {})[0]
    assert "rule not computed: broken pilot" in line and "root cause: not found" in line


def test_a_clean_attempt_prints_its_gaps_and_the_rule_s_result():
    line = M.decline_lines(attempt(), DELTA, {})[0]
    assert line.startswith("attempt 1: ") and "fraud_auc gap -0.0200" in line and "rule: go" in line


def test_every_attempt_is_listed_in_order():
    lines = M.decline_lines(attempt(1, repro_sha="different") + attempt(2, commit="c2"), DELTA, {})
    assert [ln.split(":")[0] for ln in lines] == ["attempt 1 (void)", "attempt 2"]


def test_a_decline_with_no_rows_says_so():
    assert M.decline_lines([], DELTA, {}) == ["no pilot rows exist; rule not computed"]
