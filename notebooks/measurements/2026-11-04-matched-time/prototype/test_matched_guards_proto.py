"""The go line opens the window only as itself, attempts stay apart, the budget has an exit that cannot be
reached by skipping a run, and a broken batch has one outcome."""

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


# ------------------------------------------------------------------ the go line's shape


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


def test_only_the_header_may_open_with_the_field_in_the_adr_s_own_text():
    """Read-only, from this repository. It stays true when the go or no-go is recorded in the header."""
    text = M.text_at_head(Path(__file__).resolve().parents[4], ADR_023)
    lines = M._GO_LINE_START.findall(text)
    assert len(lines) <= 1
    assert all(ln in text.split("## Disclosure")[0] for ln in lines)


def test_only_the_header_may_open_with_any_field_the_runner_reads():
    text = M.text_at_head(Path(__file__).resolve().parents[4], ADR_023)
    header_ends = text.index("## Disclosure")
    for field in (M._CLOSED, M._ATTEMPT_CAUSE, M._RUN_CAUSE, M._REAL_DATA):
        assert all(m.start() < header_ends for m in field.finditer(text))


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


def row(tag, arm, seed, auc=0.7, auprc=0.03, sha="abc", commit="c1", dirty="false", notes="", deterministic=True, **extra):
    m = {"experiment": tag, "arm": arm, "window": M.VALIDATION, "deterministic": deterministic,
         "fraud_auc": auc, "fraud_auprc": auprc, "scores_sha256": sha, **extra}
    return {"run_id": f"{tag}-{arm}-{seed}-{commit}", "model": M.MODEL, "seed": str(seed), "git_commit": commit,
            "git_dirty": dirty, "notes": notes, "metrics_json": json.dumps(m)}


def pilot_only(k=1, commit="c1", dgf1_auc=0.70, floor_auc=0.72):
    """Ten clean pilot rows, with no reproduction yet."""
    rows = []
    for arm, base in ((M.DGF1_ARM, dgf1_auc), (M.FLOOR_ARM, floor_auc)):
        rows += [row(M.pilot_tag(k), arm, s, auc=base + 0.001 * s, auprc=0.03 + 0.0001 * s, sha=f"s{s}", commit=commit)
                 for s in range(5)]
    return rows


def repro(k=1, commit="c1", dgf1_auc=0.70, floor_auc=0.72, sha="s0", **extra):
    return [row(M.repro_tag(k), arm, 0, auc=base, auprc=0.03, sha=sha, commit=commit, **extra)
            for arm, base in ((M.DGF1_ARM, dgf1_auc), (M.FLOOR_ARM, floor_auc))]


def stopped_row(arm=M.DGF1_ARM, k=1, commit="c1", n=0):
    """A run that errored with no metric written: Ctrl-C, or a run stopped for low memory."""
    m = {"experiment": M.repro_tag(k), "arm": arm, "window": M.VALIDATION, "deterministic": True}
    return {"run_id": f"stop-{arm}-{n}", "model": M.MODEL, "seed": "0", "git_commit": commit, "git_dirty": "false",
            "notes": "ERRORED", "metrics_json": json.dumps(m)}


def attempt(k=1, commit="c1", dgf1_auc=0.70, floor_auc=0.72, repro_sha="s0"):
    """A whole attempt: five seeds per arm, and seed 0's reproduction on each."""
    return pilot_only(k, commit, dgf1_auc, floor_auc) + repro(k, commit, dgf1_auc, floor_auc, repro_sha)


def test_a_clean_attempt_reaches_the_rule():
    inputs = M.clean_attempt(attempt(), 1)
    assert inputs.best_auc == pytest.approx(0.722)
    assert M.status(attempt(), 1) == "clean"


# ------------------------------------------------------------------ attempts stay apart


def test_a_reproduction_row_is_not_read_as_a_pilot_row():
    """It shares its arm and window with a pilot row, so without the filter seed 0 would count twice."""
    values = M.attempt_values(attempt(), 1, M.DGF1_ARM)
    assert sorted(values["fraud_auc"]) == [0, 1, 2, 3, 4]


def test_a_later_attempt_reads_past_a_failed_one():
    """The registry is append-only: attempt 1 has an errored row, a dirty row and another commit."""
    failed = pilot_only(1, commit="old")[:8] + [row(M.pilot_tag(1), M.FLOOR_ARM, 4, commit="old", notes="ERRORED"),
                                                row(M.pilot_tag(1), M.FLOOR_ARM, 4, commit="other", dirty="true")]
    rows = failed + attempt(2, commit="new", floor_auc=0.73)
    assert M.status(rows, 1) == "void"
    assert M.clean_attempt(rows, 2).best_auc == pytest.approx(0.732)


def test_a_void_attempt_is_never_read_by_the_rule():
    rows = attempt(1, repro_sha="different") + attempt(2, commit="c2")
    with pytest.raises(M.ReproductionFailed):
        M.clean_attempt(rows, 1)
    assert M.first_clean(rows) == 2


@pytest.mark.parametrize("spoiled", [
    row(M.pilot_tag(1), M.DGF1_ARM, 2),                      # a seed twice
    row(M.pilot_tag(1), M.DGF1_ARM, 5),                      # an extra seed
    row(M.pilot_tag(1), M.DGF1_ARM, 2, notes="ERRORED"),
    row(M.pilot_tag(1), M.FLOOR_ARM, 2, dirty="true"),
    row(M.pilot_tag(1), M.FLOOR_ARM, 2, deterministic=False),
])
def test_a_spoiled_attempt_is_void(spoiled):
    """The spoiled row comes first, so it is one of the ten that count."""
    rows = [spoiled, *attempt()]
    with pytest.raises(M.BrokenPilot):
        M.clean_attempt(rows, 1)
    assert M.status(rows, 1) == "void"


def test_a_seed_missing_makes_an_attempt_void():
    rows = [r for r in attempt() if not (r["seed"] == "3" and M.pilot_tag(1) in r["run_id"])]
    assert M.status(rows, 1) == "void"


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


# ------------------------------------------------------------------ an attempt's status is read once


@pytest.mark.parametrize("later", [
    row(M.pilot_tag(1), M.DGF1_ARM, 2, notes="ERRORED"),
    row(M.pilot_tag(1), M.DGF1_ARM, 7),
    row(M.repro_tag(1), M.DGF1_ARM, 0, sha="different"),
    row(M.repro_tag(1), M.FLOOR_ARM, 0, commit="c9"),
])
def test_a_row_added_later_cannot_unlock_a_clean_attempt(later):
    rows = [*attempt(), later]
    assert M.status(rows, 1) == "clean"
    assert M.budget_can_decline(rows) is False


def test_a_row_added_later_cannot_clean_a_void_attempt():
    rows = attempt(repro_sha="different") + repro(sha="s0")
    assert M.status(rows, 1) == "void"
    assert M.first_clean(rows) is None


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
def test_a_reproduction_not_yet_run_is_owed_on_that_arm_and_never_a_failure(arm):
    rows = [r for r in attempt() if not (M.repro_tag(1) in r["run_id"] and arm in r["run_id"])]
    with pytest.raises(M.ReproductionOwed, match=arm):
        M.clean_attempt(rows, 1)
    assert M.status(rows, 1) == "owed"


def test_ten_clean_rows_with_no_reproduction_owe_one_on_both_arms():
    with pytest.raises(M.ReproductionOwed):
        M.clean_attempt(pilot_only(), 1)
    assert M.status(pilot_only(), 1) == "owed"


def test_fewer_than_ten_rows_owe_nothing():
    assert M.status(pilot_only()[:9], 1) == "void"


@pytest.mark.parametrize("arm", M.ARMS)
def test_one_arm_s_mismatch_fails_even_with_the_other_arm_s_run_owed(arm):
    rows = pilot_only() + [row(M.repro_tag(1), arm, 0, auc=0.70 if arm == M.DGF1_ARM else 0.72, auprc=0.03, sha="different")]
    assert M.status(rows, 1) == "void"


@pytest.mark.parametrize("spoil", [{"dirty": "true"}, {"notes": "ERRORED"}, {"deterministic": False}])
def test_a_reproduction_row_must_itself_be_clean_and_is_not_called_a_failure(spoil):
    """A dirty row with a matching hash says nothing about the committed code, and it never reproduced or failed to."""
    with pytest.raises(M.BrokenPilot, match="not a clean, deterministic row") as why:
        M.clean_attempt(pilot_only() + repro(**spoil), 1)
    assert not isinstance(why.value, M.ReproductionFailed)


def test_a_reproduction_row_on_another_seed_fails():
    rows = pilot_only() + [row(M.repro_tag(1), M.DGF1_ARM, 1, sha="s1"), row(M.repro_tag(1), M.FLOOR_ARM, 0, auc=0.72, sha="s0")]
    with pytest.raises(M.ReproductionFailed, match="not seed 0"):
        M.clean_attempt(rows, 1)


def test_two_reproduction_rows_on_one_arm_fail():
    rows = pilot_only() + [repro()[0], repro()[0]]
    with pytest.raises(M.ReproductionFailed, match="one reproduction row"):
        M.clean_attempt(rows, 1)


@pytest.mark.parametrize("arm", M.ARMS)
def test_a_reproduction_stopped_once_stays_owed(arm):
    """Round 4's first finding: Ctrl-C with a gap in view must not void the attempt."""
    rows = pilot_only() + [stopped_row(arm)]
    assert M.status(rows, 1) == "owed"
    assert M.budget_can_decline(rows) is False
    with pytest.raises(M.ReproductionOwed):
        M.attempt_lines(rows, DELTA, {})


@pytest.mark.parametrize("arm", M.ARMS)
def test_a_stop_after_the_other_arm_finished_stays_owed(arm):
    finished = repro()[0] if arm == M.FLOOR_ARM else repro()[1]
    assert M.status(pilot_only() + [finished, stopped_row(arm)], 1) == "owed"


def test_a_stop_then_two_finished_rows_is_clean_and_the_go_line_names_no_stop():
    rows = pilot_only() + [stopped_row()] + repro()
    assert M.status(rows, 1) == "clean" and M.first_clean(rows) == 1
    assert "stop-" not in go_line(rows)
    assert M.go_earned(go_line(rows), rows, DELTA) is True


def test_a_second_stop_voids_the_attempt_and_prints_did_not_complete():
    rows = pilot_only() + [stopped_row(n=0), stopped_row(n=1)]
    assert M.status(rows, 1) == "void"
    assert M.budget_can_decline(rows) is True
    line = M.attempt_lines(rows, DELTA, {1: "stopped by hand"})[0]
    assert "attempt 1 (void)" in line and "fraud_auc gap -0.0" in line and "stopped by hand" in line
    assert "rule not computed: reproduction did not complete" in line and "reproduction failed" not in line


def test_a_second_stop_is_counted_across_both_arms():
    assert M.status(pilot_only() + [stopped_row(M.DGF1_ARM), stopped_row(M.FLOOR_ARM)], 1) == "void"


def test_a_row_after_the_second_stop_changes_nothing():
    rows = pilot_only() + [stopped_row(n=0), stopped_row(n=1)] + repro()
    assert M.status(rows, 1) == "void" and M.first_clean(rows) is None


def test_a_stop_after_a_clean_attempt_changes_nothing():
    rows = attempt() + [stopped_row(n=0), stopped_row(n=1)]
    assert M.status(rows, 1) == "clean"


def test_a_clean_attempt_prints_the_stop_it_ran_past():
    line = M.attempt_lines(pilot_only() + [stopped_row()] + repro(), DELTA, {1: "stopped by hand"})[0]
    assert line.startswith("attempt 1: ") and "stopped and run again: stop-" in line


def test_an_errored_row_holding_a_metric_is_no_stop():
    """Its values have been seen, so a re-run would be retrying until it matched."""
    rows = pilot_only() + [row(M.repro_tag(1), M.DGF1_ARM, 0, auc=0.70, auprc=0.03, sha="s0", notes="ERRORED")]
    assert M.status(rows, 1) == "void"
    assert "rule not computed: broken pilot" in M.attempt_lines(rows, DELTA, {})[0]


def test_a_stopped_row_on_another_commit_voids_the_attempt():
    with pytest.raises(M.BrokenPilot, match="spans commits"):
        M.clean_attempt(pilot_only() + [stopped_row(commit="c2")], 1)


def test_two_missing_hashes_do_not_count_as_a_match():
    rows = attempt()
    rows = [{**r, "metrics_json": json.dumps({k: v for k, v in json.loads(r["metrics_json"]).items() if k != "scores_sha256"})}
            for r in rows]
    with pytest.raises(M.ReproductionFailed, match="scores_sha256"):
        M.clean_attempt(rows, 1)


# ------------------------------------------------------------------ what the runner will write


def test_the_first_attempt_is_one():
    assert M.next_attempt([], "c1", {}) == 1


@pytest.mark.parametrize("rows", [attempt(), pilot_only()], ids=["clean", "owed"])
def test_no_pilot_starts_after_a_clean_attempt_or_with_a_reproduction_owed(rows):
    with pytest.raises(M.AttemptRefused, match="no further pilot"):
        M.next_attempt(rows, "c2", {1: "a cause"})


def test_a_later_attempt_needs_a_recorded_root_cause():
    with pytest.raises(M.AttemptRefused, match="root cause"):
        M.next_attempt(attempt(repro_sha="different"), "c2", {})


def test_a_later_attempt_is_refused_on_the_same_commit_never_a_retry():
    with pytest.raises(M.AttemptRefused, match="never a retry"):
        M.next_attempt(attempt(repro_sha="different"), "c1", {1: "a shuffled loader"})


@pytest.mark.parametrize("rows", [pilot_only()[:4], pilot_only() + [stopped_row(n=0), stopped_row(n=1)]], ids=["short", "stopped-twice"])
def test_an_environmental_cause_may_run_again_on_the_same_commit(rows):
    assert M.next_attempt(rows, "c1", {1: "stopped for low memory"}, environmental=[1]) == 2


def test_an_environmental_label_never_licenses_a_retry_of_a_mismatch():
    """ADR-014:22 records this exception being rejected: a finished row that differs is investigated."""
    with pytest.raises(M.AttemptRefused, match="never a retry"):
        M.next_attempt(attempt(repro_sha="different"), "c1", {1: "stopped for low memory"}, environmental=[1])


def test_an_environmental_label_never_licenses_a_retry_once_a_metric_was_seen():
    """An errored row holding a metric has been read, so running again would be retrying until it matched."""
    rows = pilot_only() + [row(M.repro_tag(1), M.DGF1_ARM, 0, auc=0.1, auprc=0.03, sha="different", notes="ERRORED")]
    with pytest.raises(M.AttemptRefused, match="never a retry"):
        M.next_attempt(rows, "c1", {1: "stopped for low memory"}, environmental=[1])


def test_every_void_attempt_s_commit_is_refused_not_only_the_last():
    rows = attempt(1, repro_sha="different") + attempt(2, commit="c2", repro_sha="different")
    with pytest.raises(M.AttemptRefused, match="attempt 1 ran on c1"):
        M.next_attempt(rows, "c1", {1: "a", 2: "b"})


def test_a_later_attempt_on_a_new_commit_is_allowed_and_counts_on():
    rows = attempt(1, repro_sha="different") + attempt(2, commit="c2", repro_sha="different")
    assert M.next_attempt(rows, "c3", {1: "a", 2: "b"}) == 3


def test_every_void_attempt_needs_its_cause_not_only_the_last():
    rows = attempt(1, repro_sha="different") + attempt(2, commit="c2", repro_sha="different")
    with pytest.raises(M.AttemptRefused, match=r"\[1\]"):
        M.next_attempt(rows, "c3", {2: "b"})


def test_the_owed_reproduction_may_run_on_its_own_commit():
    M.may_write_repro(pilot_only(), 1, M.DGF1_ARM, "c1", root_causes={}, tree_dirty=False)


def test_the_owed_reproduction_is_refused_on_another_commit():
    """Otherwise a commit, then a reproduction, is a way to void an attempt whose gap was disliked."""
    with pytest.raises(M.AttemptRefused, match="its own commit"):
        M.may_write_repro(pilot_only(), 1, M.DGF1_ARM, "c2", root_causes={}, tree_dirty=False)


@pytest.mark.parametrize("rows", [attempt(), attempt(repro_sha="different"), pilot_only()[:4]], ids=["clean", "failed", "unfinished"])
def test_only_an_owed_attempt_may_write_a_reproduction(rows):
    with pytest.raises(M.AttemptRefused, match="owes no reproduction"):
        M.may_write_repro(rows, 1, M.DGF1_ARM, "c1", root_causes={}, tree_dirty=False)


def test_a_dirty_tree_may_never_write_a_reproduction():
    with pytest.raises(M.AttemptRefused, match="uncommitted code"):
        M.may_write_repro(pilot_only(), 1, M.DGF1_ARM, "c1", root_causes={1: "a cause"}, tree_dirty=True)


def test_a_stopped_reproduction_runs_again_only_after_a_recorded_cause():
    rows = pilot_only() + [stopped_row()]
    with pytest.raises(M.AttemptRefused, match="recorded root cause"):
        M.may_write_repro(rows, 1, M.DGF1_ARM, "c1", root_causes={}, tree_dirty=False)
    M.may_write_repro(rows, 1, M.DGF1_ARM, "c1", root_causes={1: "stopped by hand"}, tree_dirty=False)


def test_a_stopped_reproduction_runs_again_only_on_its_own_commit():
    with pytest.raises(M.AttemptRefused, match="its own commit"):
        M.may_write_repro(pilot_only() + [stopped_row()], 1, M.DGF1_ARM, "c2", root_causes={1: "a cause"}, tree_dirty=False)


def test_a_finished_arm_never_runs_again():
    """A second row on it would read as two reproduction rows, and void a clean run."""
    rows = pilot_only() + [repro()[0], stopped_row(M.FLOOR_ARM)]
    with pytest.raises(M.AttemptRefused, match="finished row never runs again"):
        M.may_write_repro(rows, 1, M.DGF1_ARM, "c1", root_causes={1: "a cause"}, tree_dirty=False)
    M.may_write_repro(rows, 1, M.FLOOR_ARM, "c1", root_causes={1: "a cause"}, tree_dirty=False)


# ------------------------------------------------------------------ a later attempt closes an earlier one


def test_a_short_attempt_cannot_be_finished_after_a_later_attempt_began():
    """Round 4's should-fix 5: a tenth row added late must not move the first clean attempt back."""
    short = pilot_only()
    rows = short[:9] + attempt(2, commit="c2") + short[9:] + repro()
    assert M.status(rows, 1) == "void"
    assert M.first_clean(rows) == 2


def test_an_earlier_attempt_cannot_gain_a_reproduction_after_a_later_one_began():
    rows = pilot_only() + attempt(2, commit="c2") + repro()
    assert M.status(rows, 1) == "owed"
    assert M.first_clean(rows) == 2


def test_a_later_attempt_s_rows_never_close_a_later_one():
    assert M.status(attempt(1) + attempt(2, commit="c2"), 2) == "clean"


# ------------------------------------------------------------------ the header's lines


REAL = "**Real-data check:** passed · 2026-11-20 · c0\n"


def test_no_pilot_starts_without_one_real_data_line():
    for text in ("", f"recorded as `{REAL}`", f"  {REAL}", "**Real-data check:** failed · 2026-11-20 · c0\n", REAL + REAL):
        with pytest.raises(M.AttemptRefused, match="Real-data check"):
            M.may_start_pilot(text, [], "c1", STILL)


def test_no_pilot_starts_once_the_code_has_moved_since_the_real_data_check():
    with pytest.raises(M.AttemptRefused, match="moved since"):
        M.may_start_pilot(REAL, [], "c1", CODE_MOVED)
    assert M.may_start_pilot(REAL, [], "c1", STILL) == 1


def test_the_runner_reads_a_root_cause_from_its_own_anchored_line():
    rows = attempt(repro_sha="different")
    for quoted in ("", "  **Root cause, attempt 1:** a shuffled loader\n", "see `**Root cause, attempt 1:** x`\n"):
        with pytest.raises(M.AttemptRefused, match="root cause"):
            M.may_start_pilot(REAL + quoted, rows, "c2", STILL)
    assert M.may_start_pilot(REAL + "**Root cause, attempt 1:** a shuffled loader\n", rows, "c2", STILL) == 2


def test_an_ordinary_cause_is_not_environmental():
    text = REAL + "**Root cause, attempt 1:** a shuffled loader\n"
    with pytest.raises(M.AttemptRefused, match="never a retry"):
        M.may_start_pilot(text, pilot_only()[:4], "c1", STILL)


def test_an_environmental_cause_is_read_from_its_line():
    text = REAL + "**Root cause, attempt 1:** environmental · stopped for low memory\n"
    assert M.may_start_pilot(text, pilot_only()[:4], "c1", STILL) == 2
    assert M.root_causes_in(text) == {1: "stopped for low memory"}
    assert M.environmental_in(text) == {1}


# ------------------------------------------------------------------ the first clean attempt is final


def test_the_rule_reads_the_first_clean_attempt_and_no_other():
    rows = attempt(1, floor_auc=0.40, dgf1_auc=0.41) + attempt(2, commit="c2")
    assert M.first_clean(rows) == 1


def test_no_clean_attempt_is_none():
    assert M.first_clean(attempt(repro_sha="different")) is None
    assert M.first_clean(pilot_only()) is None
    assert M.first_clean([]) is None


# ------------------------------------------------------------------ the go line is earned


def go_line(rows, k=1, word="go"):
    ids = " ".join(r["run_id"] for r in (*M.pilot_rows(rows, k), *M.repro_rows(rows, k)))
    return f"# ADR\n**Go/no-go:** {word} · 2026-11-20 · {ids}\n"


STILL = lambda commit: []  # noqa: E731  (nothing has moved since this commit)
CODE_MOVED = lambda commit: ["scripts/x.py"]  # noqa: E731
LEDGER_MOVED = lambda commit: ["experiments/heldout_ledger.csv"]  # noqa: E731


def code_moved_since_before(commit_that_holds):
    """Every commit other than the named one has different code from HEAD."""
    return lambda commit: [] if commit == commit_that_holds else ["scripts/x.py"]


def batch_repro(commit="c9", sha="s0", dgf1_auc=0.70, floor_auc=0.72, **extra):
    return [row(M.BATCH_REPRO_TAG, arm, 0, auc=base, auprc=0.03, sha=sha, commit=commit, **extra)
            for arm, base in ((M.DGF1_ARM, dgf1_auc), (M.FLOOR_ARM, floor_auc))]


def test_a_go_line_naming_the_clean_attempt_with_a_go_rule_is_earned():
    assert M.go_earned(go_line(attempt()), attempt(), DELTA) is True


def test_a_no_go_line_naming_the_right_rows_is_not_earned():
    assert M.go_earned(go_line(attempt(), word="no-go"), attempt(), DELTA) is False


def test_a_go_line_is_not_earned_with_no_clean_attempt():
    rows = attempt(repro_sha="different")
    assert M.go_earned(go_line(rows), rows, DELTA) is False


def test_a_go_line_must_name_the_clean_attempt_s_rows():
    assert M.go_earned(GO, attempt(), DELTA) is False


def test_a_go_line_naming_only_some_of_the_rows_is_not_earned():
    text = go_line(attempt()).replace(attempt()[0]["run_id"] + " ", "")
    assert M.go_earned(text, attempt(), DELTA) is False


def test_a_go_line_naming_a_later_attempt_is_not_earned():
    """A no-go on attempt 1, then attempt 2 on a new commit, then an edited line: the first clean attempt stands."""
    rows = attempt(1, dgf1_auc=0.40, floor_auc=0.41) + attempt(2, commit="c2")
    assert M.go_earned(go_line(rows, 2), rows, DELTA) is False


def test_a_go_line_is_not_earned_when_the_rule_says_no_go():
    """The first clean attempt fails the sanity bar, and the line says go anyway."""
    rows = attempt(dgf1_auc=0.40, floor_auc=0.41)
    assert M.go_earned(go_line(rows), rows, DELTA) is False


def test_the_check_comes_before_the_opener():
    opened = []
    with pytest.raises(M.NoGoLine):
        M.open_for_batch(NO_GO, attempt(), DELTA, STILL, lambda: opened.append(1))
    assert opened == []


def test_a_go_with_the_batch_s_reproduction_opens():
    rows = attempt() + batch_repro()
    assert M.open_for_batch(go_line(attempt()), rows, DELTA, STILL, lambda: "window") == "window"


@pytest.mark.parametrize("extra", [
    [],                                        # never ran
    batch_repro(commit="c8"),                  # the code has moved since it ran
    batch_repro(sha="different"),              # the code or the environment moved
    batch_repro(dgf1_auc=0.1),
    batch_repro(dirty="true"),
    batch_repro()[:1],                         # one arm only
    batch_repro()[1:],
    batch_repro() + batch_repro(),             # twice on one commit
], ids=["none", "code-moved", "hash", "metric", "dirty", "floor-missing", "dgf1-missing", "twice"])
def test_the_batch_s_reproduction_must_pass_before_the_opener(extra):
    opened = []
    with pytest.raises(M.BrokenBatch):
        M.open_for_batch(go_line(attempt()), attempt() + extra, DELTA, code_moved_since_before("c9"), lambda: opened.append(1))
    assert opened == []


def test_a_ledger_commit_leaves_the_reproduction_standing_for_the_second_look():
    """Round 4's should-fix 1: the first look's ledger row is always committed, so HEAD always moves."""
    rows = attempt() + batch_repro()
    assert M.open_for_batch(go_line(attempt()), rows, DELTA, LEDGER_MOVED, lambda: "window") == "window"


def batch_stop(arm, commit="c9"):
    m = {"experiment": M.BATCH_REPRO_TAG, "arm": arm, "window": M.VALIDATION, "deterministic": True}
    return {"run_id": f"batch-stop-{arm}", "model": M.MODEL, "seed": "0", "git_commit": commit, "git_dirty": "false",
            "notes": "ERRORED", "metrics_json": json.dumps(m)}


@pytest.mark.parametrize("arm", M.ARMS)
def test_a_stopped_batch_reproduction_may_run_again_on_its_commit(arm):
    rows = attempt() + [batch_stop(arm)] + batch_repro()
    assert M.open_for_batch(go_line(attempt()), rows, DELTA, STILL, lambda: "window") == "window"


def test_a_stopped_batch_reproduction_alone_opens_nothing():
    rows = attempt() + [batch_stop(arm) for arm in M.ARMS]
    with pytest.raises(M.BrokenBatch, match="has none"):
        M.open_for_batch(go_line(attempt()), rows, DELTA, STILL, lambda: "window")


def cause_for(rows):
    return "".join(f"**Root cause, run {r['run_id']}:** the environment moved\n" for r in rows)


def test_a_mismatch_is_checked_again_only_with_its_root_cause_recorded():
    """Round 4's should-fix 3: a no-op commit and a re-run must not pass over the mismatch."""
    bad, good = batch_repro(commit="c8", sha="different"), batch_repro(commit="c9")
    rows, moved = attempt() + bad + good, code_moved_since_before("c9")
    with pytest.raises(M.BrokenBatch, match="no recorded root cause"):
        M.open_for_batch(go_line(attempt()), rows, DELTA, moved, lambda: "window")
    text = go_line(attempt()) + cause_for(bad)
    assert M.open_for_batch(text, rows, DELTA, moved, lambda: "window") == "window"


def test_an_earlier_row_that_matched_needs_no_cause():
    """Nothing went wrong at c8, so there is no cause to record."""
    rows = attempt() + batch_repro(commit="c8") + batch_repro(commit="c9")
    opened = M.open_for_batch(go_line(attempt()), rows, DELTA, code_moved_since_before("c9"), lambda: "window")
    assert opened == "window"


def test_a_cause_quoted_in_prose_is_not_recorded():
    bad, good = batch_repro(commit="c8", sha="different"), batch_repro(commit="c9")
    text = go_line(attempt()) + "".join(f"  {ln}\n" for ln in cause_for(bad).splitlines())
    with pytest.raises(M.BrokenBatch, match="no recorded root cause"):
        M.open_for_batch(text, attempt() + bad + good, DELTA, code_moved_since_before("c9"), lambda: "window")


def test_a_closed_line_shuts_the_window_after_a_go():
    """Round 4's should-fix 7: the go line and the rule stay "go" after a decline or a broken batch."""
    text, rows, opened = go_line(attempt()) + "**Closed:** 2026-12-01 · declined\n", attempt() + batch_repro(), []
    with pytest.raises(M.NoGoLine, match="Closed"):
        M.open_for_batch(text, rows, DELTA, STILL, lambda: opened.append(1))
    assert opened == []


def test_a_closed_line_quoted_in_prose_shuts_nothing():
    text = go_line(attempt()) + "a decline adds `**Closed:** <date>` to the header\n"
    assert M.open_for_batch(text, attempt() + batch_repro(), DELTA, STILL, lambda: "window") == "window"


# ------------------------------------------------------------------ the budget


def test_the_budget_can_decline_before_any_rows_exist():
    assert M.budget_can_decline([]) is True


def test_the_budget_cannot_decline_once_a_clean_attempt_exists():
    assert M.budget_can_decline(attempt()) is False


def test_the_budget_cannot_decline_while_a_reproduction_is_owed():
    """Ten clean rows, a gap in view, and the reproduction simply not run: that must not be an exit."""
    assert M.budget_can_decline(pilot_only()) is False


def test_the_budget_can_decline_after_a_failed_reproduction():
    """The day-13 case: every row exists, and the reproduction ran and failed."""
    assert M.budget_can_decline(attempt(repro_sha="different")) is True


def test_the_budget_can_decline_with_an_unfinished_attempt():
    assert M.budget_can_decline(pilot_only()[:4]) is True


def test_a_clean_attempt_after_a_void_one_locks_the_budget():
    assert M.budget_can_decline(attempt(1, repro_sha="different") + attempt(2, commit="c2")) is False


def test_an_owed_attempt_after_a_void_one_locks_the_budget():
    assert M.budget_can_decline(attempt(1, repro_sha="different") + pilot_only(2, commit="c2")) is False


# ------------------------------------------------------------------ what a decline and the report print


def test_nothing_prints_while_a_reproduction_is_owed():
    """So no line can say 'reproduction failed' of a run that never happened."""
    with pytest.raises(M.ReproductionOwed):
        M.attempt_lines(pilot_only(), DELTA, {})


def test_a_void_attempt_prints_its_gaps_and_that_no_rule_ran():
    line = M.attempt_lines(attempt(repro_sha="different"), DELTA, {1: "a shuffled loader"})[0]
    assert "attempt 1 (void)" in line and "fraud_auc gap -0.0200" in line and "fraud_auprc gap +0.0000" in line
    assert "rule not computed: reproduction failed" in line and "a shuffled loader" in line
    assert "rule:" not in line


def test_a_broken_attempt_says_so_differently():
    rows = [row(M.pilot_tag(1), M.DGF1_ARM, 2, notes="ERRORED"), *attempt()]
    line = M.attempt_lines(rows, DELTA, {})[0]
    assert "rule not computed: broken pilot" in line and "root cause: not found" in line


def test_a_clean_attempt_prints_its_gaps_and_the_rule_s_result():
    line = M.attempt_lines(attempt(), DELTA, {})[0]
    assert line.startswith("attempt 1: ") and "fraud_auc gap -0.0200" in line and "rule: go" in line


def test_every_attempt_is_listed_in_order():
    lines = M.attempt_lines(attempt(1, repro_sha="different") + attempt(2, commit="c2"), DELTA, {})
    assert [ln.split(":")[0] for ln in lines] == ["attempt 1 (void)", "attempt 2"]


def test_a_decline_with_no_rows_says_so():
    assert M.attempt_lines([], DELTA, {}) == ["no pilot rows exist; rule not computed"]


def test_gaps_are_read_from_the_attempt_s_counted_rows_only():
    rows = [*attempt(), row(M.pilot_tag(1), M.DGF1_ARM, 7, auc=0.1)]
    assert "fraud_auc gap -0.0200" in M.attempt_lines(rows, DELTA, {})[0]


# ------------------------------------------------------------------ the test batch's outcomes


def batch_row(arm, seed, auc=0.7, auprc=0.03, commit="c9", dirty="false", notes="", metrics=True):
    m = {"experiment": M.BATCH_TAG, "arm": arm, "window": "482-821", "deterministic": True}
    if metrics:
        m |= {"fraud_auc": auc, "fraud_auprc": auprc}
    return {"run_id": f"t-{arm}-{seed}-{notes}", "model": M.MODEL, "seed": str(seed), "git_commit": commit,
            "git_dirty": dirty, "notes": notes, "metrics_json": json.dumps(m)}


def batch(n=8):
    return [batch_row(arm, s, auc=0.7 + 0.001 * s) for arm in M.ARMS for s in range(n)]


DESIGN = M.Design(n1=8, margins={"fraud_auc": 0.018, "fraud_auprc": 0.004},
                  pilot_spreads={"fraud_auc": (0.008, 0.002), "fraud_auprc": (0.002, 0.0003)})
LOOK = {"batch": "matched-time", "tier": "scores", "by": M.ASSEMBLER}
OPENER = {"batch": "matched-time", "tier": "scores", "by": "scripts/run_dgf1_matched.py"}


def report(rows, n=8, ledger=(), moved=STILL, **kw):
    return M.batch_report(rows, n, ledger, DESIGN, moved, **kw)


def values_of(rows, arm, metric, n=8, moved=STILL):
    return M.batch_values(rows, arm, metric, n, moved)


def test_a_complete_batch_reads_and_prints_no_refusal():
    assert sorted(values_of(batch(), M.DGF1_ARM, "fraud_auc")) == list(range(8))
    assert report(batch()) is None


def test_an_errored_run_with_its_re_run_reads_the_re_run_alone():
    rows = [batch_row(M.DGF1_ARM, 3, notes="ERRORED", metrics=False), *batch()]
    assert values_of(rows, M.DGF1_ARM, "fraud_auc")[3] == pytest.approx(0.703)
    assert report(rows) is None


@pytest.mark.parametrize("arm", M.ARMS)
def test_a_seed_still_missing_is_owed_and_prints_nothing(arm):
    """Round 4's second finding: not running a seed must not reach a report that hides the look."""
    rows = [r for r in batch() if not (arm in r["run_id"] and r["seed"] == "3")]
    rows.append(batch_row(arm, 3, notes="ERRORED", metrics=False))
    for ledger in ((), (LOOK,)):
        with pytest.raises(M.SeedsMissing):
            report(rows, ledger=ledger)


def test_a_seed_missing_ends_the_batch_only_with_a_dated_reason():
    rows = [r for r in batch() if not (M.DGF1_ARM in r["run_id"] and r["seed"] == "7")]
    text = report(rows, ended="2026-12-01: a fix needs a code change")
    assert text.startswith("refused: broken batch: 2026-12-01: a fix needs a code change;")


def test_a_complete_batch_is_not_ended_by_a_reason():
    assert report(batch(), ended="2026-12-01: I would rather not") is None


@pytest.mark.parametrize("bad", [
    batch_row(M.DGF1_ARM, 3),                                 # a completed row run again
    batch_row(M.FLOOR_ARM, 3),
    batch_row(M.DGF1_ARM, 3, notes="ERRORED"),                # errored after a metric was written
    batch_row(M.FLOOR_ARM, 9, dirty="true"),
], ids=["twice-dgf1", "twice-floor", "errored-with-metric", "dirty"])
def test_anything_else_is_a_broken_batch(bad):
    assert report([*batch(), bad]).startswith("refused: broken batch")


@pytest.mark.parametrize("arm", M.ARMS)
@pytest.mark.parametrize("spoil", [{"dirty": "true"}, {"deterministic": False}], ids=["dirty", "not-deterministic"])
def test_a_row_that_is_not_clean_in_a_seed_s_place_is_a_broken_batch(arm, spoil):
    rows = [r for r in batch() if not (arm in r["run_id"] and r["seed"] == "7")]
    bad = batch_row(arm, 7, dirty=spoil.get("dirty", "false"))
    if "deterministic" in spoil:
        bad = {**bad, "metrics_json": json.dumps({**json.loads(bad["metrics_json"]), "deterministic": False})}
    assert report([*rows, bad]).startswith("refused: broken batch")


@pytest.mark.parametrize("metric", ["fraud_auc", "fraud_auprc"])
@pytest.mark.parametrize("arm", M.ARMS)
@pytest.mark.parametrize("broken", ["missing", float("nan"), None], ids=["missing", "nan", "null"])
def test_a_completed_row_with_no_finite_metric_is_a_broken_batch(arm, metric, broken):
    rows = batch()
    for i, r in enumerate(rows):
        if arm in r["run_id"] and r["seed"] == "2":
            m = json.loads(r["metrics_json"])
            m = {k: v for k, v in m.items() if k != metric} if broken == "missing" else {**m, metric: broken}
            rows[i] = {**r, "metrics_json": json.dumps(m)}
    assert report(rows).startswith("refused: broken batch")


def test_a_null_metric_in_the_pilot_voids_the_attempt_and_does_not_crash():
    rows = attempt()
    rows[2] = {**rows[2], "metrics_json": json.dumps({**json.loads(rows[2]["metrics_json"]), "fraud_auc": None})}
    assert M.status(rows, 1) == "void"
    assert "rule not computed: broken pilot" in M.attempt_lines(rows, DELTA, {})[0]


def test_an_arm_with_an_extra_seed_is_a_broken_batch():
    assert report([*batch(), batch_row(M.FLOOR_ARM, 8)]).startswith("refused: broken batch")


def test_a_row_on_code_that_has_moved_is_a_broken_batch():
    """Round 4's should-fix 1: a commit between the looks must not read clean."""
    assert report(batch(), moved=CODE_MOVED).startswith("refused: broken batch")
    assert report(batch(), moved=LEDGER_MOVED) is None


def test_rows_with_another_tag_or_arm_are_not_read():
    other = [row(M.BATCH_REPRO_TAG, M.DGF1_ARM, 3), {**batch_row("dgf1-parity-exploratory", 3)}]
    assert values_of([*other, *batch()], M.DGF1_ARM, "fraud_auc")[3] == pytest.approx(0.703)


# ------------------------------------------------------------------ a look already read prints with the refusal


def looked(n=8):
    """A first look with a clear gain on both metrics."""
    return [batch_row(M.DGF1_ARM, s, auc=0.75 + 0.001 * s, auprc=0.04 + 0.0001 * s) for s in range(n)] + \
           [batch_row(M.FLOOR_ARM, s, auc=0.70 + 0.002 * s, auprc=0.03 + 0.0001 * s) for s in range(n)]


def later_seeds(first=8, last=12):
    return [batch_row(arm, s, auc=0.1, auprc=0.001) for arm in M.ARMS for s in range(first, last)]


BROKEN = batch_row(M.FLOOR_ARM, 2, dirty="true")
LABEL = "first look, n=8, stage 1 only, stage 2 broken, supports no claim"


def test_a_second_look_not_run_is_owed():
    """The first look is read, a gain clause is unresolved, and the rest of the seeds simply do not run."""
    with pytest.raises(M.SeedsMissing):
        report(looked(), n=20, ledger=[LOOK])


def test_with_no_look_read_a_broken_batch_prints_no_gap():
    for ledger in ((), (OPENER,)):
        text = report([*looked(), BROKEN], ledger=ledger)
        assert text.startswith("refused: broken batch") and "gap" not in text and "first look" not in text


def test_a_broken_batch_after_a_look_prints_that_look_s_gaps_and_label():
    text = report([*looked(), BROKEN], ledger=[LOOK])
    assert text.startswith("refused: broken batch")
    assert LABEL in text
    assert "fraud_auc gap +0.0465" in text and "fraud_auprc gap +0.0100" in text


def test_ending_a_second_look_by_hand_prints_the_first_look():
    text = report(looked(), n=20, ledger=[LOOK], ended="2026-12-01: a fix needs a code change")
    assert "2026-12-01: a fix needs a code change" in text and LABEL in text and "fraud_auc gap +0.0465" in text


def test_the_first_look_s_line_prints_its_row_of_the_table_on_each_metric():
    from go_no_go_proto import below_margin, reading
    from gate_stats import judge

    text = report([*looked(), BROKEN], ledger=[LOOK])
    for metric in M.METRICS:
        rows = [json.loads(r["metrics_json"])[metric] for r in looked()]
        dgf1, floor = rows[:8], rows[8:]
        settings = {"n1": 8, "n2": 20, "stage": 1, "pilot_s": DESIGN.pilot_spreads[metric]}
        expected = reading(judge(dgf1, floor, **settings), below_margin(dgf1, floor, DESIGN.margins[metric], **settings))
        assert f"{metric} gap" in text and f", {expected}" in text
    assert "resolvable" in text


def test_the_first_look_s_line_never_reads_a_second_look_s_seeds():
    text = report(looked() + later_seeds(), n=20, ledger=[LOOK], ended="2026-12-01: a fix needs a code change")
    assert "fraud_auc gap +0.0465" in text and "fraud_auprc gap +0.0100" in text


def test_the_first_look_s_line_reads_each_seed_s_first_row():
    again = batch_row(M.DGF1_ARM, 3, auc=0.1, auprc=0.001)
    text = report([*looked(), again], ledger=[LOOK])
    assert text.startswith("refused: broken batch") and "fraud_auc gap +0.0465" in text


def test_the_first_look_s_line_reads_past_an_errored_row():
    stop = batch_row(M.DGF1_ARM, 3, notes="ERRORED", metrics=False)
    text = report([stop, *looked(), BROKEN], ledger=[LOOK])
    assert "fraud_auc gap +0.0465" in text


@pytest.mark.parametrize("other", [
    {**LOOK, "batch": "dgf1-gate-1"}, {**LOOK, "tier": "labels"}, OPENER,
], ids=["another-batch", "labels-tier", "the-runner"])
def test_only_the_assembler_s_scores_row_counts_as_a_look_read(other):
    assert M.look_read([other]) is False
    assert M.look_read([other, LOOK]) is True


# ------------------------------------------------------------------ may_rerun


def errored_only(commit="c9"):
    return [batch_row(M.DGF1_ARM, 3, notes="ERRORED", metrics=False, commit=commit)]


CAUSED = {errored_only()[0]["run_id"]: "stopped for low memory"}


def test_an_errored_run_may_run_again_on_its_commit():
    M.may_rerun(errored_only(), M.DGF1_ARM, 3, STILL, CAUSED)


def test_an_errored_run_may_run_again_after_a_ledger_commit():
    """Round 4's should-fix 1: only the ledger moved, so nothing about the code has changed."""
    M.may_rerun(errored_only(), M.DGF1_ARM, 3, LEDGER_MOVED, CAUSED)


def test_a_re_run_on_moved_code_is_refused():
    with pytest.raises(M.BrokenBatch, match="same code"):
        M.may_rerun(errored_only(), M.DGF1_ARM, 3, CODE_MOVED, CAUSED)


def test_a_re_run_needs_a_recorded_root_cause():
    with pytest.raises(M.BrokenBatch, match="recorded root cause"):
        M.may_rerun(errored_only(), M.DGF1_ARM, 3, STILL, {})


@pytest.mark.parametrize("rows", [
    [],                                                       # nothing to re-run
    [batch_row(M.DGF1_ARM, 3)],                               # completed
    [batch_row(M.DGF1_ARM, 3, notes="ERRORED", metrics=False), batch_row(M.DGF1_ARM, 3)],   # already re-run
], ids=["none", "completed", "already-re-run"])
def test_a_completed_row_is_never_re_run(rows):
    with pytest.raises(M.BrokenBatch, match="errored row and no completed one"):
        M.may_rerun(rows, M.DGF1_ARM, 3, STILL, CAUSED)


def test_an_errored_row_holding_a_metric_is_not_re_run():
    with pytest.raises(M.BrokenBatch, match="holds a metric"):
        M.may_rerun([batch_row(M.DGF1_ARM, 3, notes="ERRORED")], M.DGF1_ARM, 3, STILL, CAUSED)


def test_another_seed_s_error_allows_nothing_here():
    with pytest.raises(M.BrokenBatch):
        M.may_rerun(errored_only(), M.DGF1_ARM, 4, STILL, CAUSED)
    with pytest.raises(M.BrokenBatch):
        M.may_rerun(errored_only(), M.FLOOR_ARM, 3, STILL, CAUSED)


# ------------------------------------------------------------------ what may sit between the looks


@pytest.mark.parametrize("paths", [
    ["adapters/dgf1/matched.py"], ["gbe/run/session.py"], ["scripts/run_dgf1_matched.py"], ["configs/dgf1/a.yaml"],
    ["experiments/heldout_ledger.csv", "scripts/x.py"], ["pyproject.toml"], ["tests/test_x.py"], [ADR_023],
    ["notebooks/measurements/x.py"], ["experiments/other.csv"],
], ids=lambda p: p[-1])
def test_anything_off_the_list_between_the_looks_is_refused(paths):
    assert M.between_looks_ok(paths) is False


@pytest.mark.parametrize("path", ["experiments/heldout_ledger.csv", "experiments/registry.csv", "notebooks/lab/2026-12-01.md",
                                  "gates/REPORT-DGF1-matched-time.md"])
def test_the_ledger_the_registry_the_lab_and_the_report_may_sit_between_the_looks(path):
    assert M.between_looks_ok([path]) is True


def commit_file(root, git, path, content):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    git("add", path)
    git("commit", "-q", "-m", path)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, encoding="utf-8", check=True).stdout.strip()


ADR_TEXT = "# ADR\n\n**Status:** accepted\n\n## Decision\n\nThe rule.\n"


def test_moved_since_sees_code_and_not_a_header_line(repo):
    root, git = repo
    first = commit_file(root, git, ADR_023, ADR_TEXT)
    assert M.moved_since(root, first) == []
    commit_file(root, git, ADR_023, ADR_TEXT.replace("# ADR\n", f"# ADR\n{GO}\n"))
    assert M.moved_since(root, first) == []
    assert M.between_looks_ok(M.moved_since(root, first)) is True


def test_moved_since_sees_the_adr_s_body(repo):
    root, git = repo
    first = commit_file(root, git, ADR_023, ADR_TEXT)
    commit_file(root, git, ADR_023, ADR_TEXT.replace("The rule.", "Another rule."))
    assert M.moved_since(root, first) == [ADR_023]
    assert M.between_looks_ok(M.moved_since(root, first)) is False


def test_moved_since_lets_a_stray_commit_be_reverted(repo):
    root, git = repo
    first = commit_file(root, git, "scripts/x.py", "a\n")
    commit_file(root, git, "scripts/x.py", "b\n")
    assert M.moved_since(root, first) == ["scripts/x.py"]
    commit_file(root, git, "scripts/x.py", "a\n")
    assert M.moved_since(root, first) == []


def test_two_real_commits_through_the_opener_the_re_run_and_the_reading(repo):
    """Round 4's should-fix 1: the first look's ledger commit moves HEAD, and nothing about the code has moved."""
    root, git = repo
    first = commit_file(root, git, "scripts/x.py", "a\n")
    moved = lambda commit: M.moved_since(root, commit)  # noqa: E731
    stop = batch_row(M.DGF1_ARM, 3, notes="ERRORED", metrics=False, commit=first)
    tested = [batch_row(arm, s, commit=first) for arm in M.ARMS for s in range(8) if not (arm == M.DGF1_ARM and s == 3)]
    rows = attempt() + batch_repro(commit=first) + [stop] + tested
    text, causes = go_line(attempt()), {stop["run_id"]: "stopped for low memory"}

    second = commit_file(root, git, "experiments/heldout_ledger.csv", "a look\n")
    assert M.open_for_batch(text, rows, DELTA, moved, lambda: "window") == "window"
    M.may_rerun(rows, M.DGF1_ARM, 3, moved, causes)
    rows = [*rows, batch_row(M.DGF1_ARM, 3, commit=second)]
    assert sorted(M.batch_values(rows, M.DGF1_ARM, "fraud_auc", 8, moved)) == list(range(8))

    commit_file(root, git, "scripts/x.py", "b\n")
    with pytest.raises(M.BrokenBatch, match="moved"):
        M.open_for_batch(text, rows, DELTA, moved, lambda: "window")
    with pytest.raises(M.BrokenBatch, match="moved"):
        M.batch_values(rows, M.DGF1_ARM, "fraud_auc", 8, moved)
    with pytest.raises(M.BrokenBatch, match="same code"):
        M.may_rerun([stop], M.DGF1_ARM, 3, moved, causes)


# ------------------------------------------------------------------ the signed report


@pytest.mark.parametrize("text", [None, "", "# Report\n\nunsigned\n", "see `**Signed:** coderback · <date>`\n", "  **Signed:** coderback · 2026-12-01\n"])
def test_an_unsigned_report_may_be_assembled(text):
    M.may_assemble(text)


def test_a_signed_report_is_never_assembled_again():
    with pytest.raises(M.ReportSigned):
        M.may_assemble("# Report\n\n**Signed:** coderback · 2026-12-01\n")


# ------------------------------------------------------------------ the exploratory rows


HASHES = dict(zip(M.EXPLORATORY_ARMS, "ab"))


def exploratory_repro(a="a", b="b", **extra):
    return [row(M.EXPLORATORY_REPRO_TAG, M.EXPLORATORY_ARMS[0], 0, sha=a, **extra), row(M.EXPLORATORY_REPRO_TAG, M.EXPLORATORY_ARMS[1], 0, sha=b, **extra)]


def test_the_exploratory_rows_run_when_both_gated_models_reproduce():
    assert M.exploratory_rows_run(exploratory_repro(), HASHES) is True


def test_a_row_under_another_tag_does_not_count_as_the_exploratory_reproduction():
    rows = [row(M.BATCH_REPRO_TAG, M.EXPLORATORY_ARMS[0], 0, sha="a"), row(M.BATCH_REPRO_TAG, M.EXPLORATORY_ARMS[1], 0, sha="b")]
    assert M.exploratory_rows_run(rows, HASHES) is False


@pytest.mark.parametrize("rows", [
    [], exploratory_repro(a="x"), exploratory_repro(b="x"), exploratory_repro()[:1], exploratory_repro()[1:],
    exploratory_repro(dirty="true"), exploratory_repro() + exploratory_repro(),
], ids=["none", "dgf1-hash", "floor-hash", "floor-missing", "dgf1-missing", "dirty", "twice"])
def test_a_mismatch_drops_the_exploratory_rows_and_raises_nothing(rows):
    """They never decide anything, so they cannot stop the pre-registered comparison."""
    assert M.exploratory_rows_run(rows, HASHES) is False


def viewed(view="window_only", n=8, **extra):
    return [row(M.EXPLORATORY_TAG, arm, s, view=view, **extra) for arm in M.EXPLORATORY_ARMS for s in range(n)]


def test_a_view_with_all_its_clean_rows_prints():
    assert M.exploratory_view_reads(viewed(), "window_only", 8) is True


@pytest.mark.parametrize("arm", M.EXPLORATORY_ARMS)
@pytest.mark.parametrize("spoil", [
    {"notes": "ERRORED"}, {"dirty": "true"}, {"deterministic": False}, {"auc": None}, {"auc": float("nan")},
], ids=["errored", "dirty", "not-deterministic", "null", "nan"])
def test_an_errored_or_unclean_exploratory_row_drops_that_view_and_raises_nothing(arm, spoil):
    """They never decide anything, so they cannot stop the pre-registered comparison."""
    rows = [r for r in viewed() if not (arm in r["run_id"] and r["seed"] == "3")] + [row(M.EXPLORATORY_TAG, arm, 3, view="window_only", **spoil)]
    assert M.exploratory_view_reads(rows, "window_only", 8) is False


@pytest.mark.parametrize("arm", M.EXPLORATORY_ARMS)
def test_a_short_exploratory_arm_drops_that_view(arm):
    rows = [r for r in viewed() if not (arm in r["run_id"] and r["seed"] == "7")]
    assert M.exploratory_view_reads(rows, "window_only", 8) is False


def test_a_doubled_exploratory_seed_drops_that_view():
    assert M.exploratory_view_reads([*viewed(), viewed()[0]], "window_only", 8) is False


def test_one_view_s_error_does_not_drop_the_other_view():
    rows = viewed("window_only") + viewed("first_appearance", notes="ERRORED")
    assert M.exploratory_view_reads(rows, "window_only", 8) is True
    assert M.exploratory_view_reads(rows, "first_appearance", 8) is False
