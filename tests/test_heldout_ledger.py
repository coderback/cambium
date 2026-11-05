"""The held-out ledger (ADR-021 invariants I1 to I5 and I10, plan row 5a step 2).

Runs against a temporary git repository with committed authorising documents, and never writes the
real ledger. Two exceptions are read-only: ADR-015's accepted lines are parsed from this repository's
HEAD, so a format change that would silently refuse them fails here; and the real ledger's rows are
checked, because the backfill is now the only record of DGF-1's earlier looks.
"""

from __future__ import annotations

import csv
import inspect
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import heldout_ledger as cli  # noqa: E402

import adapters.dgf1.heldout as H  # noqa: E402
from gbe.run.config import git_dirty  # noqa: E402

TEST, OFFICIAL = "dgf1-temporal-test-482-821", "dgraph-official-test"
RUNNER, ASSEMBLER, OTHER = "scripts/run_gate3.py", "scripts/assemble_gate3.py", "scripts/other.py"
HAND = "scripts/heldout_ledger.py"
PREREG, PROPOSED = "decisions/ADR-900-gate3-prereg.md", "decisions/ADR-901-proposed.md"
LATE, PROSE = "decisions/ADR-902-accepted-later.md", "decisions/ADR-903-prose-only.md"
SAMEDAY, AMENDED = "decisions/ADR-904-accepted-same-day.md", "decisions/ADR-905-amended-later.md"
NIGHT, EARLY = "decisions/ADR-906-accepted-late-evening.md", "decisions/ADR-907-accepted-early-morning.md"
MORNING = "decisions/ADR-908-accepted-mid-morning.md"
BATCHES = ("gate3-stage1", "gate3", "earlier", "o", "official-1", "official-2", "other")


def at(day: int, h: int = 9) -> datetime:
    return datetime(2026, 10, day, h, tzinfo=timezone.utc)


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "core.autocrlf=false",
                    *args], cwd=cwd, check=True, capture_output=True)


def looks(*triples):
    return "".join(f"**Held-out look:** {s} · {t} · {v} · {b}\n" for s, t, v in triples for b in BATCHES)


def run_as(script, monkeypatch):
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(H.REPO_ROOT / script), raising=False)


@pytest.fixture
def root(tmp_path, monkeypatch):
    for d in ("decisions", "scripts", "experiments"):
        (tmp_path / d).mkdir()
    (tmp_path / "experiments" / "registry.csv").write_text("run_id\n", encoding="utf-8")
    for s in (RUNNER, ASSEMBLER, OTHER, HAND):
        (tmp_path / s).write_text("# a script\n", encoding="utf-8")
    head = "# x\n\n**Status:** {}\n**Date:** proposed 2026-10-01 · **accepted {}** by coderback\n"
    granted = looks((TEST, "scores", RUNNER), (TEST, "scores", ASSEMBLER), (OFFICIAL, "scores", RUNNER))
    docs = {PREREG: ("accepted", "2026-10-05", granted),
            PROPOSED: ("proposed", "2026-10-05", granted),
            LATE: ("accepted", "2026-10-20", granted),
            PROSE: ("accepted", "2026-10-05", f"Runs {RUNNER} on the {TEST} set.\n"),
            SAMEDAY: ("accepted", "2026-10-09", granted),
            AMENDED: ("accepted · amended 2026-10-09 by ADR-909, at the end", "2026-10-05", granted),
            NIGHT: ("accepted", "2026-10-08 (23:00)", granted),
            EARLY: ("accepted", "2026-10-09 (00:30)", granted),
            MORNING: ("accepted", "2026-10-09 (10:00)", granted)}
    for rel, (status, on, body) in docs.items():
        (tmp_path / rel).write_text(head.format(status, on) + body, encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "docs")
    monkeypatch.setattr(H, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(H, "LEDGER", tmp_path / "experiments" / "heldout_ledger.csv")
    monkeypatch.setattr(H, "_clock", lambda: at(20))
    run_as(RUNNER, monkeypatch)
    return tmp_path


def look(**over):
    base = dict(held_out_set=TEST, tier="scores", what="gated batch scores", by="researcher",
                authority=PREREG, batch="gate3-stage1", now=at(9))
    return {**base, **over}


def structure(**over):
    return look(tier="structure", what="edge dates", authority="engineering: the transform", batch="eng", **over)


def what_of(line: str) -> str:
    return line.split(" · ")[3].split(" (")[0]


# ------------------------------------------------------------------ I1: append, one header, on disk


def test_rows_append_under_one_header(root):
    H.record_look(**look())
    H.record_look(**structure())
    assert H.LEDGER.read_text(encoding="utf-8").count("timestamp_utc,") == 1
    rows = H.read_ledger()
    assert [r["tier"] for r in rows] == ["scores", "structure"] and set(rows[0]) == set(H.COLUMNS)


def test_appending_keeps_every_earlier_byte(root):
    H.record_look(**look())
    before = H.LEDGER.read_bytes()
    H.record_look(**look(now=at(10)))
    after = H.LEDGER.read_bytes()
    assert after.startswith(before) and len(after) > len(before)


def test_the_row_is_synced_to_disk_before_returning(root, monkeypatch):
    synced = []
    monkeypatch.setattr(H.os, "fsync", lambda fd: synced.append(fd))
    H.record_look(**look())
    assert synced


def test_a_missing_ledger_reads_as_no_rows(root):
    assert H.read_ledger() == []


def test_a_row_records_the_commit_it_ran_at_and_the_running_script(root):
    row = H.record_look(**look())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    assert row["git_commit"] == head and row["via"] == RUNNER


# ------------------------------------------------------------------ I2: bad fields and figures


@pytest.mark.parametrize("field,value,why", [
    ("held_out_set", "dgf1-validation-370-481", "unknown held-out set"), ("tier", "peek", "tier must be"),
    ("what", " ", "what is empty"), ("by", "", "by is empty"), ("authority", "", "authority is empty"),
    ("batch", "", "batch is empty"),
])
def test_bad_fields_are_refused_and_nothing_is_written(root, field, value, why):
    """At the structure tier, so no authority check can refuse the row for another reason."""
    with pytest.raises(H.LedgerError, match=why):
        H.record_look(**{**structure(), field: value})
    assert not H.LEDGER.exists()


@pytest.mark.parametrize("what", ["prevalence 1.48%", "a share of 12%", "2,717 fraud users", "12 fraud users",
                                  "mean degree 1.90"])
def test_a_figure_in_what_is_refused(root, what):
    with pytest.raises(H.LedgerError, match="figure"):
        H.record_look(**look(what=what))


@pytest.mark.parametrize("what", ["gated batch, mode gate3", "per ADR-015's batch", "the GNN arm"])
def test_digits_inside_names_are_not_figures(root, what):
    H.record_look(**look(what=what))


# ------------------------------------------------------------------ I3: authority


@pytest.mark.parametrize("tier", ["labels", "scores"])
@pytest.mark.parametrize("authority,why", [
    (PROPOSED, "not accepted"), (LATE, "night before"), (SAMEDAY, "night before"),
    (PROSE, "Held-out look"), ("decisions/ADR-999-missing.md", "not in HEAD"),
])
def test_labels_and_scores_need_a_document_that_named_the_look_first(root, tier, authority, why):
    with pytest.raises(H.LedgerError, match=why):
        H.record_look(**look(tier=tier, authority=authority))
    assert not H.LEDGER.exists()


@pytest.mark.parametrize("authority,refused,allowed", [
    (AMENDED, at(10, 9), at(10, 13)),     # the amendment's date counts, and its end of day
    (NIGHT, at(9, 9), at(9, 11)),         # the next day, but twelve hours after the recorded time
    (EARLY, at(9, 13), at(10, 9)),        # twelve hours on, but still the same calendar day
])
def test_the_wait_runs_from_the_latest_change_and_lasts_twelve_hours(root, authority, refused, allowed):
    with pytest.raises(H.LedgerError, match="night before"):
        H.record_look(**look(authority=authority, now=refused))
    H.record_look(**look(authority=authority, now=allowed))


def test_the_calendar_day_is_the_researcher_s_not_utc(root):
    """23:30 UTC on the 9th is 00:30 on the 10th in London: the next calendar day, 14 hours on."""
    H.record_look(**look(authority=MORNING, now=datetime(2026, 10, 9, 23, 30, tzinfo=timezone.utc)))


def test_a_look_may_not_be_dated_after_the_clock(root):
    with pytest.raises(H.LedgerError, match="later than the clock"):
        H.record_look(**look(now=at(21)))


def test_the_time_is_the_clock_s_unless_given(root):
    row = H.record_look(**look(now=None))
    assert row["timestamp_utc"] == at(20).isoformat()


def test_a_given_time_is_stored_in_utc_so_rows_sort_by_time(root):
    plus_one = timezone(timedelta(hours=1))
    row = H.record_look(**look(now=datetime(2026, 10, 9, 10, tzinfo=plus_one)))
    assert row["timestamp_utc"] == at(9).isoformat()


def test_a_scores_line_covers_a_labels_look_but_not_another_set_or_script(root, monkeypatch):
    H.record_look(**look(tier="labels"))
    run_as(ASSEMBLER, monkeypatch)
    with pytest.raises(H.LedgerError, match="Held-out look"):     # granted on TEST only
        H.record_look(**look(held_out_set=OFFICIAL, now=at(9, 10)))
    run_as(OTHER, monkeypatch)
    with pytest.raises(H.LedgerError, match="Held-out look"):
        H.record_look(**look(now=at(9, 11)))


def test_a_labels_line_does_not_cover_a_scores_look(root, monkeypatch):
    doc = "decisions/ADR-910-labels-only.md"
    (root / doc).write_text("**Status:** accepted\n**Date:** **accepted 2026-10-05**\n"
                            + f"**Held-out look:** {TEST} · labels · {RUNNER} · gate3-stage1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "labels only")
    H.record_look(**look(tier="labels", authority=doc))
    with pytest.raises(H.LedgerError, match="Held-out look"):
        H.record_look(**look(tier="scores", authority=doc, now=at(9, 10)))


def test_the_authority_line_must_name_the_batch(root):
    with pytest.raises(H.LedgerError, match="Held-out look"):
        H.record_look(**look(batch="unlisted"))


def test_via_comes_from_the_process_not_the_caller(root, monkeypatch):
    with pytest.raises(TypeError):
        H.record_look(**look(via=ASSEMBLER))
    with pytest.raises(TypeError):
        H.record_look(**look(commit="abc1234"))
    run_as(ASSEMBLER, monkeypatch)
    assert H.record_look(**look())["via"] == ASSEMBLER


def test_an_interactive_session_cannot_make_a_look(root, monkeypatch):
    monkeypatch.delattr(sys.modules["__main__"], "__file__", raising=False)
    with pytest.raises(H.LedgerError, match="interactive"):
        H.record_look(**structure())
    assert not H.LEDGER.exists()


@pytest.mark.parametrize("script,why", [("elsewhere.py", "not inside"), ("scripts/notes.txt", "not a Python")])
def test_a_look_is_made_by_a_python_script_in_the_repository(root, monkeypatch, tmp_path_factory, script, why):
    outside = tmp_path_factory.mktemp("outside") / script
    path = outside if script == "elsewhere.py" else root / script
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(path), raising=False)
    with pytest.raises(H.LedgerError, match=why):
        H.current_script()


def test_the_authority_is_read_from_head_not_the_working_tree(root):
    doc = root / PROPOSED
    doc.write_text(doc.read_text(encoding="utf-8").replace("proposed\n", "accepted\n", 1), encoding="utf-8")
    with pytest.raises(H.LedgerError, match="not accepted"):
        H.record_look(**look(authority=PROPOSED))


def test_a_structure_look_needs_a_reason_not_a_document(root):
    assert H.record_look(**structure())["authority"].startswith("engineering")


def test_adr_015_s_accepted_lines_authorise_its_runner_and_only_after_the_wait():
    """Read-only, from this repository's HEAD: ADR-021 clause 12's four lines and the amended date."""
    adr = "decisions/ADR-015-dgf1-official-split-positioning.md"
    runner, assembler = "scripts/run_dgf1_official.py", "scripts/assemble_dgf1_positioning.py"
    later = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
    for via, batch in ((runner, "official-certify"), (runner, "official-retune"), (runner, "official-positioning"),
                       (assembler, "official-positioning")):
        H.check_authority(OFFICIAL, "scores", adr, via, batch, later)
    with pytest.raises(H.LedgerError, match="Held-out look"):
        H.check_authority(TEST, "scores", adr, runner, "official-positioning", later)
    with pytest.raises(H.LedgerError, match="night before"):    # accepted 19:37 London on the 8th
        H.check_authority(OFFICIAL, "scores", adr, runner, "official-positioning",
                          datetime(2026, 10, 9, 7, 36, tzinfo=H.LOCAL_TZ))
    H.check_authority(OFFICIAL, "scores", adr, runner, "official-positioning",
                      datetime(2026, 10, 9, 7, 38, tzinfo=H.LOCAL_TZ))


# ------------------------------------------------------------------ I4: incidental looks


def test_an_incidental_look_is_recorded_and_says_it_had_no_authority(root):
    H.record_look(**look(authority="none: a reviewer met a disclosed figure"), incidental=True)
    assert H.read_ledger()[0]["authority"].startswith("none")


def test_an_incidental_look_cannot_claim_an_authority(root):
    with pytest.raises(H.LedgerError, match="'none: <why>'"):
        H.record_look(**look(), incidental=True)


# ------------------------------------------------------------------ the backfill is in, and its flag is gone


LEDGER_IN_THIS_REPO = REPO_ROOT / "experiments" / "heldout_ledger.csv"
ADR_012 = "decisions/ADR-012-dgf1-gate1-preregistration.md"
ADR_021_ACCEPTED = datetime(2026, 10, 8, 19, 37, tzinfo=H.LOCAL_TZ)
# ADR-021 clause 9's sources, one batch label for each look they name.
BACKFILL_BATCHES = {"snapshot-verification", "adr011-measurement", "adr011-review", "datasource-audit",
                    "feature-profiling", "timing-projection", "gate1", "audit-2026-10-07", "score-backup",
                    "reviews-2026-10-08"}


def _rows_in_this_repo() -> list[dict]:
    with LEDGER_IN_THIS_REPO.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _unknown_commits(rows: list[dict]) -> set[str]:
    asked = "".join(f"{r['git_commit']}^{{commit}}\n" for r in rows)
    out = subprocess.run(["git", "cat-file", "--batch-check"], cwd=REPO_ROOT, input=asked, capture_output=True,
                         text=True, check=True).stdout
    return {line.split()[0].removesuffix("^{commit}") for line in out.splitlines() if line.endswith("missing")}


def ledger_problems(rows: list[dict]) -> list[str]:
    """What I2 and I10 refuse at the door, asked of rows already written, plus the time and the commit."""
    problems, owner = [], {}
    for n, r in enumerate(rows, 2):
        if set(r) != set(H.COLUMNS) or not all((r.get(c) or "").strip() for c in H.COLUMNS):
            problems.append(f"line {n}: a column is missing or empty")
            continue
        if r["held_out_set"] not in H.HELD_OUT_SETS:
            problems.append(f"line {n}: unknown set")
        if r["tier"] not in H.TIERS:
            problems.append(f"line {n}: unknown tier")
        if H._FIGURE.search(r["what"]):
            problems.append(f"line {n}: a figure in what")
        try:
            if datetime.fromisoformat(r["timestamp_utc"]).utcoffset() is None:
                problems.append(f"line {n}: a time with no offset")
        except ValueError:
            problems.append(f"line {n}: a time that does not parse")
        if owner.setdefault(r["batch"], r["authority"]) != r["authority"]:
            problems.append(f"line {n}: batch {r['batch']} under two authorities")
    problems += [f"unknown commit {c}" for c in _unknown_commits([r for r in rows if (r.get("git_commit") or "").strip()])]
    return problems


GOOD_ROW = {"timestamp_utc": "2026-09-11T00:00:00+00:00", "held_out_set": TEST, "tier": "labels", "what": "a look",
            "by": "researcher", "via": "a source", "authority": "none: x", "batch": "b"}


def test_the_ledger_in_this_repository_is_well_formed():
    assert ledger_problems(_rows_in_this_repo()) == []


@pytest.mark.parametrize("change,why", [({"held_out_set": "elliptic"}, "unknown set"), ({"tier": "peek"}, "unknown tier"),
                                        ({"what": "prevalence 1.48%"}, "a figure"), ({"by": " "}, "missing or empty"),
                                        ({"timestamp_utc": "2026-09-11T00:00:00"}, "no offset"),
                                        ({"timestamp_utc": "yesterday"}, "does not parse"),
                                        ({"git_commit": "0" * 40}, "unknown commit"),
                                        ({"authority": "none: y"}, "two authorities")])
def test_the_well_formed_check_finds_each_defect(change, why):
    good = {**GOOD_ROW, "git_commit": _rows_in_this_repo()[0]["git_commit"]}
    assert ledger_problems([good]) == []
    assert any(why in p for p in ledger_problems([good, {**good, **change}]))


def test_the_backfill_holds_every_source_and_only_earlier_looks():
    backfill = [r for r in _rows_in_this_repo() if r["batch"] in BACKFILL_BATCHES]
    assert {r["batch"] for r in backfill} == BACKFILL_BATCHES
    assert all(datetime.fromisoformat(r["timestamp_utc"]) < ADR_021_ACCEPTED for r in backfill)
    assert {r["held_out_set"] for r in backfill} == {TEST, OFFICIAL}


def test_the_gate1_rows_are_the_scores_batch_adr_012_ran():
    gate1 = [r for r in _rows_in_this_repo() if r["batch"] == "gate1"]
    assert {(r["held_out_set"], r["tier"], r["authority"]) for r in gate1} == {(TEST, "scores", ADR_012)}
    assert min(r["timestamp_utc"] for r in gate1) == "2026-09-14T00:37:47+00:00"   # ERRATUM-DGF1-1:46
    assert {r["git_commit"][:7] for r in gate1} == {"09c6e72", "681a969"}


def test_what_was_seen_before_gate1_comes_from_both_sets_and_stops_at_the_batch():
    seen = H.disclosure(_rows_in_this_repo(), TEST, "gate1")
    assert seen == sorted(seen, key=lambda line: line[:25])
    assert any(OFFICIAL in line for line in seen) and any(TEST in line for line in seen)
    assert not any("Gate-1" in line for line in seen)
    assert all(line[:25] < "2026-09-14T00:37:47+00:00" for line in seen)


# ------------------------------------------------------------------ I5: git_dirty, with the real writer


def test_a_batch_s_opener_row_leaves_the_tree_clean_and_a_rewrite_does_not(root):
    H.record_look(**look())
    assert git_dirty(root) is False, "the first row, before the ledger is committed"
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "ledger")
    H.record_look(**look(now=at(10)))
    assert git_dirty(root) is False, "an appended row"
    H.LEDGER.write_text(H.LEDGER.read_text(encoding="utf-8").replace("researcher", "someone"), encoding="utf-8")
    assert git_dirty(root) is True, "a rewritten row"


# ------------------------------------------------------------------ I10: disclosure and batch labels


def test_disclosure_covers_overlapping_sets_earlier_rows_in_time_order(root):
    H.record_look(**look(what="second look", batch="earlier", now=at(9, 11)))
    H.record_look(**look(what="first look", batch="earlier", now=at(9, 10)))
    H.record_look(**look(held_out_set=OFFICIAL, what="official look", batch="o", now=at(9, 9)))
    H.record_look(**look(what="the batch itself", batch="gate3", now=at(9, 12)))
    lines = H.disclosure(H.read_ledger(), TEST, batch="gate3")
    assert [what_of(ln) for ln in lines] == ["official look", "first look", "second look"]


def test_two_batches_under_one_document_stay_apart(root):
    H.record_look(**look(what="first report", batch="official-1", now=at(9, 9)))
    H.record_look(**look(what="second report", batch="official-2", now=at(9, 10)))
    assert [what_of(ln) for ln in H.disclosure(H.read_ledger(), TEST, batch="official-2")] == ["first report"]


def test_disclosure_stops_at_the_batch_s_first_row(root):
    H.record_look(**look(what="batch opens", batch="gate3", now=at(9, 10)))
    H.record_look(**look(what="another batch", batch="other", now=at(9, 11)))
    H.record_look(**look(what="batch assembles", batch="gate3", now=at(9, 12)))
    assert H.disclosure(H.read_ledger(), TEST, batch="gate3") == []


def test_a_batch_with_no_opener_row_cannot_be_disclosed(root):
    H.record_look(**look())
    with pytest.raises(H.LedgerError, match="no opener row"):
        H.disclosure(H.read_ledger(), TEST, batch="never-opened")


def test_a_document_lists_every_look_so_far_before_its_batch_opens(root):
    H.record_look(**look(what="second look", batch="earlier", now=at(9, 11)))
    H.record_look(**look(what="first look", batch="earlier", now=at(9, 10)))
    H.record_look(**look(held_out_set=OFFICIAL, what="official look", batch="o", now=at(9, 9)))
    expected = ["official look", "first look", "second look"]
    assert [what_of(ln) for ln in H.seen_so_far(H.read_ledger(), TEST)] == expected
    assert [what_of(ln) for ln in H.seen_so_far(H.read_ledger(), OFFICIAL)] == expected


def test_the_look_so_far_leaves_out_a_set_that_does_not_overlap(root, monkeypatch):
    monkeypatch.setitem(H.HELD_OUT_SETS, "unrelated-set", frozenset())
    H.record_look(**look(what="a test-window look", batch="earlier"))
    H.record_look(**look(held_out_set="unrelated-set", tier="structure", what="an unrelated look",
                         authority="engineering: another model", batch="eng"))
    assert [what_of(ln) for ln in H.seen_so_far(H.read_ledger(), TEST)] == ["a test-window look"]
    assert [what_of(ln) for ln in H.seen_so_far(H.read_ledger(), "unrelated-set")] == ["an unrelated look"]


def test_the_look_so_far_of_an_empty_ledger_is_empty(root):
    assert H.seen_so_far(H.read_ledger(), TEST) == []


def test_the_look_so_far_refuses_an_unknown_set(root):
    with pytest.raises(H.LedgerError, match="unknown held-out set"):
        H.seen_so_far([], "not-a-set")


def test_a_batch_label_belongs_to_one_authority(root):
    H.record_look(**look())
    with pytest.raises(H.LedgerError, match="another authority"):
        H.record_look(**look(tier="structure", what="edge dates", authority="engineering"))
    assert len(H.read_ledger()) == 1


# ------------------------------------------------------------------ the hand path's command line


BASE = ["record", "--set", TEST, "--tier", "labels", "--by", "researcher"]


def test_the_hand_path_records_an_incidental_look(root, monkeypatch, capsys):
    run_as(HAND, monkeypatch)
    rc = cli.main([*BASE, "--what", "a reviewer met a figure", "--authority", "none: a slip", "--batch", "slip",
                   "--incidental", "--when", "2026-10-09T09:00:00+00:00"])
    row = H.read_ledger()[0]
    assert rc == 0 and row["via"] == HAND and row["timestamp_utc"] == at(9).isoformat()
    assert "recorded" in capsys.readouterr().out


def test_the_backfill_flag_is_gone(root, monkeypatch):
    """ADR-021 clause 9. From here, an earlier look is recorded as incidental, with its authority as "none"."""
    assert not {"backfill", "via", "commit"} & set(inspect.signature(H.record_look).parameters)
    assert not hasattr(H, "BACKFILL_BEFORE")
    run_as(HAND, monkeypatch)
    for flags in (["--backfill"], ["--via", "a source"], ["--commit", "abc1234"]):
        with pytest.raises(SystemExit):
            cli.main([*BASE, "--what", "a look", "--authority", "none: x", "--batch", "x", "--incidental", *flags])
    assert not H.LEDGER.exists()


def test_the_hand_path_refuses_what_record_look_refuses(root, monkeypatch, capsys):
    run_as(HAND, monkeypatch)
    rc = cli.main([*BASE, "--what", "prevalence", "--authority", PREREG, "--batch", "gate3"])
    assert rc == 1 and "refused" in capsys.readouterr().err and not H.LEDGER.exists()


def test_the_hand_path_refuses_a_time_with_no_offset(root, monkeypatch):
    run_as(HAND, monkeypatch)
    with pytest.raises(SystemExit):
        cli.main([*BASE, "--what", "a look", "--authority", "none: x", "--batch", "x", "--incidental",
                  "--when", "2026-10-09T09:00:00"])
    assert not H.LEDGER.exists()


def test_the_hand_path_prints_a_disclosure(root, monkeypatch, capsys):
    H.record_look(**look(what="earlier look", batch="earlier", now=at(9, 9)))
    H.record_look(**look(what="the batch", batch="gate3", now=at(9, 10)))
    assert cli.main(["disclose", "--set", TEST, "--batch", "gate3"]) == 0
    assert "earlier look" in capsys.readouterr().out
    assert cli.main(["disclose", "--set", TEST, "--batch", "earlier"]) == 0
    assert "nothing seen" in capsys.readouterr().out
    assert cli.main(["disclose", "--set", TEST, "--batch", "never"]) == 1


def test_the_hand_path_prints_the_look_so_far_without_a_batch(root, capsys):
    assert cli.main(["disclose", "--set", TEST]) == 0
    assert "nothing seen" in capsys.readouterr().out
    H.record_look(**look(what="earlier look", batch="earlier", now=at(9, 9)))
    H.record_look(**look(what="the batch", batch="gate3", now=at(9, 10)))
    assert cli.main(["disclose", "--set", TEST]) == 0
    out = capsys.readouterr().out
    assert "earlier look" in out and "the batch" in out


def test_the_hand_path_keeps_a_batch_s_own_rows_out_of_its_disclosure(root, capsys):
    H.record_look(**look(what="earlier look", batch="earlier", now=at(9, 9)))
    H.record_look(**look(what="the batch", batch="gate3", now=at(9, 10)))
    assert cli.main(["disclose", "--set", TEST, "--batch", "gate3"]) == 0
    out = capsys.readouterr().out
    assert "earlier look" in out and "the batch" not in out


def test_the_hand_path_refuses_an_unknown_set_without_a_batch(root, capsys):
    assert cli.main(["disclose", "--set", "not-a-set"]) == 1
    assert "refused" in capsys.readouterr().err
