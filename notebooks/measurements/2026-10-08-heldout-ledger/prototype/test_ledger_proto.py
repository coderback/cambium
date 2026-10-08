"""Tests for the held-out ledger prototype. Run: python -m pytest -q test_ledger_proto.py"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ledger_proto as L  # noqa: E402

# The repo root: four levels up when run in place; mutate.py passes it when it runs a copy.
REPO = Path(os.environ.get("CAMBIUM_REPO") or Path(__file__).resolve().parents[4])
SET = "dgf1-temporal-test-482-821"
ACCEPTED, PROPOSED = "decisions/ADR-900-accepted.md", "decisions/ADR-901-proposed.md"


@pytest.fixture
def root(tmp_path):
    (tmp_path / "decisions").mkdir()
    (tmp_path / ACCEPTED).write_text("# x\n\n**Status:** accepted\n", encoding="utf-8")
    (tmp_path / PROPOSED).write_text("# y\n\n**Status:** proposed\n", encoding="utf-8")
    return tmp_path


def ok(root, **over):
    base = dict(held_out_set=SET, tier="labels", what="fraud count and prevalence of the window",
                by="main session", via="scripts/measure_dgf1_temporal_split.py", authority=ACCEPTED,
                git_commit="abc1234", repo_root=root)
    return {**base, **over}


def at(h: int, m: int = 0) -> datetime:
    return datetime(2026, 10, 9, h, m, tzinfo=timezone.utc)


# ---- record_look -------------------------------------------------------------------------------

def test_rows_append_under_one_header(root):
    p = root / "ledger.csv"
    L.record_look(p, **ok(root), now=at(9))
    L.record_look(p, **ok(root, tier="structure", what="edge dates in the window"), now=at(10))
    assert p.read_text(encoding="utf-8").count("timestamp_utc,") == 1
    rows = L.read_ledger(p)
    assert [r["tier"] for r in rows] == ["labels", "structure"]
    assert set(rows[0]) == set(L.COLUMNS)


def test_an_existing_row_survives_a_later_write(root):
    p = root / "ledger.csv"
    L.record_look(p, **ok(root), now=at(9))
    first = p.read_text(encoding="utf-8")
    L.record_look(p, **ok(root), now=at(10))
    assert L.is_append_only(first, p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("field,value", [
    ("held_out_set", "dgf1-validation-370-481"),
    ("tier", "peek"),
    ("what", "  "),
    ("by", ""),
    ("authority", ""),
])
def test_bad_fields_are_refused_and_nothing_is_written(root, field, value):
    p = root / "ledger.csv"
    with pytest.raises(L.LedgerError):
        L.record_look(p, **ok(root, **{field: value}))
    assert not p.exists()


@pytest.mark.parametrize("what", ["prevalence 1.48%", "2717 fraud users", "mean degree 1.90"])
def test_a_figure_in_what_is_refused(root, what):
    with pytest.raises(L.LedgerError):
        L.record_look(root / "ledger.csv", **ok(root, what=what))


# ---- authority: labels and scores need an accepted document ------------------------------------

@pytest.mark.parametrize("tier", ["labels", "scores"])
@pytest.mark.parametrize("authority", [PROPOSED, "decisions/ADR-999-missing.md", "ADR-011"])
def test_labels_and_scores_without_an_accepted_document_are_refused(root, tier, authority):
    p = root / "ledger.csv"
    with pytest.raises(L.LedgerError):
        L.record_look(p, **ok(root, tier=tier, authority=authority))
    assert not p.exists()


def test_a_structure_look_needs_a_reason_but_not_an_accepted_document(root):
    L.record_look(root / "ledger.csv", **ok(root, tier="structure", what="edge dates",
                                             authority="engineering: the feature transform"))


def test_an_incidental_look_is_recorded_and_says_it_had_no_authority(root):
    p = root / "ledger.csv"
    L.record_look(p, **ok(root, authority="none: a reviewer met a disclosed figure"), incidental=True)
    assert L.read_ledger(p)[0]["authority"].startswith("none")


def test_an_incidental_look_cannot_claim_an_authority(root):
    with pytest.raises(L.LedgerError):
        L.record_look(root / "ledger.csv", **ok(root), incidental=True)


# ---- append-only -------------------------------------------------------------------------------

def _rows(*hours):
    return [{"timestamp_utc": at(h).isoformat(), "held_out_set": SET, "tier": "labels",
             "what": "a look", "by": "x", "via": "x", "authority": "ADR-011", "git_commit": "c"}
            for h in hours]


def test_append_only_accepts_an_appended_row():
    assert L.is_append_only(L.csv_text(_rows(1, 2)), L.csv_text(_rows(1, 2, 3)))


@pytest.mark.parametrize("working", [
    L.csv_text(_rows(1)),                        # a row deleted
    L.csv_text(_rows(2, 1)),                     # rows reordered
    L.csv_text(_rows(0, 1, 2)),                  # a row inserted before old ones
    L.csv_text(_rows(1, 2)).replace("ADR-011", "ADR-012", 1),   # an old row edited
])
def test_append_only_refuses_any_other_change(working):
    assert not L.is_append_only(L.csv_text(_rows(1, 2)), working)


# ---- the guard ---------------------------------------------------------------------------------

def modes(doc=ACCEPTED):
    return {"gate3": L.GatedMode(document=doc, held_out_set=SET)}


def test_the_guard_records_one_scores_row_naming_its_document(root):
    p = root / "ledger.csv"
    L.open_held_out("gate3", modes(), p, git_commit="def5678", repo_root=root)
    (row,) = L.read_ledger(p)
    assert row["tier"] == "scores" and row["authority"] == ACCEPTED
    assert row["held_out_set"] == SET and row["git_commit"] == "def5678"


def test_a_refused_mode_records_nothing(root):
    p = root / "ledger.csv"
    with pytest.raises(SystemExit):
        L.open_held_out("gate", modes(), p, git_commit="def5678", repo_root=root)
    assert not p.exists()


def test_the_guard_cannot_record_for_an_unaccepted_document(root):
    p = root / "ledger.csv"
    with pytest.raises(L.LedgerError):
        L.open_held_out("gate3", modes(PROPOSED), p, git_commit="def5678", repo_root=root)
    assert not p.exists()


def test_the_row_exists_before_the_caller_reads_held_out_data(root):
    p = root / "ledger.csv"

    def run_batch():
        L.open_held_out("gate3", modes(), p, git_commit="def5678", repo_root=root)
        return L.read_ledger(p)          # stands in for the first held-out read

    assert len(run_batch()) == 1


# ---- the lint ----------------------------------------------------------------------------------

READER = 'data = load_dgraph(root)\nsplits = dgf1_splits()\nt = splits["test"]\n'
OFFICIAL = 'data = DGraphFin(root=r)[0]\nn = int(data.test_mask.sum())\n'
ASSEMBLER = 'gs = load_scores(row["scores_path"])\n'
VAL_ONLY = 'data = load_dgraph(root)\nv = splits["val"]\n'


def test_the_lint_flags_each_kind_of_unrecorded_reader():
    srcs = {"a.py": READER, "b.py": OFFICIAL, "c.py": ASSEMBLER, "d.py": VAL_ONLY}
    assert L.unrecorded_readers(srcs) == ["a.py", "b.py", "c.py"]


def test_the_lint_accepts_a_reader_that_records_or_passes_the_guard():
    srcs = {"a.py": READER + "record_look(LEDGER, ...)\n",
            "b.py": OFFICIAL + "require_gated_preregistration(mode, p, False)\n"}
    assert L.unrecorded_readers(srcs) == []


def test_only_the_hand_path_may_record_an_incidental_look():
    srcs = {"scripts/measure.py": READER + "record_look(LEDGER, ..., incidental=True)\n",
            L.HAND_PATH: "record_look(LEDGER, ..., incidental=True)\n"}
    assert L.scripts_claiming_incidental(srcs) == ["scripts/measure.py"]


def test_the_lint_on_the_committed_scripts_matches_the_inventory():
    """Measured cross-check: the four outside-guard readers of the 2026-10-08 inventory."""
    names = subprocess.run(["git", "ls-files", "scripts/*.py"], cwd=REPO, capture_output=True,
                           text=True, check=True).stdout.split()
    srcs = {n: (REPO / n).read_text(encoding="utf-8") for n in names}
    assert L.unrecorded_readers(srcs) == [
        "scripts/assemble_gate_dgf1_1.py", "scripts/audit_dgf1_datasource.py",
        "scripts/measure_dgf1_temporal_split.py", "scripts/verify_dgraph_snapshot.py"]


# ---- disclosure --------------------------------------------------------------------------------

def test_disclosure_lists_earlier_looks_at_this_set_only_in_time_order(root):
    p = root / "ledger.csv"
    L.record_look(p, **ok(root, what="second look"), now=at(11))
    L.record_look(p, **ok(root, what="first look"), now=at(10))
    L.record_look(p, **ok(root, held_out_set="dgraph-official-test", what="other set"), now=at(9))
    L.record_look(p, **ok(root, tier="scores", what="the batch itself"), now=at(12))
    lines = L.disclosure(L.read_ledger(p), SET, before=at(12).isoformat(timespec="seconds"))
    assert [ln.split(" · ")[2].split(" (")[0] for ln in lines] == ["first look", "second look"]


# ---- the dirty-tree interaction ------------------------------------------------------------------

def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_a_ledger_append_dirties_the_tree_unless_git_dirty_ignores_it(root):
    sys.path.insert(0, str(REPO))
    from gbe.run.config import IGNORED_DIRTY_PATHS, git_dirty

    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "experiments").mkdir()
    ledger = root / "experiments" / "heldout_ledger.csv"
    L.record_look(ledger, **ok(root), now=at(9))
    (root / "code.py").write_text("x = 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "init")
    assert not git_dirty(root)

    L.record_look(ledger, **ok(root), now=at(10))      # the guard's row, at the start of a batch
    assert git_dirty(root), "today's ignore set treats the ledger as code"
    assert not git_dirty(root, ignore=IGNORED_DIRTY_PATHS | {"experiments/heldout_ledger.csv"})
