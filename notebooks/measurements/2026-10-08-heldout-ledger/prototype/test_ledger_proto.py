"""Tests for the held-out ledger prototype v2. Run: python -m pytest -q test_ledger_proto.py"""
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
TEST, OFFICIAL = "dgf1-temporal-test-482-821", "dgraph-official-test"
RUNNER, ASSEMBLER = "scripts/run_gate3.py", "scripts/assemble_gate3.py"
PREREG, PROPOSED = "decisions/ADR-900-gate3-prereg.md", "decisions/ADR-901-proposed.md"
LATE, SILENT = "decisions/ADR-902-accepted-later.md", "decisions/ADR-903-names-nothing.md"
CONFIG = {"val_min": 370, "val_max": 481, "test_min": 482, "test_max": 821}


def at(day: int, h: int = 9) -> datetime:
    return datetime(2026, 10, day, h, tzinfo=timezone.utc)


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "decisions"
    d.mkdir()
    head = "# x\n\n**Status:** {}\n**Date:** proposed 2026-10-01 · **accepted {}** by coderback\n"
    docs = {PREREG: ("accepted", "2026-10-05", f"Runs {RUNNER}; {ASSEMBLER}.\n"),
            PROPOSED: ("proposed", "2026-10-05", f"Runs {RUNNER}.\n"),
            LATE: ("accepted", "2026-10-20", f"Runs {RUNNER}.\n"),
            SILENT: ("accepted", "2026-10-05", "Runs something else.\n")}
    for rel, (status, on, body) in docs.items():
        (tmp_path / rel).write_text(head.format(status, on) + body, encoding="utf-8")
    return tmp_path


def look(root, **over):
    base = dict(held_out_set=TEST, tier="scores", what="gated batch scores", by="researcher",
                via=RUNNER, authority=PREREG, git_commit="abc1234", repo_root=root, now=at(9))
    return {**base, **over}


# ---- record_look -------------------------------------------------------------------------------

def test_rows_append_under_one_header(root):
    p = root / "ledger.csv"
    L.record_look(p, **look(root))
    L.record_look(p, **look(root, tier="structure", what="edge dates", authority="engineering"), )
    assert p.read_text(encoding="utf-8").count("timestamp_utc,") == 1
    rows = L.read_ledger(p)
    assert [r["tier"] for r in rows] == ["scores", "structure"] and set(rows[0]) == set(L.COLUMNS)


@pytest.mark.parametrize("field,value", [
    ("held_out_set", "dgf1-validation-370-481"), ("tier", "peek"), ("what", " "), ("by", ""),
    ("authority", ""),
])
def test_bad_fields_are_refused_and_nothing_is_written(root, field, value):
    p = root / "ledger.csv"
    with pytest.raises(L.LedgerError):
        L.record_look(p, **look(root, **{field: value}))
    assert not p.exists()


@pytest.mark.parametrize("what", ["prevalence 1.48%", "a share of 12%", "2,717 fraud users",
                                  "12 fraud users", "mean degree 1.90"])
def test_a_figure_in_what_is_refused(root, what):
    with pytest.raises(L.LedgerError):
        L.record_look(root / "ledger.csv", **look(root, what=what))


@pytest.mark.parametrize("what", ["gated batch, mode gate3", "per ADR-015's batch", "the GNN arm"])
def test_digits_inside_names_are_not_figures(root, what):
    L.record_look(root / "ledger.csv", **look(root, what=what))


def test_the_row_is_flushed_to_disk_before_returning(root, monkeypatch):
    synced = []
    monkeypatch.setattr(L.os, "fsync", lambda fd: synced.append(fd))
    L.record_look(root / "ledger.csv", **look(root))
    assert synced


# ---- authority: accepted, accepted by then, and names the look's script ---------------------------

@pytest.mark.parametrize("tier", ["labels", "scores"])
@pytest.mark.parametrize("authority", [PROPOSED, LATE, SILENT, "decisions/ADR-999-missing.md"])
def test_labels_and_scores_need_a_document_that_named_the_look_first(root, tier, authority):
    p = root / "ledger.csv"
    with pytest.raises(L.LedgerError):
        L.record_look(p, **look(root, tier=tier, authority=authority))
    assert not p.exists()


def test_a_structure_look_needs_a_reason_not_a_document(root):
    L.record_look(root / "ledger.csv", **look(root, tier="structure", what="edge dates",
                                               authority="engineering: the feature transform"))


def test_an_incidental_look_is_recorded_and_says_it_had_no_authority(root):
    p = root / "ledger.csv"
    L.record_look(p, **look(root, authority="none: a reviewer met a disclosed figure"), incidental=True)
    assert L.read_ledger(p)[0]["authority"].startswith("none")


def test_an_incidental_look_cannot_claim_an_authority(root):
    with pytest.raises(L.LedgerError):
        L.record_look(root / "ledger.csv", **look(root), incidental=True)


# ---- the chokepoint -------------------------------------------------------------------------------

def test_the_ordinary_paths_carry_no_held_out_part():
    assert set(L.splits(CONFIG)) == {"val"}
    assert "test_mask" not in L.load({"x": 1, "train_mask": 2, "val_mask": 3, "test_mask": 4})


def test_opening_the_test_window_records_first_then_returns_it(root):
    p = root / "ledger.csv"
    w = L.open_test_window(CONFIG, tier="scores", authority=PREREG, by="researcher", via=RUNNER,
                           ledger=p, repo_root=root, git_commit="c", what="gated batch", now=at(9))
    assert (w.lo, w.hi) == (482, 821)
    assert L.read_ledger(p)[0]["held_out_set"] == TEST


def test_opening_the_official_mask_records_before_the_data_is_loaded(root):
    p = root / "ledger.csv"
    seen = []

    def load_raw():
        seen.append(len(L.read_ledger(p)) if p.exists() else 0)
        return {"test_mask": "mask"}

    m = L.open_official_test_mask(load_raw, tier="scores", authority=PREREG, by="r", via=RUNNER,
                                  ledger=p, repo_root=root, git_commit="c", what="positioning",
                                  now=at(9))
    assert m == "mask" and seen == [1]


@pytest.mark.parametrize("opener", ["window", "mask"])
def test_an_unauthorised_open_records_nothing_and_returns_nothing(root, opener):
    p = root / "ledger.csv"
    loaded = []
    with pytest.raises(L.LedgerError):
        if opener == "window":
            L.open_test_window(CONFIG, tier="labels", authority=PROPOSED, by="r", via=RUNNER,
                               ledger=p, repo_root=root, git_commit="c", what="a count", now=at(9))
        else:
            L.open_official_test_mask(lambda: loaded.append(1) or {"test_mask": 1}, tier="labels",
                                      authority=PROPOSED, by="r", via=RUNNER, ledger=p,
                                      repo_root=root, git_commit="c", what="a count", now=at(9))
    assert not p.exists() and not loaded


# ---- append-only, and what git_dirty may ignore ------------------------------------------------------

def _rows(*hours):
    return [{"timestamp_utc": at(9, h).isoformat(), "held_out_set": TEST, "tier": "labels",
             "what": "a look", "by": "x", "via": "x", "authority": "ADR-011", "git_commit": "c"}
            for h in hours]


def test_append_only_accepts_an_appended_row():
    assert L.is_append_only(L.csv_text(_rows(1, 2)), L.csv_text(_rows(1, 2, 3)))


@pytest.mark.parametrize("working", [
    L.csv_text(_rows(1)), L.csv_text(_rows(2, 1)), L.csv_text(_rows(0, 1, 2)),
    L.csv_text(_rows(1, 2)).replace("ADR-011", "ADR-012", 1),
])
def test_append_only_refuses_any_other_change(working):
    assert not L.is_append_only(L.csv_text(_rows(1, 2)), working)


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_git_dirty_may_ignore_an_append_to_the_ledger_and_nothing_else(root):
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "experiments").mkdir()
    ledger = root / L.LEDGER_REL
    L.record_look(ledger, **look(root))
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "init")
    assert L.ledger_change_is_append(root)
    L.record_look(ledger, **look(root, now=at(10)))          # a guard row at batch start
    assert L.ledger_change_is_append(root)
    ledger.write_text(ledger.read_text(encoding="utf-8").replace("researcher", "someone"),
                      encoding="utf-8")
    assert not L.ledger_change_is_append(root), "a rewritten row must make the tree dirty"


def test_today_git_dirty_counts_a_ledger_append_as_code(root):
    sys.path.insert(0, str(REPO))
    from gbe.run.config import git_dirty
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "experiments").mkdir()
    ledger = root / L.LEDGER_REL
    L.record_look(ledger, **look(root))
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "init")
    L.record_look(ledger, **look(root, now=at(10)))
    assert git_dirty(root)


# ---- disclosure -----------------------------------------------------------------------------------

def test_disclosure_covers_overlapping_sets_earlier_rows_in_time_order(root):
    p = root / "ledger.csv"
    L.record_look(p, **look(root, what="second look", now=at(9, 11)))
    L.record_look(p, **look(root, what="first look", now=at(9, 10)))
    L.record_look(p, **look(root, held_out_set=OFFICIAL, what="official look", now=at(9, 9)))
    L.record_look(p, **look(root, what="the batch itself", now=at(9, 12)))
    lines = L.disclosure(L.read_ledger(p), TEST, before=at(9, 12).isoformat(timespec="seconds"))
    assert [ln.split(" · ")[3].split(" (")[0] for ln in lines] == ["official look", "first look", "second look"]


# ---- the backstop lint ----------------------------------------------------------------------------

def test_the_backstop_finds_each_raw_route_and_ignores_strings():
    srcs = {
        "scripts/a.py": "data = DGraphFin(root=r)[0]\n",
        "scripts/b.py": "m = data.official_test_mask\n",
        "scripts/c.py": "m = d['test_mask']\n",
        "scripts/d.py": "s = TemporalSplit(train_max=1, test_min=2, test_max=3)\n",
        "scripts/e.py": "g = load_scores(p)\n",
        "scripts/f.py": "record_look(p, incidental=flag)\n",
        "notebooks/g.py": "PAT = 'DGraphFin( and test_mask'\n",
        "notebooks/k.py": "data = DGraphFin(root=r)[0]\n",
        "scripts/h.py": "w = open_test_window(c, tier='scores')\ng = load_scores(p)\n",
        "tests/i.py": "data = DGraphFin(root=r)[0]\n",
        "scripts/test_j.py": "data = DGraphFin(root=r)[0]\n",
        "notebooks/measurements/x/prototype/p.py": "m = raw['test_mask']\n",
        "adapters/dgf1/eval.py": "s = TemporalSplit(1, 2, 3)\n",
        L.HAND_PATH: "record_look(p, incidental=True)\n",
    }
    assert L.backstop(srcs) == [
        "notebooks/k.py: PyG loader", "scripts/a.py: PyG loader", "scripts/b.py: official test mask",
        "scripts/c.py: official test mask", "scripts/d.py: hand-built window",
        "scripts/e.py: score files without an accessor", "scripts/f.py: claims an incidental look",
    ]


def test_the_backstop_on_committed_code_finds_the_known_routes():
    """Measured: the routes in committed code today, before row 5a moves them behind accessors."""
    names = subprocess.run(["git", "ls-files", "scripts/*.py", "adapters/dgf1/*.py", "notebooks/*.py"],
                           cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    srcs = {n: (REPO / n).read_text(encoding="utf-8") for n in names}
    assert L.backstop(srcs) == [
        "scripts/assemble_gate_dgf1_1.py: score files without an accessor",
        "scripts/measure_dgf1_temporal_split.py: PyG loader",
        "scripts/measure_dgf1_temporal_split.py: official test mask",
        "scripts/run_dgf1_gnn.py: score files without an accessor",
        "scripts/verify_dgraph_snapshot.py: PyG loader",
        "scripts/verify_dgraph_snapshot.py: official test mask",
    ]
