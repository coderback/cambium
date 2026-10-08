"""Tests for the held-out ledger prototype v5. Run: python -m pytest -q test_ledger_proto.py"""
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
RUNNER, ASSEMBLER, OTHER = "scripts/run_gate3.py", "scripts/assemble_gate3.py", "scripts/other.py"
PREREG, PROPOSED = "decisions/ADR-900-gate3-prereg.md", "decisions/ADR-901-proposed.md"
LATE, PROSE = "decisions/ADR-902-accepted-later.md", "decisions/ADR-903-prose-only.md"
SAMEDAY = "decisions/ADR-904-accepted-same-day.md"
AMENDED, NIGHT = "decisions/ADR-905-amended-later.md", "decisions/ADR-906-accepted-late-evening.md"
EARLY = "decisions/ADR-908-accepted-early-morning.md"
BATCHES = ("gate3-stage1", "gate3", "earlier", "o", "official-1", "official-2", "other")
CONFIG = {"val_min": 370, "val_max": 481, "test_min": 482, "test_max": 821}
RAW = {"y": [0, 1, 1, 0, 3], "node_time": [100, 400, 500, 700, 600],
       "train_mask": [1, 0, 1, 0, 0], "val_mask": [0, 1, 0, 0, 0], "test_mask": [0, 0, 0, 1, 0],
       "x": "feats"}


def at(day: int, h: int = 9) -> datetime:
    return datetime(2026, 10, day, h, tzinfo=timezone.utc)


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def looks(*triples):
    return "".join(f"**Held-out look:** {s} · {t} · {v} · {b}\n" for s, t, v in triples for b in BATCHES)


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "decisions").mkdir()
    (tmp_path / "scripts").mkdir()
    for s in (RUNNER, ASSEMBLER, OTHER):
        (tmp_path / s).write_text("# a script\n", encoding="utf-8")
    head = "# x\n\n**Status:** {}\n**Date:** proposed 2026-10-01 · **accepted {}** by coderback\n"
    granted = looks((TEST, "scores", RUNNER), (TEST, "scores", ASSEMBLER), (OFFICIAL, "scores", RUNNER))
    docs = {PREREG: ("accepted", "2026-10-05", granted),
            PROPOSED: ("proposed", "2026-10-05", granted),
            LATE: ("accepted", "2026-10-20", granted),
            PROSE: ("accepted", "2026-10-05", f"Runs {RUNNER} on the {TEST} set.\n"),
            SAMEDAY: ("accepted", "2026-10-09", granted),
            AMENDED: ("accepted · amended 2026-10-09 by ADR-907, at the end", "2026-10-05", granted),
            NIGHT: ("accepted", "2026-10-08 (23:00)", granted),
            EARLY: ("accepted", "2026-10-09 (00:30)", granted)}
    for rel, (status, on, body) in docs.items():
        (tmp_path / rel).write_text(head.format(status, on) + body, encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "docs")
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(tmp_path / RUNNER), raising=False)
    monkeypatch.setattr(L, "_clock", lambda: at(20))
    return tmp_path


def look(root, **over):
    base = dict(held_out_set=TEST, tier="scores", what="gated batch scores", by="researcher",
                via=RUNNER, authority=PREREG, batch="gate3-stage1", git_commit="abc1234",
                repo_root=root, now=at(9))
    return {**base, **over}


def opener(root, **over):
    base = dict(tier="scores", authority=PREREG, by="researcher", what="gated batch",
                batch="gate3-stage1", ledger=root / "ledger.csv", repo_root=root, git_commit="c")
    return {**base, **over}


# ---- record_look -------------------------------------------------------------------------------

def test_rows_append_under_one_header(root):
    p = root / "ledger.csv"
    L.record_look(p, **look(root))
    L.record_look(p, **look(root, tier="structure", what="edge dates", authority="engineering", batch="eng"))
    assert p.read_text(encoding="utf-8").count("timestamp_utc,") == 1
    rows = L.read_ledger(p)
    assert [r["tier"] for r in rows] == ["scores", "structure"] and set(rows[0]) == set(L.COLUMNS)


@pytest.mark.parametrize("field,value,why", [
    ("held_out_set", "dgf1-validation-370-481", "unknown held-out set"), ("tier", "peek", "tier must be"),
    ("what", " ", "what is empty"), ("by", "", "by is empty"), ("authority", "", "authority is empty"),
    ("batch", "", "batch is empty"),
])
def test_bad_fields_are_refused_and_nothing_is_written(root, field, value, why):
    """At the structure tier, so no authority check can refuse the row for another reason."""
    p = root / "ledger.csv"
    base = look(root, tier="structure", what="edge dates", authority="engineering")
    with pytest.raises(L.LedgerError, match=why):
        L.record_look(p, **{**base, field: value})
    assert not p.exists()


@pytest.mark.parametrize("what", ["prevalence 1.48%", "a share of 12%", "2,717 fraud users",
                                  "12 fraud users", "mean degree 1.90"])
def test_a_figure_in_what_is_refused(root, what):
    with pytest.raises(L.LedgerError, match="figure"):
        L.record_look(root / "ledger.csv", **look(root, what=what))


@pytest.mark.parametrize("what", ["gated batch, mode gate3", "per ADR-015's batch", "the GNN arm"])
def test_digits_inside_names_are_not_figures(root, what):
    L.record_look(root / "ledger.csv", **look(root, what=what))


def test_the_row_is_flushed_to_disk_before_returning(root, monkeypatch):
    synced = []
    monkeypatch.setattr(L.os, "fsync", lambda fd: synced.append(fd))
    L.record_look(root / "ledger.csv", **look(root))
    assert synced


# ---- authority -------------------------------------------------------------------------------------

@pytest.mark.parametrize("tier", ["labels", "scores"])
@pytest.mark.parametrize("authority,why", [
    (PROPOSED, "not accepted"), (LATE, "night before"), (SAMEDAY, "night before"),
    (PROSE, "Held-out look"), ("decisions/ADR-999-missing.md", "not in HEAD"),
])
def test_labels_and_scores_need_a_document_that_named_the_look_first(root, tier, authority, why):
    p = root / "ledger.csv"
    with pytest.raises(L.LedgerError, match=why):
        L.record_look(p, **look(root, tier=tier, authority=authority))
    assert not p.exists()


def test_a_scores_line_covers_a_labels_look_but_not_another_set_or_script(root, monkeypatch):
    L.record_look(root / "ledger.csv", **look(root, tier="labels"))
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(root / ASSEMBLER))
    with pytest.raises(L.LedgerError, match="Held-out look"):     # granted on TEST only
        L.record_look(root / "ledger.csv", **look(root, held_out_set=OFFICIAL, via=ASSEMBLER,
                                                   now=at(9, 10)))
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(root / OTHER))
    with pytest.raises(L.LedgerError, match="Held-out look"):
        L.record_look(root / "ledger.csv", **look(root, via=OTHER, now=at(9, 11)))


@pytest.mark.parametrize("authority,refused,allowed", [
    (AMENDED, at(10, 9), at(10, 13)),     # the amendment's date counts, and its end of day
    (NIGHT, at(9, 9), at(9, 11)),         # the next day, but twelve hours after the recorded time
    (EARLY, at(9, 13), at(10, 9)),        # twelve hours on, but still the same calendar day
])
def test_the_wait_runs_from_the_latest_change_and_lasts_twelve_hours(root, authority, refused, allowed):
    p = root / "ledger.csv"
    with pytest.raises(L.LedgerError, match="night before"):
        L.record_look(p, **look(root, authority=authority, now=refused))
    L.record_look(p, **look(root, authority=authority, now=allowed))


def test_the_authority_line_must_name_the_batch(root):
    with pytest.raises(L.LedgerError, match="Held-out look"):
        L.record_look(root / "ledger.csv", **look(root, batch="unlisted"))


def test_a_batch_label_belongs_to_one_authority(root):
    p = root / "ledger.csv"
    L.record_look(p, **look(root))
    with pytest.raises(L.LedgerError, match="another authority"):
        L.record_look(p, **look(root, tier="structure", what="edge dates", authority="engineering"))
    assert len(L.read_ledger(p)) == 1


def test_a_look_may_not_be_dated_after_the_clock(root):
    with pytest.raises(L.LedgerError, match="later than the clock"):
        L.record_look(root / "ledger.csv", **look(root, now=at(21)))


def test_via_must_be_the_running_script(root):
    with pytest.raises(L.LedgerError, match="running script"):
        L.record_look(root / "ledger.csv", **look(root, via=ASSEMBLER))


def test_an_interactive_session_cannot_make_a_look(root, monkeypatch):
    monkeypatch.delattr(sys.modules["__main__"], "__file__", raising=False)
    with pytest.raises(L.LedgerError, match="interactive"):
        L.current_script(root)


def test_the_authority_is_read_from_head_not_the_working_tree(root):
    doc = root / PROPOSED
    doc.write_text(doc.read_text(encoding="utf-8").replace("proposed\n", "accepted\n", 1), encoding="utf-8")
    with pytest.raises(L.LedgerError, match="not accepted"):
        L.record_look(root / "ledger.csv", **look(root, authority=PROPOSED))


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


# ---- backfill ---------------------------------------------------------------------------------------

def test_a_backfill_row_keeps_its_authority_unchecked(root):
    p = root / "ledger.csv"
    L.record_look(p, **look(root, authority="ADR-012", via="backfill: gate batch", now=at(1)), backfill=True)
    assert L.read_ledger(p)[0]["authority"] == "ADR-012"


def test_a_backfill_row_still_refuses_a_figure(root):
    with pytest.raises(L.LedgerError, match="figure"):
        L.record_look(root / "ledger.csv", **look(root, what="prevalence 1.48%", now=at(1)), backfill=True)


def test_a_backfill_row_cannot_record_a_look_from_after_the_cutoff(root):
    with pytest.raises(L.LedgerError, match="before ADR-021"):
        L.record_look(root / "ledger.csv", **look(root, authority="ADR-012", now=L.BACKFILL_BEFORE),
                      backfill=True)


# ---- the default paths ---------------------------------------------------------------------------

def test_the_default_paths_carry_no_held_out_label_or_mask():
    assert set(L.splits(CONFIG)) == {"val"}
    data = L.load(RAW, CONFIG)
    assert not {"train_mask", "val_mask", "test_mask"} & set(data), "every official mask is label-derived"
    assert data["y"] == [0, 1, L.HIDDEN, L.HIDDEN, L.HIDDEN]
    assert data["node_time"] == RAW["node_time"] and data["x"] == RAW["x"]


def test_labelled_mask_marks_no_user_after_the_validation_window():
    """Round 4's B1: built from the raw labels, it would say which hidden users are labelled."""
    data = L.load(RAW, CONFIG)
    late = [t > CONFIG["val_max"] for t in data["node_time"]]
    assert L.labelled(RAW["y"]) != L.labelled(data["y"]), "the fixture must tell the two apart"
    assert data["labelled_mask"] == L.labelled(data["y"])
    assert not any(m and t for m, t in zip(data["labelled_mask"], late))


def test_with_labels_rebuilds_labelled_mask_so_the_window_has_targets(root):
    """A runner that put back only y would score no test user: targets come from labelled_mask."""
    data = L.load(RAW, CONFIG)
    w, y = L.open_test_window(CONFIG, lambda: RAW, **opener(root))
    scored = L.with_labels(data, y)
    targets = [i for i, (m, t) in enumerate(zip(scored["labelled_mask"], scored["node_time"]))
               if m and w.lo <= t <= w.hi]
    assert scored["y"] == RAW["y"] and targets == [2, 3]


def test_the_hidden_value_is_no_real_label():
    assert isinstance(L.HIDDEN, int) and L.HIDDEN not in (0, 1, 2, 3)


def test_the_integrity_check_says_only_pass_or_fail(monkeypatch):
    assert L.integrity_ok(lambda: RAW) is False, "no pinned totals, no pass"
    monkeypatch.setattr(L, "PUBLISHED", {0: 2, 1: 2, 3: 1})
    assert L.integrity_ok(lambda: RAW) is True
    monkeypatch.setattr(L, "PUBLISHED", {0: 2, 1: 3})
    assert L.integrity_ok(lambda: RAW) is False


# ---- the accessors --------------------------------------------------------------------------------

def _counting_loader(p):
    seen = []

    def load_raw():
        seen.append(len(L.read_ledger(p)) if p.exists() else 0)
        return RAW
    return load_raw, seen


def test_a_scores_open_of_the_window_records_first_then_returns_labels(root):
    p = root / "ledger.csv"
    load_raw, seen = _counting_loader(p)
    w, y = L.open_test_window(CONFIG, load_raw, **opener(root))
    assert (w.lo, w.hi) == (482, 821) and y == RAW["y"] and seen == [1]
    row = L.read_ledger(p)[0]
    assert row["via"] == RUNNER and row["timestamp_utc"] == at(20).isoformat(), "the clock's time"


def test_the_openers_take_no_time_from_their_caller(root):
    for fn in (L.open_test_window, L.open_official_track):
        with pytest.raises(TypeError):
            fn(CONFIG, lambda: RAW, **opener(root), now=at(1)) if fn is L.open_test_window \
                else fn(lambda: RAW, **opener(root), now=at(1))


def test_an_open_records_the_running_script_as_via(root, monkeypatch):
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(root / ASSEMBLER))
    p = root / "ledger.csv"
    L.open_test_window(CONFIG, lambda: RAW, **opener(root, what="assembly"))
    assert L.read_ledger(p)[0]["via"] == ASSEMBLER


def test_a_structure_open_of_the_window_returns_no_labels(root):
    w, y = L.open_test_window(CONFIG, lambda: RAW, **opener(root, tier="structure",
                                                              authority="engineering: views"))
    assert y is None


def test_the_official_track_returns_every_label_and_mask_after_recording(root):
    p = root / "ledger.csv"
    load_raw, seen = _counting_loader(p)
    out = L.open_official_track(load_raw, **opener(root, what="positioning", batch="official-1"))
    assert out["y"] == RAW["y"] and all(out[m] == RAW[m] for m in L.OFFICIAL_MASKS) and seen == [1]
    assert L.read_ledger(p)[0]["held_out_set"] == OFFICIAL


def test_the_official_track_refuses_a_structure_open(root):
    with pytest.raises(L.LedgerError, match="label-derived"):
        L.open_official_track(lambda: RAW, **opener(root, tier="structure", authority="engineering"))


@pytest.mark.parametrize("which", ["window", "official"])
def test_an_unauthorised_open_records_nothing_and_loads_nothing(root, which):
    p = root / "ledger.csv"
    loaded = []

    def load_raw():
        loaded.append(1)
        return RAW

    with pytest.raises(L.LedgerError):
        if which == "window":
            L.open_test_window(CONFIG, load_raw, **opener(root, tier="labels", authority=PROPOSED))
        else:
            L.open_official_track(load_raw, **opener(root, tier="labels", authority=PROPOSED))
    assert not p.exists() and not loaded


# ---- append-only, and what git_dirty may ignore ------------------------------------------------------

def _rows(*hours):
    return [{"timestamp_utc": at(9, h).isoformat(), "held_out_set": TEST, "tier": "labels",
             "what": "a look", "by": "x", "via": "x", "authority": "ADR-011", "batch": "b",
             "git_commit": "c"} for h in hours]


def test_append_only_accepts_an_appended_row():
    assert L.is_append_only(L.csv_text(_rows(1, 2)), L.csv_text(_rows(1, 2, 3)))


@pytest.mark.parametrize("working", [
    L.csv_text(_rows(1)), L.csv_text(_rows(2, 1)), L.csv_text(_rows(0, 1, 2)),
    L.csv_text(_rows(1, 2)).replace("ADR-011", "ADR-012", 1),
])
def test_append_only_refuses_any_other_change(working):
    assert not L.is_append_only(L.csv_text(_rows(1, 2)), working)


def test_git_dirty_may_ignore_an_append_to_the_ledger_and_nothing_else(root):
    (root / "experiments").mkdir()
    ledger = root / L.LEDGER_REL
    L.record_look(ledger, **look(root))
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "ledger")
    assert L.ledger_change_is_append(root)
    L.record_look(ledger, **look(root, now=at(10)))          # an opener row at batch start
    assert L.ledger_change_is_append(root)
    ledger.write_text(ledger.read_text(encoding="utf-8").replace("researcher", "someone"),
                      encoding="utf-8")
    assert not L.ledger_change_is_append(root), "a rewritten row must make the tree dirty"


def test_today_git_dirty_counts_a_ledger_append_as_code(root):
    sys.path.insert(0, str(REPO))
    from gbe.run.config import git_dirty
    (root / "experiments").mkdir()
    ledger = root / L.LEDGER_REL
    L.record_look(ledger, **look(root))
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "ledger")
    L.record_look(ledger, **look(root, now=at(10)))
    assert git_dirty(root)


# ---- disclosure -----------------------------------------------------------------------------------

def test_disclosure_covers_overlapping_sets_earlier_rows_in_time_order(root):
    p = root / "ledger.csv"
    L.record_look(p, **look(root, what="second look", batch="earlier", now=at(9, 11)))
    L.record_look(p, **look(root, what="first look", batch="earlier", now=at(9, 10)))
    L.record_look(p, **look(root, held_out_set=OFFICIAL, what="official look", batch="o", now=at(9, 9)))
    L.record_look(p, **look(root, what="the batch itself", batch="gate3", now=at(9, 12)))
    lines = L.disclosure(L.read_ledger(p), TEST, batch="gate3")
    assert [ln.split(" · ")[3].split(" (")[0] for ln in lines] == ["official look", "first look", "second look"]


def test_two_batches_under_one_document_stay_apart(root):
    p = root / "ledger.csv"
    L.record_look(p, **look(root, what="first report", batch="official-1", now=at(9, 9)))
    L.record_look(p, **look(root, what="second report", batch="official-2", now=at(9, 10)))
    lines = L.disclosure(L.read_ledger(p), TEST, batch="official-2")
    assert [ln.split(" · ")[3].split(" (")[0] for ln in lines] == ["first report"]


def test_disclosure_stops_at_the_batch_s_first_row(root):
    p = root / "ledger.csv"
    L.record_look(p, **look(root, what="batch opens", batch="gate3", now=at(9, 10)))
    L.record_look(p, **look(root, what="another batch", batch="other", now=at(9, 11)))
    L.record_look(p, **look(root, what="batch assembles", batch="gate3", now=at(9, 12)))
    assert L.disclosure(L.read_ledger(p), TEST, batch="gate3") == []


def test_a_batch_with_no_opener_row_cannot_be_disclosed(root):
    p = root / "ledger.csv"
    L.record_look(p, **look(root))
    with pytest.raises(L.LedgerError, match="no opener row"):
        L.disclosure(L.read_ledger(p), TEST, batch="never-opened")


# ---- the backstop lint ----------------------------------------------------------------------------

def test_the_backstop_finds_each_raw_route_and_ignores_strings():
    srcs = {
        "scripts/a.py": "data = DGraphFin(root=r)[0]\n",
        "scripts/b.py": "m = data.official_test_mask\n",
        "scripts/b2.py": "m = data.official_train_mask | data.official_val_mask\n",
        "scripts/c.py": "m = d['test_mask']\n",
        "scripts/d.py": "s = TemporalSplit(train_max=1, test_min=2, test_max=3)\n",
        "scripts/e.py": "g = load_scores(p)\n",
        "scripts/f.py": "record_look(p, incidental=flag)\n",
        "scripts/f2.py": "record_look(p, backfill=True)\n",
        "scripts/f3.py": "record_look(p, now=t)\n",
        "scripts/al.py": "from torch_geometric.datasets import DGraphFin as D\n",
        "scripts/al2.py": "from gbe.eval.scores import load_scores as ls\ng = ls(p)\n",
        "scripts/al3.py": "import x as tsplit\nfrom m import TemporalSplit as TS\ns = TS(1, 2, 3)\n",
        "scripts/raw.py": "from adapters.dgf1.heldout import _load_raw\n",
        "scripts/raw2.py": "import adapters.dgf1.heldout as h\nr = h._load_raw()\n",
        "scripts/sp.py": "a = np.load('data/dgraph/raw/dgraphfin.npz')\n",
        "scripts/sp2.py": "d = ROOT / 'experiments' / 'scores'\n",
        "scripts/sp3.py": "d = 'experiments\\\\scores\\\\run.npz'\n",
        "scripts/nsp.py": "d = ROOT / 'scores' / 'experiments'\nNEVER = ('data/', 'experiments/')\n",
        "scripts/ok.py": "t = data.labelled_mask & train_seed_mask(d, s) & window_target_mask(d, s)\n",
        "notebooks/g.py": "PAT = 'DGraphFin( and test_mask'\n",
        "notebooks/k.py": "data = DGraphFin(root=r)[0]\n",
        "scripts/h.py": "w = open_test_window(c, tier='scores')\ng = load_scores(p)\n",
        "tests/i.py": "data = DGraphFin(root=r)[0]\n",
        "scripts/test_j.py": "data = DGraphFin(root=r)[0]\n",
        "notebooks/measurements/x/prototype/p.py": "m = raw['test_mask']\n",
        "scripts/prototype/q.py": "m = raw['test_mask']\n",
        "adapters/dgf1/eval.py": "s = TemporalSplit(1, 2, 3)\n",
        L.HAND_PATH: "record_look(p, incidental=True, backfill=True)\n",
    }
    hand = "sets a look's authority or time by hand"
    assert L.backstop(srcs) == sorted([
        "notebooks/k.py: PyG loader", "scripts/a.py: PyG loader", "scripts/b.py: an official mask",
        "scripts/b2.py: an official mask", "scripts/c.py: an official mask",
        "scripts/d.py: hand-built window", "scripts/e.py: score files without an accessor",
        f"scripts/f.py: {hand}", f"scripts/f2.py: {hand}", f"scripts/f3.py: {hand}",
        "scripts/al.py: PyG loader", "scripts/al2.py: score files without an accessor",
        "scripts/al3.py: hand-built window",
        "scripts/raw.py: the raw loader", "scripts/raw2.py: the raw loader",
        "scripts/sp.py: a held-out store path", "scripts/sp2.py: a held-out store path",
        "scripts/sp3.py: a held-out store path",
        "scripts/test_j.py: PyG loader",
        "scripts/prototype/q.py: an official mask",
    ])


def test_the_backstop_on_committed_code_finds_the_known_routes():
    """Measured: the routes in committed code today, before row 5a moves them behind accessors."""
    names = subprocess.run(["git", "ls-files", "scripts/*.py", "adapters/dgf1/*.py", "notebooks/*.py"],
                           cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    srcs = {n: (REPO / n).read_text(encoding="utf-8") for n in names}
    assert L.backstop(srcs) == EXPECTED_TODAY


EXPECTED_TODAY = [     # the store paths move into the accessor modules in row 5a (ADR-021 I11)
    "scripts/assemble_gate_dgf1_1.py: score files without an accessor",
    "scripts/audit_dgf1_datasource.py: a held-out store path",
    "scripts/measure_dgf1_temporal_split.py: PyG loader",
    "scripts/measure_dgf1_temporal_split.py: a held-out store path",
    "scripts/measure_dgf1_temporal_split.py: an official mask",
    "scripts/run_dgf1_floor.py: a held-out store path",
    "scripts/run_dgf1_gnn.py: a held-out store path",
    "scripts/run_dgf1_gnn.py: score files without an accessor",
    "scripts/verify_dgf1_temporal_sampler.py: a held-out store path",
    "scripts/verify_dgraph_snapshot.py: PyG loader",
    "scripts/verify_dgraph_snapshot.py: a held-out store path",
    "scripts/verify_dgraph_snapshot.py: an official mask",
]
