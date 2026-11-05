"""What the default paths leave out, and the accessors that hand it over (ADR-021 I6 to I9, row 5a step 3).

The guarantee is what the default load leaves out, not who calls what. Everything here runs on a
synthetic snapshot and a temporary git repository: it reads no real data and writes no real ledger.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import adapters.dgf1.datasource_dgraph as DS  # noqa: E402
import adapters.dgf1.heldout as H  # noqa: E402
from adapters.dgf1.eval import dgf1_splits  # noqa: E402

TEST, OFFICIAL = H.TEMPORAL_TEST, H.OFFICIAL_TEST
RUNNER, ASSEMBLER, OTHER = "scripts/run_gate3.py", "scripts/assemble_gate3.py", "scripts/other.py"
PREREG, PROPOSED = "decisions/ADR-900-gate3-prereg.md", "decisions/ADR-901-proposed.md"
MASKS = ("train_mask", "val_mask", "test_mask",
         "official_train_mask", "official_val_mask", "official_test_mask")
# Users 0 to 7. Node times: two at 100 (training), two at 400 (validation), four at 500 or 700 (test).
Y = [0, 1, 0, 2, 1, 0, 3, 1]
H_ = DS.HIDDEN


def raw_snapshot() -> Data:
    return Data(
        x=torch.zeros(8, DS.EXPECTED_FEATURES),
        edge_index=torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]]),
        edge_time=torch.tensor([100, 400, 500, 700]),
        edge_type=torch.ones(4, dtype=torch.long),
        y=torch.tensor(Y),
        train_mask=torch.tensor([True, True, False, False, False, False, False, False]),
        val_mask=torch.tensor([False, False, True, False, False, False, False, False]),
        test_mask=torch.tensor([False, False, False, False, True, True, False, True]),
        num_nodes=8,
    )


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "core.autocrlf=false",
                    *args], cwd=cwd, check=True, capture_output=True)


def run_as(script, monkeypatch):
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(H.REPO_ROOT / script), raising=False)


@pytest.fixture
def snapshot(monkeypatch):
    """The synthetic snapshot behind both modules, counting how often it is loaded."""
    loads = []

    def load(root=None):
        loads.append(True)
        return raw_snapshot()

    monkeypatch.setattr(DS, "_load_raw", load)
    monkeypatch.setattr(H, "_load_raw", load)
    return loads


@pytest.fixture
def root(tmp_path, monkeypatch, snapshot):
    for d in ("decisions", "scripts", "experiments"):
        (tmp_path / d).mkdir()
    for s in (RUNNER, ASSEMBLER, OTHER):
        (tmp_path / s).write_text("# a script\n", encoding="utf-8")
    head = "# x\n\n**Status:** {}\n**Date:** proposed 2026-10-01 · **accepted 2026-10-05** by coderback\n"
    lines = "".join(f"**Held-out look:** {s} · scores · {v} · {b}\n"
                    for s, v, b in ((TEST, RUNNER, "gate3"), (TEST, ASSEMBLER, "gate3"), (OFFICIAL, RUNNER, "official-1")))
    (tmp_path / PREREG).write_text(head.format("accepted") + lines, encoding="utf-8")
    (tmp_path / PROPOSED).write_text(head.format("proposed") + lines, encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "docs")
    monkeypatch.setattr(H, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(H, "LEDGER", tmp_path / "experiments" / "heldout_ledger.csv")
    monkeypatch.setattr(H, "_clock", lambda: datetime(2026, 10, 20, 9, tzinfo=timezone.utc))
    run_as(RUNNER, monkeypatch)
    return tmp_path


def window_look(**over):
    return {**dict(tier="scores", what="gated batch scores", by="researcher", authority=PREREG, batch="gate3"), **over}


def official_look(**over):
    return {**dict(tier="scores", what="positioning scores", by="researcher", authority=PREREG, batch="official-1"), **over}


# ------------------------------------------------------------------ I6: what the default paths carry


def test_the_default_splits_give_validation_only():
    assert set(dgf1_splits()) == {"val"}


def test_the_default_load_hides_every_label_after_the_validation_window(snapshot):
    assert DS.load_dgraph(strict=False).y.tolist() == [0, 1, 0, 2, H_, H_, H_, H_]


def test_labelled_mask_marks_no_user_after_the_validation_window(snapshot):
    assert DS.load_dgraph(strict=False).labelled_mask.tolist() == [True, True, True, False, False, False, False, False]


def test_the_default_load_carries_no_official_mask(snapshot):
    assert not set(MASKS) & set(DS.load_dgraph(strict=False).keys())


def test_the_hiding_starts_right_after_the_validation_window():
    node_time = torch.tensor([481, 482])
    assert DS.hide_later_labels(torch.tensor([0, 1]), node_time, 481).tolist() == [0, DS.HIDDEN]


def test_the_hidden_value_is_no_real_label():
    assert isinstance(DS.HIDDEN, int) and DS.HIDDEN not in (DS.NORMAL, DS.FRAUD, *DS.BACKGROUND)


def test_hiding_changes_no_training_or_validation_row(snapshot):
    raw = raw_snapshot()
    full = DS.prepare_dgraph(raw.x, raw.edge_index, raw.edge_time, raw.edge_type, raw.y)
    hidden = DS.load_dgraph(strict=False)
    val = dgf1_splits()["val"]
    for mask in (DS.train_seed_mask, DS.window_target_mask):
        assert torch.equal(mask(full, val), mask(hidden, val))
        assert torch.equal(full.y[mask(full, val)], hidden.y[mask(hidden, val)])
    assert torch.equal(full.node_time, hidden.node_time)
    assert int(DS.window_target_mask(hidden, val).sum()) == 1, "the fixture has a validation target"


# ------------------------------------------------------------------ I7: open_test_window


def test_a_scores_open_records_first_then_returns_the_window_and_labels(root, monkeypatch):
    rows_at_load = []

    def load():
        rows_at_load.append(len(H.read_ledger()))
        return raw_snapshot()

    monkeypatch.setattr(H, "_load_raw", load)
    window, y = H.open_test_window(**window_look())
    assert rows_at_load == [1], "the row is on disk before the data is read"
    assert (window.train_max, window.test_min, window.test_max) == (369, 482, 821)
    assert y.tolist() == Y
    assert H.read_ledger()[0]["held_out_set"] == TEST


def test_a_labels_open_returns_the_labels_under_a_scores_line(root):
    _, y = H.open_test_window(**window_look(tier="labels"))
    assert y.tolist() == Y


def test_a_structure_open_returns_the_window_and_reads_no_labels(root, snapshot):
    window, y = H.open_test_window(**window_look(tier="structure", what="edge dates", authority="engineering: a size",
                                                 batch="eng"))
    assert y is None and not snapshot, "a structure look loads nothing"
    assert window.test_min == 482 and H.read_ledger()[0]["tier"] == "structure"


def test_with_labels_rebuilds_labelled_mask_so_the_window_has_targets(root, snapshot):
    hidden = DS.load_dgraph(strict=False)
    window, y = H.open_test_window(**window_look())
    assert not DS.window_target_mask(hidden, window).any(), "no target before the labels go back"
    scored = DS.with_labels(hidden, y)
    assert scored.y.tolist() == Y
    assert DS.window_target_mask(scored, window).tolist() == [False, False, False, False, True, True, False, True]
    assert hidden.y.tolist() == [0, 1, 0, 2, H_, H_, H_, H_], "the default graph keeps its hidden labels"
    assert not hidden.labelled_mask[4:].any()


def test_an_open_records_the_running_script_as_via(root, monkeypatch):
    run_as(ASSEMBLER, monkeypatch)
    H.open_test_window(**window_look(what="assembly"))
    assert H.read_ledger()[0]["via"] == ASSEMBLER


@pytest.mark.parametrize("opener,look", [(H.open_test_window, window_look), (H.open_official_track, official_look)])
@pytest.mark.parametrize("hand", ["now", "incidental", "backfill", "via", "commit"])
def test_the_openers_take_nothing_only_the_hand_path_may_give(root, opener, look, hand):
    with pytest.raises(TypeError):
        opener(**look(), **{hand: None})


@pytest.mark.parametrize("opener,look", [(H.open_test_window, window_look), (H.open_official_track, official_look)])
def test_an_unauthorised_open_records_nothing_and_loads_nothing(root, snapshot, opener, look):
    with pytest.raises(H.LedgerError, match="accepted"):
        opener(**look(authority=PROPOSED))
    assert not H.LEDGER.exists() and not snapshot


def test_a_script_the_document_does_not_name_cannot_open_the_window(root, snapshot, monkeypatch):
    run_as(OTHER, monkeypatch)
    with pytest.raises(H.LedgerError, match="Held-out look"):
        H.open_test_window(**window_look())
    assert not H.LEDGER.exists() and not snapshot


# ------------------------------------------------------------------ I8: open_official_track


def test_the_official_track_records_first_then_returns_every_label_and_mask(root, monkeypatch):
    rows_at_load = []

    def load():
        rows_at_load.append(len(H.read_ledger()))
        return raw_snapshot()

    monkeypatch.setattr(H, "_load_raw", load)
    out = H.open_official_track(**official_look())
    raw = raw_snapshot()
    assert rows_at_load == [1]
    assert out["y"].tolist() == Y
    assert torch.equal(out["official_train_mask"], raw.train_mask)
    assert torch.equal(out["official_val_mask"], raw.val_mask)
    assert torch.equal(out["official_test_mask"], raw.test_mask)
    assert H.read_ledger()[0]["held_out_set"] == OFFICIAL


def test_the_official_track_refuses_the_structure_tier(root, snapshot):
    with pytest.raises(H.LedgerError, match="label-derived"):
        H.open_official_track(**official_look(tier="structure", authority="engineering: a size"))
    assert not H.LEDGER.exists() and not snapshot


# ------------------------------------------------------------------ I9: the integrity check


TOTALS = {"normal": 3, "fraud": 3, "background": 2}


def test_the_integrity_check_takes_no_argument():
    assert not inspect.signature(H.integrity_ok).parameters


def test_the_integrity_check_says_only_pass_or_fail(snapshot, monkeypatch):
    monkeypatch.setattr(H, "PUBLISHED_TOTALS", TOTALS)
    assert H.integrity_ok() is True


@pytest.mark.parametrize("wrong", ["normal", "fraud", "background"])
def test_the_integrity_check_fails_on_any_wrong_total(snapshot, monkeypatch, wrong):
    monkeypatch.setattr(H, "PUBLISHED_TOTALS", {**TOTALS, wrong: TOTALS[wrong] + 1})
    assert H.integrity_ok() is False


def test_the_integrity_check_records_no_look(root, monkeypatch):
    monkeypatch.setattr(H, "PUBLISHED_TOTALS", TOTALS)
    H.integrity_ok()
    assert not H.LEDGER.exists()


def test_the_pinned_totals_are_the_published_ones():
    line = (REPO_ROOT / "decisions" / "ADR-010-dgraph-snapshot-reconciliation.md").read_text(encoding="utf-8").splitlines()[23]
    for name, total in H.PUBLISHED_TOTALS.items():
        assert f"{total:,} {name}" in line
