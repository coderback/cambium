"""Guard: git_dirty flags uncommitted code/config, not the append-only registry.

The registry is a run output; a tracked registry.csv self-dirties the tree after the
first append in a batch. dirty must mean uncommitted *inputs* (ADR-002). The parser is
tested directly, without a repo. The wrapper that runs git is tested against a temporary
repo, because the audit of 2026-10-07 (A2 E-S8) found a wrapper that always answered
"clean" left the suite green: every other test monkeypatches it.
"""

from __future__ import annotations

import subprocess

import pytest

from gbe.run.config import IGNORED_DIRTY_PATHS, _dirty_paths, git_dirty


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                    "-c", "core.autocrlf=false", *args],
                   cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A committed repo holding one code file and a tracked registry, as cambium does."""
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "registry.csv").write_text("run_id\n", encoding="utf-8")
    (tmp_path / "model.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_wrapper_reports_a_committed_tree_clean(repo):
    assert git_dirty(repo) is False


def test_wrapper_reports_a_code_edit_dirty(repo):
    (repo / "model.py").write_text("x = 2\n", encoding="utf-8")
    assert git_dirty(repo) is True


def test_wrapper_ignores_a_registry_append(repo):
    (repo / "experiments" / "registry.csv").write_text("run_id\nr1\n", encoding="utf-8")
    assert git_dirty(repo) is False


def test_wrapper_reports_an_untracked_source_file_dirty(repo):
    (repo / "new_head.py").write_text("", encoding="utf-8")
    assert git_dirty(repo) is True


def test_wrapper_treats_an_unknown_git_state_as_dirty(tmp_path):
    """Outside a repository nothing can be verified, so the run must not claim it was clean."""
    assert git_dirty(tmp_path) is True


# ------------------------------------------------------------------ the held-out ledger (ADR-021 I5)

LEDGER = "experiments/heldout_ledger.csv"
ROWS = b"timestamp_utc,what\n2026-10-09,first\n2026-10-10,second\n"


def _write(repo, data: bytes, rel: str = LEDGER):
    (repo / rel).write_bytes(data)


@pytest.fixture
def ledger_repo(repo):
    """The repo above, with a committed ledger and a subdirectory to run from."""
    _write(repo, ROWS)
    (repo / "sub").mkdir()
    (repo / "sub" / "keep.py").write_text("", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "ledger")
    return repo


def test_an_unchanged_ledger_is_clean(ledger_repo):
    assert git_dirty(ledger_repo) is False


def test_a_ledger_append_is_clean(ledger_repo):
    _write(ledger_repo, ROWS + b"2026-10-11,third\n")
    assert git_dirty(ledger_repo) is False


def test_the_first_ledger_row_before_any_commit_of_it_is_clean(repo):
    _write(repo, b"timestamp_utc,what\n2026-10-09,first\n")
    assert git_dirty(repo) is False


@pytest.mark.parametrize("rewritten", [
    ROWS.replace(b"first", b"edited"),                                         # a row rewritten
    b"timestamp_utc,what\n2026-10-10,second\n2026-10-09,first\n",              # rows reordered
    b"timestamp_utc,what\n2026-10-09,first\n",                                 # a row removed
    b"timestamp_utc,what\n2026-10-08,earlier\n2026-10-09,first\n2026-10-10,second\n",  # inserted above
    ROWS.replace(b"second", b"other!") + b"2026-10-11,third\n",                # a rewrite the same length, and longer
    b"",                                                                       # emptied
])
def test_any_other_ledger_change_is_dirty(ledger_repo, rewritten):
    _write(ledger_repo, rewritten)
    assert git_dirty(ledger_repo) is True


def test_a_deleted_ledger_is_dirty(ledger_repo):
    (ledger_repo / LEDGER).unlink()
    assert git_dirty(ledger_repo) is True


def test_a_ledger_append_does_not_hide_a_code_edit(ledger_repo):
    _write(ledger_repo, ROWS + b"2026-10-11,third\n")
    (ledger_repo / "model.py").write_text("x = 2\n", encoding="utf-8")
    assert git_dirty(ledger_repo) is True


def test_a_crlf_working_copy_of_an_lf_commit_is_still_an_append(ledger_repo):
    """On this project's checkouts core.autocrlf stores LF and the working copy holds CRLF."""
    assert b"\r" not in ROWS
    _write(ledger_repo, ROWS.replace(b"\n", b"\r\n"))
    assert git_dirty(ledger_repo) is False
    _write(ledger_repo, (ROWS + b"2026-10-11,third\n").replace(b"\n", b"\r\n"))
    assert git_dirty(ledger_repo) is False
    _write(ledger_repo, ROWS.replace(b"first", b"edited").replace(b"\n", b"\r\n"))
    assert git_dirty(ledger_repo) is True


def test_extending_the_last_row_of_a_ledger_with_no_final_newline_is_dirty(repo):
    _write(repo, b"timestamp_utc,what\n2026-10-09,first")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "ledger")
    _write(repo, b"timestamp_utc,what\n2026-10-09,first and edited\n")
    assert git_dirty(repo) is True


def test_the_ledger_is_judged_from_a_subdirectory_too(ledger_repo):
    sub = ledger_repo / "sub"
    assert git_dirty(sub) is False
    _write(ledger_repo, ROWS + b"2026-10-11,third\n")
    assert git_dirty(sub) is False
    _write(ledger_repo, ROWS.replace(b"first", b"edited"))
    assert git_dirty(sub) is True


def test_renaming_code_onto_the_ledger_path_is_dirty(repo):
    """Git reports only the new name of a rename, which would hide the source file's removal."""
    _git(repo, "mv", "model.py", LEDGER)
    assert git_dirty(repo) is True


def test_a_ledger_with_no_head_to_compare_is_dirty(tmp_path):
    (tmp_path / "experiments").mkdir()
    _write(tmp_path, ROWS)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    assert git_dirty(tmp_path) is True


def test_append_only_is_a_parameter(ledger_repo):
    _write(ledger_repo, ROWS + b"2026-10-11,third\n")
    assert git_dirty(ledger_repo, append_only=frozenset()) is True
    _write(ledger_repo, ROWS)
    (ledger_repo / "model.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    assert git_dirty(ledger_repo, append_only=frozenset({"model.py"})) is False


def test_the_registry_stays_ignored_outright(ledger_repo):
    """ADR-002 is unchanged: the registry's own append-only rule is not git_dirty's job."""
    (ledger_repo / "experiments" / "registry.csv").write_text("rewritten\n", encoding="utf-8")
    assert git_dirty(ledger_repo) is False


# ------------------------------------------------------------------ the porcelain parser


def test_registry_only_change_is_not_dirty():
    porcelain = " M experiments/registry.csv\n"
    assert _dirty_paths(porcelain, IGNORED_DIRTY_PATHS) == []


def test_code_change_is_dirty():
    porcelain = " M gbe/run/config.py\n M experiments/registry.csv\n"
    assert _dirty_paths(porcelain, IGNORED_DIRTY_PATHS) == ["gbe/run/config.py"]


def test_untracked_and_renamed_paths_counted():
    porcelain = '?? adapters/ell1/new_head.py\nR  old.py -> gbe/run/session.py\n'
    got = _dirty_paths(porcelain, IGNORED_DIRTY_PATHS)
    assert got == ["adapters/ell1/new_head.py", "gbe/run/session.py"]


def test_empty_tree_is_clean():
    assert _dirty_paths("", IGNORED_DIRTY_PATHS) == []
