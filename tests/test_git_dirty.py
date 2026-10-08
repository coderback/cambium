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
