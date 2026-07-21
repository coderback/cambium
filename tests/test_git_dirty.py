"""Guard: git_dirty flags uncommitted code/config, not the append-only registry.

The registry is a run output; a tracked registry.csv self-dirties the tree after the
first append in a batch. dirty must mean uncommitted *inputs* (ADR-002). Tested on the
porcelain-parsing helper directly — fast, no git repo needed.
"""

from __future__ import annotations

from gbe.run.config import IGNORED_DIRTY_PATHS, _dirty_paths


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
