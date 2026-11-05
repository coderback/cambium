"""Guard: nothing under gbe/ imports a domain library or an adapter.

The core-inclusion boundary is enforced, not advisory: if gbe/ imports edgartools,
yfinance, kaggle, or anything under adapters/, the interface has leaked.

Dataset loaders can also hide inside a library the core legitimately uses: PyG is a core
dependency, but `torch_geometric.datasets` holds the Elliptic and DGraph loaders CLAUDE.md
bans from gbe/. Checking only the top-level name let `from torch_geometric.datasets import
DGraphFin` through (audit of 2026-10-07, A finding S7), so denied submodules are matched on
the full dotted path, in every import form.
"""

from __future__ import annotations

import ast
from pathlib import Path

GBE_DIR = Path(__file__).resolve().parents[1] / "gbe"

DENYLIST = {"edgartools", "yfinance", "kaggle", "adapters"}
# Submodules of allowed packages that are themselves domain loaders.
DENIED_MODULES = ("torch_geometric.datasets",)


def _top_level(module: str | None) -> str | None:
    if not module:
        return None
    return module.split(".")[0]


def _imported_top_levels(tree: ast.AST) -> set[str]:
    tops: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = _top_level(alias.name)
                if top:
                    tops.add(top)
        elif isinstance(node, ast.ImportFrom):
            # Ignore relative imports (level > 0): they can only reach within gbe/.
            if node.level == 0:
                top = _top_level(node.module)
                if top:
                    tops.add(top)
    return tops


def _imported_modules(tree: ast.AST) -> set[str]:
    """Every dotted name an import can bind: ``import a.b``, ``from a.b import c`` (both ``a.b``
    and ``a.b.c``, since ``c`` may be a submodule), and ``from a import b``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _denied_modules(tree: ast.AST) -> set[str]:
    return {name for name in _imported_modules(tree)
            for denied in DENIED_MODULES if name == denied or name.startswith(denied + ".")}


def test_no_domain_or_adapter_imports_under_gbe():
    violations: dict[str, set[str]] = {}
    for py in sorted(GBE_DIR.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        bad = (_imported_top_levels(tree) & DENYLIST) | _denied_modules(tree)
        if bad:
            violations[str(py)] = bad
    assert not violations, f"forbidden imports under gbe/: {violations}"


def test_denied_submodules_are_caught_in_every_import_form():
    """The lint has to fail on the imports it exists to stop, not merely pass on today's tree."""
    for source in ("from torch_geometric.datasets import DGraphFin",
                   "import torch_geometric.datasets",
                   "import torch_geometric.datasets.dgraph as d",
                   "from torch_geometric import datasets"):
        assert _denied_modules(ast.parse(source)), f"lint missed: {source}"
    for source in ("from torch_geometric.nn import SAGEConv",
                   "from torch_geometric.loader import NeighborLoader"):
        assert not _denied_modules(ast.parse(source)), f"lint flagged an allowed import: {source}"
