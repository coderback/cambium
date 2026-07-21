"""Guard: nothing under gbe/ imports a domain library or an adapter.

The core-inclusion boundary is enforced, not advisory: if gbe/ imports edgartools,
yfinance, kaggle, or anything under adapters/, the interface has leaked.
"""

from __future__ import annotations

import ast
from pathlib import Path

GBE_DIR = Path(__file__).resolve().parents[1] / "gbe"

DENYLIST = {"edgartools", "yfinance", "kaggle", "adapters"}


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


def test_no_domain_or_adapter_imports_under_gbe():
    violations: dict[str, set[str]] = {}
    for py in sorted(GBE_DIR.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        bad = _imported_top_levels(tree) & DENYLIST
        if bad:
            violations[str(py)] = bad
    assert not violations, f"forbidden imports under gbe/: {violations}"
