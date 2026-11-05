"""Guard: no BatchNorm anywhere in the core, gbe/.

BatchNorm normalises with the statistics of the batch it sees. At scoring time that mixes
evaluation users' statistics into every prediction, which is a known leakage route under a
full-graph pass. CLAUDE.md and gbe/CLAUDE.md state the ban for gbe/gnn/. This lint covers all of
gbe/, by the researcher's decision of 2026-10-07, because the input encoder in gbe/features/
runs inside every GNN forward pass and its own docstring says the ban covers it. Before, nothing
checked that folder (audit of 2026-10-07, A finding S8).

The check reads the code's syntax tree, not its text. BatchNorm used as a layer, a class, an
imported name or the functional `batch_norm` is flagged. Docstrings and comments that mention it,
like the encoder's, are not code and are left alone.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

GBE_DIR = Path(__file__).resolve().parents[1] / "gbe"
GNN_DIR = GBE_DIR / "gnn"

_LAYER = re.compile(r"^(Sync)?BatchNorm\w*$")   # nn.BatchNorm1d/2d/3d, SyncBatchNorm, PyG BatchNorm
_FUNCTIONAL = "batch_norm"                         # F.batch_norm, torch.batch_norm


def _is_batchnorm(name: str) -> bool:
    return bool(_LAYER.match(name)) or name == _FUNCTIONAL


def _offences(source: str) -> list[str]:
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and _is_batchnorm(node.id):
            hits.append(f"L{node.lineno}: {node.id}")
        elif isinstance(node, ast.Attribute) and _is_batchnorm(node.attr):
            hits.append(f"L{node.lineno}: .{node.attr}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if _is_batchnorm(alias.name.split(".")[-1]):
                    hits.append(f"L{node.lineno}: import {alias.name}")
    return hits


def test_gnn_dir_exists():
    assert GNN_DIR.is_dir(), f"expected package dir at {GNN_DIR}"


def test_no_batchnorm_under_gbe():
    offenders: dict[str, list[str]] = {}
    for py in sorted(GBE_DIR.rglob("*.py")):
        hits = _offences(py.read_text(encoding="utf-8"))
        if hits:
            offenders[str(py)] = hits
    assert not offenders, f"BatchNorm used under gbe/: {offenders}"


def test_the_lint_catches_every_form_and_ignores_prose():
    """It has to fail on the code it exists to stop, not merely pass on today's tree."""
    for source in ("self.bn = nn.BatchNorm1d(16)",
                   "layer = torch.nn.SyncBatchNorm(16)",
                   "from torch_geometric.nn import BatchNorm",
                   "from torch.nn import BatchNorm2d as BN",
                   "y = F.batch_norm(x, mean, var)",
                   "y = torch.batch_norm(x, None, None, None, None, True, 0.1, 1e-5, False)"):
        assert _offences(source), f"lint missed: {source}"
    for source in ('"""LayerNorm, never BatchNorm: it would mix statistics."""',
                   "# a BatchNorm here would leak",
                   "self.norm = nn.LayerNorm(16)",
                   "from torch_geometric.nn import GraphNorm"):
        assert not _offences(source), f"lint flagged something that is not BatchNorm: {source}"
