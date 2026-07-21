"""Guard: no BatchNorm anywhere under gbe/gnn/.

BatchNorm statistics computed over test-period nodes are an architectural leakage
vector under a full-graph pass (part of the Elliptic 0.807->0.12 collapse). The
tree is empty today; this test still runs and stays meaningful as gnn/ fills in.
"""

from __future__ import annotations

from pathlib import Path

GNN_DIR = Path(__file__).resolve().parents[1] / "gbe" / "gnn"

# Case-insensitive substrings that indicate a BatchNorm usage in torch / PyG.
_FORBIDDEN = ("batchnorm", "batch_norm")


def _offending_lines(text: str) -> list[str]:
    hits = []
    for i, line in enumerate(text.splitlines(), start=1):
        low = line.lower()
        if any(tok in low for tok in _FORBIDDEN):
            hits.append(f"L{i}: {line.strip()}")
    return hits


def test_gnn_dir_exists():
    assert GNN_DIR.is_dir(), f"expected package dir at {GNN_DIR}"


def test_no_batchnorm_reference_under_gnn():
    offenders: dict[str, list[str]] = {}
    for py in sorted(GNN_DIR.rglob("*.py")):
        lines = _offending_lines(py.read_text(encoding="utf-8"))
        if lines:
            offenders[str(py)] = lines
    assert not offenders, f"BatchNorm reference(s) found under gbe/gnn/: {offenders}"
