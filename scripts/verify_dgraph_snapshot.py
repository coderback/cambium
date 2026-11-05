"""Verify the DGraph-Fin snapshot against doc-02 §2.1 — before any DGF-1 code is written.

    python scripts/verify_dgraph_snapshot.py

**Obtaining the data (manual, cannot be automated).** PyG's `DGraphFin` loader does not download:
it raises and tells you to fetch `DGraphFin.zip` yourself from https://dgraph.xinye.com, which is
registration-gated. Place it under the adapter's ``DGRAPH_ROOT``, at ``raw/DGraphFin.zip``; PyG extracts ``dgraphfin.npz``
and caches ``processed/data.pt`` on first access. ``data/`` is gitignored, so nothing is committed.

**Scope: verification only.** This script defines no `DataSource`, writes no adapter config, touches
nothing under `gbe/`, and appends no registry row. It exists so that the three known doc-vs-data
questions below are answered *before* EXTRACT designs an interface around assumptions about them.

**ADR-001 precedent.** When ELL-1's data disagreed with its doc (167 CSV columns, so 165 features,
not the doc's 166), the loader escalated rather than quietly accepting the number — the doc was
amended by ADR and only then did the constant change. Same discipline here: a mismatch **raises**
and names the doc section to reconcile. Do not edit a constant to make this pass.

**The three questions this answers:**

1. **Shape.** PyG's docstring claims 3,700,550 nodes / 4,300,999 edges / 17 features. doc-02 §2.1
   says ~3.7M / ~4M / 17 and its class breakdown sums to exactly 3,700,550. Consistent — but
   assert it.
2. **Labels.** `DGraphFin.num_classes` returns 2, yet doc-02 §2.1 describes 15,509 fraud +
   1,210,092 normal + **2,474,949 background**. Background nodes therefore carry distinct label
   values that a binary reading would silently swallow. doc-02 §2.2.2 requires keeping them for
   message passing and masking them from the loss — the ELL-1 ``labelled_mask`` pattern. The totals
   are checked as pass or fail only (ADR-021 I9), through the adapter.
3. **Is the official split temporal?** ADR-010 answered this once from this script. Its comparison
   across the official masks is retired, because it read them (ADR-021 clause 8), and it is in the
   repository's history at ``cb3ace6``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Pinned from doc-02 §2.1 + the PyG loader docstring. Treat as the doc's claim, not as truth:
# a mismatch is a doc-vs-data discrepancy to reconcile, never a constant to edit.
#
# `EXPECTED_EDGE_TYPES` was 12 ("subtypes 0-11", per doc-02 and PyG's own docstring) until this
# script escalated on the real snapshot: the data has 11 contiguous types, 1..11, with no type 0.
# The constant changed only *after* doc-02 §0/§2.1/§2.2/§9 were amended by ADR-010 — that ordering
# is the point (ADR-001 precedent). Editing it first would have been the failure this script exists
# to prevent.
EXPECTED_NODES = 3_700_550
EXPECTED_EDGES = 4_300_999
EXPECTED_FEATURES = 17
EXPECTED_FRAUD = 15_509
EXPECTED_NORMAL = 1_210_092
EXPECTED_BACKGROUND = 2_474_949
EXPECTED_EDGE_TIME = (1, 821)
EXPECTED_EDGE_TYPES = 11          # subtypes 1..11 — NOT 0..11 (ADR-010)
DOWNLOAD_URL = "https://dgraph.xinye.com"


def _escalate(what: str, loaded, expected, doc_ref: str) -> None:
    raise AssertionError(
        f"doc-vs-data discrepancy on {what}: loaded {loaded!r}, {doc_ref} pins {expected!r}.\n"
        "Reconcile the build doc first (doc-first, ADR-001 precedent) — amend it via an ADR if the "
        "data is right. Do NOT edit the constant in this script to make the check pass."
    )


def _require_data(root: Path) -> None:
    raw = root / "raw" / "DGraphFin.zip"
    if raw.exists() or (root / "processed" / "data.pt").exists():
        return
    raise SystemExit(
        "DGraph-Fin snapshot not found.\n\n"
        f"  expected: {raw}\n"
        f"  obtain from: {DOWNLOAD_URL}  (registration-gated; PyG will not download it)\n\n"
        "This is the long-pole prerequisite for DGF-1 Phase 0 — request access first, everything "
        "else can proceed in parallel."
    )


def _node_time_proxy(edge_index: torch.Tensor, edge_time: torch.Tensor, n: int) -> torch.Tensor:
    """Earliest incident-edge time per node; ``-1`` where a node has no edges.

    Edges are directed, so both endpoints are credited — an edge's timestamp is evidence about
    when *either* endpoint became active. Min (rather than mean) is the least assumption-laden
    proxy for "when did this node first appear".
    """
    out = torch.full((n,), torch.iinfo(torch.long).max, dtype=torch.long)
    for endpoint in (edge_index[0], edge_index[1]):
        out.scatter_reduce_(0, endpoint, edge_time, reduce="amin")
    out[out == torch.iinfo(torch.long).max] = -1
    return out


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    from adapters.dgf1.datasource_dgraph import DGRAPH_ROOT, load_dgraph
    from adapters.dgf1.heldout import integrity_ok

    _require_data(DGRAPH_ROOT)
    print(f"[verify] loading DGraph-Fin through the adapter, from {DGRAPH_ROOT} (first run extracts + caches, ~minutes)")
    data = load_dgraph(strict=False)   # this script's own checks below carry its doc-first message

    # --- 1. shape -----------------------------------------------------------------------
    n_nodes, n_features = data.x.shape
    n_edges = data.edge_index.size(1)
    if n_nodes != EXPECTED_NODES:
        _escalate("node count", n_nodes, EXPECTED_NODES, "doc-02 §2.1")
    if n_features != EXPECTED_FEATURES:
        _escalate("feature count", n_features, EXPECTED_FEATURES, "doc-02 §2.1")
    if n_edges != EXPECTED_EDGES:
        _escalate("edge count", n_edges, EXPECTED_EDGES, "doc-02 §2.1")
    print(f"[1/3] shape OK — {n_nodes:,} nodes · {n_edges:,} edges · {n_features} features")
    # Report BOTH conventions explicitly. doc-02 §0/§2.1's "avg degree ~1.16" is E/N — mean
    # *out*-degree on the directed graph. But §2.2.4 makes reverse edges the default ("doubles
    # usable connectivity"), so the graph the GNN actually samples has mean *total* degree 2E/N.
    print(f"      mean out-degree  E/N  = {n_edges / n_nodes:.3f}   (doc-02 §0/§2.1 'avg degree ~1.16')")
    print(f"      mean total degree 2E/N = {2 * n_edges / n_nodes:.3f}   "
          "(what reverse-edges-ON gives — the graph actually sampled)")

    # --- 2. labels: pass or fail only (ADR-021 I9, not a look) ----------------------------
    if not integrity_ok():
        _escalate("whole-snapshot label totals", "a different total", "the published totals", "doc-02 §2.1")
    print("\n[2/3] label totals match the published ones (pass or fail only; ADR-021 I9)")

    # --- 3. edge metadata -----------------------------------------------------------------
    t_lo, t_hi = int(data.edge_time.min()), int(data.edge_time.max())
    n_types = int(data.edge_type.unique().numel())
    if (t_lo, t_hi) != EXPECTED_EDGE_TIME:
        _escalate("edge_time range", (t_lo, t_hi), EXPECTED_EDGE_TIME, "doc-02 §2.1")
    if n_types != EXPECTED_EDGE_TYPES:
        _escalate("edge_type count", n_types, EXPECTED_EDGE_TYPES, "doc-02 §2.1")
    print(f"\n[3/3] edge metadata OK — time {t_lo}..{t_hi} · {n_types} contact subtypes")

    print("\n[verify] all pinned expectations from doc-02 §2.1 hold. No held-out set opened; no registry row written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
