"""Verify the DGraph-Fin snapshot against doc-02 §2.1 — before any DGF-1 code is written.

    python scripts/verify_dgraph_snapshot.py [--data-root data/dgraph]

**Obtaining the data (manual, cannot be automated).** PyG's `DGraphFin` loader does not download:
it raises and tells you to fetch `DGraphFin.zip` yourself from https://dgraph.xinye.com, which is
registration-gated. Place it at ``<data-root>/raw/DGraphFin.zip``; PyG extracts ``dgraphfin.npz``
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
   message passing and masking them from the loss — the ELL-1 ``labelled_mask`` pattern. Report the
   actual distribution of ``y``.
3. **Is the official split temporal?** doc-02 §2.3 makes this an explicit verification task ("do not
   assume"). The loader exposes ``train_mask``/``val_mask``/``test_mask``, but **timestamps live on
   edges (``edge_time``), not nodes** — while `gbe.eval.temporal.split_masks` takes a *per-node*
   time tensor. So this script derives a node-time proxy from incident edge times and compares its
   distribution across the three masks. That is evidence, not proof, and it is also the first
   concrete input to a real EXTRACT design decision: DGF-1 needs either a node-time derivation or a
   harness extension.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

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


def _describe(name: str, values: torch.Tensor) -> None:
    if values.numel() == 0:
        print(f"    {name:<8} (empty)")
        return
    v = values.to(torch.float)
    q = torch.quantile(v, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    print(f"    {name:<8} n={values.numel():>9,}  min={q[0]:>6.0f}  p25={q[1]:>6.0f}  "
          f"median={q[2]:>6.0f}  p75={q[3]:>6.0f}  max={q[4]:>6.0f}  mean={v.mean():>7.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data" / "dgraph"))
    args = ap.parse_args()

    root = Path(args.data_root)
    _require_data(root)

    from torch_geometric.datasets import DGraphFin   # imported late: only needed on the happy path

    print(f"[verify] loading DGraph-Fin from {root} (first run extracts + caches, ~minutes)")
    data = DGraphFin(root=str(root))[0]
    print(f"[verify] loaded: {data}\n")

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
    # Printing only the latter next to the doc's 1.16 reads as a contradiction when it is a
    # definitional difference; printing only the former understates what message passing sees.
    print(f"      mean out-degree  E/N  = {n_edges / n_nodes:.3f}   (doc-02 §0/§2.1 'avg degree ~1.16')")
    print(f"      mean total degree 2E/N = {2 * n_edges / n_nodes:.3f}   "
          "(what reverse-edges-ON gives — the graph actually sampled)")

    # --- 2. labels ----------------------------------------------------------------------
    labels, counts = data.y.unique(return_counts=True)
    print(f"\n[2/3] label distribution ({labels.numel()} distinct values in y):")
    for value, count in zip(labels.tolist(), counts.tolist()):
        print(f"      y={value}: {count:>9,} ({100 * count / n_nodes:5.2f}%)")

    by_value = dict(zip(labels.tolist(), counts.tolist()))
    n_fraud = by_value.get(1, 0)
    n_normal = by_value.get(0, 0)
    n_background = n_nodes - n_fraud - n_normal
    if n_fraud != EXPECTED_FRAUD:
        _escalate("fraud count (y==1)", n_fraud, EXPECTED_FRAUD, "doc-02 §2.1")
    if n_normal != EXPECTED_NORMAL:
        _escalate("normal count (y==0)", n_normal, EXPECTED_NORMAL, "doc-02 §2.1")
    if n_background != EXPECTED_BACKGROUND:
        _escalate("background count", n_background, EXPECTED_BACKGROUND, "doc-02 §2.1")

    prevalence = n_fraud / (n_fraud + n_normal)
    print(f"      labelled prevalence = {prevalence:.4%} "
          f"(ADR-007 assumed 1.27% when fixing the metric pair)")
    if labels.numel() > 2:
        print(f"      NOTE: y carries {labels.numel()} values but DGraphFin.num_classes == 2. "
              "Background nodes need\n            the ELL-1 labelled_mask treatment "
              "(doc-02 §2.2.2: keep in graph, mask from loss).")

    # --- 3. edge metadata + is the official split temporal? -----------------------------
    t_lo, t_hi = int(data.edge_time.min()), int(data.edge_time.max())
    n_types = int(data.edge_type.unique().numel())
    if (t_lo, t_hi) != EXPECTED_EDGE_TIME:
        _escalate("edge_time range", (t_lo, t_hi), EXPECTED_EDGE_TIME, "doc-02 §2.1")
    if n_types != EXPECTED_EDGE_TYPES:
        _escalate("edge_type count", n_types, EXPECTED_EDGE_TYPES, "doc-02 §2.1")

    print(f"\n[3/3] edge metadata OK — time {t_lo}..{t_hi} · {n_types} contact subtypes")
    print("      official split sizes: "
          f"train={int(data.train_mask.sum()):,} "
          f"val={int(data.val_mask.sum()):,} "
          f"test={int(data.test_mask.sum()):,}")

    node_time = _node_time_proxy(data.edge_index, data.edge_time, n_nodes)
    isolated = int((node_time == -1).sum())
    print(f"\n      node-time proxy = earliest incident edge time ({isolated:,} nodes have no "
          "edges and are excluded):")
    for name, mask in (("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)):
        _describe(name, node_time[mask & (node_time >= 0)])

    medians = [
        float(node_time[m & (node_time >= 0)].to(torch.float).median())
        for m in (data.train_mask, data.val_mask, data.test_mask)
    ]
    spread = max(medians) - min(medians)
    ordered = medians[0] <= medians[1] <= medians[2]
    print(f"\n      medians train/val/test = {medians[0]:.0f} / {medians[1]:.0f} / {medians[2]:.0f}"
          f"  (spread {spread:.0f} of {t_hi - t_lo} steps, monotonic={ordered})")
    print("\n      READ THIS AS EVIDENCE, NOT PROOF (doc-02 §2.3 'do not assume'):")
    print("        · near-identical distributions  -> official split is a RANDOM mask, so the")
    print("          leaderboard number is NOT a temporal holdout; doc-02 requires reporting our")
    print("          own temporal split separately and labelled as such.")
    print("        · ordered, well-separated       -> consistent with a temporal split.")
    print("      Either way, EXTRACT must decide how a per-node time reaches")
    print("      gbe.eval.temporal.split_masks, which takes node times while DGraph dates edges.")

    print("\n[verify] all pinned expectations from doc-02 §2.1 hold. No registry row written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
