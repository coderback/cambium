"""Snapshot audit for the DGF-1 DataSource — ADR-011 clause 5, run on the real graph.

    python scripts/audit_dgf1_datasource.py

The pytest suite runs on synthetic fixtures (the snapshot is gitignored), so this script is where
the DataSource is checked against the **real** DGraph-Fin snapshot, through the adapter's own code
path (`adapters.dgf1.datasource_dgraph`) rather than a reimplementation. Read-only: no model, no
score, no registry row.

It compares the config's windows and ADR-011's training and validation counts with the pinned values.
A mismatch **escalates** (exit 1, naming the ADR): it is a doc-vs-data discrepancy to reconcile in the
ADR, never a constant to edit here (ADR-001 precedent).

**Retired, because they read held-out data (ADR-021 clause 8):** re-deriving the windows from all labels,
the test window's counts and view, and the count of later edges between training users, which reached
the test window's dates. They are in the repository's history at ``cb3ace6``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gbe.eval import assert_no_temporal_leakage, edges_as_of  # noqa: E402
from adapters.dgf1.datasource_dgraph import (  # noqa: E402
    FRAUD,
    graph_view,
    load_dgraph,
    node_time_earliest_edge,
    train_seed_mask,
    window_target_mask,
)
from adapters.dgf1.eval import dgf1_splits  # noqa: E402

# Every number below is stated in ADR-011 (Context / clause 2), measured by
# scripts/measure_dgf1_temporal_split.py before the DataSource existed.
PINNED = {
    'train_max': 369,
    'val_max': 481,
    'train labelled': 858_702,
    'train fraud': 10_317,
    'val labelled': 183_430,
    'val fraud': 2_475,
    'forward edges dated <= 369': 1_743_128,
}

failures: list[str] = []


def _fmt(v) -> str:
    # bool is a subclass of int, so test it first — otherwise True would print as "1".
    if isinstance(v, bool) or not isinstance(v, int):
        return str(v)
    return f"{v:,}"


def check(name: str, got, want) -> None:
    ok = got == want
    line = f"  [{'ok' if ok else 'MISMATCH'}] {name}: {_fmt(got)}"
    if not ok:
        line += f"  (ADR-011 pins {_fmt(want)})"
        failures.append(f"{name}: got {got!r}, ADR-011 pins {want!r}")
    print(line)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    data = load_dgraph(strict=True)  # raises on shape / edge-type / isolated nodes
    print(f"[audit] loaded through the adapter: {data.num_nodes:,} users, "
          f"{data.edge_index.size(1):,} forward edges, zero isolated (asserted by the loader)")
    val = dgf1_splits()["val"]
    t_node = data.node_time

    print("\n[1] windows: the config's values vs ADR-011's")
    check("train_max (config)", val.train_max, PINNED["train_max"])
    check("val_max (config)", val.test_max, PINNED["val_max"])

    print("\n[2] per-window support (clause 2)")
    seeds = train_seed_mask(data, val)
    check("train labelled", int(seeds.sum()), PINNED["train labelled"])
    check("train fraud", int((seeds & (data.y == FRAUD)).sum()), PINNED["train fraud"])
    tgt = window_target_mask(data, val)
    check("val labelled", int(tgt.sum()), PINNED["val labelled"])
    check("val fraud", int((tgt & (data.y == FRAUD)).sum()), PINNED["val fraud"])

    print("\n[3] training view at 369 (clause 3; clause-5 test 1 on the real graph)")
    view = graph_view(data, val.train_max)
    fwd = edges_as_of(data.edge_index, edge_time=data.edge_time, t_max=val.train_max)
    check("forward edges dated <= 369", fwd.size(1), PINNED["forward edges dated <= 369"])
    check("view == edges_as_of, reverse-doubled", torch.equal(view.edge_index,
          torch.cat([fwd, fwd.flip(0)], dim=1)) if data.reverse_edges else torch.equal(view.edge_index, fwd), True)
    assert_no_temporal_leakage(view.edge_index, t_node, val, edge_time=view.edge_time)
    print("  [ok] core leakage check silent on the training view")

    print("\n[4] prefix-determinism on the real graph (clause-5 test 4)")
    for b in (val.train_max, val.test_max):
        keep = data.edge_time <= b
        t_b = node_time_earliest_edge(data.edge_index[:, keep], data.edge_time[keep], data.num_nodes)
        check(f"window membership at {b} from edges <= {b} alone",
              torch.equal((t_b >= 0) & (t_b <= b), t_node <= b), True)

    print("\n[5] the validation scoring view (clause-5 test 5) and view features")
    v = graph_view(data, val.test_max)
    check(f"view <= {val.test_max}: no later edge", int(v.edge_time.max()) <= val.test_max, True)
    hist_sum = v.node_features[:, :11].sum(dim=1)
    fwd_v = v.edge_index[:, : v.edge_index.size(1) // 2] if data.reverse_edges else v.edge_index
    degree = torch.bincount(fwd_v.reshape(-1), minlength=data.num_nodes).float()
    check(f"view <= {val.test_max}: histogram rows sum to degree", torch.equal(hist_sum, degree), True)
    raw_missing = float((data.x[t_node <= val.test_max] == -1).float().mean())
    print(f"  [info] share of raw feature values equal to -1, users up to {val.test_max}: {raw_missing:.2%} "
          "(likely missing-value sentinel; the feature transform's concern)")

    print()
    if failures:
        print("[audit] FAIL — doc-vs-data discrepancy. Reconcile ADR-011 first; do NOT edit a pinned")
        print("        value here to make this pass (ADR-001 precedent):")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("[audit] PASS — the DGF-1 DataSource reproduces ADR-011's training and validation counts. No row written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
