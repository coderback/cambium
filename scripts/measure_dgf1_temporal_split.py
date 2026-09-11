"""Measure the DGF-1 temporal split proposed in ADR-011 — read-only provenance for that ADR.

    python scripts/measure_dgf1_temporal_split.py [--data-root data/dgraph]

**Scope: measurement only.** No model, no score, no registry row, nothing under `gbe/` or
`adapters/`. It computes what the ADR-011 rule *implies* on the snapshot — window boundaries,
support, prevalence, and how many edges each candidate training/eval graph would admit — so the
ADR states measured facts rather than assumed ones (ADR-010 precedent).

**The rule was fixed before this script was run** (ADR-011 *Disclosure*):

* node time = **earliest incident edge time** (both endpoints credited, as in
  `verify_dgraph_snapshot._node_time_proxy`);
* ``train_max`` = the first step at which the cumulative **labelled** count reaches 70%;
  ``val_max`` = the first step at which it reaches 85%; test = the remainder to step 821 —
  mirroring the official split's 70/15/15 over labelled nodes, so the two tracks are
  size-comparable.

Nothing printed here may be used to move those fractions. The prevalence per window is reported
because AUPRC's chance level travels with it (ADR-007), not as an input to the cutoffs.

Sections 5-6 are **label-free** (they use the labelled-vs-background mask, never the fraud/normal
values):
* section 5 measures how much of each node's neighbourhood arrives after its first appearance, and
  whether the training view (as of 369) matches the test view (as of 821) in neighbourhood age and
  degree (ADR-011 clauses 2 and 4);
* section 6 confirms on Elliptic that an edge-date filter, using the shared step as the edge date,
  reproduces the node-induced train graph exactly, with the same edges in the same order (ADR-011
  clause 6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # for the gbe / adapters imports in section 6

# ADR-011 rule — fixed before measurement. Do not tune.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85


def node_time_min(edge_index: torch.Tensor, edge_time: torch.Tensor, n: int) -> torch.Tensor:
    """Earliest incident-edge time per node; ``-1`` where a node has no edges."""
    out = torch.full((n,), torch.iinfo(torch.long).max, dtype=torch.long)
    for endpoint in (edge_index[0], edge_index[1]):
        out.scatter_reduce_(0, endpoint, edge_time, reduce="amin")
    out[out == torch.iinfo(torch.long).max] = -1
    return out


def cumulative_cutoff(times: torch.Tensor, frac: float) -> int:
    """First integer step T at which ``#(times <= T) >= frac * len(times)``."""
    counts = torch.bincount(times)
    cum = counts.cumsum(0)
    return int((cum >= frac * times.numel()).nonzero()[0])


def _window(name: str, mask: torch.Tensor, y: torch.Tensor) -> None:
    fraud = int((mask & (y == 1)).sum())
    normal = int((mask & (y == 0)).sum())
    labelled = fraud + normal
    background = int((mask & ((y == 2) | (y == 3))).sum())
    prev = fraud / labelled if labelled else float("nan")
    print(f"    {name:<6} labelled={labelled:>9,} ({fraud:>6,} fraud, prevalence {prev:.4%})  "
          f"background={background:>9,}  all={int(mask.sum()):>9,}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data" / "dgraph"))
    args = ap.parse_args()

    from torch_geometric.datasets import DGraphFin

    data = DGraphFin(root=args.data_root)[0]
    n = data.num_nodes
    ei, et, y = data.edge_index, data.edge_time.view(-1).long(), data.y.view(-1)
    src, dst = ei[0], ei[1]

    t_node = node_time_min(ei, et, n)
    assert int((t_node < 0).sum()) == 0, "ADR-010 measured zero isolated nodes; this changed"
    labelled = (y == 0) | (y == 1)

    # --- 1. cutoffs from the fixed rule -----------------------------------------------
    t_tr = cumulative_cutoff(t_node[labelled], TRAIN_FRAC)
    t_va = cumulative_cutoff(t_node[labelled], VAL_FRAC)
    t_max = int(et.max())
    print(f"[1] rule: train <= {t_tr} | val {t_tr + 1}-{t_va} | test {t_va + 1}-{t_max}  "
          f"(cumulative labelled {TRAIN_FRAC:.0%} / {VAL_FRAC:.0%})")

    train = t_node <= t_tr
    val = (t_node > t_tr) & (t_node <= t_va)
    test = t_node > t_va
    print("\n[2] support and prevalence per window (AUPRC chance level = prevalence, ADR-007):")
    _window("train", train, y)
    _window("val", val, y)
    _window("test", test, y)
    _window("<=481", t_node <= t_va, y)
    print("      ^ REJECTED alternative (ADR-011 clause 2): train on all pre-test time. Gate runs "
          "train on <=369, role-matched to the official split")
    print("    -- official (random) split, for contrast:")
    _window("o.trn", data.train_mask, y)
    _window("o.val", data.val_mask, y)
    _window("o.tst", data.test_mask, y)

    # --- 3. training graph: node-induced vs edge-time-filtered --------------------------
    # 369 is the one training cutoff (retune, pilot and gate runs). 481 is the rejected
    # train-on-all-pre-test alternative, measured for scale only.
    for label, cutoff in (("gate", t_tr), ("rejected-481", t_va)):
        in_window = t_node <= cutoff
        both_in = in_window[src] & in_window[dst]
        edge_pre = et <= cutoff
        # By construction of the min proxy, an edge dated <= T has both endpoints at time <= T.
        assert bool((both_in | ~edge_pre).all()), "min-proxy invariant violated"
        leak = both_in & ~edge_pre
        print(f"\n[3:{label}] training graph at train_max={cutoff} "
              "(directed edges, before reverse-edges):")
        print(f"    edge-time filtered  (t_e <= T)            : {int(edge_pre.sum()):>9,}")
        print(f"    node-induced        (both endpoints <= T) : {int(both_in.sum()):>9,}")
        print(f"    difference = post-cutoff edges between in-window nodes: {int(leak.sum()):>9,} "
              f"({int(leak.sum()) / max(int(both_in.sum()), 1):.2%} of the node-induced graph)")
        print("      ^ what the pre-ADR-011 core (node-induced train graph, node-time-only "
              "leakage check) admitted without complaint")
        leak_nodes = torch.zeros(n, dtype=torch.bool)
        leak_nodes[src[leak]] = True
        leak_nodes[dst[leak]] = True
        print(f"    labelled in-window nodes touched by such an edge: "
              f"{int((leak_nodes & labelled & in_window).sum()):,} of "
              f"{int((labelled & in_window).sum()):,}")

    # --- 4. eval graph: window-induced (ELL-1 literal) vs graph-as-of-window-end ---------
    print("\n[4] eval graph for the TEST window:")
    in_test_induced = test[src] & test[dst]
    touches_test = test[src] | test[dst]
    print(f"    edges incident to a test node                : {int(touches_test.sum()):>9,}")
    print(f"    ... of which both endpoints in test window   : {int(in_test_induced.sum()):>9,} "
          f"({int(in_test_induced.sum()) / max(int(touches_test.sum()), 1):.2%})")
    deg_induced = torch.zeros(n, dtype=torch.long)
    deg_induced.index_add_(0, src[in_test_induced], torch.ones_like(src[in_test_induced]))
    deg_induced.index_add_(0, dst[in_test_induced], torch.ones_like(dst[in_test_induced]))
    iso = labelled & test & (deg_induced == 0)
    print(f"    labelled test nodes ISOLATED under window-induced eval: {int(iso.sum()):,} of "
          f"{int((labelled & test).sum()):,} ({int(iso.sum()) / max(int((labelled & test).sum()), 1):.2%})")

    # --- 5. neighbourhood growth and train/test view match (label-free) -------------------
    # Uses the labelled-vs-background mask only — never the fraud/normal values.
    lag = torch.cat([et - t_node[src], et - t_node[dst]])
    print(f"\n[5] incident-edge lag after the endpoint's first appearance "
          f"({lag.numel():,} incidences):")
    print(f"    lag == 0: {(lag == 0).float().mean():.2%}   lag > 0: {(lag > 0).float().mean():.2%}"
          f"   lag > 100: {(lag > 100).float().mean():.2%}")

    def degree_as_of(cut: int) -> torch.Tensor:
        m = et <= cut
        deg = torch.zeros(n, dtype=torch.long)
        deg.index_add_(0, src[m], torch.ones(int(m.sum()), dtype=torch.long))
        deg.index_add_(0, dst[m], torch.ones(int(m.sum()), dtype=torch.long))
        return deg

    def _q(v: torch.Tensor) -> str:
        v = v.float()
        return f"median={torch.quantile(v, 0.5):.0f} mean={v.mean():.2f}"

    views = (
        ("train <=369, as of 369 (gate)", labelled & (t_node <= t_tr), t_tr),
        ("train <=481, as of 481 (rejected)", labelled & (t_node <= t_va), t_va),
        ("test 482-821, as of 821", labelled & (t_node > t_va), t_max),
    )
    for name, sel, view in views:
        print(f"    {name:<34} age-at-scoring {_q(view - t_node[sel])}   "
              f"degree {_q(degree_as_of(view)[sel])}")
    first = torch.zeros(n, dtype=torch.long)
    for e in (src, dst):
        hit = et == t_node[e]
        first.index_add_(0, e[hit], torch.ones(int(hit.sum()), dtype=torch.long))
    sel_te = labelled & (t_node > t_va)
    print(f"    {'test 482-821, at first appearance':<34} degree {_q(first[sel_te])}")

    # --- 6. Elliptic: edge date = shared step reproduces the node-induced graph exactly ----
    from adapters.ell1.datasource_elliptic import derive_edge_time, load_elliptic
    from gbe.eval import edges_as_of

    ell = load_elliptic(REPO_ROOT / "data" / "elliptic", strict=True)
    ts, eie = ell.time_step, ell.edge_index
    print(f"\n[6] Elliptic edges within one step: {int((ts[eie[0]] == ts[eie[1]]).sum()):,} "
          f"/ {eie.size(1):,}")
    derived = derive_edge_time(eie, ts)  # raises if any edge spans two steps
    for cut in (29, 34):
        # The retired node-induced reading, written out: both endpoints at or before the cutoff.
        # (Before retirement this line called gbe.eval.induced_train_subgraph and gave the same
        # result; it applied exactly this mask via torch_geometric.utils.subgraph.)
        node_induced = eie[:, (ts[eie[0]] <= cut) & (ts[eie[1]] <= cut)]
        print(f"    T={cut}: node-induced == edge-date filter (same edges, same order): "
              f"{torch.equal(node_induced, edges_as_of(eie, edge_time=derived, t_max=cut))}")

    print("\n[measure] read-only. No registry row written; no model trained or scored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
