"""What remains of ADR-011's split measurement: the parts that read no held-out data (ADR-021 clause 8).

    python scripts/measure_dgf1_temporal_split.py

**Scope: measurement only.** No model, no score, no registry row. It reports training and validation
support, what an edge-date filter does to the training graph, and neighbourhood age and degree in the
training views. It also confirms on Elliptic that an edge-date filter, using the shared step as the
edge date, reproduces the node-induced train graph exactly (ADR-011 clause 6).

**Retired, because they read held-out data:** re-deriving the cutoffs from all labels, the test window's
support and views, and the official split. They ran once, as ADR-011's provenance, and the script that
ran them is in the repository's history at ``cb3ace6``. The windows now come from the config, and every
edge dated after the validation window is dropped before anything is counted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _window(name: str, mask: torch.Tensor, y: torch.Tensor) -> None:
    fraud = int((mask & (y == 1)).sum())
    normal = int((mask & (y == 0)).sum())
    labelled = fraud + normal
    background = int((mask & ((y == 2) | (y == 3))).sum())
    prev = fraud / labelled if labelled else float("nan")
    print(f"    {name:<6} labelled={labelled:>9,} ({fraud:>6,} fraud, prevalence {prev:.4%})  "
          f"background={background:>9,}  all={int(mask.sum()):>9,}")


def elliptic_check() -> None:
    """Elliptic: edge date = shared step reproduces the node-induced graph exactly."""
    from adapters.ell1.datasource_elliptic import derive_edge_time, load_elliptic
    from gbe.eval import edges_as_of

    ell = load_elliptic(REPO_ROOT / "data" / "elliptic", strict=True)
    ts, eie = ell.time_step, ell.edge_index
    print(f"\n[4] Elliptic edges within one step: {int((ts[eie[0]] == ts[eie[1]]).sum()):,} / {eie.size(1):,}")
    derived = derive_edge_time(eie, ts)  # raises if any edge spans two steps
    for cut in (29, 34):
        # The retired node-induced reading, written out: both endpoints at or before the cutoff.
        node_induced = eie[:, (ts[eie[0]] <= cut) & (ts[eie[1]] <= cut)]
        print(f"    T={cut}: node-induced == edge-date filter (same edges, same order): "
              f"{torch.equal(node_induced, edges_as_of(eie, edge_time=derived, t_max=cut))}")


def main() -> int:
    from adapters.dgf1.datasource_dgraph import load_dgraph
    from adapters.dgf1.eval import dgf1_splits

    val = dgf1_splits()["val"]
    t_tr, t_va = val.train_max, val.test_max
    data = load_dgraph(strict=True)
    keep = data.edge_time <= t_va
    src, dst = data.edge_index[0, keep], data.edge_index[1, keep]
    et, y, t_node, n = data.edge_time[keep], data.y, data.node_time, data.num_nodes
    labelled = data.labelled_mask

    print(f"[1] windows, from the config: train <= {t_tr} | val {val.test_min}-{t_va}")

    print("\n[2] support and prevalence per window (AUPRC chance level = prevalence, ADR-007):")
    _window("train", t_node <= t_tr, y)
    _window("val", (t_node > t_tr) & (t_node <= t_va), y)

    in_window = t_node <= t_tr
    both_in = in_window[src] & in_window[dst]
    edge_pre = et <= t_tr
    # By construction of the min proxy, an edge dated <= T has both endpoints at time <= T.
    assert bool((both_in | ~edge_pre).all()), "min-proxy invariant violated"
    late = both_in & ~edge_pre
    print(f"\n[3] training graph at train_max={t_tr} (directed edges, before reverse-edges; edges dated after "
          f"{t_va} are not counted):")
    print(f"    edge-time filtered  (t_e <= T)            : {int(edge_pre.sum()):>9,}")
    print(f"    node-induced        (both endpoints <= T) : {int(both_in.sum()):>9,}")
    print(f"    difference, later edges between in-window nodes: {int(late.sum()):>9,}")

    def degree_as_of(cut: int) -> torch.Tensor:
        m = et <= cut
        deg = torch.zeros(n, dtype=torch.long)
        deg.index_add_(0, src[m], torch.ones(int(m.sum()), dtype=torch.long))
        deg.index_add_(0, dst[m], torch.ones(int(m.sum()), dtype=torch.long))
        return deg

    def _q(v: torch.Tensor) -> str:
        v = v.float()
        return f"median={torch.quantile(v, 0.5):.0f} mean={v.mean():.2f}"

    for name, sel, view in (("train <=369, as of 369", labelled & (t_node <= t_tr), t_tr),
                            ("val 370-481, as of 481", labelled & (t_node > t_tr) & (t_node <= t_va), t_va)):
        print(f"    {name:<34} age-at-scoring {_q(view - t_node[sel])}   degree {_q(degree_as_of(view)[sel])}")

    elliptic_check()

    print("\n[measure] read-only. No registry row written; no model trained or scored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
