"""Decide whether DGF-1's first-appearance row may be produced: is the temporal sampler deterministic
on the real graph, and does it respect every seed's time? (ADR-011 clause 4, clause-5 test 10)

    python scripts/verify_dgf1_temporal_sampler.py [--batch-size 1024]

Runs on the **validation** window only (targets first appearing 370–481, eligible edges as of 481):
determinism and the temporal constraint are properties of the sampler, not of the window, and this
keeps the test window untouched before the Gate-1 pre-registration. No model, no score, no label
values, no registry row.

For each mode — ``exact`` (every qualifying neighbour) and ``fanout`` (ADR-003's [25, 10] extended
to three hops as ELL-1 does) — it makes two full passes at seed 0 under ``seed_everything`` (strict
``use_deterministic_algorithms``) and compares every batch's signature exactly; checks every batch
with the per-batch temporal guard; and, for ``fanout``, a pass at seed 1 must differ (else the seed is
inert and "reproducible from the recorded seed" would be vacuous). Exit 1 on any failure: the row is
then **not produced**, and there is no non-deterministic fallback (ADR-011).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from gbe.run.seeding import seed_everything  # noqa: E402
from adapters.dgf1.datasource_dgraph import graph_view, load_dgraph, window_target_mask  # noqa: E402
from adapters.dgf1.eval import dgf1_splits  # noqa: E402
from adapters.dgf1.sampling import (  # noqa: E402
    assert_temporal_batch,
    batch_signature,
    first_appearance_loader,
)

MODES = {"exact": [-1, -1, -1], "fanout": [25, 10, 10]}


def _digest(batch) -> str:
    h = hashlib.sha256()
    for t in batch_signature(batch):
        h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


def one_pass(data, view, targets, fanout, seed, batch_size):
    seed_everything(seed)
    loader = first_appearance_loader(data.x, view, data.node_time, targets, fanout, batch_size)
    digests, n_edges, max_nodes, t0 = [], 0, 0, time.perf_counter()
    for batch in loader:
        assert_temporal_batch(batch)
        digests.append(_digest(batch))
        n_edges += batch.edge_index.size(1)
        max_nodes = max(max_nodes, batch.num_nodes)
    return digests, n_edges, max_nodes, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    data = load_dgraph(REPO_ROOT / "data" / "dgraph", strict=True)
    val = dgf1_splits()["val"]
    view = graph_view(data, val.test_max)
    targets = window_target_mask(data, val).nonzero().view(-1)
    print(f"[sampler] val window {val.test_min}-{val.test_max}: {targets.numel():,} targets, "
          f"{view.edge_index.size(1):,} eligible edges (as of {val.test_max}); "
          f"deterministic algorithms will be pinned per pass")

    ok = True
    for mode, fanout in MODES.items():
        a, n_edges, max_nodes, secs = one_pass(data, view, targets, fanout, 0, args.batch_size)
        b, _, _, _ = one_pass(data, view, targets, fanout, 0, args.batch_size)
        same = a == b
        line = (f"[{mode}] fan-out {fanout}: {len(a)} batches, {n_edges:,} sampled edges, "
                f"max {max_nodes:,} nodes/batch, {secs:.1f}s/pass | temporal guard: every batch ok | "
                f"seed 0 twice identical: {same}")
        ok &= same
        if mode == "fanout":
            c, _, _, _ = one_pass(data, view, targets, fanout, 1, args.batch_size)
            responds = a != c
            line += f" | seed 1 differs: {responds}"
            ok &= responds
        print(line)
        assert torch.are_deterministic_algorithms_enabled()

    if ok:
        print("\n[sampler] PASS — deterministic under ADR-005 and every sampled edge respects its "
              "seed's time. The first-appearance row may be produced (ADR-011 clause 4).")
        return 0
    print("\n[sampler] FAIL — the first-appearance row must NOT be produced; record this. "
          "No non-deterministic fallback (ADR-011).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
