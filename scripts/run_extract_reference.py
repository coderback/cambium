"""Create the EXTRACT regression reference for the GCN backbone (ADR-008 clause 3).

    python scripts/run_extract_reference.py [--device cuda|cpu] [--seeds 0,1,2]

**Why this exists.** ADR-008 makes EXTRACT's acceptance condition "the refactored core reproduces
ELL-1's recorded numbers bit-for-bit". The Gate-0 (RF/LR) and Gate-3 (GraphSAGE × 5 arms) batches
already supply reproducible references. **The GCN backbone does not have one:** the only GCN rows
in the registry are Gate 1's, which predate ADR-005 and ran on non-deterministic CUDA, so equality
against them is impossible. Without this batch `BACKBONES["gcn"]` would pass through the entire
refactor unchecked — and GCN is a named DGF-1 comparator (doc-02 §5).

So: GraphSAGE's frozen ADR-003 config with ``backbone=gcn`` (exactly the capacity-matched
comparator Gate 1 used), 3 seeds, **deterministic** (ADR-005), on committed code. Three registry
rows tagged ``experiment="extract_reference"``, which is also the first live exercise of
`PROVENANCE_KEYS` — committed 2026-07-25 but never yet written to a row.

This script **assembles no gate file and reads none**. It only appends rows. In particular it is
deliberately *not* `run_ell1_gnn.py`, which would rewrite the dated `gates/GATE-ELL1-1.md`.
"""

from __future__ import annotations

import argparse

from gbe.run.config import git_dirty
from adapters.ell1.datasource_elliptic import load_elliptic
from adapters.ell1.eval import ell1_eval_split
from adapters.ell1.train_gnn import frozen_hparams, resolve_device, run_gnn

BACKBONE = "gcn"
EXPERIMENT = "extract_reference"
METRICS = ("illicit_f1", "illicit_recall", "illicit_precision", "illicit_auc", "illicit_auprc")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="cuda | cpu | auto (default: config)")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds (ADR-008: 0,1,2)")
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run despite uncommitted code. The rows will be git_dirty=true and are then NOT "
             "usable as an ADR-008 reference. For smoke-testing only.",
    )
    args = ap.parse_args()

    # A reference row must be reproducible from a commit, so a dirty tree disqualifies the batch
    # before it costs 4 minutes. This has bitten the project twice — the Gate-0 and Gate-1 batches
    # both had to be re-run after logging dirty=true (ADR-002, lab 2026-07-21 / 2026-07-24) — so
    # the rule is enforced here rather than remembered.
    if git_dirty() and not args.allow_dirty:
        raise SystemExit(
            "refusing to run: the working tree has uncommitted code/config, so these rows would "
            "log git_dirty=true and could not serve as an ADR-008 reference (clause 3: 'on "
            "committed code'). Commit first, then re-run. Override with --allow-dirty only for a "
            "smoke test whose rows you intend to discard."
        )

    seeds = tuple(int(s) for s in args.seeds.split(","))
    hp, base_cfg, cfg_device = frozen_hparams()
    device = resolve_device(args.device or cfg_device)
    base_cfg = {**base_cfg, "experiment": EXPERIMENT}

    data = load_elliptic()
    split = ell1_eval_split()

    print(f"[extract-ref] backbone={BACKBONE} seeds={seeds} device={device}")
    print(f"[extract-ref] frozen ADR-003 config | test window {split.test_min}-{split.test_max}")
    # Do NOT print the *global* determinism flag here: RunSession pins it per-run at entry, so
    # before the first run it reads False and a log line saying so would read as "this reference
    # is not deterministic" — exactly the wrong conclusion. Each row records its own state
    # (ADR-005 clause 3); trust the row, not a banner.
    print("[extract-ref] determinism: pinned per-run by RunSession at entry (ADR-005); "
          "every row records the state it ran under\n")

    results = []
    for seed in seeds:
        run_id, metrics = run_gnn(
            backbone=BACKBONE,
            seed=seed,
            data=data,
            split=split,
            hp=hp,
            base_cfg_values=base_cfg,
            device=str(device),
        )
        results.append((seed, run_id, metrics))
        got = "  ".join(f"{k.replace('illicit_', '')}={metrics[k]:.6f}" for k in METRICS if k in metrics)
        print(f"  seed {seed}: {got}")
        print(f"           run_id={run_id}  experiment={metrics.get('experiment')}")

    print(f"\n[extract-ref] {len(results)} reference rows appended to experiments/registry.csv.")
    print("[extract-ref] Full precision (this is what the post-refactor re-run must match "
          "EXACTLY — ADR-008 clause 2):")
    for seed, run_id, m in results:
        vals = {k: m[k] for k in METRICS if k in m}
        print(f"  seed {seed}: {vals}")


if __name__ == "__main__":
    main()
