"""Run DGF-1 GNN batches — the retune grid, the validation pilot, or the gate batch (ADR-012).

    python scripts/run_dgf1_gnn.py --mode retune                       # 9 configs x 1 seed, val
    python scripts/run_dgf1_gnn.py --mode pilot  --lr L --batch-size B # 5 seeds, val
    python scripts/run_dgf1_gnn.py --mode gate   --lr L --batch-size B \
        --preregistration decisions/ADR-012-dgf1-gate1-preregistration.md

The three modes are the execution order ADR-012 clause 8 fixes, and each carries its own guard:

* **retune** (clause 3) — the nine `lr` x `batch_size` configurations, one seed each, scored on the
  **validation** window and selected on AUPRC. The winner is the researcher's to record in the ADR;
  this script prints the table and never edits a document.
* **pilot** (clause 4) — 5 seeds at the winning configuration, on **validation**, persisting scores.
  Its variance is what mechanically fixes the stage-1 seed count (clause 5).
* **gate** — the **test** window, and only with a pre-registration that is accepted *and* carries
  `**Stage-1 seeds:** <integer>`. **The seed count comes from that document, not from a flag**, so
  the batch cannot run a different `n` than the one pre-registered.

Assembles no gate file: the verdict is the researcher's (`gates/GATE-DGF1-1.md`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preregistration import require_accepted_preregistration, require_clean_tree  # noqa: E402

SCORES_DIR = REPO_ROOT / "experiments" / "scores"
PILOT_SEEDS = 5   # ADR-012 clause 4


def resolve_batch(mode: str, args) -> tuple[list[tuple[float, int]], list[int], str, str]:
    """``(configs, seeds, window, experiment_tag)`` for one mode — the whole policy in one place."""
    from adapters.dgf1.eval import dgf1_retune_grid

    if mode == "retune":
        return dgf1_retune_grid(), [0], "val", "retune"

    if args.lr is None or args.batch_size is None:
        raise SystemExit(f"--mode {mode} needs --lr and --batch-size (the retune winner).")
    config = [(float(args.lr), int(args.batch_size))]

    if mode == "pilot":
        return config, list(range(PILOT_SEEDS)), "val", "pilot"

    # gate: the count comes from the pre-registration, never from the command line
    n_seeds = require_accepted_preregistration(args.preregistration)
    print(f"[dgf1] test window unlocked by {Path(args.preregistration).name}: "
          f"stage-1 seeds = {n_seeds} (from the ADR, not a flag)")
    return config, list(range(n_seeds)), "test", "gate"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("retune", "pilot", "gate"), required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--preregistration", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    configs, seeds, window, tag = resolve_batch(args.mode, args)
    require_clean_tree(args.allow_dirty)

    from gbe.gnn import GNNHParams
    from adapters.dgf1.datasource_dgraph import load_dgraph
    from adapters.dgf1.eval import dgf1_base_config, dgf1_hparams, dgf1_splits
    from adapters.dgf1.train_gnn import run_dgf1

    base_hp, cfg_device = dgf1_hparams()
    data = load_dgraph(REPO_ROOT / "data" / "dgraph", strict=True)
    split = dgf1_splits()[window]
    base_cfg = {**dgf1_base_config(), "phase": "P1", "experiment": tag}
    # Retune runs are selection runs, not results: they persist no score vectors (clause 10 asks
    # for them on pilot and gate runs).
    scores_dir = None if args.mode == "retune" else SCORES_DIR

    print(f"[dgf1] mode={args.mode} window={window} ({split.test_min}-{split.test_max}) "
          f"configs={len(configs)} seeds={len(seeds)} device={args.device or cfg_device}")

    results = []
    for lr, batch_size in configs:
        hp = GNNHParams(**{**base_hp.as_dict(), "lr": lr, "batch_size": batch_size,
                           "fan_out": tuple(base_hp.fan_out)})
        for seed in seeds:
            run_id, m = run_dgf1(seed, data, split, hp, base_cfg, device=args.device or cfg_device,
                                 scores_dir=scores_dir)
            results.append((lr, batch_size, seed, m))
            print(f"[dgf1] lr={lr:.2e} batch={batch_size} seed={seed}: "
                  f"ROC-AUC={m['fraud_auc']:.4f} AUPRC={m['fraud_auprc']:.4f} "
                  f"(prevalence {m['prevalence']:.4%}, {m['n_score_fraud']:,} fraud)  {run_id}")

    if args.mode == "retune":
        print("\n== retune, validation AUPRC (ADR-012 clause 3: selection metric) ==")
        for lr, batch_size, _, m in sorted(results, key=lambda r: -r[3]["fraud_auprc"]):
            print(f"  lr={lr:.3e}  batch={batch_size:<5} AUPRC={m['fraud_auprc']:.4f}  "
                  f"ROC-AUC={m['fraud_auc']:.4f}")
        best = max(results, key=lambda r: r[3]["fraud_auprc"])
        print(f"\n  winner: lr={best[0]:.6e} batch_size={best[1]} "
              f"(AUPRC {best[3]['fraud_auprc']:.4f})")
        print("  Record it in ADR-012 as a dated amendment; this script edits no document.")
    elif args.mode == "pilot":
        print("\n[dgf1] pilot done. Compute s_GNN and Δ_val against the floor's pilot rows, apply "
              "clause 5's rule, and record '**Stage-1 seeds:** <n>' in ADR-012 before any gate run.")


if __name__ == "__main__":
    main()
