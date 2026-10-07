"""Run the DGF-1 tabular floor — one registry row per (model, feature set, seed) on one window.

    python scripts/run_dgf1_floor.py --mode retune                          # 9 configs x 1 seed, val
    python scripts/run_dgf1_floor.py --mode pilot --max-depth D --subsample S

The modes mirror the GNN runner's, so the two cannot drift: **retune** is ADR-012 clause 2's nine
`max_depth` x `subsample` configurations on the **validation** window, one seed each, selected on
AUPRC (no score files — selection runs are not results); **pilot** runs the winning configuration's
seeds on validation, persisting scores; **gate** ran Gate 1's **test** batch at `09c6e72` and is now
**closed** (ADR-016): the shared guard refuses it before any flag is read, whatever
``--preregistration`` names, ADR-012 included.

Two refusals, encoded rather than remembered:

* **The test window opens only through a mapped mode.** ``scripts/preregistration.py``, shared with
  the GNN runner, maps each gated mode to the one accepted pre-registration that may open it, and the
  batch reads its seed count from that document rather than from a flag. The map is empty since
  ADR-016; a later gated batch adds its own mode there. The same guard refuses a dirty tree, and
  ``--allow-dirty``, in any gated mode.
* **A dirty tree is refused** (``--allow-dirty`` for smoke runs whose rows you intend to discard):
  rows must be reproducible from a commit — dirty rows have forced full re-runs twice.

Assembles no gate file and writes none. Every row carries ROC-AUC, AUPRC, and the window's prevalence
and positive count (ADR-007).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The guard lives in one module so this runner and the GNN runner cannot drift apart.
from preregistration import require_clean_tree, require_gated_preregistration  # noqa: E402


SCORES_DIR = REPO_ROOT / "experiments" / "scores"
PILOT_SEEDS = 5          # ADR-012 clause 4; the floor runs 3 when its winner is deterministic
MODES = ("retune", "pilot", "gate")


def resolve_floor_batch(mode: str, args) -> tuple[list[dict], list[int], str, str]:
    """``(param overrides, seeds, window, experiment tag)`` — the mode policy in one place.

    Mirrors the GNN runner deliberately: the same modes, the same shared guard for any gated mode,
    and every mode named, so none falls through to the test window (ADR-016 clause 4).
    """
    from adapters.dgf1.baselines_tabular import FLOOR_GRID

    if mode == "retune":
        grid = [{"max_depth": d, "subsample": s}
                for d in FLOOR_GRID["max_depth"] for s in FLOOR_GRID["subsample"]]
        return grid, [0], "val", "retune"

    if mode == "pilot":
        if args.max_depth is None or args.subsample is None:
            raise SystemExit(f"--mode {mode} needs --max-depth and --subsample (the retune winner).")
        winner = [{"max_depth": int(args.max_depth), "subsample": float(args.subsample)}]
        # Clause 2: 3 seeds when the winner is deterministic (enough to evidence identical rows),
        # 5 when it subsamples — and then clause 5 requires it to match the GNN's count.
        n = 3 if float(args.subsample) == 1.0 else PILOT_SEEDS
        return winner, list(range(n)), "val", "pilot"

    if mode == "gate":
        # Closed (ADR-016 clause 3). The shared guard refuses it before any flag is read, whatever
        # --preregistration names; the raise below only makes that visible here.
        require_gated_preregistration(mode, args.preregistration, args.allow_dirty)
        raise SystemExit("refusing --mode gate: closed by ADR-016.")

    raise SystemExit(f"refusing --mode {mode!r}: not one of {', '.join(MODES)} (ADR-016 clause 4).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--models", default="xgboost")
    ap.add_argument("--feature-sets", default="parity,raw17")
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--subsample", type=float, default=None)
    ap.add_argument("--preregistration", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    configs, seeds, window, tag = resolve_floor_batch(args.mode, args)
    require_clean_tree(args.allow_dirty)

    from adapters.dgf1.baselines_tabular import run_floor
    from adapters.dgf1.datasource_dgraph import load_dgraph
    from adapters.dgf1.eval import dgf1_base_config, dgf1_splits

    data = load_dgraph(REPO_ROOT / "data" / "dgraph", strict=True)
    split = dgf1_splits()[window]
    base = {**dgf1_base_config(), "experiment": tag}
    # Retune runs are selection runs, not results: no score vectors (clause 10 asks for them on
    # pilot and gate runs).
    scores_dir = None if args.mode == "retune" else SCORES_DIR

    print(f"[floor] mode={args.mode} window={window} ({split.test_min}-{split.test_max}) "
          f"models={args.models} feature-sets={args.feature_sets} "
          f"configs={len(configs)} seeds={len(seeds)}")

    results = []
    for model in args.models.split(","):
        for feature_set in args.feature_sets.split(","):
            for overrides in configs:
                for seed in seeds:
                    run_id, m = run_floor(model, feature_set, split, seed, data, base,
                                          overrides=overrides if model == "xgboost" else None,
                                          scores_dir=scores_dir)
                    results.append((model, feature_set, overrides, seed, m))
                    shown = " ".join(f"{k}={v}" for k, v in overrides.items()) if model == "xgboost" else ""
                    print(f"[floor] {model}/{feature_set} {shown} seed={seed}: "
                          f"ROC-AUC={m['fraud_auc']:.4f} AUPRC={m['fraud_auprc']:.4f} "
                          f"(prevalence {m['prevalence']:.4%}, {m['n_score_fraud']:,} fraud)  {run_id}")

    if args.mode == "retune":
        print("\n== floor retune, validation AUPRC (ADR-012 clause 2: selection metric) ==")
        gated = [r for r in results if r[1] == "parity"]
        for model, fs, ov, _, m in sorted(gated, key=lambda r: -r[4]["fraud_auprc"]):
            print(f"  {model}/{fs} {ov}  AUPRC={m['fraud_auprc']:.4f}  ROC-AUC={m['fraud_auc']:.4f}")
        if gated:
            best = max(gated, key=lambda r: r[4]["fraud_auprc"])
            print(f"\n  winner (parity floor): {best[2]}  AUPRC {best[4]['fraud_auprc']:.4f}")
            print("  Record it in ADR-012 as a dated amendment; this script edits no document.")
            if best[2]["subsample"] == 1.0:
                print("  NOTE: the winner does not subsample -> the floor is deterministic and "
                      "s_floor = 0 (clause 2); the pilot must verify its rows are identical.")


if __name__ == "__main__":
    main()
