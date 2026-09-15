"""Run DGF-1 GNN batches — the retune grid, the validation pilot, the gate batch, or the repro check.

    python scripts/run_dgf1_gnn.py --mode retune                       # 9 configs x 1 seed, val
    python scripts/run_dgf1_gnn.py --mode pilot  --lr L --batch-size B # 5 seeds, val
    python scripts/run_dgf1_gnn.py --mode gate   --lr L --batch-size B \
        --preregistration decisions/ADR-012-dgf1-gate1-preregistration.md
    python scripts/run_dgf1_gnn.py --mode repro-check                  # 1 seed, val, no flags

The first three modes are the execution order ADR-012 clause 8 fixes, and each carries its own guard:

* **retune** (clause 3) — the nine `lr` x `batch_size` configurations, one seed each, scored on the
  **validation** window and selected on AUPRC. The winner is the researcher's to record in the ADR;
  this script prints the table and never edits a document.
* **pilot** (clause 4) — 5 seeds at the winning configuration, on **validation**, persisting scores.
  Its variance is what mechanically fixes the stage-1 seed count (clause 5).
* **gate** — the **test** window, and only with a pre-registration that is accepted *and* carries
  `**Stage-1 seeds:** <integer>`. **The seed count comes from that document, not from a flag**, so
  the batch cannot run a different `n` than the one pre-registered.

The fourth is ADR-013 clause 3's re-certification of the trainer after the reported views left it:

* **repro-check** — hard-wired to the **validation** window, seed 0 and ADR-012's winning
  configuration; it accepts no configuration flag and cannot select the test window. Before
  training, it refuses unless its configuration hashes to the reference pilot row's. After, it
  reads its own row back and exits non-zero unless every compared field equals the pilot row's
  **bit-for-bit**, the score-file SHA-256 included. On a mismatch: root cause, never retry (ADR-008
  clause 2). The row is written either way (`RunSession` writes on error); the comparison decides.

Assembles no gate file: the verdict is the researcher's (`gates/GATE-DGF1-1.md`).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preregistration import require_accepted_preregistration, require_clean_tree  # noqa: E402

SCORES_DIR = REPO_ROOT / "experiments" / "scores"
PILOT_SEEDS = 5   # ADR-012 clause 4

# -- ADR-013 clause 3: the re-certification's fixed reference -------------------------------------
REPRO_TAG = "repro_check"
REPRO_REFERENCE_RUN = "dgf1-20260913T232506Z-1acd5067"   # the validation pilot's seed 0
REPRO_REFERENCE_TAG = "pilot"
# ADR-012 clause 3's dated amendment of 2026-09-14 records the winner; written out, not read from a
# flag, so the check cannot be pointed at a different configuration.
REPRO_LR = 3.318335548548489e-4
REPRO_BATCH_SIZE = 2048
# What must be identical. Written out rather than derived from either row, so a key that goes
# missing is a mismatch instead of silently leaving the comparison. `config_hash`, `git_commit`,
# `experiment`, `scores_path`, timestamps and wall-clock differ by construction; the configuration
# is checked through the hash rebuild before training instead.
REPRO_METRIC_KEYS: tuple[str, ...] = (
    "fraud_auc", "fraud_auprc", "fraud_f1", "fraud_precision", "fraud_recall",
    "n_score", "n_score_fraud", "prevalence", "n_train", "scores_sha256",
    "arm", "window", "features", "backbone", "lr", "batch_size", "epochs",
    "deterministic", "cublas_workspace_config",
)
REPRO_ROW_COLUMNS: tuple[str, ...] = ("model", "phase", "seed", "data_snapshot_id")


def resolve_batch(mode: str, args) -> tuple[list[tuple[float, int]], list[int], str, str]:
    """``(configs, seeds, window, experiment_tag)`` for one mode — the whole policy in one place."""
    from adapters.dgf1.eval import dgf1_retune_grid

    if mode == "retune":
        return dgf1_retune_grid(), [0], "val", "retune"

    if mode == "repro-check":
        given = [flag for flag, value in (("--lr", args.lr), ("--batch-size", args.batch_size),
                                          ("--preregistration", args.preregistration))
                 if value is not None]
        if given:
            raise SystemExit(f"refusing --mode repro-check with {', '.join(given)}: it is hard-wired "
                             "to ADR-012's winner, seed 0 and the validation window (ADR-013 clause 3).")
        return [(REPRO_LR, REPRO_BATCH_SIZE)], [0], "val", REPRO_TAG

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


# -- repro check: pure helpers, unit-tested without a run -------------------------------------------
def read_row(registry_path: Path, run_id: str) -> dict[str, Any]:
    """The one registry row with ``run_id``, its ``metrics_json`` parsed into ``metrics``."""
    with Path(registry_path).open(newline="", encoding="utf-8") as fh:
        found = [r for r in csv.DictReader(fh) if r["run_id"] == run_id]
    if len(found) != 1:
        raise SystemExit(f"expected exactly one registry row {run_id}, found {len(found)}.")
    row = dict(found[0])
    row["metrics"] = json.loads(row.pop("metrics_json"))
    return row


def compare_repro(reference: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    """Every way ``observed`` fails to reproduce ``reference``; empty means it reproduces.

    Exact equality, deliberately (ADR-008): a 1-ULP drift is still a changed computation.
    """
    problems: list[str] = []
    for col in REPRO_ROW_COLUMNS:
        if observed.get(col) != reference.get(col):
            problems.append(f"{col}: reference {reference.get(col)!r} != observed {observed.get(col)!r}")

    ref_m, obs_m = reference["metrics"], observed["metrics"]
    for key in REPRO_METRIC_KEYS:
        if key not in ref_m or key not in obs_m:
            problems.append(f"{key}: missing from the {'reference' if key not in ref_m else 'observed'} row")
        elif obs_m[key] != ref_m[key]:
            problems.append(f"{key}: reference {ref_m[key]!r} != observed {obs_m[key]!r}")

    if ref_m.get("experiment") != REPRO_REFERENCE_TAG:
        problems.append(f"reference row is tagged {ref_m.get('experiment')!r}, not {REPRO_REFERENCE_TAG!r}")
    if obs_m.get("experiment") != REPRO_TAG:
        problems.append(f"observed row is tagged {obs_m.get('experiment')!r}, not {REPRO_TAG!r}")
    if observed.get("git_dirty") != "false":
        problems.append(f"observed row has git_dirty={observed.get('git_dirty')!r}")
    if "ERRORED" in observed.get("notes", ""):
        problems.append("observed row is marked ERRORED")
    leaked = sorted(k for k in obs_m if k.startswith("window_only_") or k.startswith("first_appearance_"))
    if leaked:
        problems.append(f"observed row carries withdrawn reported-view keys (ADR-013): {leaked}")
    return problems


def score_file_problems(row: dict[str, Any]) -> list[str]:
    """Re-hash the score file a row points to; the row's hash was computed before the file was
    written, so this confirms the file on disk is the vector the row describes."""
    from gbe.eval import load_scores
    from gbe.eval.scores import content_hash

    path = Path(row["metrics"].get("scores_path", ""))
    if not path.is_file():
        return [f"score file {path} does not exist"]
    s = load_scores(path)
    on_disk = content_hash(s["node_ids"], s["y_true"], s["proba"])
    if on_disk != row["metrics"].get("scores_sha256"):
        return [f"score file {path.name} hashes to {on_disk}, row says {row['metrics'].get('scores_sha256')}"]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("retune", "pilot", "gate", "repro-check"), required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--preregistration", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    configs, seeds, window, tag = resolve_batch(args.mode, args)
    if args.mode == "repro-check" and args.allow_dirty:
        raise SystemExit("refusing --mode repro-check with --allow-dirty: a dirty row certifies nothing.")
    require_clean_tree(args.allow_dirty)

    from gbe.gnn import GNNHParams
    from gbe.run.config import resolve_config
    from gbe.run.registry import default_registry_path
    from adapters.dgf1.datasource_dgraph import load_dgraph
    from adapters.dgf1.eval import dgf1_base_config, dgf1_hparams, dgf1_splits
    from adapters.dgf1.train_gnn import run_config_values, run_dgf1

    base_hp, cfg_device = dgf1_hparams()
    split = dgf1_splits()[window]
    base_cfg = {**dgf1_base_config(), "phase": "P1", "experiment": tag}
    registry_path = default_registry_path(REPO_ROOT)

    def hparams(lr: float, batch_size: int) -> GNNHParams:
        return GNNHParams(**{**base_hp.as_dict(), "lr": lr, "batch_size": batch_size,
                             "fan_out": tuple(base_hp.fan_out)})

    reference = None
    if args.mode == "repro-check":
        # Before loading 4M nodes or training: the check must run the reference's configuration.
        reference = read_row(registry_path, REPRO_REFERENCE_RUN)
        (lr, batch_size), = configs
        as_reference = run_config_values(seeds[0], split, hparams(lr, batch_size),
                                         {**base_cfg, "experiment": REPRO_REFERENCE_TAG})
        rebuilt = resolve_config(as_reference).config_hash
        if rebuilt != reference["config_hash"]:
            raise SystemExit(f"refusing repro-check: this configuration hashes to {rebuilt}, the "
                             f"reference row to {reference['config_hash']}. Something moved "
                             "(config.yaml, hyperparameters or split); find it before running.")
        print(f"[dgf1] repro-check: configuration rebuilds reference hash {rebuilt[:12]}")

    data = load_dgraph(REPO_ROOT / "data" / "dgraph", strict=True)
    # Retune runs are selection runs, not results: they persist no score vectors (clause 10 asks
    # for them on pilot and gate runs).
    scores_dir = None if args.mode == "retune" else SCORES_DIR

    print(f"[dgf1] mode={args.mode} window={window} ({split.test_min}-{split.test_max}) "
          f"configs={len(configs)} seeds={len(seeds)} device={args.device or cfg_device}")

    results = []
    for lr, batch_size in configs:
        hp = hparams(lr, batch_size)
        for seed in seeds:
            run_id, m = run_dgf1(seed, data, split, hp, base_cfg, device=args.device or cfg_device,
                                 scores_dir=scores_dir)
            results.append((lr, batch_size, seed, m, run_id))
            print(f"[dgf1] lr={lr:.2e} batch={batch_size} seed={seed}: "
                  f"ROC-AUC={m['fraud_auc']:.4f} AUPRC={m['fraud_auprc']:.4f} "
                  f"(prevalence {m['prevalence']:.4%}, {m['n_score_fraud']:,} fraud)  {run_id}")

    if args.mode == "retune":
        print("\n== retune, validation AUPRC (ADR-012 clause 3: selection metric) ==")
        for lr, batch_size, _, m, _ in sorted(results, key=lambda r: -r[3]["fraud_auprc"]):
            print(f"  lr={lr:.3e}  batch={batch_size:<5} AUPRC={m['fraud_auprc']:.4f}  "
                  f"ROC-AUC={m['fraud_auc']:.4f}")
        best = max(results, key=lambda r: r[3]["fraud_auprc"])
        print(f"\n  winner: lr={best[0]:.6e} batch_size={best[1]} "
              f"(AUPRC {best[3]['fraud_auprc']:.4f})")
        print("  Record it in ADR-012 as a dated amendment; this script edits no document.")
    elif args.mode == "pilot":
        print("\n[dgf1] pilot done. Compute s_GNN and Δ_val against the floor's pilot rows, apply "
              "clause 5's rule, and record '**Stage-1 seeds:** <n>' in ADR-012 before any gate run.")
    elif args.mode == "repro-check":
        observed = read_row(registry_path, results[0][4])   # what was written, not what is in memory
        problems = compare_repro(reference, observed) + score_file_problems(observed)
        print(f"\n== repro-check (ADR-013 clause 3): {observed['run_id']} vs {REPRO_REFERENCE_RUN} ==")
        if problems:
            for p in problems:
                print(f"  MISMATCH  {p}")
            print("  Root-cause it; never retry (ADR-008 clause 2). No later DGF-1 run may rely on the "
                  "trainer until this passes.")
            raise SystemExit(1)
        print(f"  REPRODUCED: {len(REPRO_METRIC_KEYS)} metric fields and {len(REPRO_ROW_COLUMNS)} "
              "row columns bit-for-bit, score file hash included.")
        print("  Record the result in the lab notebook and the timeline; Gate 3's ADR cites it.")


if __name__ == "__main__":
    main()
