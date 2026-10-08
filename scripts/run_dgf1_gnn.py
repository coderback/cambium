"""Run DGF-1 GNN batches — the retune grid, the validation pilot, or the repro check.

    python scripts/run_dgf1_gnn.py --mode retune                       # 9 configs x 1 seed, val
    python scripts/run_dgf1_gnn.py --mode pilot  --lr L --batch-size B # 5 seeds, val
    python scripts/run_dgf1_gnn.py --mode repro-check                  # 1 seed, val, no flags

Retune, pilot and gate are the execution order ADR-012 clause 8 fixes, and each carries its own guard:

* **retune** (clause 3) — the nine `lr` x `batch_size` configurations, one seed each, scored on the
  **validation** window and selected on AUPRC. The winner is the researcher's to record in the ADR;
  this script prints the table and never edits a document.
* **pilot** (clause 4) — 5 seeds at the winning configuration, on **validation**, persisting scores.
  Its variance is what mechanically fixes the stage-1 seed count (clause 5).
* **gate** — **closed** (ADR-016). It ran Gate 1's one pre-registered **test** batch at `09c6e72`,
  and the verdict is signed (`gates/GATE-DGF1-1.md`). The mode is kept so its refusal says why: the
  shared guard in `scripts/preregistration.py` refuses it before any flag is read, whatever
  `--preregistration` names, ADR-012 included. A later gated batch opens the test window only through
  its own mode, mapped there to its own accepted pre-registration, which fixes its seed count.

The fourth is ADR-013 clause 3's re-certification of the trainer after the reported views left it:

* **repro-check** — hard-wired to the **validation** window, seed 0 and ADR-012's winning
  configuration; it accepts no configuration flag and cannot select the test window. Before
  training, it refuses unless its configuration hashes to the reference pilot row's. After, it
  reads its own row back and exits non-zero unless every compared field equals the pilot row's
  **bit-for-bit**, the score-file SHA-256 included. On a mismatch: root cause, never retry (ADR-008
  clause 2). A run that *raises* still writes its row (`RunSession` writes on error), but the
  comparison is not reached — the traceback is the report, and the row is the record.

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

from preregistration import require_clean_tree, require_gated_preregistration  # noqa: E402

SCORES_DIR = REPO_ROOT / "experiments" / "scores"
PIN_PATH = REPO_ROOT / "experiments" / "extract_reference_env.txt"   # ADR-008 clause 4
PILOT_SEEDS = 5   # ADR-012 clause 4
MODES = ("retune", "pilot", "gate", "repro-check")

# -- ADR-013 clause 3: the re-certification's fixed reference -------------------------------------
REPRO_TAG = "repro_check"
REPRO_REFERENCE_RUN = "dgf1-20260913T232506Z-1acd5067"   # the validation pilot's seed 0
REPRO_REFERENCE_TAG = "pilot"
# ADR-012 clause 3's dated amendment of 2026-09-14 records the winner; written out, not read from a
# flag, so the check cannot be pointed at a different configuration.
REPRO_LR = 3.318335548548489e-4
REPRO_BATCH_SIZE = 2048
# What must be identical. Written out rather than derived from either row, so a key that goes
# missing *from a row* is a mismatch instead of silently leaving the comparison. A literal list
# has its own failure mode — a key quietly dropped from the list itself — so both tuples are pinned
# literally in `tests/test_dgf1_runner.py`, each entry is shown to be compared, and a row key that
# belongs in neither tuple nor DELIBERATELY_UNCOMPARED fails there too.
# `config_hash`, `git_commit`, `experiment`, `scores_path`, timestamps and wall-clock differ by
# construction; the configuration is checked through the hash rebuild before training instead.
REPRO_METRIC_KEYS: tuple[str, ...] = (
    "fraud_auc", "fraud_auprc", "fraud_f1", "fraud_precision", "fraud_recall",
    "n_score", "n_score_fraud", "prevalence", "n_train", "scores_sha256",
    "arm", "window", "features", "backbone", "lr", "batch_size", "epochs",
    "deterministic", "cublas_workspace_config",
)
REPRO_ROW_COLUMNS: tuple[str, ...] = ("model", "phase", "seed", "data_snapshot_id")
# Row metric keys the comparison deliberately skips, each for a stated reason. Anything a DGF-1 row
# carries that is in neither this set nor REPRO_METRIC_KEYS fails the coverage test, so a metric
# added to the trainer cannot slip past the check unnoticed.
DELIBERATELY_UNCOMPARED: frozenset[str] = frozenset({
    "experiment",    # differs by construction: 'pilot' vs 'repro_check', checked separately
    "scores_path",   # absolute and run-id specific; the file's content is checked by its hash
})


def resolve_batch(mode: str, args) -> tuple[list[tuple[float, int]], list[int], str, str]:
    """``(configs, seeds, window, experiment_tag)`` for one mode — the whole policy in one place.

    Every mode is named, so none falls through to the test window (ADR-016 clause 4).
    """
    from adapters.dgf1.eval import dgf1_retune_grid

    if mode == "retune":
        return dgf1_retune_grid(), [0], "val", "retune"

    if mode == "repro-check":
        # Every flag that could redirect the check, refused in one place, before any side effect.
        # `--device` is here because the device is in neither the hashed config nor the row: a CPU
        # run would clear the pre-run hash guard and then fail every float comparison, condemning a
        # sound trainer under a rule (ADR-008 clause 2) that forbids retrying.
        given = [flag for flag, redirected in (("--lr", args.lr is not None),
                                               ("--batch-size", args.batch_size is not None),
                                               ("--preregistration", args.preregistration is not None),
                                               ("--device", args.device is not None),
                                               ("--allow-dirty", bool(getattr(args, "allow_dirty", False))))
                 if redirected]
        if given:
            raise SystemExit(f"refusing --mode repro-check with {', '.join(given)}: it is hard-wired "
                             "to ADR-012's winner, seed 0, the validation window and the configured "
                             "device, and a dirty row certifies nothing (ADR-013 clause 3).")
        return [(REPRO_LR, REPRO_BATCH_SIZE)], [0], "val", REPRO_TAG

    if mode == "pilot":
        if args.lr is None or args.batch_size is None:
            raise SystemExit(f"--mode {mode} needs --lr and --batch-size (the retune winner).")
        return [(float(args.lr), int(args.batch_size))], list(range(PILOT_SEEDS)), "val", "pilot"

    if mode == "gate":
        # Closed (ADR-016 clause 3). The shared guard refuses it before any flag is read, whatever
        # --preregistration names; the raise below only makes that visible here.
        require_gated_preregistration(mode, args.preregistration, args.allow_dirty)
        raise SystemExit("refusing --mode gate: closed by ADR-016.")

    raise SystemExit(f"refusing --mode {mode!r}: not one of {', '.join(MODES)} (ADR-016 clause 4).")


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
    """Re-hash the score file a row points to, and report any disagreement with the row.

    A row's ``scores_sha256`` is computed from the in-memory arrays, not from the bytes on disk
    (`gbe/eval/scores.py`: ``np.savez`` then ``content_hash``), so re-hashing the file is what ties
    the vector still on disk to the run that recorded it.
    """
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


def pin_runtime(pin_path: Path) -> dict[str, str]:
    """The environment pin's ``[runtime]`` block as ``{field: value}``.

    `check_extract_regression._pinned_versions` reads only the pin's ``[pip freeze]`` section, so the
    runtime fields need their own parser. Two shapes to allow for, both present in the real file: the
    block is CRLF, and ``cublas_workspace_config`` carries a trailing prose comment after the value.
    """
    fields: dict[str, str] = {}
    in_runtime = False
    for raw in pin_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_runtime = line == "[runtime]"
            continue
        if not in_runtime or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        fields[key.strip()] = value.strip().split("  ")[0].strip()   # drop any trailing comment
    return fields


def device_problems(pin_path: Path, resolved_device: str) -> list[str]:
    """Refuse a device that cannot reproduce the reference. Empty means the device is usable.

    **Deliberately not guarded by `torch.cuda.is_available()`.** The bug this closes is exactly the
    case where CUDA has gone away: `environment_drift` puts its GPU and CUDA comparisons behind that
    call, and `resolve_device("auto")` then falls back to CPU silently, so the check would report no
    drift, train on CPU, differ on every float, and record a sound trainer as a regression under a
    rule that forbids retrying (ADR-008 clause 2). A conditional check is vacuous precisely when it
    is needed.

    `adapters/dgf1/config.yaml`'s `device:` reaches the same place with no driver failure: it is
    returned by `dgf1_hparams()`, not `dgf1_base_config()`, so it sits in neither the hashed config
    nor the row, and editing it passes the configuration-hash guard untouched.
    """
    pinned_gpu = pin_runtime(pin_path).get("gpu", "")
    if not pinned_gpu:
        return [f"{pin_path.name} records no gpu in its [runtime] block, so the device cannot be "
                "verified against the reference"]
    if resolved_device.split(":")[0] == "cpu":
        return [f"resolved device {resolved_device!r}, but the reference was produced on "
                f"{pinned_gpu!r}. A CPU run cannot reproduce a CUDA reference: every float would "
                "differ, and the result would be recorded as a regression rather than as the "
                "environment change it is."]
    return []


def rebuilt_reference_hash(split, hp, base_cfg: dict[str, Any]) -> str:
    """The config hash this check's run *would* carry if it were tagged as the reference pilot.

    The tag override is the whole point: the check's own row is tagged ``repro_check``, and
    ``experiment`` is inside the hashed config, so comparing hashes at all requires rebuilding the
    reference's tag. Everything else — split, hyperparameters, snapshot, seed — must match.
    """
    from gbe.run.config import resolve_config
    from adapters.dgf1.train_gnn import run_config_values

    values = run_config_values(0, split, hp, {**base_cfg, "experiment": REPRO_REFERENCE_TAG})
    return resolve_config(values).config_hash


def repro_check_reference(registry_path: Path, rebuilt: str) -> dict[str, Any]:
    """The reference pilot row — refused unless this check would hash to the same configuration.

    Called before the dataset is loaded, so a drifted `config.yaml`, hyperparameter or split costs
    a second rather than a training run that was never going to reproduce anything.
    """
    reference = read_row(registry_path, REPRO_REFERENCE_RUN)
    if rebuilt != reference["config_hash"]:
        raise SystemExit(f"refusing repro-check: this configuration hashes to {rebuilt}, the "
                         f"reference row to {reference['config_hash']}. Something moved "
                         "(config.yaml, hyperparameters or split); find it before running.")
    return reference


def repro_check_verdict(reference: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    """ADR-013 clause 3's bar, both halves: the row reproduces the reference **and** its score file
    still hashes to what the row says. Empty means certified."""
    return compare_repro(reference, observed) + score_file_problems(observed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--preregistration", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    configs, seeds, window, tag = resolve_batch(args.mode, args)   # refuses repro-check's flags
    require_clean_tree(args.allow_dirty)

    from gbe.gnn import GNNHParams
    from gbe.run.registry import default_registry_path
    from adapters.dgf1.datasource_dgraph import load_dgraph
    from adapters.dgf1.eval import dgf1_base_config, dgf1_hparams, dgf1_splits
    from adapters.dgf1.train_gnn import run_dgf1

    base_hp, cfg_device = dgf1_hparams()
    split = dgf1_splits()[window]
    base_cfg = {**dgf1_base_config(), "phase": "P1", "experiment": tag}
    registry_path = default_registry_path(REPO_ROOT)

    def hparams(lr: float, batch_size: int) -> GNNHParams:
        return GNNHParams(**{**base_hp.as_dict(), "lr": lr, "batch_size": batch_size,
                             "fan_out": tuple(base_hp.fan_out)})

    device = args.device or cfg_device
    reference = None
    if args.mode == "repro-check":
        # Both refusals run before 4M nodes are loaded or anything is trained. The device is checked
        # first: it is the cheaper of the two and the likelier to have moved.
        from gbe.run import resolve_device

        device = str(resolve_device(device))   # resolve once, run on what was checked
        problems = device_problems(PIN_PATH, device)
        if problems:
            raise SystemExit("refusing repro-check: " + " ".join(problems))
        print(f"[dgf1] repro-check: device {device} matches the reference environment")

        (lr, batch_size), = configs
        rebuilt = rebuilt_reference_hash(split, hparams(lr, batch_size), base_cfg)
        reference = repro_check_reference(registry_path, rebuilt)
        print(f"[dgf1] repro-check: configuration rebuilds reference hash {rebuilt[:12]}")

    data = load_dgraph(REPO_ROOT / "data" / "dgraph", strict=True)
    # Retune runs are selection runs, not results: they persist no score vectors (clause 10 asks
    # for them on pilot and gate runs).
    scores_dir = None if args.mode == "retune" else SCORES_DIR

    print(f"[dgf1] mode={args.mode} window={window} ({split.test_min}-{split.test_max}) "
          f"configs={len(configs)} seeds={len(seeds)} device={device}")

    results = []
    for lr, batch_size in configs:
        hp = hparams(lr, batch_size)
        for seed in seeds:
            run_id, m = run_dgf1(seed, data, split, hp, base_cfg, device=device,
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
        problems = repro_check_verdict(reference, observed)
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
