"""Run the one-time ELL-1 HPO sweep (doc-00 §6.4) and report the winner.

    python scripts/run_ell1_hpo.py [--trials N] [--device cuda|cpu] [--storage URL]

Tunes GraphSAGE on the inner temporal split (train 1-29 / val 30-34, strict inductive),
maximising validation illicit-F1. Trials are NOT registry rows (the sweep is a meta-process,
doc §6.4 "one lab entry + an ADR"); the full study is saved to experiments/ell1_hpo_study.csv
for provenance, and the best config is printed for ADR-003 / config.yaml. The 35-49 test
window is never touched here.

Resumable: the study persists to SQLite (``--storage``), so an intermittent CUDA fault on the
4 GB laptop GPU loses only the in-flight trial. Each invocation optimises the *remaining*
trials up to ``--trials`` total and exits 0 once the target is met — so an outer
``until scripts/run_ell1_hpo.py ...; do :; done`` loop drives the sweep to completion across
process restarts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adapters.ell1.datasource_elliptic import load_elliptic
from adapters.ell1.hpo import HPO_TRIALS, INNER_SPLIT, build_study, make_objective
from adapters.ell1.train_gnn import resolve_device

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_CSV = REPO_ROOT / "experiments" / "ell1_hpo_study.csv"
DEFAULT_DB = REPO_ROOT / "experiments" / "ell1_hpo.db"


def _n_finished(study) -> int:
    return sum(1 for t in study.trials if t.state.is_finished())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=HPO_TRIALS, help="target TOTAL finished trials")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--storage", default=f"sqlite:///{DEFAULT_DB}", help="Optuna storage URL")
    args = ap.parse_args()

    device = resolve_device(args.device)
    study = build_study(storage=args.storage)
    done = _n_finished(study)
    remaining = max(0, args.trials - done)
    print(f"[hpo] device={device} inner=train<=({INNER_SPLIT.train_max}) "
          f"val[{INNER_SPLIT.test_min}-{INNER_SPLIT.test_max}] | {done}/{args.trials} done, "
          f"{remaining} to go")

    if remaining > 0:
        data = load_elliptic(REPO_ROOT / "data" / "elliptic", strict=True)
        study.optimize(make_objective(data, device), n_trials=remaining, show_progress_bar=False)

    # reached the target (this line only executes if optimize did not crash the process)
    df = study.trials_dataframe()
    STUDY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(STUDY_CSV, index=False)
    pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    print(f"[hpo] {_n_finished(study)} finished ({pruned} pruned); study -> "
          f"{STUDY_CSV.relative_to(REPO_ROOT)}")
    print(f"[best] val illicit-F1 = {study.best_value:.4f}  (trial {study.best_trial.number})")
    print("[best params]")
    for k, v in sorted(study.best_params.items()):
        print(f"    {k}: {v}")
    print("\nFreeze these in ADR-003 + adapters/ell1/config.yaml (gnn: block), then run "
          "scripts/run_ell1_gnn.py for the 3-seed Gate-1 batch.")


if __name__ == "__main__":
    main()
