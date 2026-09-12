"""Run the DGF-1 tabular floor — one registry row per (model, feature set, seed) on one window.

    python scripts/run_dgf1_floor.py --window val  [--models rf,lr] [--feature-sets parity,raw17] [--seeds 0,1,2]
    python scripts/run_dgf1_floor.py --window test --preregistration decisions/ADR-0NN-....md ...

Two refusals, encoded rather than remembered:

* **The test window needs an accepted pre-registration.** CLAUDE.md: DGF-1's Gate-1
  pre-registration ADR must be accepted before any run touches 482–821. ``--window test`` is refused
  unless ``--preregistration`` names an ADR file whose ``**Status:**`` line reads ``accepted``.
* **A dirty tree is refused** (``--allow-dirty`` for smoke runs whose rows you intend to discard):
  rows must be reproducible from a commit — dirty rows have forced full re-runs twice.

Assembles no gate file and writes none. Every row carries ROC-AUC, AUPRC, and the window's prevalence
and positive count (ADR-007).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gbe.run.config import git_dirty  # noqa: E402

_STATUS = re.compile(r"^\*\*Status:\*\*\s*(\w+)", re.MULTILINE)
# ADR-012 clause 8 fixes this line's exact shape, so the guard is not left guessing a format.
_SEEDS = re.compile(r"^\*\*Stage-1 seeds:\*\*\s*(\d+)", re.MULTILINE)


def require_accepted_preregistration(path: str | Path | None) -> int:
    """Exit unless ``path`` is an ADR that is **accepted** *and* carries a stage-1 seed count.

    Both conditions guard the test window (ADR-012 clause 8). The second one matters on its own:
    ADR-012 fixes the *rule* for the seed count and leaves the *number* to a validation pilot, so an
    accepted ADR whose count has not yet been derived must still keep 482–821 shut. Returns the
    count, so a caller can record what it ran under.
    """
    if path is None:
        raise SystemExit(
            "refusing --window test: no --preregistration given. DGF-1's Gate-1 pre-registration ADR "
            "must be accepted before any run touches the test window (CLAUDE.md)."
        )
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"refusing --window test: pre-registration {p} does not exist.")
    text = p.read_text(encoding="utf-8")

    status = _STATUS.search(text)
    if not status or status.group(1).lower() != "accepted":
        raise SystemExit(
            f"refusing --window test: {p.name} status is "
            f"{status.group(1) if status else 'missing'!r}, not 'accepted'."
        )

    seeds = _SEEDS.search(text)
    if not seeds:
        raise SystemExit(
            f"refusing --window test: {p.name} carries no stage-1 seed count. ADR-012 clause 5 "
            "derives it from the validation pilot; record it as '**Stage-1 seeds:** <integer>' "
            "before any test-window run."
        )
    return int(seeds.group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", choices=("val", "test"), required=True)
    ap.add_argument("--models", default="rf,lr")
    ap.add_argument("--feature-sets", default="parity,raw17")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--preregistration", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    if args.window == "test":
        n_seeds = require_accepted_preregistration(args.preregistration)
        print(f"[floor] test window unlocked by {Path(args.preregistration).name}: "
              f"stage-1 seeds = {n_seeds}")
    if git_dirty() and not args.allow_dirty:
        raise SystemExit("refusing to run: uncommitted code/config, rows would log git_dirty=true.")

    from adapters.dgf1.baselines_tabular import run_floor
    from adapters.dgf1.datasource_dgraph import load_dgraph
    from adapters.dgf1.eval import dgf1_base_config, dgf1_splits

    data = load_dgraph(REPO_ROOT / "data" / "dgraph", strict=True)
    split = dgf1_splits()[args.window]
    base = dgf1_base_config()
    for model in args.models.split(","):
        for feature_set in args.feature_sets.split(","):
            for seed in (int(s) for s in args.seeds.split(",")):
                run_id, m = run_floor(model, feature_set, split, seed, data, base)
                print(f"[floor] {model}/{feature_set} seed={seed} window={m['window']}: "
                      f"ROC-AUC={m['fraud_auc']:.4f} AUPRC={m['fraud_auprc']:.4f} "
                      f"(prevalence {m['prevalence']:.4%}, {m['n_score_fraud']:,} fraud)  {run_id}")


if __name__ == "__main__":
    main()
