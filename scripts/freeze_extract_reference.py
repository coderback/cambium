"""Freeze the EXTRACT regression reference set to an immutable manifest (ADR-008).

    python scripts/freeze_extract_reference.py [--out experiments/extract_reference_manifest.json]

**Run this ONCE, before the refactor starts.** It is a one-shot tool, not a maintained utility.

**Why a frozen manifest rather than resolving identities at check time.** ADR-008's bar is
bit-for-bit equality against 49 recorded rows. Two of the three reference batches cannot be
identified from their rows alone:

* **Gate-3's 40 rows carry no ``ablation`` key** — they predate `PROVENANCE_KEYS` (added
  2026-07-25). Their arm lives only inside the *hashed config*, so recovering it means rebuilding
  each ``(arm, seed)`` config exactly as `run_gnn` built it and matching ``config_hash`` — which is
  what `run_ell1_ablations.arm_run_ids` does, using `frozen_hparams()` and `GNNHParams` from the
  adapter.
* **That recovery cannot survive EXTRACT.** A refactor changes config shape; changed config content
  changes the hash; and the adapter code the rebuild depends on is precisely what moves. So the
  identities must be resolved while the *pre-refactor* code still exists, and then never re-derived.

After this script runs, the manifest is standalone: `check_extract_regression.py` pairs rows by
``(source, arm, seed)`` and never touches a config hash. ``config_hash`` is stored per row for
audit only.

Nothing here trains, and nothing appends a registry row — this script only reads.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# One-shot import of the pre-refactor arm-recovery logic. `scripts/` is not a package, but
# sys.path[0] is this directory when the file is run as a script; the insert makes the import work
# under `python -c` and pytest too. Deliberately reusing `arm_run_ids` rather than reimplementing
# it: a second copy of the hash-rebuild rule could drift from the one that assembled GATE-ELL1-3,
# and drift is the exact failure this manifest exists to prevent.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ell1_ablations import ARMS, arm_run_ids  # noqa: E402

from gbe.run.config import git_commit  # noqa: E402
from gbe.run.registry import default_registry_path  # noqa: E402
from adapters.ell1.train_gnn import frozen_hparams  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "experiments" / "extract_reference_manifest.json"
ENV_PIN = "experiments/extract_reference_env.txt"

GATE3_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]     # ADR-006 stage 1
GCN_SEEDS = [0, 1, 2]                       # ADR-008 clause 3
BASELINES = ("rf", "lr")                    # Gate-0 clean batch
GATE0_SEEDS = [0, 1, 2]

# The four metrics every reference row carries. `illicit_auprc` exists only on the GCN rows
# (created after ADR-007) and is picked up opportunistically below — ADR-008: a new key is not a
# regression, so each row is compared over the metrics it actually recorded.
BASE_METRICS = ("illicit_f1", "illicit_recall", "illicit_precision", "illicit_auc")
OPTIONAL_METRICS = ("illicit_auprc",)

EXPECTED_TOTAL = 49


def _rows(registry_path: Path) -> list[dict]:
    with registry_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _entry(source: str, arm: str, row: dict) -> dict:
    """One manifest entry: semantic identity + the metrics that must reproduce exactly."""
    m = json.loads(row["metrics_json"])
    metrics = {k: float(m[k]) for k in BASE_METRICS if k in m}
    metrics.update({k: float(m[k]) for k in OPTIONAL_METRICS if k in m})
    missing = set(BASE_METRICS) - set(metrics)
    if missing:
        raise RuntimeError(
            f"row {row['run_id']} ({source}/{arm}) is missing required metrics {sorted(missing)}; "
            "refusing to write an incomplete manifest."
        )
    return {
        "source": source,
        "arm": arm,
        "seed": int(row["seed"]),
        "run_id": row["run_id"],
        "config_hash": row["config_hash"],   # audit only — never used for pairing
        "git_commit": row["git_commit"],
        "deterministic": json.loads(row["metrics_json"]).get("deterministic"),
        "metrics": metrics,
    }


def collect_gate0(rows: list[dict]) -> list[dict]:
    """Gate-0's clean tabular floor: 6 rows, identified directly by ``baseline`` + seed."""
    out = []
    for baseline in BASELINES:
        for seed in GATE0_SEEDS:
            match = [
                r for r in rows
                if r["phase"] == "P0" and r["git_dirty"] == "false"
                and int(r["seed"]) == seed
                and json.loads(r["metrics_json"]).get("baseline") == baseline
            ]
            if len(match) != 1:
                raise RuntimeError(
                    f"expected exactly 1 clean Gate-0 row for baseline={baseline} seed={seed}, "
                    f"found {len(match)}."
                )
            out.append(_entry("gate0", baseline, match[0]))
    return out


def collect_gate3(registry_path: Path, rows: list[dict]) -> list[dict]:
    """Gate-3's 40 deterministic rows, arm recovered by pre-refactor config-hash rebuild."""
    hp, base_cfg, _ = frozen_hparams()
    ids = arm_run_ids(registry_path, hp, base_cfg, GATE3_SEEDS)   # raises on any gap
    by_id = {r["run_id"]: r for r in rows}
    out = []
    for arm in ARMS:
        for seed, run_id in zip(GATE3_SEEDS, ids[arm]):
            row = by_id[run_id]
            if json.loads(row["metrics_json"]).get("deterministic") is not True:
                raise RuntimeError(
                    f"Gate-3 reference row {run_id} (arm={arm} seed={seed}) is not marked "
                    "deterministic; bit-for-bit equality would be unenforceable against it."
                )
            out.append(_entry("gate3", arm, row))
    return out


def collect_gcn(rows: list[dict]) -> list[dict]:
    """The GCN reference created for ADR-008 clause 3: 3 rows, tagged ``extract_reference``."""
    out = []
    for seed in GCN_SEEDS:
        match = [
            r for r in rows
            if r["git_dirty"] == "false" and int(r["seed"]) == seed
            and json.loads(r["metrics_json"]).get("experiment") == "extract_reference"
            and json.loads(r["metrics_json"]).get("backbone") == "gcn"
        ]
        if len(match) != 1:
            raise RuntimeError(
                f"expected exactly 1 extract_reference GCN row for seed={seed}, "
                f"found {len(match)}. Run scripts/run_extract_reference.py first."
            )
        out.append(_entry("gcn_ref", "gcn", match[0]))
    return out


def build_manifest(registry_path: Path) -> dict:
    rows = _rows(registry_path)
    entries = collect_gate0(rows) + collect_gate3(registry_path, rows) + collect_gcn(rows)

    if len(entries) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"expected {EXPECTED_TOTAL} reference rows (6 gate0 + 40 gate3 + 3 gcn_ref), "
            f"built {len(entries)}. Refusing to write a partial manifest."
        )
    seen = {(e["source"], e["arm"], e["seed"]) for e in entries}
    if len(seen) != len(entries):
        raise RuntimeError("duplicate (source, arm, seed) identity in the reference set.")

    return {
        "adr": "ADR-008",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(REPO_ROOT),
        "env_pin": ENV_PIN,
        "note": (
            "Frozen pre-refactor. Pair post-refactor rows by (source, arm, seed) — never by "
            "config_hash, which a refactor changes. Bar is exact float equality (ADR-008 clause 2)."
        ),
        "counts": {
            "gate0": len([e for e in entries if e["source"] == "gate0"]),
            "gate3": len([e for e in entries if e["source"] == "gate3"]),
            "gcn_ref": len([e for e in entries if e["source"] == "gcn_ref"]),
            "total": len(entries),
        },
        "rows": entries,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="manifest output path")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest (default: refuse)")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(
            f"{out} already exists. The manifest is the frozen reference and should be written "
            "once, before the refactor — re-freezing after code has changed would silently "
            "rebase the bar onto new numbers. Pass --force only if you are certain."
        )

    manifest = build_manifest(default_registry_path(REPO_ROOT))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    c = manifest["counts"]
    print(f"[freeze] wrote {out.relative_to(REPO_ROOT)}")
    print(f"[freeze] {c['total']} reference rows: {c['gate0']} gate0 + {c['gate3']} gate3 "
          f"+ {c['gcn_ref']} gcn_ref")
    print(f"[freeze] commit {manifest['git_commit'][:8]} | env pin {manifest['env_pin']}")
    n_auprc = sum(1 for e in manifest["rows"] if "illicit_auprc" in e["metrics"])
    print(f"[freeze] {n_auprc}/{c['total']} rows carry illicit_auprc (post-ADR-007 rows only; "
          "a new key is not a regression)")


if __name__ == "__main__":
    main()
