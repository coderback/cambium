"""EXTRACT's acceptance test: does the refactored core reproduce ELL-1 bit-for-bit? (ADR-008)

    python scripts/check_extract_regression.py --run            # run the 49-row batch, then compare
    python scripts/check_extract_regression.py --compare-only   # compare HEAD's rows in the registry
    python scripts/check_extract_regression.py --compare-only --commit d614671   # an earlier batch

**A comparison certifies one commit.** Only rows that ran at that commit, on a clean tree, under
deterministic kernels and without error count; an identity that ran twice there is a retry and
fails the check; and a check that compares nothing is not a pass.

ADR-008 replaced doc-02's "ELL-1 still passes its gates on the refactored core" — unsatisfiable,
since Gate 1 **failed** — with: *the refactored core reproduces ELL-1's recorded numbers bit-for-bit
over a fixed reference set, in an unchanged environment.* This script is that test.

**Pairing is by ``(arm, seed)``, never by config hash.** The reference identities are frozen in
`experiments/extract_reference_manifest.json` (see `freeze_extract_reference.py` for why: Gate-3's
40 rows carry no ``ablation`` key, so their arm was only recoverable via a pre-refactor hash
rebuild that the refactor itself breaks). The 8 arm names are globally unique across the three
sources — ``rf``/``lr``, ``real``/``scrambled``/``random``/``no_edges``/``config``, ``gcn`` — so
``(arm, seed)`` identifies a reference row unambiguously. That is asserted, not assumed.

**Post-refactor rows are tagged ``experiment="extract_check"``** via `PROVENANCE_KEYS`, which is what
separates them from the reference rows they are being compared against — those share the same
``(arm, seed)`` and would otherwise be indistinguishable in an append-only registry.

*The extra ``experiment``/``arm`` config keys change each row's config hash and nothing else:
`resolve_config` feeds provenance and the seed, while training reads only ``hp``, the graph and the
split. The keys are numerically inert — which is exactly why pairing must not depend on the hash.*

**The bar is exact float equality, and a mismatch is never retried** (ADR-008 clause 2): a
differing row is a regression to be investigated to root cause. If the change turns out to be
*intended*, that needs its own ADR and a new batch — the dated `gates/GATE-ELL1-*.md` files are
never edited.

This script assembles no gate file and rewrites none. (`run_ell1_ablations.py`,
`run_ell1_baselines.py` and `run_ell1_gnn.py` each overwrite a dated gate file, so none of them may
be reused to produce this batch — but the adapter-level functions they call are reused here, so the
code path under test is the real one.)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from gbe.eval import (
    configuration_model_edges,
    random_graph_edges,
    remove_edges,
    scramble_edges,
)
from gbe.run.config import git_dirty
from gbe.run.registry import default_registry_path
from adapters.ell1.baselines_tabular import prepare_labelled_split, run_baseline
from adapters.ell1.datasource_elliptic import load_elliptic
from adapters.ell1.eval import ell1_eval_split
from adapters.ell1.train_gnn import frozen_hparams, resolve_device, run_gnn

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "experiments" / "extract_reference_manifest.json"
CHECK_TAG = "extract_check"

# Gate-3's five arms. Imported from `gbe.eval` — the core, stable location — rather than from
# run_ell1_ablations, so this script depends on the code under test and not on a sibling script.
GATE3_ARMS = {
    "real": None,
    "scrambled": scramble_edges,
    "random": random_graph_edges,
    "no_edges": lambda ei, ts, seed: remove_edges(ei),
    "config": configuration_model_edges,
}

# ADR-008 clause 4: equality is only meaningful against the pinned stack.
CRITICAL_PACKAGES = ("torch", "torch-geometric", "pyg-lib", "numpy", "scikit-learn")


# --------------------------------------------------------------------------- comparison (pure)


@dataclass(frozen=True)
class Mismatch:
    arm: str
    seed: int
    metric: str
    reference: float
    observed: float

    def __str__(self) -> str:
        return (f"{self.arm}/seed{self.seed} {self.metric}: "
                f"reference {self.reference!r} != observed {self.observed!r}")


def compare(reference_rows: list[dict], observed: dict[tuple[str, int], dict]) -> tuple[list[Mismatch], list[str]]:
    """Diff reference metrics against observed ones. Pure, so it is unit-testable.

    Args:
        reference_rows: manifest ``rows`` entries (``arm``, ``seed``, ``metrics``).
        observed: ``{(arm, seed): {metric: value}}`` from the post-refactor batch.

    Returns:
        ``(mismatches, missing)`` — ``missing`` names reference identities with no observed
        counterpart. Comparison is over the metrics the *manifest* recorded: a metric present only
        in the observed row (e.g. ``illicit_auprc`` on a pre-ADR-007 reference) is not a regression.
    """
    mismatches: list[Mismatch] = []
    missing: list[str] = []
    for entry in reference_rows:
        key = (entry["arm"], int(entry["seed"]))
        got = observed.get(key)
        if got is None:
            missing.append(f"{entry['arm']}/seed{entry['seed']}")
            continue
        for metric, ref_value in entry["metrics"].items():
            if metric not in got:
                missing.append(f"{entry['arm']}/seed{entry['seed']}:{metric}")
                continue
            # Exact equality, deliberately. A 1-ULP drift passes math.isclose and is still a
            # regression: it means the arithmetic changed.
            if got[metric] != ref_value:
                mismatches.append(
                    Mismatch(entry["arm"], int(entry["seed"]), metric, ref_value, got[metric])
                )
    return mismatches, missing


# --------------------------------------------------------------------------- environment


def _canonical(name: str) -> str:
    """PEP 503 name normalisation: lowercase, runs of ``-_.`` collapsed to ``-``.

    Not cosmetic. `pip freeze` emits the distribution's own spelling — ``pyg_lib`` with an
    underscore — while this module and the docs refer to ``pyg-lib``. Comparing the raw strings
    reports drift on a package that has not moved, and since drift makes the check
    *inconclusive by design*, the acceptance test could then never pass. Normalise both sides.
    """
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _pinned_versions(pin_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    in_freeze = False
    for line in pin_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "[pip freeze]":
            in_freeze = True
            continue
        if in_freeze and "==" in line:
            name, _, ver = line.partition("==")
            out[_canonical(name)] = ver.strip()
    return out


def environment_drift(pin_path: Path) -> list[str]:
    """Differences between the pinned stack and the live one, for the packages that matter."""
    pinned = _pinned_versions(pin_path)
    drift = []
    for pkg in CRITICAL_PACKAGES:
        want = pinned.get(_canonical(pkg))
        try:
            have = version(pkg)
        except PackageNotFoundError:
            have = None
        if want is None:
            drift.append(f"{pkg}: absent from the pin file (cannot verify)")
        elif have != want:
            drift.append(f"{pkg}: pinned {want} != installed {have}")

    runtime = pin_path.read_text(encoding="utf-8")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        if gpu not in runtime:
            drift.append(f"gpu: {gpu} not the pinned device")
        if f"cuda_runtime     = {torch.version.cuda}" not in runtime:
            drift.append(f"cuda_runtime: {torch.version.cuda} differs from the pin")
    return drift


# --------------------------------------------------------------------------- running the batch


def run_batch(registry_path: Path, device_arg: str | None) -> None:
    """Produce the 49-row post-refactor batch, tagged ``experiment=extract_check``."""
    hp, base_cfg, cfg_device = frozen_hparams()
    device = resolve_device(device_arg or cfg_device)
    data = load_elliptic(REPO_ROOT / "data" / "elliptic", strict=True)
    split = ell1_eval_split()
    test_window = f"{split.test_min}-{split.test_max}"

    print(f"[check] device={device} | frozen ADR-003 | tagging experiment={CHECK_TAG}")

    # --- Gate-0 tabular floor (6 rows). phase comes from config.yaml (P0), as in the original.
    X_tr, y_tr, X_te, y_te, meta = prepare_labelled_split(data, split)
    floor_cfg = {**base_cfg, "phase": "P0", "experiment": CHECK_TAG}
    for baseline in ("rf", "lr"):
        for seed in (0, 1, 2):
            _, m = run_baseline(
                baseline, seed, X_tr, y_tr, X_te, y_te, meta,
                {**floor_cfg, "arm": baseline}, test_window, registry_path=registry_path,
            )
            print(f"[check] {baseline:<10} seed={seed}: F1={m['illicit_f1']:.6f}")

    # --- Gate-3 ablation arms (40 rows). The ablation seed tracks the run seed, exactly as the
    # original batch did — otherwise the rewiring realisation differs and nothing would match.
    for arm, transform in GATE3_ARMS.items():
        for seed in range(8):
            if transform is None:
                data_arm = data
            else:
                data_arm = data.clone()
                data_arm.edge_index = transform(data.edge_index, data.time_step, seed)
            cfg = {**base_cfg, "phase": "P3", "ablation": arm, "arm": arm,
                   "experiment": CHECK_TAG}
            _, m = run_gnn("graphsage", seed, data_arm, split, hp, cfg,
                           registry_path=registry_path, device=str(device))
            print(f"[check] {arm:<10} seed={seed}: F1={m['illicit_f1']:.6f}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # --- GCN reference (3 rows).
    for seed in (0, 1, 2):
        cfg = {**base_cfg, "arm": "gcn", "experiment": CHECK_TAG}
        _, m = run_gnn("gcn", seed, data, split, hp, cfg,
                       registry_path=registry_path, device=str(device))
        print(f"[check] {'gcn':<10} seed={seed}: F1={m['illicit_f1']:.6f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()


@dataclass(frozen=True)
class Selection:
    """The ``extract_check`` rows that can certify one commit, and what was set aside."""
    observed: dict[tuple[str, int], dict]
    retried: dict[tuple[str, int], list[str]]   # identity -> run ids, when it ran more than once
    excluded: int                               # rows at the commit that cannot certify anything


def observed_rows(registry_path: Path, commit: str) -> Selection:
    """``{(arm, seed): metrics}`` for the ``experiment=extract_check`` rows that certify ``commit``.

    A row counts only if it ran **at that commit**, on a clean tree, under deterministic kernels,
    and did not error. The registry is append-only and holds a batch per certified commit, so
    without the commit filter a row from an older commit would stand in for one never produced at
    this one, and a PASS would certify code that was not run.

    Two counted rows for one identity mean the identity was re-run. ADR-008 clause 2 forbids
    retrying a mismatch, so the duplicates are reported for the caller to refuse, never collapsed
    by letting the later row win.
    """
    observed: dict[tuple[str, int], dict] = {}
    run_ids: dict[tuple[str, int], list[str]] = {}
    excluded = 0
    with registry_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            m = json.loads(row["metrics_json"])
            if m.get("experiment") != CHECK_TAG:
                continue
            arm = m.get("arm")
            if arm is None or row["git_commit"] != commit:
                continue
            if (row["git_dirty"] != "false" or "ERRORED" in row["notes"]
                    or m.get("deterministic") is not True):
                excluded += 1
                continue
            key = (arm, int(row["seed"]))
            run_ids.setdefault(key, []).append(row["run_id"])
            observed[key] = {k: v for k, v in m.items() if k.startswith("illicit_")}
    retried = {key: ids for key, ids in run_ids.items() if len(ids) > 1}
    return Selection(observed, retried, excluded)


def resolve_commit(ref: str) -> str:
    """The full hash ``ref`` names, so a short hash on the command line matches the rows' field."""
    try:
        return subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        raise SystemExit(f"--commit {ref!r} does not name a commit in this repository.")


# --------------------------------------------------------------------------- entry point


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="run the 49-row batch, then compare")
    group.add_argument("--compare-only", action="store_true", help="compare existing rows only")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: config)")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run despite uncommitted code (rows will be git_dirty=true, so they "
                         "cannot certify anything)")
    ap.add_argument("--commit", default=None,
                    help="with --compare-only: the commit to certify (default HEAD)")
    args = ap.parse_args()
    if args.run and args.commit:
        raise SystemExit("refusing --run with --commit: a batch certifies the commit it ran at, "
                         "which is HEAD.")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(
            f"no manifest at {manifest_path}. Run scripts/freeze_extract_reference.py — and note "
            "it must be frozen BEFORE the refactor, or the bar is rebased onto new numbers."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = manifest["rows"]

    arms = [e["arm"] for e in reference]
    identities = {(e["arm"], int(e["seed"])) for e in reference}
    if len(identities) != len(reference):
        raise SystemExit("manifest (arm, seed) identities are not unique — pairing is ambiguous.")

    registry_path = default_registry_path(REPO_ROOT)

    print(f"[check] manifest: {len(reference)} reference rows, "
          f"{len(set(arms))} distinct arms, frozen at commit {manifest['git_commit'][:8]}")

    # --- ADR-008 clause 4 -----------------------------------------------------------------
    drift = environment_drift(REPO_ROOT / manifest["env_pin"])
    if drift:
        print("\n" + "!" * 78)
        print("ENVIRONMENT DRIFT — bit-for-bit equality is VOID (ADR-008 clause 4):")
        for d in drift:
            print(f"  - {d}")
        print("Reproducibility here is same-environment. A version change alters the arithmetic")
        print("for reasons unrelated to the refactor and is indistinguishable from a regression.")
        print("This run CANNOT report a pass. Restore the pinned stack, or re-establish the")
        print("reference set on the new one and record the environment delta.")
        print("!" * 78 + "\n")
    else:
        print("[check] environment matches the pin (ADR-008 clause 4) — equality is enforceable")

    if args.run:
        if git_dirty() and not args.allow_dirty:
            raise SystemExit(
                "refusing to run: uncommitted code/config, so these rows would log "
                "git_dirty=true. Commit first (this has forced full re-runs twice already)."
            )
        run_batch(registry_path, args.device)

    commit = resolve_commit(args.commit or "HEAD")
    selection = observed_rows(registry_path, commit)
    observed = selection.observed
    print(f"[check] certifying commit {commit[:8]}: {len(observed)} clean, deterministic "
          f"{CHECK_TAG} rows at it")
    if selection.excluded:
        print(f"[check] set aside {selection.excluded} row(s) at this commit that were dirty, "
              "errored or nondeterministic; they cannot certify it")
    if not observed:
        print(f"\n[check] NOT A PASS — no {CHECK_TAG} rows at {commit[:8]}, so nothing was "
              "compared. Run the batch at this commit with --run.")
        return 1
    if selection.retried:
        print(f"\n[check] FAIL — {len(selection.retried)} identit(y/ies) ran more than once at this "
              "commit. ADR-008 clause 2 forbids retrying, so no row is chosen between them:")
        for (arm, seed), ids in sorted(selection.retried.items()):
            print(f"  {arm}/seed{seed}: {', '.join(ids)}")
        return 1

    mismatches, missing = compare(reference, observed)

    print(f"\n[check] compared {len(reference) - len(missing)} / {len(reference)} reference rows")
    if missing:
        print(f"[check] MISSING ({len(missing)}): {', '.join(missing[:12])}"
              + (" ..." if len(missing) > 12 else ""))

    if mismatches:
        print(f"\n[check] {len(mismatches)} MISMATCH(ES) — this is a regression (ADR-008 clause 2):\n")
        for mm in mismatches:
            print(f"  {mm}")
        print("\nInvestigate to root cause. Do NOT re-run until it matches — that is the")
        print("optional-stopping error in another costume. If the change is *intended*, it needs")
        print("its own ADR and a new batch; the dated gate files are not edited.")

    ok = not mismatches and not missing and not drift
    print()
    if ok:
        print("[check] PASS — the refactored core reproduces every reference row bit-for-bit.")
    elif drift and not mismatches and not missing:
        print("[check] INCONCLUSIVE — every row matched, but the environment drifted, so the")
        print("        equality bar does not hold. Not a pass.")
    else:
        print("[check] FAIL — see above. DGF-1 Gate 0 cannot be signed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
