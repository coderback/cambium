"""ELL-1 Phase-3 defending ablations — the Gate-3 batch (ADR-006, doc-01 §4/§7, doc-00 §8).

    python scripts/run_ell1_ablations.py [--device cuda|cpu] [--seeds 0,1,2,3,4,5,6,7]

Runs GraphSAGE under the frozen ADR-003 config, identical in every respect except the edge set,
and compares each ablated arm against a **real arm re-run in the same batch** (never against the
Gate-1 rows, whose code path and determinism settings differ):

| arm        | what it destroys                                   | role (ADR-006) |
|------------|----------------------------------------------------|----------------|
| `real`     | —                                                  | the baseline every clause is measured against |
| `scrambled`| node↔position correspondence; topology intact      | **clause 1, gated** |
| `random`   | wiring + degree sequence (Erdős–Rényi, same count) | **clause 2, gated** |
| `no_edges` | message passing entirely, params/budget matched    | **clause 3, gated** |
| `config`   | wiring only; every node keeps its own degree       | clause 5, **reported, not gated** |

Every rewiring stays **within a time step**: an edge forged across the 34/35 cutoff would either
leak the future into training or be silently dropped by induction, and either way the arm would
stop being comparable to the baseline it controls for (`tests/test_edge_scramble.py` pins this).

Pass condition per ADR-006: for each gated clause, `mean(real) − mean(ablated)` is positive and
**resolvable**, i.e. exceeds `2 × SE_diff` where `SE_diff = sqrt(s_r²/n_r + s_a²/n_a)`. Primary
metric is illicit-F1; recall and AUC are reported but are **not** pass/fail inputs.

`--assemble-only` skips training and writes `gates/GATE-ELL1-3.md` from the clean P3 rows already
in the registry, carrying ADR-006's disclosure verbatim. **The verdict is always left blank** —
a gate is decided by the researcher, never by this script and never by Claude.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_ind

from gbe.eval import (
    configuration_model_edges,
    random_graph_edges,
    remove_edges,
    scramble_edges,
)
from gbe.run.registry import default_registry_path
from adapters.ell1.datasource_elliptic import load_elliptic
from adapters.ell1.eval import ell1_eval_split
from adapters.ell1.train_gnn import frozen_hparams, resolve_device, run_gnn

REPO_ROOT = Path(__file__).resolve().parents[1]
METRIC_COLS = ("illicit_f1", "illicit_recall", "illicit_auc")
GATE_METRIC = "illicit_f1"  # doc-01 §5; ADR-006 keeps this primary over AUC deliberately

# arm -> (edge transform, is a gated clause). `real` transforms nothing.
ARMS: dict[str, tuple] = {
    "real": (None, False),
    "scrambled": (scramble_edges, True),
    "random": (random_graph_edges, True),
    "no_edges": (lambda ei, ts, seed: remove_edges(ei), True),
    "config": (configuration_model_edges, False),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda|cpu (default: config/auto)")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7",
                    help="ADR-006: 8 per arm; >=5 is a hard floor")
    ap.add_argument("--assemble-only", action="store_true",
                    help="skip training; assemble gates/GATE-ELL1-3.md from clean P3 rows")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.assemble_only:
        hp, base_cfg, _ = frozen_hparams()
        registry_path = default_registry_path(REPO_ROOT)
        ids = arm_run_ids(registry_path, hp, base_cfg, seeds)
        _report(registry_path, ids)
        assemble(registry_path, ids, REPO_ROOT / "gates" / "GATE-ELL1-3.md")
        return

    if len(seeds) < 5:
        raise SystemExit(
            f"ADR-006 clause 4 sets a hard floor of 5 seeds per arm; got {len(seeds)}. "
            "Stage 1 is n=8; stage 2 (n=20, all arms, at most once) is the ONLY permitted "
            "top-up, and only if a gated clause is unresolvable at stage 1."
        )

    hp, base_cfg, cfg_device = frozen_hparams()
    device = resolve_device(args.device or cfg_device)
    data = load_elliptic(REPO_ROOT / "data" / "elliptic", strict=True)
    split = ell1_eval_split()
    registry_path = default_registry_path(REPO_ROOT)
    print(f"[ablations] device={device} | frozen ADR-003 | test window "
          f"{split.test_min}-{split.test_max} | {len(seeds)} seeds x {len(ARMS)} arms")

    ids: dict[str, list[str]] = {a: [] for a in ARMS}
    for arm, (transform, _) in ARMS.items():
        for seed in seeds:
            if transform is None:
                data_arm = data
            else:
                # the ablation seed tracks the run seed, so the seeds vary the realisation of
                # the rewiring as well as the init -- otherwise we measure a single draw
                data_arm = data.clone()
                data_arm.edge_index = transform(data.edge_index, data.time_step, seed)

            cfg = {**base_cfg, "phase": "P3", "ablation": arm}
            run_id, m = run_gnn(
                "graphsage", seed, data_arm, split, hp, cfg,
                registry_path=registry_path, device=str(device),
            )
            ids[arm].append(run_id)
            print(f"[run] {arm:<10} seed={seed}: F1={m['illicit_f1']:.4f} "
                  f"recall={m['illicit_recall']:.4f} AUC={m['illicit_auc']:.4f}  ({run_id})")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _report(registry_path, ids)


def arm_run_ids(registry_path: Path, hp, base_cfg: dict, seeds: list[int]) -> dict[str, list[str]]:
    """Recover which registry row belongs to which arm, from the registry alone.

    The batch that produced GATE-ELL1-3 predates `ablation` being logged into `metrics_json`,
    so the arm is not directly readable from those rows. It *is* inside the hashed config, so
    it is recovered exactly by rebuilding each (arm, seed) config the way :func:`run_gnn` built
    it and matching `config_hash`. This is provenance, not guesswork: a mismatch raises rather
    than silently mislabelling an arm.
    """
    from gbe.run.config import resolve_config
    from adapters.ell1.train_gnn import GNNHParams

    hp_arm = GNNHParams(**{**hp.as_dict(), "backbone": "graphsage", "fan_out": tuple(hp.fan_out)})
    with registry_path.open(newline="", encoding="utf-8") as fh:
        clean = [r for r in csv.DictReader(fh)
                 if r["phase"] == "P3" and r["git_dirty"] == "false"]
    by_hash = {r["config_hash"]: r for r in clean}

    ids: dict[str, list[str]] = {}
    for arm in ARMS:
        run_ids = []
        for seed in seeds:
            cfg = resolve_config({**base_cfg, "phase": "P3", "ablation": arm,
                                  "backbone": "graphsage", "seed": seed,
                                  "gnn": hp_arm.as_dict()})
            row = by_hash.get(cfg.config_hash)
            if row is None:
                raise RuntimeError(
                    f"no clean P3 registry row for arm={arm} seed={seed} "
                    f"(config_hash {cfg.config_hash[:12]}...). Refusing to assemble a gate "
                    "file from an incomplete batch."
                )
            run_ids.append(row["run_id"])
        ids[arm] = run_ids
    return ids


def _stats(registry_path: Path, ids: dict[str, list[str]]) -> dict[str, dict]:
    wanted = {i for v in ids.values() for i in v}
    with registry_path.open(newline="", encoding="utf-8") as fh:
        rows = {r["run_id"]: r for r in csv.DictReader(fh) if r["run_id"] in wanted}
    out: dict[str, dict] = {}
    for arm, run_ids in ids.items():
        vals = {c: [float(json.loads(rows[r]["metrics_json"])[c]) for r in run_ids]
                for c in METRIC_COLS}
        # ddof=1 (sample std) is pinned by ADR-006 so the criterion cannot depend on which
        # script evaluates it. The Gate-0/Gate-1 scripts keep ddof=0 so they still reproduce
        # their dated gate files; see the note in scripts/run_ell1_gnn.py.
        out[arm] = {"n": len(run_ids), "raw": vals,
                    **{c: (float(np.mean(v)), float(np.std(v, ddof=1))) for c, v in vals.items()}}
    return out


def _report(registry_path: Path, ids: dict[str, list[str]]) -> None:
    st = _stats(registry_path, ids)

    print("\n== Phase-3 ablations, steps 35-49, illicit class (mean +/- std) ==\n")
    print(f"{'arm':<11}{'n':>3}  {'F1':<20}{'recall':<20}{'AUC':<20}")
    for arm in ARMS:
        s = st[arm]
        cells = "".join(f"{s[c][0]:.4f} +/- {s[c][1]:.4f}   " for c in METRIC_COLS)
        print(f"{arm:<11}{s['n']:>3}  {cells}")

    r_m, r_s = st["real"][GATE_METRIC]
    r_n = st["real"]["n"]
    print(f"\n== ADR-006 check on {GATE_METRIC} vs real ({r_m:.4f}) -- mechanical, not the verdict ==\n")
    gated_ok = []
    for arm, (_, is_gated) in ARMS.items():
        if arm == "real":
            continue
        a_m, a_s = st[arm][GATE_METRIC]
        a_n = st[arm]["n"]
        se = float(np.sqrt(r_s**2 / r_n + a_s**2 / a_n))
        drop = r_m - a_m
        resolvable = drop > 2 * se
        tag = "GATED " if is_gated else "report"
        verdict = ("PASS" if resolvable else "FAIL") if is_gated else "--"
        # Welch p reported for context only; the 2*SE_diff rule above is the criterion (ADR-006).
        _, p = ttest_ind(st["real"]["raw"][GATE_METRIC], st[arm]["raw"][GATE_METRIC],
                         equal_var=False)
        print(f"  [{tag}] {arm:<10} drop {drop:+.4f}   2*SE_diff {2*se:.4f}   "
              f"resolvable={str(resolvable):<5} {verdict}   (Welch p={p:.4f})")
        if is_gated:
            gated_ok.append(resolvable)

    print(f"\n  All gated clauses: {'PASS' if all(gated_ok) else 'FAIL'}  "
          f"({sum(gated_ok)}/{len(gated_ok)} resolvable)")
    print("\nIf a gated clause is unresolvable: ADR-006 permits exactly ONE top-up -- rerun ALL")
    print("arms at n=20, and that result is final either way. Never top up repeatedly (optional")
    print("stopping), and never soften the clause.")
    print("Verdict and gates/GATE-ELL1-3.md are the researcher's; the gate file must repeat")
    print("ADR-006's disclosure that clause 1 is informed, not blind.")


DISCLOSURE = """> **Disclosure — this is a partial pre-registration, and one clause is not blind.**
> Reproduced verbatim from ADR-006, which requires this gate file to carry it.
>
> ADR-004 worked because it fixed the bar while **no** test-window number existed. That is not
> the situation here. A **provisional edge-scramble batch had already been run and seen** before
> the bar was set: 6 seeds per arm on CUDA, `git_dirty=true`, run ids
> `ell1-20260724T202110Z-70e9fab9` … `ell1-20260724T202947Z-f7dc7b7b`, observed real illicit-F1
> 0.6763 ± 0.0790 vs scrambled 0.5597 ± 0.0288.
>
> - **Clause 1 (edge-scramble) is _informed_, not blind.** Its bar is derived from a principle —
>   directional sign plus statistical resolvability — and **not** from the observed 0.117
>   magnitude. No threshold in ADR-006 is a number the provisional result happens to clear.
> - **Clauses 2 and 3 (random-graph control, GNN-removed) are genuinely blind.** Neither had been
>   run when the criteria were fixed. They carry ADR-004's full force.
> - The provisional rows remain in the append-only registry, unused for this gate.
>
> A reader is entitled to discount clause 1 accordingly."""


def _arm_table(title: str, note: str, rows: list[dict]) -> str:
    lines = [f"### {title}", "", f"_{note}_", "",
             "| seed | illicit F1 | illicit recall | illicit AUC | run_id |",
             "|------|-----------|----------------|-------------|--------|"]
    for r in sorted(rows, key=lambda r: int(r["seed"])):
        m = json.loads(r["metrics_json"])
        lines.append(f"| {r['seed']} | {m['illicit_f1']:.4f} | {m['illicit_recall']:.4f} "
                     f"| {m['illicit_auc']:.4f} | {r['run_id']} |")
    vals = {c: [float(json.loads(r["metrics_json"])[c]) for r in rows] for c in METRIC_COLS}
    st = {c: (float(np.mean(v)), float(np.std(v, ddof=1))) for c, v in vals.items()}
    lines.append("| **mean ± std** | " + " | ".join(
        f"**{st[c][0]:.4f} ± {st[c][1]:.4f}**" for c in METRIC_COLS) + " | |")
    return "\n".join(lines) + "\n"


ARM_NOTES = {
    "real": "The true graph. Every clause below is measured against this arm, re-run in the "
            "same batch — never against the Gate-1 rows, whose code path and determinism "
            "settings differ.",
    "scrambled": "Node identities permuted within each time step. Topology, degree sequence and "
                 "the temporal split are preserved exactly; only the correspondence between "
                 "graph position and node content is destroyed. **Clause 1, gated.**",
    "random": "Erdős–Rényi within each time step: same per-step edge count, degree sequence "
              "destroyed. **Clause 2, gated.**",
    "no_edges": "Empty edge set — message passing disabled at identical parameter count and "
                "training budget. **Clause 3, gated.**",
    "config": "Degree-preserving rewire by double-edge swaps: every node keeps its own degree "
              "and its own features, and only *who it connects to* is randomised. "
              "**Reported, not gated** (ADR-006 clause 5).",
}


def assemble(registry_path: Path, ids: dict[str, list[str]], out_path: Path) -> None:
    wanted = {i for v in ids.values() for i in v}
    with registry_path.open(newline="", encoding="utf-8") as fh:
        rows = {r["run_id"]: r for r in csv.DictReader(fh) if r["run_id"] in wanted}
    st = _stats(registry_path, ids)
    any_row = rows[ids["real"][0]]
    meta = json.loads(any_row["metrics_json"])

    r_m, r_s = st["real"][GATE_METRIC]
    r_n = st["real"]["n"]
    checks, gated_ok = [], []
    for arm, (_, is_gated) in ARMS.items():
        if arm == "real":
            continue
        a_m, a_s = st[arm][GATE_METRIC]
        se = float(np.sqrt(r_s**2 / r_n + a_s**2 / st[arm]["n"]))
        drop, resolvable = r_m - a_m, (r_m - a_m) > 2 * float(np.sqrt(
            r_s**2 / r_n + a_s**2 / st[arm]["n"]))
        _, p = ttest_ind(st["real"]["raw"][GATE_METRIC], st[arm]["raw"][GATE_METRIC],
                         equal_var=False)
        role = "**clause " + {"scrambled": "1", "random": "2", "no_edges": "3"}[arm] + ", gated**" \
            if is_gated else "reported"
        checks.append(f"| {arm} | {role} | {drop:+.4f} | {2*se:.4f} | "
                      f"{'**yes**' if resolvable else '**no**'} | {p:.4f} |")
        if is_gated:
            gated_ok.append(resolvable)

    body = "\n".join([
        "# GATE-ELL1-3 — is the gain structural? (defending ablations)",
        "",
        f"**Date assembled:** {date.today().isoformat()}",
        "**Phase / doc:** ELL-1 Phase 3 — docs/01-elliptic-embedding-model-BUILD.md §4, §7; "
        "doc-00 §8",
        f"**Git commit:** {any_row['git_commit']}  ·  **Data snapshot:** "
        f"{any_row['data_snapshot_id']}  ·  **Config:** ADR-003  ·  **Criteria:** ADR-006",
        f"**Determinism:** all rows `deterministic=true`, "
        f"`CUBLAS_WORKSPACE_CONFIG={meta.get('cublas_workspace_config', '')}` (ADR-005) — "
        "these numbers are bit-for-bit reproducible on the recorded environment.",
        f"**Batch:** stage 1 of ADR-006 clause 4 — {st['real']['n']} seeds × {len(ARMS)} arms, "
        "all `git_dirty=false`. No stage-2 top-up was triggered.",
        "",
        DISCLOSURE,
        "",
        "## Question",
        "Gate 1 established that GraphSAGE does not beat the RF tabular floor. It did **not** "
        "establish whether the GNN's score comes from the graph or from the 165 node features "
        "alone — features 94–164 are hand-built one-hop aggregates, so a GNN could score well "
        "while ignoring message passing. Does destroying the structure cost performance?",
        "",
        "## Pass condition (doc-01 §4/§7 as amended by ADR-006; operationalised in ADR-006)",
        "> **Gate 3:** edge-scramble drops **materially and resolvably below the real-graph "
        "run**; real graph ≫ random graph; removing the GNN removes the gain.",
        "",
        "A difference is **resolvable** iff `mean(real) − mean(ablated) > 2 × SE_diff`, where "
        "`SE_diff = sqrt(s_r²/n_r + s_a²/n_a)` and `s` is the **sample** standard deviation "
        "(ddof=1, pinned by ADR-006). Primary metric is illicit-F1; recall and AUC are reported "
        "but are **not** pass/fail inputs. Gate 3 passes iff clauses 1–3 all hold.",
        "",
        "## Results (8 seeds per arm — a gate does not pass on a single seed)",
        "",
    ] + [_arm_table(arm, ARM_NOTES[arm], [rows[i] for i in ids[arm]]) for arm in ARMS] + [
        f"**Support:** train = {meta['n_train']} labelled nodes (steps 1–34); test = "
        f"{meta['n_test']} labelled nodes (steps 35–49), of which {meta['n_test_illicit']} "
        "illicit. Every arm rewires **within a time step**, so no ablation forges an edge "
        "across the 34/35 cutoff and the strict inductive protocol is identical across arms.",
        "",
        "## ADR-006 criterion check — mechanical, not the verdict",
        "",
        f"Real-graph arm: **{r_m:.4f} ± {r_s:.4f}** illicit-F1 (n={r_n}).",
        "",
        "| arm | role | drop vs real | 2×SE_diff | resolvable | Welch p |",
        "|---|---|---|---|---|---|",
    ] + checks + [
        "",
        f"**All gated clauses (1–3): {'PASS' if all(gated_ok) else 'FAIL'}** "
        f"({sum(gated_ok)}/{len(gated_ok)} resolvable). This is the pre-registered mechanical "
        "result; the verdict below is the researcher's.",
        "",
        "Two properties of the conjunction, so the result is not over-read: the three tests "
        "share the same `real` arm, so they are **correlated, not independent**; and requiring "
        "all three makes this gate stricter than any single clause.",
        "",
        "_All numbers read from `experiments/registry.csv`. No cell is filled by estimate, "
        "extrapolation, or smoothing. Arms were matched to rows by rebuilding each (arm, seed) "
        "config hash, not by position._",
        "",
        "## Verdict",
        "<!-- Left blank. Decided by the researcher, not by Claude. -->",
        "",
        "---",
        "_Verdict, seeds, and table are sacred once dated. Papers are assembled from gate "
        "files; nothing is reported that is not in one._",
        "",
    ])
    out_path.write_text(body, encoding="utf-8")
    print(f"[gate] wrote {out_path} from {len(wanted)} registry rows")


if __name__ == "__main__":
    main()
