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

This script computes the mechanical check and prints it. **It does not write the gate file and
does not decide the verdict** — `gates/GATE-ELL1-3.md` is assembled and signed by the researcher,
and must carry ADR-006's disclosure that clause 1 is informed rather than blind.
"""

from __future__ import annotations

import argparse
import csv
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
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
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


if __name__ == "__main__":
    main()
