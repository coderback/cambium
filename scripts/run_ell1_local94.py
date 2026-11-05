"""ELL-1 Phase-3 diagnostic: has message passing replaced the hand-built aggregates?

    python scripts/run_ell1_local94.py [--device cuda|cpu] [--seeds 0,...,7]

Doc-01 §3, "the sharpest experiment this model can run": Elliptic's features 94-164 are
**hand-built one-hop neighbour aggregates** computed by the dataset authors — a manual proxy for
one round of message passing. A GNN trained on all 165 therefore partly *re-computes* what its
input already contains, which is why beating the tabular floor is a harder bar than it looks.

So train the GNN on the **94 local features only** (ADR-001: local = CSV columns 2..95, with
`time_step` excluded as ever) and compare against **RF on all 165**. If GNN-on-local-94 matches
RF-on-165, message passing has genuinely learned what the dataset authors hand-computed — a
cleaner claim than the headline comparison, and a good paper paragraph either way.

Three arms, all in one batch at one commit so the comparison is not split across code states:

| arm           | features | graph | role |
|---------------|----------|-------|------|
| `gnn_local94` | 94 local | real  | the experiment |
| `gnn_all165`  | all 165  | real  | reference — what the aggregates are worth to the GNN |
| `rf_all165`   | all 165  | none  | the tabular floor, re-run at the same seed count |

**Reported, not gated** (ADR-006 clause 5). There is no pass condition here and none may be
added retroactively; doc-01 §7 lists this as "reported". Runs are deterministic (ADR-005).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from gbe.run.registry import default_registry_path
from adapters.ell1.baselines_tabular import prepare_labelled_split, run_baseline
from adapters.ell1.datasource_elliptic import load_elliptic
from adapters.ell1.eval import ell1_eval_split
from adapters.ell1.train_gnn import frozen_hparams, resolve_device, run_gnn

REPO_ROOT = Path(__file__).resolve().parents[1]
METRIC_COLS = ("illicit_f1", "illicit_recall", "illicit_auc")
N_LOCAL = 94  # ADR-001: columns 2..95 of the raw CSV are the local features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda|cpu (default: config/auto)")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    hp, base_cfg, cfg_device = frozen_hparams()
    device = resolve_device(args.device or cfg_device)
    data = load_elliptic(REPO_ROOT / "data" / "elliptic", strict=True)
    split = ell1_eval_split()
    registry_path = default_registry_path(REPO_ROOT)
    print(f"[local94] device={device} | frozen ADR-003 | test window "
          f"{split.test_min}-{split.test_max} | {len(seeds)} seeds")

    ids: dict[str, list[str]] = {"gnn_local94": [], "gnn_all165": [], "rf_all165": []}

    # -- GNN arms: identical but for how many feature columns the encoder sees ---------------
    for arm, n_feat in (("gnn_local94", N_LOCAL), ("gnn_all165", data.x.size(1))):
        for seed in seeds:
            data_arm = data.clone()
            data_arm.x = data.x[:, :n_feat].contiguous()
            cfg = {**base_cfg, "phase": "P3", "experiment": "local94_vs_rf165",
                   "arm": arm, "features": n_feat}
            run_id, m = run_gnn("graphsage", seed, data_arm, split, hp, cfg,
                                registry_path=registry_path, device=str(device))
            ids[arm].append(run_id)
            print(f"[run] {arm:<12} seed={seed}: F1={m['illicit_f1']:.4f} "
                  f"AUC={m['illicit_auc']:.4f}  ({run_id})")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # -- RF floor on all 165, re-run at the same seed count ---------------------------------
    X_tr, y_tr, X_te, y_te, meta = prepare_labelled_split(data, split)
    for seed in seeds:
        cfg = {**base_cfg, "phase": "P3", "experiment": "local94_vs_rf165",
               "arm": "rf_all165", "features": data.x.size(1)}
        run_id, m = run_baseline("rf", seed, X_tr, y_tr, X_te, y_te, meta, cfg,
                                 f"{split.test_min}-{split.test_max}",
                                 registry_path=registry_path)
        ids["rf_all165"].append(run_id)
        print(f"[run] {'rf_all165':<12} seed={seed}: F1={m['illicit_f1']:.4f} "
              f"AUC={m['illicit_auc']:.4f}  ({run_id})")

    _report(registry_path, ids)


def _report(registry_path: Path, ids: dict[str, list[str]]) -> None:
    wanted = {i for v in ids.values() for i in v}
    with registry_path.open(newline="", encoding="utf-8") as fh:
        rows = {r["run_id"]: r for r in csv.DictReader(fh) if r["run_id"] in wanted}

    st = {}
    for arm, run_ids in ids.items():
        vals = {c: [float(json.loads(rows[r]["metrics_json"])[c]) for r in run_ids]
                for c in METRIC_COLS}
        st[arm] = {c: (float(np.mean(v)), float(np.std(v, ddof=1))) for c, v in vals.items()}

    print("\n== GNN-on-local-94 vs RF-on-165, steps 35-49, illicit class "
          "(mean +/- std, ddof=1) ==\n")
    print(f"{'arm':<13}{'F1':<21}{'recall':<21}{'AUC':<21}")
    for arm in ids:
        s = st[arm]
        print(f"{arm:<13}" + "".join(f"{s[c][0]:.4f} +/- {s[c][1]:.4f}    " for c in METRIC_COLS))

    gl, rf, ga = (st[a]["illicit_f1"][0] for a in ("gnn_local94", "rf_all165", "gnn_all165"))
    print(f"\n  doc-01 §3 question: does GNN-on-local-94 match RF-on-165?")
    print(f"    GNN local-94 {gl:.4f}  vs  RF all-165 {rf:.4f}   gap {gl - rf:+.4f}")
    print(f"    what the hand-built aggregates are worth to the GNN: "
          f"{ga - gl:+.4f} (all-165 {ga:.4f} - local-94 {gl:.4f})")
    print("\n  Reported, not gated (ADR-006 clause 5 / doc-01 §7). No pass condition exists")
    print("  for this comparison and none may be added after the fact.")


if __name__ == "__main__":
    main()
