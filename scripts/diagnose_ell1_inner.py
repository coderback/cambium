"""Inner-window headroom diagnostic — how much illicit-F1 the Gate-1 protocol left on the table.

    python scripts/diagnose_ell1_inner.py [--device cuda|cpu] [--seeds 0,1,2]

**This script never touches steps 35-49.** It trains on steps 1-29 and measures on the 30-34
inner validation window — the same leak-free split the HPO sweep used (`adapters.ell1.hpo`,
pinned by `tests/test_hpo_no_eval_peek.py`). Every number it prints is a *validation* number
and must never be reported as a result (ADR-003 precedent: the sweep's numbers live in their
own CSV, not `experiments/registry.csv`).

It quantifies three handicaps found in the Gate-1 implementation review, all of which are
blind to the test window:

  1. **No model selection.** The trainer returns whatever exists at the last epoch. Measured
     as: best-epoch val-F1 minus last-epoch val-F1.
  2. **Uncalibrated operating point.** The loss upweights illicit ~7.6x, then eval takes a
     plain argmax (0.5). Measured as: best-threshold val-F1 minus val-F1 at 0.5. Reported two
     ways — an *oracle* bound (threshold picked and scored on the same window) and a
     *transferable* estimate (threshold picked on 30-32, scored on 33-34).
  3. **Heavy-tailed features into an MLP.** Run as a second arm with a fit-on-train rank
     transform instead of standardisation.

Nothing here changes the frozen ADR-003 config or the Gate-1 verdict. If the headroom turns
out to be worth chasing, the protocol change goes through an ADR *before* any re-run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve

from gbe.eval import edges_as_of, split_masks
from gbe.run.seeding import seed_everything
from adapters.ell1.datasource_elliptic import ILLICIT, derive_edge_time, load_elliptic
from adapters.ell1.hpo import INNER_SPLIT
from adapters.ell1.train_gnn import (
    FEATURE_TRANSFORMS,
    build_model,
    frozen_hparams,
    predict_window,
    resolve_device,
    train_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = REPO_ROOT / "experiments" / "ell1_inner_diag.csv"
# The 30-34 window is split here for the transferable threshold estimate: pick on <=32, score
# on >32. Both halves are post-training and pre-cutoff, so this stays leak-free.
THR_PICK_MAX = 32


def f1_at(y_true: np.ndarray, proba: np.ndarray, thr: float) -> float:
    """Illicit-class F1 when the positive class is called at ``proba >= thr``."""
    pred = (proba >= thr).astype(int)
    return float(f1_score(y_true, pred, pos_label=ILLICIT, zero_division=0))


def pick_and_score(
    y: np.ndarray, proba: np.ndarray, pick: np.ndarray, score: np.ndarray
) -> tuple[float, float, float]:
    """Pick the F1-optimal threshold on one slice, then spend it on another.

    Returns ``(f1_at_0.5, f1_at_picked_threshold, threshold)`` on the ``score`` slice. When
    ``pick`` and ``score`` are the same slice this is an *oracle* ceiling, not a transfer
    estimate — the caller is responsible for labelling it as such.
    """
    if not pick.any() or not score.any():
        return float("nan"), float("nan"), float("nan")
    _, thr = best_f1_threshold(y[pick], proba[pick])
    return f1_at(y[score], proba[score], 0.5), f1_at(y[score], proba[score], thr), thr


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Exact best achievable illicit-F1 over all thresholds, and the threshold achieving it."""
    prec, rec, thr = precision_recall_curve(y_true, proba, pos_label=ILLICIT)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(prec), where=denom > 0)
    if thr.size == 0:
        return 0.0, 0.5
    i = int(np.argmax(f1[:-1]))  # last point has no corresponding threshold
    return float(f1[i]), float(thr[i])


def run_arm(transform: str, seed: int, data, hp, device) -> list[dict]:
    """Train on <=29 with one feature transform; score the 30-34 window after every epoch."""
    seed_everything(seed)
    x = FEATURE_TRANSFORMS[transform](data.x, data.time_step, INNER_SPLIT.train_max)
    # ADR-011: same edges, same order as the retired induced_train_subgraph on Elliptic.
    edge_train = edges_as_of(
        data.edge_index,
        edge_time=derive_edge_time(data.edge_index, data.time_step),
        t_max=INNER_SPLIT.train_max,
    )
    train_node_mask, _ = split_masks(data.time_step, INNER_SPLIT)
    seed_mask = train_node_mask & data.labelled_mask

    model = build_model(x.size(1), hp, device)
    rows: list[dict] = []

    def on_epoch(epoch: int, m) -> None:
        y_l, proba_l, pred_l, time_l, meta = predict_window(
            m, x, data.edge_index, data.time_step, data.y, INNER_SPLIT, device
        )
        f1_half = f1_at(y_l, proba_l, 0.5)
        # sanity: the 0.5 operating point must reproduce the argmax the gate runs used
        assert abs(f1_half - f1_score(y_l, pred_l, pos_label=ILLICIT, zero_division=0)) < 1e-9

        f1_oracle, thr_oracle = best_f1_threshold(y_l, proba_l)

        # Slices of the val window. 30-32 is positive-dense (20.8% illicit); 33-34 sits at
        # 6.3%, which is the only pre-cutoff regime resembling the 35-49 test window (6.5%).
        early = time_l <= THR_PICK_MAX
        late = ~early
        s33, s34 = time_l == 33, time_l == 34

        # (a) the original mis-matched transfer: threshold picked at 20.8% prevalence, spent at 6.3%
        f1_hold_half, f1_hold_thr, thr_early = pick_and_score(y_l, proba_l, early, late)
        # (b) ORACLE ceiling on the matched-prevalence slice (picked and scored on 33-34)
        _, f1_late_oracle, thr_late = pick_and_score(y_l, proba_l, late, late)
        # (c) genuine transfer *within* the matched regime, both directions (tiny n - see report)
        f34_half, f34_thr, _ = pick_and_score(y_l, proba_l, s33, s34)
        f33_half, f33_thr, _ = pick_and_score(y_l, proba_l, s34, s33)

        rows.append(
            {
                "transform": transform,
                "seed": seed,
                "epoch": epoch,
                "val_f1_at_0.5": round(f1_half, 6),
                "val_f1_oracle_thr": round(f1_oracle, 6),
                "oracle_thr": round(thr_oracle, 6),
                "picked_thr_le32": round(thr_early, 6),
                "holdout_f1_at_0.5": round(f1_hold_half, 6),
                "holdout_f1_at_picked_thr": round(f1_hold_thr, 6),
                "late_oracle_f1": round(f1_late_oracle, 6),
                "late_oracle_thr": round(thr_late, 6),
                "fold_s34_at_0.5": round(f34_half, 6),
                "fold_s34_at_thr_from_s33": round(f34_thr, 6),
                "fold_s33_at_0.5": round(f33_half, 6),
                "fold_s33_at_thr_from_s34": round(f33_thr, 6),
                "n_val": meta["n_test"],
                "n_val_illicit": meta["n_test_illicit"],
            }
        )

    train_model(model, x, edge_train, data.y, seed_mask, hp, device, on_epoch=on_epoch)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def summarise(rows: list[dict], transform: str) -> dict[str, tuple[float, float]]:
    """Per-seed summary -> mean/std across seeds for each headroom quantity."""
    per_seed: dict[int, list[dict]] = {}
    for r in rows:
        if r["transform"] == transform:
            per_seed.setdefault(r["seed"], []).append(r)

    keys = ("last", "best_epoch", "oracle_thr", "hold_half", "hold_thr",
            "late_half", "late_oracle", "fold_half", "fold_thr")
    cols: dict[str, list[float]] = {k: [] for k in keys}
    for seed_rows in per_seed.values():
        ep = sorted(seed_rows, key=lambda r: r["epoch"])
        last = ep[-1]
        best = max(ep, key=lambda r: r["val_f1_at_0.5"])
        cols["last"].append(last["val_f1_at_0.5"])
        cols["best_epoch"].append(best["val_f1_at_0.5"])
        cols["oracle_thr"].append(best["val_f1_oracle_thr"])
        cols["hold_half"].append(last["holdout_f1_at_0.5"])
        cols["hold_thr"].append(last["holdout_f1_at_picked_thr"])
        cols["late_half"].append(last["holdout_f1_at_0.5"])
        cols["late_oracle"].append(last["late_oracle_f1"])
        # the two folds are pooled: each is one direction of a 33<->34 threshold transfer
        cols["fold_half"].append(np.mean([last["fold_s33_at_0.5"], last["fold_s34_at_0.5"]]))
        cols["fold_thr"].append(
            np.mean([last["fold_s33_at_thr_from_s34"], last["fold_s34_at_thr_from_s33"]])
        )
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in cols.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda|cpu (default: config/auto)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--from-csv", action="store_true",
                    help="re-print the summary from an existing ell1_inner_diag.csv (no training)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.from_csv:
        def _num(v: str):
            """Every column except ``transform`` is numeric; keep it schema-agnostic so adding
            a measurement column never silently leaves strings in the summary maths."""
            try:
                return float(v)
            except ValueError:
                return v

        with OUT_CSV.open(newline="", encoding="utf-8") as fh:
            cached = [
                {k: (v if k == "transform" else
                     int(v) if k in ("seed", "epoch", "n_val", "n_val_illicit") else _num(v))
                 for k, v in r.items()}
                for r in csv.DictReader(fh)
            ]
        report(cached, sorted({r["seed"] for r in cached}))
        return

    hp, _, cfg_device = frozen_hparams()
    device = resolve_device(args.device or cfg_device)
    data = load_elliptic(REPO_ROOT / "data" / "elliptic", strict=True)
    print(f"[diag] device={device} | frozen ADR-003 config | INNER split "
          f"train<={INNER_SPLIT.train_max} val {INNER_SPLIT.test_min}-{INNER_SPLIT.test_max} "
          f"| 35-49 NEVER touched")

    rows: list[dict] = []
    for transform in FEATURE_TRANSFORMS:
        for seed in seeds:
            r = run_arm(transform, seed, data, hp, device)
            rows += r
            last = r[-1]
            print(f"[run] {transform:<11} seed={seed}  last-epoch val F1@0.5="
                  f"{last['val_f1_at_0.5']:.4f}  oracle-thr={last['val_f1_oracle_thr']:.4f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[out] {OUT_CSV.relative_to(REPO_ROOT)}  ({len(rows)} rows)")

    report(rows, seeds)


def report(rows: list[dict], seeds: list[int]) -> None:
    """Print the headroom summary. ASCII only — the Windows console is cp1252."""
    fmt = lambda t: f"{t[0]:.4f} +/- {t[1]:.4f}"
    print("\n== inner-window (30-34) headroom, mean +/- std over seeds "
          f"{seeds} -- VALIDATION NUMBERS, NOT RESULTS ==\n")
    for transform in FEATURE_TRANSFORMS:
        s = summarise(rows, transform)
        print(f"--- {transform} ---")
        print(f"  last-epoch F1 @0.5      (current protocol) : {fmt(s['last'])}")
        print(f"  best-epoch F1 @0.5      (+model selection) : {fmt(s['best_epoch'])}"
              f"   [delta {s['best_epoch'][0] - s['last'][0]:+.4f}]")
        print(f"  best-epoch F1 @best thr (+calibration, ORACLE bound) : {fmt(s['oracle_thr'])}"
              f"   [delta {s['oracle_thr'][0] - s['last'][0]:+.4f}]")
        print(f"  -- threshold, measured on the matched-prevalence slice 33-34 (6.3% illicit) --")
        print(f"      F1 @0.5                                  : {fmt(s['late_half'])}")
        print(f"      F1 @thr picked on 30-32 (20.8% illicit)  : {fmt(s['hold_thr'])}"
              f"   [delta {s['hold_thr'][0] - s['late_half'][0]:+.4f}]  MIS-MATCHED pick")
        print(f"      F1 @thr picked on 33-34 itself           : {fmt(s['late_oracle'])}"
              f"   [delta {s['late_oracle'][0] - s['late_half'][0]:+.4f}]  ORACLE ceiling")
        print(f"      33<->34 transfer fold, pooled both ways  :")
        print(f"          F1 @0.5        : {fmt(s['fold_half'])}")
        print(f"          F1 @picked thr : {fmt(s['fold_thr'])}"
              f"   [delta {s['fold_thr'][0] - s['fold_half'][0]:+.4f}]  (n_illicit 23 / 37 - very noisy)")
        print()


if __name__ == "__main__":
    main()
