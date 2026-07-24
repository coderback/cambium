"""Run the ELL-1 Gate-1 batch (GraphSAGE + GCN, 3 seeds each) and assemble Gate 1.

    python scripts/run_ell1_gnn.py [--device cuda|cpu]

Loads the frozen config (ADR-003, adapters/ell1/config.yaml `gnn:` block), trains GraphSAGE
and the capacity-matched GCN comparator on steps 1-34 and evaluates on 35-49 **under the
strict inductive protocol** (adapters/ell1/train_gnn), 3 seeds each -> 6 real registry rows.
Then reads those rows back plus the RF floor (Gate-0 clean batch) and writes
gates/GATE-ELL1-1.md: per-seed + mean±std tables, the **pre-registered ADR-004 criterion
check** (mechanical), and a blank verdict for the researcher (CLAUDE.md: never mark a gate
passed).

This is the first script to touch the 35-49 test window. Nothing here tunes anything — the
config is frozen upstream.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml

from gbe.run.config import git_commit
from gbe.run.registry import default_registry_path
from adapters.ell1.datasource_elliptic import load_elliptic
from adapters.ell1.eval import CONFIG_PATH, ell1_eval_split
from adapters.ell1.train_gnn import GNNHParams, resolve_device, run_gnn

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
BACKBONES = ("graphsage", "gcn")
GATE_PATH = REPO_ROOT / "gates" / "GATE-ELL1-1.md"
METRIC_COLS = ("illicit_f1", "illicit_recall", "illicit_auc")


def load_frozen() -> tuple[GNNHParams, dict, str]:
    """Read the frozen GNN config + experiment metadata from adapters/ell1/config.yaml."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    g = cfg["gnn"]
    hp = GNNHParams(
        backbone=g["backbone"], num_layers=g["num_layers"], hidden_dim=g["hidden_dim"],
        aggr=g["aggr"], dropout=g["dropout"], lr=float(g["lr"]), fan_out=tuple(g["fan_out"]),
        encoder_layers=g["encoder_layers"], norm=g["norm"], epochs=g["epochs"],
        batch_size=g["batch_size"], weight_decay=float(g["weight_decay"]),
    )
    base_cfg = {
        "model": cfg["model"],
        "phase": "P1",  # these are Phase-1 runs (config.yaml's default P0 was for Gate 0)
        "data_snapshot_id": cfg["data_snapshot_id"],
        "split": cfg["split"],
        "symmetrise": cfg["symmetrise"],
        "features": 165,  # ADR-001
    }
    return hp, base_cfg, str(cfg.get("device", "auto"))


def _read_rows(registry_path: Path, run_ids: set[str]) -> list[dict]:
    with registry_path.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["run_id"] in run_ids]


def _rf_floor(registry_path: Path) -> dict[str, tuple[float, float]]:
    """RF floor from the Gate-0 clean batch (baseline=rf, git_dirty=false) — from the registry.

    Returns ``{metric: (mean, std)}`` over the 3 clean RF seeds. This is the exact floor
    GATE-ELL1-0 cites and ADR-004 pre-registers (0.8058 ± 0.0024 F1).
    """
    with registry_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rf = []
    for r in rows:
        m = json.loads(r["metrics_json"])
        if m.get("baseline") == "rf" and r["git_dirty"] == "false":
            rf.append(m)
    if len(rf) != 3:
        raise RuntimeError(f"expected 3 clean RF rows for the floor, found {len(rf)}")
    return {c: _stats([m[c] for m in rf]) for c in METRIC_COLS}


def _stats(values) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _result_block(title: str, backbone_rows: list[dict]) -> tuple[str, tuple[float, float]]:
    """Per-seed table + mean±std for one backbone. Seed is a registry column, not a metric."""
    lines = [
        f"### {title} (illicit = positive class, steps 35–49, strict inductive)",
        "",
        "| seed | illicit F1 | illicit recall | illicit AUC | run_id |",
        "|------|-----------|----------------|-------------|--------|",
    ]
    rows = sorted(backbone_rows, key=lambda r: int(r["seed"]))
    cols: dict[str, list[float]] = {c: [] for c in METRIC_COLS}
    for r in rows:
        m = json.loads(r["metrics_json"])
        for c in METRIC_COLS:
            cols[c].append(float(m[c]))
        lines.append(f"| {r['seed']} | {m['illicit_f1']:.4f} | {m['illicit_recall']:.4f} "
                     f"| {m['illicit_auc']:.4f} | {r['run_id']} |")
    stats = {c: _stats(v) for c, v in cols.items()}
    lines.append(f"| **mean ± std** | **{_fmt(*stats['illicit_f1'])}** "
                 f"| **{_fmt(*stats['illicit_recall'])}** | **{_fmt(*stats['illicit_auc'])}** | |")
    lines.append("")
    return "\n".join(lines), stats["illicit_f1"]


def _criterion_block(gs_f1: tuple[float, float], gcn_f1: tuple[float, float],
                     rf_f1: tuple[float, float]) -> str:
    gs_m, gs_s = gs_f1
    gcn_m, _ = gcn_f1
    rf_m, rf_s = rf_f1
    clause1 = (gs_m - gs_s) > (rf_m + rf_s)
    clause2 = gs_m >= gcn_m
    tick = lambda b: "PASS" if b else "FAIL"
    return "\n".join([
        "## Pre-registered criterion check (ADR-004) — mechanical, not the verdict",
        "",
        f"- **RF floor band** (Gate-0 clean batch): [{rf_m - rf_s:.4f}, {rf_m + rf_s:.4f}]  "
        f"(mean ± std = {_fmt(rf_m, rf_s)})",
        f"- **GraphSAGE band**: [{gs_m - gs_s:.4f}, {gs_m + gs_s:.4f}]  (mean ± std = {_fmt(gs_m, gs_s)})",
        f"- **GCN mean**: {gcn_m:.4f}",
        "",
        f"1. Beats RF, non-overlapping bands — `(GS_mean − GS_std) > (RF_mean + RF_std)`: "
        f"`{gs_m - gs_s:.4f} > {rf_m + rf_s:.4f}` → **{tick(clause1)}**",
        f"2. At least matches GCN — `GS_mean ≥ GCN_mean`: "
        f"`{gs_m:.4f} ≥ {gcn_m:.4f}` → **{tick(clause2)}**",
        "",
        f"**Both clauses (ADR-004): {tick(clause1 and clause2)}.** "
        "This is the pre-registered mechanical result; the verdict below is the researcher's.",
        "",
    ])


def _assemble_gate(rows: list[dict], gs_ids: list[str], gcn_ids: list[str],
                   rf_floor: dict, meta: dict) -> str:
    rid2row = {r["run_id"]: r for r in rows}
    gs_rows = [rid2row[i] for i in gs_ids]
    gcn_rows = [rid2row[i] for i in gcn_ids]
    commit = rows[0]["git_commit"] if rows else git_commit(REPO_ROOT)
    snapshot = rows[0]["data_snapshot_id"] if rows else "unknown"

    gs_block, gs_f1 = _result_block("GraphSAGE (ELL-1, frozen ADR-003 config)", gs_rows)
    gcn_block, gcn_f1 = _result_block("GCN (capacity-matched comparator)", gcn_rows)
    rf_block = "\n".join([
        "### RF on raw 165 features — the floor (from GATE-ELL1-0, no graph)",
        "",
        f"| **mean ± std** | **{_fmt(*rf_floor['illicit_f1'])}** "
        f"| **{_fmt(*rf_floor['illicit_recall'])}** | **{_fmt(*rf_floor['illicit_auc'])}** |",
        "", "(F1 | recall | AUC — carried from the Gate-0 clean batch, not recomputed here.)", "",
    ])

    return "\n".join([
        "# GATE-ELL1-1 — structure vs the tabular floor (strict inductive)",
        "",
        f"**Date assembled:** {date.today().isoformat()}",
        "**Phase / doc:** ELL-1 Phase 1 — docs/01-elliptic-embedding-model-BUILD.md §4, §5, §7",
        f"**Assembled from registry rows:** {', '.join(gs_ids + gcn_ids)}",
        f"**Git commit:** {commit}  ·  **Data snapshot:** {snapshot}  ·  **Config:** ADR-003",
        "",
        "## Question",
        "Under the strict inductive protocol, does the structure-aware GNN beat the tabular "
        "floor (RF on raw 165 features) convincingly, and at least match GCN, on steps 35–49?",
        "",
        "## Pass condition (from the doc — quoted; operationalised in ADR-004)",
        "> **Gate 1 (the load-bearing gate):** the GNN model **beats the tabular floor "
        "(Random Forest / LR on raw 165 features) convincingly**, and at least "
        "**matches/beats plain GCN**, on the held-out steps 35–49. (§4, Phase 1)",
        ">",
        "> Gate 1 — *\"GNN > RF on raw features and ≥ GCN on steps 35–49.\"* (§7)",
        "",
        "ADR-004 pre-registers \"convincingly\" as **non-overlapping mean±std F1 bands** vs the "
        "RF floor, **and** GraphSAGE mean ≥ GCN mean; ≥3 seeds; strict inductive.",
        "",
        "## Results (≥3 seeds — a gate does not pass on a single seed)",
        "",
        rf_block,
        gcn_block,
        gs_block,
        f"**Support:** train = {meta['n_train']} labelled nodes (steps 1–34); "
        f"test = {meta['n_test']} labelled nodes (steps 35–49), of which "
        f"{meta['n_test_illicit']} illicit. Message passing: train over the ≤34 induced "
        "subgraph; eval over the 35–49 induced subgraph only (no full-graph forward).",
        "",
        _criterion_block(gs_f1, gcn_f1, rf_floor["illicit_f1"]),
        "_All GNN numbers copied from `experiments/registry.csv`. No cell is filled by "
        "estimate, extrapolation, or smoothing. RF floor carried from GATE-ELL1-0._",
        "",
        "## Verdict",
        "<!-- Left blank. Decided by the researcher, not by Claude. -->",
        "",
        "---",
        "_Verdict, seeds, and table are sacred once dated. Papers are assembled from gate "
        "files; nothing is reported that is not in one._",
        "",
    ])


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda|cpu (default: config/auto)")
    args = ap.parse_args()

    hp, base_cfg, cfg_device = load_frozen()
    device = resolve_device(args.device or cfg_device)
    data = load_elliptic(REPO_ROOT / "data" / "elliptic", strict=True)
    split = ell1_eval_split()
    registry_path = default_registry_path(REPO_ROOT)
    print(f"[gnn] device={device} config=ADR-003 (frozen) | test window {split.test_min}-{split.test_max}")

    ids: dict[str, list[str]] = {b: [] for b in BACKBONES}
    meta: dict = {}
    for backbone in BACKBONES:
        for seed in SEEDS:
            run_id, m = run_gnn(backbone, seed, data, split, hp, base_cfg,
                                registry_path=registry_path, device=str(device))
            ids[backbone].append(run_id)
            meta = {"n_train": m["n_train"], "n_test": m["n_test"], "n_test_illicit": m["n_test_illicit"]}
            print(f"[run] {backbone} seed={seed}: F1={m['illicit_f1']:.4f} "
                  f"recall={m['illicit_recall']:.4f} AUC={m['illicit_auc']:.4f}  ({run_id})")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    all_ids = set(ids["graphsage"] + ids["gcn"])
    rows = _read_rows(registry_path, all_ids)
    rf_floor = _rf_floor(registry_path)
    GATE_PATH.write_text(_assemble_gate(rows, ids["graphsage"], ids["gcn"], rf_floor, meta),
                         encoding="utf-8")
    print(f"[gate] wrote {GATE_PATH.relative_to(REPO_ROOT)} from {len(rows)} registry rows")


if __name__ == "__main__":
    main()