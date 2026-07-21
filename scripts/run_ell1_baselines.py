"""Run the ELL-1 tabular floor (RF + LR, 3 seeds each) and assemble Gate 0.

    python scripts/run_ell1_baselines.py

Loads the Elliptic graph, splits on steps 1-34 / 35-49, runs 6 baselines through
`gbe.run` (6 real rows appended to experiments/registry.csv), then reads those rows back
and writes gates/GATE-ELL1-0.md — per-seed illicit F1/recall/AUC + mean±std, verdict left
blank for the researcher (CLAUDE.md: never mark a gate passed).
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from gbe.run.config import git_commit
from gbe.run.registry import default_registry_path
from adapters.ell1.baselines_tabular import BASELINES, prepare_labelled_split, run_baseline
from adapters.ell1.datasource_elliptic import load_elliptic
from adapters.ell1.eval import CONFIG_PATH, ell1_eval_split

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
GATE_PATH = REPO_ROOT / "gates" / "GATE-ELL1-0.md"
METRIC_COLS = ("illicit_f1", "illicit_recall", "illicit_auc")


def _base_cfg_values() -> dict:
    """Experiment config shared by every baseline run (from adapters/ell1/config.yaml)."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "model": cfg["model"],
        "phase": cfg["phase"],
        "data_snapshot_id": cfg["data_snapshot_id"],
        "split": cfg["split"],
        "symmetrise": cfg["symmetrise"],
        "features": 165,  # ADR-001: time_step excluded from x
    }


def _read_batch_rows(registry_path: Path, run_ids: set[str]) -> list[dict]:
    with registry_path.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["run_id"] in run_ids]


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _result_block(title: str, rows: list[dict]) -> str:
    """One markdown block (per-seed table + mean±std) for a single baseline."""
    rows = sorted(rows, key=lambda r: int(r["seed"]))
    lines = [
        f"### {title} (illicit = positive class, steps 35–49)",
        "",
        "| seed | illicit F1 | illicit recall | illicit AUC | notes |",
        "|------|-----------|----------------|-------------|-------|",
    ]
    cols: dict[str, list[float]] = {c: [] for c in METRIC_COLS}
    for r in rows:
        m = json.loads(r["metrics_json"])
        for c in METRIC_COLS:
            cols[c].append(float(m[c]))
        lines.append(
            f"| {r['seed']} | {m['illicit_f1']:.4f} | {m['illicit_recall']:.4f} "
            f"| {m['illicit_auc']:.4f} | {r['run_id']} |"
        )
    stats = {c: (float(np.mean(v)), float(np.std(v))) for c, v in cols.items()}
    lines.append(
        f"| **mean ± std** | **{_fmt(*stats['illicit_f1'])}** "
        f"| **{_fmt(*stats['illicit_recall'])}** | **{_fmt(*stats['illicit_auc'])}** | |"
    )
    lines.append("")
    return "\n".join(lines)


def _assemble_gate(rows: list[dict], run_ids: list[str], meta: dict) -> str:
    by = {b: [r for r in rows if json.loads(r["metrics_json"]).get("baseline") == b] for b in BASELINES}
    commit = rows[0]["git_commit"] if rows else git_commit(REPO_ROOT)
    snapshot = rows[0]["data_snapshot_id"] if rows else "unknown"
    return "\n".join(
        [
            "# GATE-ELL1-0 — ELL-1 tabular floor (RF + LR) logged",
            "",
            f"**Date assembled:** {date.today().isoformat()}",
            "**Phase / doc:** ELL-1 Phase 0 — docs/01-elliptic-embedding-model-BUILD.md §4, §5, §8",
            f"**Assembled from registry rows:** {', '.join(run_ids)}",
            f"**Git commit:** {commit}  ·  **Data snapshot:** {snapshot}",
            "",
            "## Question",
            "Is the ELL-1 pipeline standing (clean temporal graph) and are reproducible "
            "tabular-floor numbers logged on the steps 35–49 holdout?",
            "",
            "## Pass condition (from the doc — quote it, do not invent)",
            "> **Gate 0:** clean PyG graph at scale + reproducible baseline numbers logged. (§4, Phase 0)",
            ">",
            "> Gate 0 — *\"Pipeline works, numbers reproducible.\"* (§7)",
            "",
            "## Results (≥3 seeds — a gate does not pass on a single seed)",
            "",
            _result_block("Random Forest on raw 165 features", by["rf"]),
            _result_block("Logistic Regression on standardised 165 features", by["lr"]),
            f"**Support:** train = {meta['n_train']} labelled nodes (steps 1–34); "
            f"test = {meta['n_test']} labelled nodes (steps 35–49), of which "
            f"{meta['n_test_illicit']} illicit.",
            "",
            "_All numbers copied from `experiments/registry.csv`. No cell is filled by "
            "estimate, extrapolation, or smoothing._",
            "",
            "_Notes: **165** features per ADR-001 (`time_step` excluded); illicit is the "
            "positive class; accuracy deliberately not reported (doc §5); RF on raw features, "
            "LR on a train-fit StandardScaler; edges unused — this is the no-graph floor._",
            "",
            "## Verdict",
            "<!-- Left blank. Decided by the researcher, not by Claude. -->",
            "",
            "---",
            "_Verdict, seeds, and table are sacred once dated. Papers are assembled from gate "
            "files; nothing is reported that is not in one._",
            "",
        ]
    )


def main() -> None:
    data = load_elliptic(REPO_ROOT / "data" / "elliptic", strict=True)
    split = ell1_eval_split()
    X_train, y_train, X_test, y_test, meta = prepare_labelled_split(data, split)
    print(f"[data] train={meta['n_train']} test={meta['n_test']} "
          f"(illicit in test={meta['n_test_illicit']}) | features={X_train.shape[1]}")

    base_cfg = _base_cfg_values()
    test_window = f"{split.test_min}-{split.test_max}"
    registry_path = default_registry_path(REPO_ROOT)

    run_ids: list[str] = []
    for baseline in BASELINES:
        for seed in SEEDS:
            run_id, m = run_baseline(
                baseline, seed, X_train, y_train, X_test, y_test, meta,
                base_cfg, test_window, registry_path=registry_path,
            )
            run_ids.append(run_id)
            print(f"[run] {baseline} seed={seed}: F1={m['illicit_f1']:.4f} "
                  f"recall={m['illicit_recall']:.4f} AUC={m['illicit_auc']:.4f}  ({run_id})")

    rows = _read_batch_rows(registry_path, set(run_ids))
    GATE_PATH.write_text(_assemble_gate(rows, run_ids, meta), encoding="utf-8")
    print(f"[gate] wrote {GATE_PATH.relative_to(REPO_ROOT)} from {len(rows)} registry rows")


if __name__ == "__main__":
    main()
