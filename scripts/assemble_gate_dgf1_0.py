"""Assemble gates/GATE-DGF1-0.md from the registry (doc-02 §4 Phase 0 and §7; ADR-007, ADR-008, ADR-010).

    python scripts/assemble_gate_dgf1_0.py

Gate 0 has three conditions (doc-02 §7): sampled training runs without OOM, the baseline logs
ROC-AUC **and** AUPRC, and ELL-1's reference set reproduces bit-for-bit on the refactored core.
ADR-008 clause 5 requires the third to be "performed programmatically against
`experiments/registry.csv` — not read off by eye" and recorded in this file, so the comparison is
re-run here against **each certifying commit's own rows**, rather than quoted from a console log or
from the checker's latest-row-wins view.

Every number is computed from the registry (plus `git`, for commit identities and which paths
changed). The script writes the gate file and nothing else. **The verdict is always left blank**, and
a gate file whose Verdict is filled in is never overwritten.

This gate is assembled **after** Gate 1 was run and signed. The file says so at the top and checks
that every row it relies on predates the first test-window row.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_extract_regression import (  # noqa: E402
    CHECK_TAG,
    CRITICAL_PACKAGES,
    MANIFEST,
    _canonical,
    _pinned_versions,
    compare,
    environment_drift,
)

REGISTRY = REPO_ROOT / "experiments" / "registry.csv"
GATE_PATH = REPO_ROOT / "gates" / "GATE-DGF1-0.md"
GATE_1_PATH = REPO_ROOT / "gates" / "GATE-DGF1-1.md"
DOC_02 = REPO_ROOT / "docs" / "02-dgraph-fin-embedding-model-BUILD.md"

VAL, TEST = "370-481", "482-821"
GNN_ARM, FLOOR_ARM = "dgf1-parity", "xgboost-parity"
FLOOR_RETUNE_WINNER = "dgf1-20260912T235406Z-7e30aa90"  # ADR-012 clause 2 amendment

# The three ADR-008 batches, in order. The last is the certificate: it ran after the final
# extraction commit (the training loop moving into gbe.gnn.train).
CERTIFYING_COMMITS = ("26774a0", "d9787e9", "d614671")
# A change to any of these after the certificate would void it: the core, the ELL-1 adapter the
# reference set exercises, and the bar itself (manifest, environment pin, checker).
CERTIFIED_PATHS = ("gbe", "adapters/ell1", "scripts/check_extract_regression.py",
                   "experiments/extract_reference_manifest.json",
                   "experiments/extract_reference_env.txt")
EXTRA_PACKAGES = ("xgboost", "pypdf")  # installed during EXTRACT, outside ADR-008's pinned five


def m(r: dict) -> dict:
    return json.loads(r["metrics_json"])


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def fmt(x: float, p: int = 4) -> str:
    return f"{x:.{p}f}"


def mean_sd(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1))


def utc(r: dict) -> datetime:
    return datetime.fromisoformat(r["timestamp_utc"])


def is_clean(r: dict) -> bool:
    return (r["git_dirty"] == "false" and m(r).get("deterministic") is True
            and "ERRORED" not in r["notes"])


# ----------------------------------------------------------------------------- guards and quotes


def refuse_to_overwrite_a_signed_gate() -> None:
    if not GATE_PATH.exists():
        return
    text = GATE_PATH.read_text(encoding="utf-8")
    verdict = text.split("## Verdict", 1)[-1].split("\n---", 1)[0]
    if re.sub(r"<!--.*?-->", "", verdict, flags=re.S).strip():
        raise SystemExit(f"refusing: {GATE_PATH.name} already carries a verdict. Dated gate files are "
                         "never regenerated over a signed verdict.")


def doc_02_line(prefix: str) -> str:
    hits = [ln for ln in DOC_02.read_text(encoding="utf-8").splitlines() if ln.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"refusing: expected one doc-02 line starting {prefix!r}, found {len(hits)}.")
    return hits[0]


def gate_1_verdict() -> str:
    text = GATE_1_PATH.read_text(encoding="utf-8").split("## Verdict", 1)[-1]
    hit = re.search(r"\*\*(PASSED|FAILED)\*\*\s*—\s*(\d{4}-\d{2}-\d{2})", text)
    if not hit:
        raise SystemExit("refusing: GATE-DGF1-1 carries no signed verdict; the ordering disclosure "
                         "below would be false.")
    return f"{hit.group(1)} {hit.group(2)}"


# ----------------------------------------------------------------------------- ADR-008


def adr_008_batches(rows: list[dict], reference: list[dict]) -> list[dict]:
    checks = [r for r in rows if m(r).get("experiment") == CHECK_TAG]
    out = []
    for short in CERTIFYING_COMMITS:
        full = git("rev-parse", short)
        batch = [r for r in checks if r["git_commit"] == full]
        observed: dict[tuple[str, int], dict] = {}
        dupes = 0
        for r in batch:
            key = (m(r)["arm"], int(r["seed"]))
            dupes += key in observed
            observed[key] = {k: v for k, v in m(r).items() if k.startswith("illicit_")}
        mismatches, missing = compare(reference, observed)
        stamps = sorted(utc(r) for r in batch)
        out.append({
            "short": short, "full": full, "rows": batch, "dupes": dupes,
            "mismatches": mismatches, "missing": missing,
            "clean": sum(is_clean(r) for r in batch),
            "span": (f"{stamps[0]:%Y-%m-%d %H:%M}–{stamps[-1]:%H:%M}Z" if stamps else "—"),
            "pin_in_tree": bool(git("ls-tree", full, "experiments/extract_reference_env.txt")),
            "green": (len(batch) == len(reference) and not dupes and not mismatches
                      and not missing and all(is_clean(r) for r in batch)),
        })
    return out


def commits_touching_certified_paths(since: str, until: str) -> list[str]:
    log = git("log", "--format=%h %s", f"{since}..{until}", "--", *CERTIFIED_PATHS)
    return [ln for ln in log.splitlines() if ln]


def identical_twin(r: dict, pool: list[dict]) -> tuple[dict | None, int]:
    """An echoed row whose every float metric equals ``r``'s — recovers a pre-echo row's config."""
    floats = {k: v for k, v in m(r).items() if isinstance(v, float)}
    for other in pool:
        om = m(other)
        if other is not r and "batch_size" in om and all(om.get(k) == v for k, v in floats.items()):
            return other, len(floats)
    return None, len(floats)


# ----------------------------------------------------------------------------- assembly


def main() -> None:
    refuse_to_overwrite_a_signed_gate()
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    dgf1 = [r for r in rows if r["model"] == "dgf1"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reference = manifest["rows"]
    n_metrics = sum(len(e["metrics"]) for e in reference)
    arm_counts = Counter(e["arm"] for e in reference)

    batches = adr_008_batches(rows, reference)
    certificate = batches[-1]
    test_rows = [r for r in dgf1 if m(r).get("window") == TEST]
    gate_1_commits = sorted({r["git_commit"] for r in test_rows})
    if len(gate_1_commits) != 1:
        raise SystemExit(f"refusing: test-window rows span {len(gate_1_commits)} commits, expected 1.")
    gate_1_commit = gate_1_commits[0]
    touched_to_gate_1 = commits_touching_certified_paths(certificate["full"], gate_1_commit)
    touched_to_head = commits_touching_certified_paths(certificate["full"], "HEAD")

    pin_path = REPO_ROOT / manifest["env_pin"]
    drift = environment_drift(pin_path)
    pinned = _pinned_versions(pin_path)
    extras = []
    for pkg in EXTRA_PACKAGES:
        try:
            have = version(pkg)
        except PackageNotFoundError:
            continue
        want = pinned.get(_canonical(pkg))
        if want != have:
            extras.append(f"`{pkg} {have}` (pin: {want or 'absent'})")

    gnn_val = sorted([r for r in dgf1 if m(r)["arm"] == GNN_ARM and m(r).get("window") == VAL], key=utc)
    floor_val = sorted([r for r in dgf1 if m(r)["arm"] == FLOOR_ARM and m(r).get("window") == VAL], key=utc)
    floor_retune = [r for r in floor_val if m(r)["experiment"] == "retune"]
    floor_pilot = sorted([r for r in floor_val if m(r)["experiment"] == "pilot"], key=lambda r: int(r["seed"]))
    floor_all = [r for r in dgf1 if m(r)["arm"].startswith("xgboost")]
    floor_with_both = sum("fraud_auc" in m(r) and "fraud_auprc" in m(r) for r in floor_all)
    raw17_windows = Counter(m(r)["window"] for r in dgf1 if m(r)["arm"] == "xgboost-raw17")
    if not floor_pilot or len({(m(r)["n_score"], m(r)["n_score_fraud"]) for r in floor_val}) != 1:
        raise SystemExit("refusing: validation floor rows are missing or disagree on the scored population.")
    n_score, n_fraud = m(floor_val[0])["n_score"], m(floor_val[0])["n_score_fraud"]

    evidence = gnn_val + floor_val + [r for b in batches for r in b["rows"]]
    last_evidence = max(evidence, key=utc)
    first_test = min(test_rows, key=utc)
    if utc(last_evidence) >= utc(first_test):
        raise SystemExit("refusing: a row this gate relies on postdates the first test-window row.")
    unclean = [r["run_id"] for r in gnn_val + floor_val if not is_clean(r)]
    errored = sum("ERRORED" in r["notes"] for r in dgf1)
    temporal = {VAL, TEST}
    off_temporal = [r for r in dgf1 if m(r).get("window") not in temporal]
    breakdown = Counter((m(r)["experiment"], m(r)["arm"], m(r).get("window")) for r in dgf1)

    L: list[str] = []
    L += [
        "# GATE-DGF1-0 — does DGF-1 run at scale, with its baseline logged, on a core that still reproduces ELL-1?",
        "",
        f"**Date assembled:** {date.today().isoformat()}",
        "**Phase / doc:** DGF-1 Phase 0 — docs/02-dgraph-fin-embedding-model-BUILD.md §4 (Phase 0, Gate 0), §7, §8  ·  "
        "**Criteria:** doc-02 §7 Gate-0 row (metrics ADR-007, reproduction ADR-008, splits ADR-010)",
        f"**Certifying commit (ADR-008):** {certificate['full']}  ·  **Data snapshot:** "
        f"{floor_val[0]['data_snapshot_id']}  ·  **ELL-1 reference manifest frozen at:** {manifest['git_commit']}",
        f"**Assembled from registry rows:** ADR-008 `experiment={CHECK_TAG}` batches at "
        + ", ".join(f"`{b['short']}` ({len(b['rows'])} rows)" for b in batches)
        + f"; DGF-1 GNN on {VAL}: " + ", ".join(r["run_id"] for r in gnn_val)
        + f"; parity floor on {VAL}: " + ", ".join(r["run_id"] for r in floor_val) + ".",
        "",
        f"> **Disclosure — this gate is assembled after Gate 1.** Gate 1 was run and signed "
        f"(**{gate_1_verdict()}**, `gates/GATE-DGF1-1.md`) before this file existed. doc-02 places Gate 0 "
        "before Phase 1, and ADR-008 clause 5 says DGF-1's Gate 0 \"cannot be signed until this check is "
        "green\" and that the result \"is recorded in `gates/GATE-DGF1-0.md`\". The check was green before "
        "any Gate-1 run, but no gate file recorded it until now. **Every row this file relies on predates the "
        f"first test-window row** (latest: `{last_evidence['run_id']}` at {utc(last_evidence):%Y-%m-%d %H:%M:%S}Z; "
        f"first test row: `{first_test['run_id']}` at {utc(first_test):%Y-%m-%d %H:%M:%S}Z — checked in code). "
        "The rows are uninformed by test-window results; the assembly is not, and a reader should weigh that.",
        "",
        "## Question",
        "Does sampled DGF-1 training run at DGraph-Fin scale without running out of memory, is its tabular "
        "baseline logged on ROC-AUC **and** AUPRC, and does the refactored `gbe/` core still reproduce "
        "ELL-1's recorded numbers bit-for-bit?",
        "",
        "## Pass condition (doc-02 — quoted)",
        "> " + doc_02_line("- **Gate 0:**")[2:] + " (§4, Phase 0)",
        ">",
    ]
    cells = [c.strip() for c in doc_02_line("| 0 |").strip().strip("|").split("|")]
    L += [
        f"> Gate 0 — *{cells[1]}* — Pass: {cells[2]} (§7)",
        "",
        "Doc-02 names no split in either statement. The Phase-0 task list and §8 step 3 do — see section 4.",
        "",
        "## 1. ELL-1's reference set reproduces bit-for-bit on the refactored core (ADR-008)",
        "",
        f"_Reference set frozen pre-refactor at `{manifest['git_commit'][:7]}`: {len(reference)} rows ("
        + ", ".join(f"{a}×{n}" for a, n in arm_counts.items())
        + f"), {n_metrics} metric values. Bar: exact float equality, paired on `(arm, seed)`, never retried "
        "(clause 2). **Each batch is compared against its own commit's rows only**, not the checker's "
        "latest-row-wins view, so no batch can borrow another's pass._",
        "",
        "| batch | commit | rows written (UTC) | rows | clean + deterministic | metrics compared | mismatches | missing | result |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, b in enumerate(batches, 1):
        compared = n_metrics - sum(":" in x for x in b["missing"])
        L.append(f"| {i} | `{b['short']}` | {b['span']} | {len(b['rows'])} | {b['clean']}/{len(b['rows'])} | "
                 f"{compared} | {len(b['mismatches'])} | {len(b['missing'])} | "
                 f"{'**PASS**' if b['green'] else '**NOT GREEN**'} |")
    L += [
        "",
        f"**The certificate is batch {len(batches)}, at `{certificate['short']}`** — the commit that moved the "
        "training loop into `gbe.gnn.train`, the last extraction step.",
        f"- Commits touching {', '.join(f'`{p}`' for p in CERTIFIED_PATHS)} after `{certificate['short']}` "
        f"and up to the Gate-1 commit `{gate_1_commit[:7]}`: "
        + ("**none**" if not touched_to_gate_1 else "; ".join(f"`{c}`" for c in touched_to_gate_1)) + ".",
        "- The same, up to the assembly-time `HEAD`: "
        + ("**none**" if not touched_to_head else "; ".join(f"`{c}`" for c in touched_to_head)) + ".",
        f"- **Environment (clause 4).** The pin `{manifest['env_pin']}` is present in every certifying commit's tree: "
        + ("yes" if all(b["pin_in_tree"] for b in batches) else "**no**") + ". At assembly the live stack "
        + ("**matches the pin** for " + ", ".join(f"`{p}`" for p in CRITICAL_PACKAGES) + " and the GPU/CUDA runtime"
           if not drift else "**drifts from the pin**: " + "; ".join(drift))
        + ". The registry records no package versions per row, so the environment **at run time** rests on the "
        "checker's console output for each batch (\"environment matches the pin\"), which is not persisted in the repo."
        + (" Installed outside the pinned five: " + ", ".join(extras) + "." if extras else ""),
        "",
        "## 2. Sampled training at scale runs without running out of memory",
        "",
        f"_DGF-1 trains GraphSAGE through PyG `NeighborLoader` on the graph as of step 369 "
        f"({m(gnn_val[-1])['n_train']:,} training seeds) and scores {n_score:,} validation users per run. "
        f"Every DGF-1 GNN row on the validation window {VAL}:_",
        "",
        "| experiment | seed | lr | batch_size | epochs | commit | wall-clock (min) | ROC-AUC | AUPRC | run_id |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    twins = []
    for r in gnn_val:
        mm = m(r)
        if "batch_size" in mm:
            knobs = f"{mm['lr']:.4g} | {mm['batch_size']} | {mm['epochs']}"
        else:
            knobs = "not echoed | not echoed | not echoed"
            twins.append((r, *identical_twin(r, gnn_val)))
        L.append(f"| {mm['experiment']} | {r['seed']} | {knobs} | `{r['git_commit'][:7]}` | "
                 f"{float(r['wall_clock_s']) / 60:.1f} | {fmt(mm['fraud_auc'])} | {fmt(mm['fraud_auprc'])} | {r['run_id']} |")
    by_batch = Counter(m(r)["batch_size"] for r in gnn_val if "batch_size" in m(r))
    L += [""]
    for r, twin, n in twins:
        if twin is not None:
            tm = m(twin)
            L.append(f"- `{r['run_id']}` predates the knob echo; all {n} of its float metrics are identical to "
                     f"`{twin['run_id']}` (`lr={tm['lr']:.4g}`, `batch_size={tm['batch_size']}`), so it is that "
                     "configuration, reproduced bit-for-bit.")
        else:
            L.append(f"- `{r['run_id']}` predates the knob echo and has no bit-identical echoed twin.")
    L += [
        f"- **Completed:** {len(gnn_val)} runs on {VAL}, "
        + ", ".join(f"{n} with echoed `batch_size={b}`" for b, n in sorted(by_batch.items()))
        + f"; plus {sum(m(r)['arm'] == GNN_ARM for r in test_rows)} on {TEST} (GATE-DGF1-1). "
        f"Unclean or non-deterministic among the rows above: {len(unclean)}. DGF-1 rows marked errored: {errored}.",
        "",
        "> **Disclosure — runs that wrote no row.** Two retune batches were stopped by the Claude Code harness "
        "for low **host** memory, each during configuration 2 (lab Session 29). Killed runs write no row, so "
        "nothing above comes from them. The diagnosis there was measured before any code changed — no "
        "per-epoch growth, and a flat footprint across the three scoring views — and the grid then completed "
        "in the researcher's own terminal. **The registry records no memory figure**, so this condition rests "
        "on completion: every run allowed to finish did finish. Whether a harness kill on host memory counts "
        "against \"no OOM\" is the verdict's to decide.",
        "",
        "## 3. Baseline ROC-AUC + AUPRC logged (ADR-007)",
        "",
        f"_Validation window {VAL}: {n_score:,} scored users, {n_fraud:,} fraud, **prevalence "
        f"{100 * n_fraud / n_score:.4f}%** (the AUPRC chance level). Never compared with test-window or ELL-1 AUPRCs._",
        "",
        "### Parity floor — validation pilot at the selected configuration",
        "",
        "_XGBoost on the 30 parity inputs, `max_depth=8`, `subsample=0.8` (ADR-012 clause 2). Sample std, `ddof=1`._",
        "",
        "| seed | ROC-AUC | AUPRC | recall @argmax | precision @argmax | run_id |",
        "|------|---------|-------|----------------|-------------------|--------|",
    ]
    for r in floor_pilot:
        mm = m(r)
        L.append(f"| {r['seed']} | {fmt(mm['fraud_auc'])} | {fmt(mm['fraud_auprc'])} | "
                 f"{fmt(mm['fraud_recall'])} | {fmt(mm['fraud_precision'])} | {r['run_id']} |")
    stats_ = {k: mean_sd([m(r)[k] for r in floor_pilot])
              for k in ("fraud_auc", "fraud_auprc", "fraud_recall", "fraud_precision")}
    L.append("| **mean ± std** | " + " | ".join(f"**{fmt(a)} ± {fmt(s)}**" for a, s in stats_.values()) + " | |")
    L += [
        "",
        "### Parity floor — validation retune grid (selection runs, one seed each)",
        "",
        "| configuration | commit | ROC-AUC | AUPRC | run_id |",
        "|---|---|---|---|---|",
    ]
    for r in floor_retune:
        mm = m(r)
        conf = (f"`max_depth={mm['max_depth']}`, `subsample={mm['subsample']}`" if "max_depth" in mm
                else "not echoed" + (" — ADR-012 clause 2's winner, `max_depth=8`, `subsample=0.8`"
                                     if r["run_id"] == FLOOR_RETUNE_WINNER else ""))
        L.append(f"| {conf} | `{r['git_commit'][:7]}` | {fmt(mm['fraud_auc'])} | {fmt(mm['fraud_auprc'])} | {r['run_id']} |")
    first_floor, first_gnn = min(floor_val, key=utc), min(gnn_val, key=utc)
    L += [
        "",
        f"- **Both metrics on every floor row:** {floor_with_both}/{len(floor_all)} DGF-1 XGBoost rows carry "
        "`fraud_auc` and `fraud_auprc`, so Gate 1's AUPRC clause had a comparator from the first floor row on.",
        f"- **Logged before the first DGF-1 GNN row:** first floor row {utc(first_floor):%Y-%m-%d %H:%M}Z; "
        f"first GNN row {utc(first_gnn):%Y-%m-%d %H:%M}Z. (Session 27's timing run scored the GNN without "
        "writing a row, so this orders rows, not every number ever computed.)",
        "- The retune rows predate the config echo; their configurations were recovered 9/9 by config-hash "
        "rebuild (lab Session 28), and the winner is recorded in ADR-012.",
        "- **The raw-17 floor was never logged on validation**: its rows exist only on "
        + ", ".join(f"{w} ({n} rows)" for w, n in sorted(raw17_windows.items())) + ", reported in GATE-DGF1-1.",
        "",
        "## 4. The official (random) split — not run",
        "",
        "| experiment | arm | window | rows |",
        "|---|---|---|---|",
    ]
    for (exp, arm, window), n in sorted(breakdown.items()):
        L.append(f"| {exp} | {arm} | {window} | {n} |")
    L += [
        "",
        f"**DGF-1 rows on any split other than the temporal windows {VAL} / {TEST}: {len(off_temporal)}.** "
        "No official-split number exists for either arm. doc-02's Gate-0 statements (quoted above) do not name "
        "a split, but its Phase-0 task list does, and §8 step 3 reads:",
        "",
        "> " + doc_02_line("3. Run **XGBoost**")[3:],
        "",
        "ADR-010 clause 1 makes those numbers **reported, never gated** — \"They acquire no pass condition and "
        "may not acquire one retroactively.\" They are owed either way; whether their absence bears on this "
        "gate is the verdict's to decide.",
        "",
        "_All numbers computed by `scripts/assemble_gate_dgf1_0.py` from `experiments/registry.csv`, the frozen "
        "ELL-1 manifest and `git`. No cell is filled by estimate, extrapolation, or smoothing._",
        "",
        "## Verdict",
        "<!-- Left blank. Decided by the researcher, not by Claude. -->",
        "",
        "---",
        "_Verdict, seeds, and table are sacred once dated. Papers are assembled from gate files; nothing is "
        "reported that is not in one._",
        "",
    ]
    GATE_PATH.write_text("\n".join(L), encoding="utf-8")

    print(f"[assemble] wrote {GATE_PATH}")
    for b in batches:
        print(f"  ADR-008 {b['short']}: rows {len(b['rows'])} mismatches {len(b['mismatches'])} "
              f"missing {len(b['missing'])} clean {b['clean']} -> {'PASS' if b['green'] else 'NOT GREEN'}")
    print(f"  certified paths touched since {certificate['short']}: to Gate-1 commit {len(touched_to_gate_1)}, "
          f"to HEAD {len(touched_to_head)} | env drift now: {drift or 'none'}")
    print(f"  GNN val rows {len(gnn_val)} | floor val rows {len(floor_val)} | floor rows with both metrics "
          f"{floor_with_both}/{len(floor_all)} | off-temporal rows {len(off_temporal)} | unclean {len(unclean)}")


if __name__ == "__main__":
    main()
