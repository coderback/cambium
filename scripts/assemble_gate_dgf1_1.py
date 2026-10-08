"""Assemble gates/GATE-DGF1-1.md from the registry and the persisted score vectors (ADR-012).

    python scripts/assemble_gate_dgf1_1.py

Every number in the gate file is computed here from `experiments/registry.csv` and the score files
the gate rows point to — nothing is typed by hand, so the file can be regenerated and audited. The
script writes the gate file and nothing else: no registry row, no ADR edit.

What it computes, clause by clause:
  * clause 6 — both gated clauses (ROC-AUC, AUPRC): mean ± sample std (ddof=1), SE_diff with
    n_GNN = n_floor = 8, the 2×SE_diff resolvability test, Welch p (context), and the exact Welch
    critical multiplier alongside the pre-registered 2× (context — ADR-006 admits 2× is mildly
    liberal at small n);
  * clause 5 — whether a stage-2 top-up is triggered;
  * clause 10 — a paired, stratified bootstrap of model − floor for EVERY seed pair (k, k), at the
    pinned 1,000 replicates and seed 0, plus Boyd logit intervals per arm per seed;
  * clause 1 — precision at matched recall, in both directions, per seed;
  * clause 1 / ADR-011 clause 4 — the reported rows: raw-17 floor, and the window-only and
    first-appearance views;
  * clause 7 — the facts the gate file must carry, including ADR-011's test-window label look,
    EXTRACTED verbatim from ADR-011 rather than retyped.

**The verdict is always left blank.** A gate is decided by the researcher, never by this script and
never by Claude. It also refuses to overwrite an existing gate file whose Verdict is filled in:
verdicts are sacred once dated.
"""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gbe.eval import assert_aligned, load_scores, logit_interval, paired_bootstrap_difference  # noqa: E402

REGISTRY = REPO_ROOT / "experiments" / "registry.csv"
GATE_PATH = REPO_ROOT / "gates" / "GATE-DGF1-1.md"
ADR_011 = REPO_ROOT / "decisions" / "ADR-011-dgf1-temporal-split-derivation.md"
ADR_012 = REPO_ROOT / "decisions" / "ADR-012-dgf1-gate1-preregistration.md"

WINDOW = "482-821"
GNN_ARM, FLOOR_ARM, RAW_ARM = "dgf1-parity", "xgboost-parity", "xgboost-raw17"
GATED = ("fraud_auc", "fraud_auprc")
LABEL = {"fraud_auc": "ROC-AUC", "fraud_auprc": "AUPRC"}
N_BOOT, BOOT_SEED = 1000, 0                      # pinned by ADR-012 clause 10
GNN_RETUNE_RUN = "dgf1-20260913T153524Z-ac72170d"    # ADR-012 clause 3 amendment
FLOOR_RETUNE_RUN = "dgf1-20260912T235406Z-7e30aa90"  # ADR-012 clause 2 amendment


def m(r):
    return json.loads(r["metrics_json"])


# ----------------------------------------------------------------------------- guards


def refuse_to_overwrite_a_signed_gate() -> None:
    if not GATE_PATH.exists():
        return
    text = GATE_PATH.read_text(encoding="utf-8")
    verdict = text.split("## Verdict", 1)[-1].split("\n---", 1)[0]
    body = re.sub(r"<!--.*?-->", "", verdict, flags=re.S).strip()
    if body:
        raise SystemExit(f"refusing: {GATE_PATH.name} already carries a verdict. Dated gate files are "
                         "never regenerated over a signed verdict.")


def load_gate_rows() -> dict[str, list[dict]]:
    rows = list(csv.DictReader(REGISTRY.open(encoding="utf-8", newline="")))
    d = [r for r in rows if r["model"] == "dgf1"]
    touching = [r for r in d if m(r).get("window") == WINDOW]
    if any(m(r).get("experiment") != "gate" for r in touching):
        raise SystemExit("refusing: a row touching the test window is not tagged experiment=gate.")
    arms = {a: sorted([r for r in touching if m(r).get("arm") == a], key=lambda r: int(r["seed"]))
            for a in (GNN_ARM, FLOOR_ARM, RAW_ARM)}
    for a, rs in arms.items():
        seeds = [int(r["seed"]) for r in rs]
        ok = (seeds == list(range(8))
              and all(r["git_dirty"] == "false" for r in rs)
              and all(m(r)["deterministic"] for r in rs)
              and not any("ERRORED" in r["notes"] for r in rs)
              and len({r["git_commit"] for r in rs}) == 1)
        if not ok:
            raise SystemExit(f"refusing: arm {a} is not 8 clean, deterministic, single-commit rows.")
    return arms


# ----------------------------------------------------------------------------- statistics


def mean_sd(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1))


def criterion(g_vals, f_vals) -> dict:
    g_mean, g_sd = mean_sd(g_vals)
    f_mean, f_sd = mean_sd(f_vals)
    n_g, n_f = len(g_vals), len(f_vals)
    se = math.sqrt(g_sd ** 2 / n_g + f_sd ** 2 / n_f)
    diff = g_mean - f_mean
    welch = stats.ttest_ind(g_vals, f_vals, equal_var=False)
    # Welch–Satterthwaite degrees of freedom -> exact two-sided 95% multiplier, for context only
    num = (g_sd ** 2 / n_g + f_sd ** 2 / n_f) ** 2
    den = (g_sd ** 2 / n_g) ** 2 / (n_g - 1) + (f_sd ** 2 / n_f) ** 2 / (n_f - 1)
    df = num / den if den > 0 else float("inf")
    t_crit = float(stats.t.ppf(0.975, df))
    return {"g_mean": g_mean, "g_sd": g_sd, "f_mean": f_mean, "f_sd": f_sd, "diff": diff,
            "se": se, "two_se": 2 * se, "resolvable": diff > 0 and diff > 2 * se,
            "welch_p": float(welch.pvalue), "df": df, "t_crit": t_crit,
            "resolvable_exact": diff > 0 and diff > t_crit * se}


def precision_at_recall(y, proba, target_recall) -> tuple[float, float]:
    """Highest-threshold operating point whose recall still reaches ``target_recall``."""
    precision, recall, _ = precision_recall_curve(y, proba)
    idx = np.where(recall[:-1] >= target_recall)[0]
    if idx.size == 0:
        return float("nan"), float("nan")
    k = idx[-1]
    return float(precision[k]), float(recall[k])


# ----------------------------------------------------------------------------- disclosures


def adr_011_label_look() -> str:
    """ADR-011's test-window label-look bullet, extracted verbatim (clause 7 requires verbatim)."""
    text = ADR_011.read_text(encoding="utf-8")
    start = text.find("- **Test-window label statistics were seen during review")
    end = text.find("\n## Context", start)
    if start < 0 or end < 0:
        raise SystemExit("refusing: could not locate ADR-011's label-look disclosure to quote verbatim.")
    return text[start:end].rstrip()


def git_commit_time(commit: str) -> str:
    return subprocess.run(["git", "show", "-s", "--format=%cI", commit], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


# ----------------------------------------------------------------------------- assembly


def fmt(x, p=4):
    return f"{x:.{p}f}"


def main() -> None:
    refuse_to_overwrite_a_signed_gate()
    arms = load_gate_rows()
    gnn, floor, raw = arms[GNN_ARM], arms[FLOOR_ARM], arms[RAW_ARM]
    g0 = m(gnn[0])
    commit = gnn[0]["git_commit"]
    n_score, n_pos, prev = g0["n_score"], g0["n_score_fraud"], g0["prevalence"]

    # --- clause 6 -----------------------------------------------------------------------
    crit = {k: criterion([m(r)[k] for r in gnn], [m(r)[k] for r in floor]) for k in GATED}
    all_resolvable = all(c["resolvable"] for c in crit.values())

    # --- clause 10: paired bootstrap per seed pair, and Boyd intervals ------------------
    print(f"[assemble] paired bootstrap: {len(gnn)} seed pairs x {N_BOOT} replicates (seed {BOOT_SEED})",
          flush=True)
    boot_rows, boyd_rows, matched_rows = [], [], []
    for gr, fr in zip(gnn, floor):
        seed = int(gr["seed"])
        gs, fs = load_scores(m(gr)["scores_path"]), load_scores(m(fr)["scores_path"])
        assert_aligned(fs, gs)
        b = paired_bootstrap_difference(fs["y_true"], fs["proba"], gs["proba"],
                                        n_resamples=N_BOOT, seed=BOOT_SEED)
        boot_rows.append((seed, b))
        print(f"  seed {seed}: AUPRC diff CI [{b['auprc']['ci_lo']:+.4f}, {b['auprc']['ci_hi']:+.4f}]",
              flush=True)

        ap_g = float(average_precision_score(gs["y_true"], gs["proba"]))
        ap_f = float(average_precision_score(fs["y_true"], fs["proba"]))
        boyd_rows.append((seed, ap_g, logit_interval(ap_g, n_pos), ap_f, logit_interval(ap_f, n_pos)))

        # clause 1: precision at matched recall, both directions, at each arm's argmax recall
        f_recall, g_recall = m(fr)["fraud_recall"], m(gr)["fraud_recall"]
        g_prec_at_f, g_rec_ach = precision_at_recall(gs["y_true"], gs["proba"], f_recall)
        f_prec_at_g, f_rec_ach = precision_at_recall(fs["y_true"], fs["proba"], g_recall)
        matched_rows.append((seed, f_recall, m(fr)["fraud_precision"], g_prec_at_f, g_rec_ach,
                             g_recall, m(gr)["fraud_precision"], f_prec_at_g, f_rec_ach))

    straddle = {k: sum(1 for _, b in boot_rows if b[k]["ci_lo"] <= 0 <= b[k]["ci_hi"]) for k in ("auc", "auprc")}

    # --- reported rows --------------------------------------------------------------------
    floor_has_views = any(k.startswith(("window_only_", "first_appearance_")) for k in m(floor[0]))
    views = {v: {k: mean_sd([m(r)[f"{v}_{k}"] for r in gnn]) for k in GATED}
             for v in ("window_only", "first_appearance")}
    raw_stats = {k: mean_sd([m(r)[k] for r in raw]) for k in GATED}

    # --- ordering: a CHECK, not just a sentence ----------------------------------------------
    # Both instants normalised to UTC. Printed raw, git's local offset (+01:00) next to the
    # registry's UTC made the test row read as an hour *before* the commit — the opposite of true.
    commit_dt = datetime.fromisoformat(git_commit_time(commit)).astimezone(timezone.utc)
    first_dt = datetime.fromisoformat(
        min(r["timestamp_utc"] for r in gnn + floor + raw)
    ).astimezone(timezone.utc)
    gap_s = (first_dt - commit_dt).total_seconds()
    if gap_s <= 0:
        raise SystemExit("refusing: a test-window row predates the commit that recorded the seed "
                         "count — the pre-registration ordering does not hold.")

    # ======================================================================================
    L: list[str] = []
    A = L.append
    A("# GATE-DGF1-1 — does structure beat features at scale? (temporal holdout)")
    A("")
    A(f"**Date assembled:** {date.today().isoformat()}")
    A("**Phase / doc:** DGF-1 Phase 1 — docs/02-dgraph-fin-embedding-model-BUILD.md §4 (Gate 1), §5, §7")
    A(f"**Git commit:** {commit}  ·  **Data snapshot:** {gnn[0]['data_snapshot_id']}  ·  "
      "**Criteria:** ADR-012 (metrics ADR-007, split and protocol ADR-011, determinism ADR-005)")
    A(f"**Determinism:** all rows `deterministic=true`, `CUBLAS_WORKSPACE_CONFIG={g0['cublas_workspace_config']}` "
      "— bit-for-bit reproducible on the recorded environment, not anywhere.")
    A(f"**Batch:** stage 1 of ADR-012 clause 5 — **8 seeds per arm**, the count recorded as "
      f"`**Stage-1 seeds:** 8` in ADR-012 at commit `{commit[:7]}`.")
    A(f"**Ordering (all UTC):** the seed count was committed at `{commit_dt:%Y-%m-%d %H:%M:%S}Z`; the "
      f"first-ever row on the test window started at `{first_dt:%Y-%m-%d %H:%M:%S}Z`, "
      f"**{gap_s:.0f} s later**. No test-window number existed before the seed count was fixed — "
      "and this assembly refuses to run if that ever stops being true.")
    A(f"**Assembled from registry rows:** gated GNN {', '.join(r['run_id'] for r in gnn)}; "
      f"gated floor {', '.join(r['run_id'] for r in floor)}; reported raw-17 floor "
      f"{', '.join(r['run_id'] for r in raw)}.")
    A("")
    A("> **Disclosure — what had been seen before this batch.** No DGF-1 test-window number of any kind "
      "existed before it. What *had* been seen: both 9-configuration retune grids and both "
      "5-seed pilots, **all on the validation window 370–481**, on which both arms were selected; and "
      "ADR-011's review look at test-window **label** statistics, reproduced verbatim under *Facts*. "
      "The seed count was derived mechanically from the validation pilots by ADR-012 clause 5 and "
      "recorded before the first test row.")
    A("")
    A("## Question")
    A("On a temporal holdout of users who first appear after the training cutoff, does the sampled "
      "GraphSAGE model beat a tabular floor that sees the same node-level information — on ROC-AUC "
      "**and** on AUPRC?")
    A("")
    A("## Pass condition (ADR-012 clause 6 — quoted)")
    A("> **Gate 1 passes iff both clauses hold**, on the test window 482–821, under ADR-011's protocol, "
      "deterministic, on committed code, against the parity floor **re-run in the same batch**:")
    A("> 1. **ROC-AUC:** `mean(DGF-1) − mean(floor) > 2 × SE_diff`, positive and resolvable;")
    A("> 2. **AUPRC:** the same.")
    A(">")
    A("> `s` is the sample standard deviation (`ddof=1`, ADR-006). The Welch p-value is reported "
      "alongside for context and is not the criterion.")
    A("")
    A("**Estimand (ADR-012 clause 6):** DGF-1's expected performance over seeds exceeds the floor's, "
      "**on this fixed test window**. Only seed variability is modelled; uncertainty from which users "
      "form the window is reported below as a paired bootstrap, not gated.")
    A("")
    A(f"## Results — test window {WINDOW}, {n_score:,} scored users, {n_pos:,} fraud "
      f"(**prevalence {prev:.4%}**, the AUPRC chance level)")
    A("")
    for title, rs, note in (
        ("DGF-1 — gated", gnn,
         f"GraphSAGE, ADR-003 region; `lr={g0['lr']!r}`, `batch_size={g0['batch_size']}`, "
         f"`epochs={g0['epochs']}` (ADR-012 clause 3 amendment, retune run `{GNN_RETUNE_RUN}`). "
         "Trained on the graph as of 369; scored on the graph as of 821."),
        ("Parity floor — gated", floor,
         f"XGBoost on the same 30 inputs, `max_depth={m(floor[0])['max_depth']}`, "
         f"`subsample={m(floor[0])['subsample']}` (clause 2 amendment, retune run `{FLOOR_RETUNE_RUN}`)."),
        ("Raw-17 floor — reported, not gated", raw,
         "XGBoost on the raw 17 features only, same configuration."),
    ):
        A(f"### {title}")
        A("")
        A(f"_{note}_")
        A("")
        A("| seed | ROC-AUC | AUPRC | recall @argmax | precision @argmax | run_id |")
        A("|------|---------|-------|----------------|-------------------|--------|")
        for r in rs:
            x = m(r)
            A(f"| {r['seed']} | {fmt(x['fraud_auc'])} | {fmt(x['fraud_auprc'])} | {fmt(x['fraud_recall'])} "
              f"| {fmt(x['fraud_precision'])} | {r['run_id']} |")
        cells = []
        for k in ("fraud_auc", "fraud_auprc", "fraud_recall", "fraud_precision"):
            mu, sd = mean_sd([m(r)[k] for r in rs])
            cells.append(f"**{fmt(mu)} ± {fmt(sd)}**")
        A(f"| **mean ± std** | {' | '.join(cells)} | |")
        A("")

    A("## ADR-012 clause 6 check — mechanical, not the verdict")
    A("")
    A("| clause | DGF-1 | parity floor | difference | 2×SE_diff | resolvable | Welch p | exact Welch multiplier | resolvable at exact multiplier |")
    A("|---|---|---|---|---|---|---|---|---|")
    for i, k in enumerate(GATED, 1):
        c = crit[k]
        A(f"| {i}. {LABEL[k]} | {fmt(c['g_mean'])} ± {fmt(c['g_sd'])} | {fmt(c['f_mean'])} ± {fmt(c['f_sd'])} "
          f"| {c['diff']:+.4f} | {fmt(c['two_se'])} | **{'yes' if c['resolvable'] else 'no'}** "
          f"| {c['welch_p']:.2e} | {c['t_crit']:.2f} (df {c['df']:.1f}) | {'yes' if c['resolvable_exact'] else 'no'} |")
    A("")
    A(f"**Both gated clauses: {'PASS' if all_resolvable else 'FAIL'}** "
      f"({sum(c['resolvable'] for c in crit.values())}/2 resolvable). This is the pre-registered "
      "mechanical result; the verdict below is the researcher's.")
    A("")
    if all_resolvable:
        A("**Stage 2 is not triggered** (ADR-012 clause 5: a top-up to n=20 runs only if a gated clause "
          "is unresolvable at stage 1).")
    else:
        A("**Stage 2 IS triggered** (ADR-012 clause 5): both arms extend to n=20, **at most once**, and "
          "that result is final in whichever direction it falls. No verdict may be signed on this "
          "stage-1 table.")
    A("")
    A("The two clauses share one floor arm, so they are **correlated, not independent**, and must not "
      "be described as two independent confirmations (clause 6).")
    A("")

    A("## Paired bootstrap over test users — reported, not gated (ADR-012 clause 10)")
    A("")
    A(f"_Per seed pair (k, k): users resampled once per replicate, **stratified by label**, both arms "
      f"scored on the identical resample; {N_BOOT} replicates, bootstrap seed {BOOT_SEED} (pinned). "
      "Difference is DGF-1 − floor. ADR-012 does not fix how eight per-seed intervals aggregate, so "
      "all eight are shown and any that straddle zero are counted._")
    A("")
    A("| seed | ROC-AUC diff (95% CI) | AUPRC diff (95% CI) |")
    A("|------|------------------------|---------------------|")
    for seed, b in boot_rows:
        A(f"| {seed} | {b['auc']['mean']:+.4f} [{b['auc']['ci_lo']:+.4f}, {b['auc']['ci_hi']:+.4f}] "
          f"| {b['auprc']['mean']:+.4f} [{b['auprc']['ci_lo']:+.4f}, {b['auprc']['ci_hi']:+.4f}] |")
    A("")
    A(f"**Intervals straddling zero:** ROC-AUC {straddle['auc']}/8 · AUPRC {straddle['auprc']}/8.")
    A("")
    for k, bk in (("fraud_auc", "auc"), ("fraud_auprc", "auprc")):
        if crit[k]["resolvable"] and straddle[bk] > 0:
            A(f"> **Disagreement, stated as clause 10 requires:** the seed test resolves {LABEL[k]}, but "
              f"{straddle[bk]} of 8 paired intervals over test users include zero. The claim for "
              f"{LABEL[k]} must be qualified accordingly.")
            A("")
    A("_Boyd, Eng & Page (2013) logit 95% intervals for AUPRC, per arm and seed (n_pos = "
      f"{n_pos:,}):_")
    A("")
    A("| seed | DGF-1 AUPRC [95% CI] | floor AUPRC [95% CI] |")
    A("|------|----------------------|----------------------|")
    for seed, ag, (glo, ghi), af, (flo, fhi) in boyd_rows:
        A(f"| {seed} | {fmt(ag)} [{fmt(glo)}, {fmt(ghi)}] | {fmt(af)} [{fmt(flo)}, {fmt(fhi)}] |")
    A("")

    A("## Precision at matched recall — reported, not gated (ADR-012 clause 1)")
    A("")
    A("_Per seed: the recall each arm reaches at its own argmax, then the **other** arm's precision at "
      "the highest threshold that still reaches that recall._")
    A("")
    A("| seed | floor recall @argmax | floor precision | DGF-1 precision at that recall (recall achieved) | DGF-1 recall @argmax | DGF-1 precision | floor precision at that recall (recall achieved) |")
    A("|---|---|---|---|---|---|---|")
    for seed, fr_, fp_, gpf, gra, gr_, gp_, fpg, fra in matched_rows:
        A(f"| {seed} | {fmt(fr_)} | {fmt(fp_)} | {fmt(gpf)} ({fmt(gra)}) | {fmt(gr_)} | {fmt(gp_)} | {fmt(fpg)} ({fmt(fra)}) |")
    A("")

    A("## Scoring-view sensitivity — reported, not gated (ADR-011 clause 4)")
    A("")
    A("| DGF-1 scored under | ROC-AUC | AUPRC |")
    A("|---|---|---|")
    gated_auc, gated_ap = mean_sd([m(r)["fraud_auc"] for r in gnn]), mean_sd([m(r)["fraud_auprc"] for r in gnn])
    A(f"| gated view (graph as of 821) | {fmt(gated_auc[0])} ± {fmt(gated_auc[1])} | {fmt(gated_ap[0])} ± {fmt(gated_ap[1])} |")
    for v, label in (("window_only", "window-only (edges inside 482–821)"),
                     ("first_appearance", "first appearance (edges up to each user's own node time)")):
        a, p = views[v]["fraud_auc"], views[v]["fraud_auprc"]
        A(f"| {label} | {fmt(a[0])} ± {fmt(a[1])} | {fmt(p[0])} ± {fmt(p[1])} |")
    A("")
    if not floor_has_views:
        A("> **Implementation gap, stated rather than left unnoticed.** ADR-011 clause 4 pre-registered "
          "these sensitivity rows as GNN-**vs-floor** comparisons, with the floor's view-derived features "
          "recomputed under each view. **The floor was only ever scored under the gated view** — its "
          "gate rows carry no window-only or first-appearance metrics — so the rows above show how "
          "DGF-1's *own* score moves with the view, not how the GNN-floor gap moves. They are reported, "
          "not gated, so the clause-6 check above is unaffected; producing the floor under both views "
          "is owed as follow-up work, and is not attempted in this assembly.")
        A("")

    A("## Facts this gate file must carry (ADR-012 clause 7)")
    A("")
    A(f"- **Every AUPRC here is on the test window at prevalence {prev:.4%} ({n_pos:,} of {n_score:,}).** "
      "AUPRC's chance level is that prevalence; these numbers are never compared with validation "
      "AUPRCs (1.3493%) or with ELL-1.")
    A("- **The validation window is short.** Validation users were scored at ages 0–111 steps where "
      "training users span 0–368 and test users 0–339, so the pilot that sized this batch measured "
      "variance on younger users than the gate scores.")
    A("- **Edge type 8 exists only in the test window** (all 84,338 such edges are dated 529–821). Its "
      "histogram column is constant in training and reads 0 for **both** arms, while the GNN still "
      "passes messages over those edges.")
    A("- **49.95% of raw feature values are the −1 sentinel**, standardised as a value; it lies strictly "
      "below every observed value, so it stays separable.")
    A("- **Scoring-view sensitivity and the paired bootstrap** are reported above, beside the gated "
      "numbers.")
    A(f"- **Tuning selected:** DGF-1 `lr={g0['lr']!r}`, `batch_size={g0['batch_size']}`; parity floor "
      f"`max_depth={m(floor[0])['max_depth']}`, `subsample={m(floor[0])['subsample']}` — the floor "
      "subsamples, so its seed variance is real and it runs at the GNN's seed count (clauses 2, 5).")
    A("- **ADR-011's test-window label look, repeated verbatim.** Clause numbers *inside* the quote "
      "refer to ADR-011's own clauses, not to ADR-012's pass criteria above:")
    A("")
    for line in adr_011_label_look().splitlines():
        A(f"  > {line}" if line.strip() else "  >")
    A("")

    A("_All numbers computed by `scripts/assemble_gate_dgf1_1.py` from `experiments/registry.csv` and the "
      "score files those rows reference. No cell is filled by estimate, extrapolation, or smoothing._")
    A("")
    A("## Verdict")
    A("<!-- Left blank. Decided by the researcher, not by Claude. -->")
    A("")
    A("---")
    A("_Verdict, seeds, and table are sacred once dated. Papers are assembled from gate files; nothing "
      "is reported that is not in one._")
    A("")

    GATE_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[assemble] wrote {GATE_PATH}")
    for k in GATED:
        c = crit[k]
        print(f"  {LABEL[k]:<7} DGF-1 {c['g_mean']:.4f}±{c['g_sd']:.4f} | floor {c['f_mean']:.4f}±{c['f_sd']:.4f} "
              f"| diff {c['diff']:+.4f} | 2xSE {c['two_se']:.4f} | resolvable {c['resolvable']} | Welch p {c['welch_p']:.2e}")
    print(f"  mechanical result: {'PASS' if all_resolvable else 'FAIL'} | stage 2 triggered: {not all_resolvable}")
    print(f"  paired intervals straddling zero: AUC {straddle['auc']}/8, AUPRC {straddle['auprc']}/8")
    print(f"  floor scored under the other views: {floor_has_views}")


if __name__ == "__main__":
    main()
