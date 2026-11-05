# ERRATUM-ELL1-1 — GATE-ELL1-1's stated cause, and its feature range

**Dated:** 2026-10-08 · **accepted 2026-10-08 (15:30) by coderback**
**Source:** research audit 2026-10-07, §2 (`notebooks/audit/2026-10-07/README.md`; findings B-S3 and
A2 E-S1)
**Concerns:** `gates/GATE-ELL1-1.md`, signed FAILED 2026-07-24. That file is **not edited**; dated
gate files never are.

**This is not a gate file and carries no verdict.** Gate 1's verdict (FAILED), its tables, its bands
and its criterion check stand exactly as recorded.

## What is corrected, and why

| GATE-ELL1-1 passage | lines | status |
|---|---|---|
| the causal clause: the GNN does not beat the floor "because Elliptic's features 94–164 already encode one hop and RF on them is a hard 0.806 bar" | 69–70 | **partly superseded.** As the explanation of the GNN's deficit it is not supported (below). That RF on all the features is a hard bar stands. |
| the feature range "94–164" | 70 | **off by one**: that range holds 71 columns; Elliptic has 72 aggregate features (ADR-001's erratum) |
| "EDR-1 proceeds regardless" | 71–72 | **superseded by a later decision**, not an error when written: EDR-1's start condition is ADR-020's |

**Why the "because" is superseded.** The GNN's deficit does not come from the aggregates.
- **GATE-ELL1-3, signed 2026-07-25, locates the deficit in the neural feature path.** Without any
  edges the GNN sits about 0.21 F1 below a tree on identical features, and the structural gain does
  not cover that (GATE-ELL1-3:168-170).
- **The local-94 diagnostic agrees** (lab 2026-07-25:6). It is reported, not gated, and ran on a dirty
  tree: the rows are at `c4bc923` with `git_dirty=true`, and `scripts/run_ell1_local94.py` was first
  committed afterwards, at `b065779`.
  - **What it changed:** removing 71 of the 72 aggregates did not resolvably change the GNN's F1.
    One aggregate remained, if the CSV keeps Weber et al.'s column order (ADR-001's erratum).
  - **What it left:** the GNN stayed about 0.21 below RF on all 165 features.

**What stands: the aggregates matter to a tree, on outside evidence only.**
- **The registry holds no ELL-1 random forest on the local features alone.** Its ELL-1 random-forest
  rows are the Gate-0 floor (which GATE-ELL1-0 describes as RF on all 165), its ADR-008 reproductions
  (`extract_check`) and the local-94 batch's `rf_all165` arm.
- **Weber et al.'s Table 1** (arXiv:1908.02591v1, p.5), on the same 1–34 / 35–49 split, reports a
  random forest on the first 94 features (the local ones, `time_step` included) at illicit F1 0.694,
  against 0.788 on all features. That is a different run and protocol from ELL-1's floor.

## How to state the cause now

> Under the strict inductive protocol, a tree beat the neural model on identical features. Message
> passing helped that model (+0.070 over no graph, GATE-ELL1-3) but not by enough. The protocol is
> exonerated, because zero edges cross the cutoff, and the absence of structural signal is excluded.

This is CLAUDE.md's two-part statement: **"trees beat neural nets on this tabular data; the graph
helps but not by enough."** ELL-1's Gate 1 is an engine gate in both directions (ADR-019): its
failure is not a thesis refutation.

Any write-up drawing on GATE-ELL1-1 reads this notice with it, and with ADR-004's erratum, which
corrects the leakage citation behind the strict inductive protocol.
