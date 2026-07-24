# ADR-004 — Gate 1 pass criterion, pre-registered before any 35–49 run

**Status:** accepted
**Date:** 2026-07-24
**Deciders:** coderback
**Docs affected:** none (operationalises an existing gate; recorded in `gates/GATE-ELL1-1.md`)

## Context

Doc-01 §4/§7 states Gate 1 as: the GNN "**beats the tabular floor (RF on raw 165 features)
convincingly, and at least matches/beats plain GCN**," on steps 35–49, under the **strict
inductive protocol**. "Convincingly" is a soft word. The constitution forbids weakening a
gate to make a run pass, and the safest way to honour that is to fix the operational bar
**before** any test-window number exists — otherwise the threshold can be (even unconsciously)
set to whatever the observed GraphSAGE number happens to clear.

This ADR is written while the HPO sweep (inner split, 30–34 validation only) is still running
and **no GraphSAGE/GCN number on 35–49 has been produced**. It pre-registers the bar.

## Decision

**Gate 1 PASSES iff, on steps 35–49, illicit class, strict inductive protocol, ≥3 seeds:**

1. **Beats the RF floor convincingly — non-overlapping bands.**
   `mean(GraphSAGE illicit-F1) − std(GraphSAGE illicit-F1)  >  mean(RF illicit-F1) + std(RF illicit-F1)`
   i.e. the mean±std bands of GraphSAGE and the RF floor do not overlap, GraphSAGE above.
   Equivalently, GraphSAGE beats RF by more than the combined seed std. The RF floor is
   **0.8058 ± 0.0024** (GATE-ELL1-0, frozen). **AND**
2. **At least matches GCN.** `mean(GraphSAGE illicit-F1) ≥ mean(GCN illicit-F1)`, GCN run
   under the identical protocol and seeds.

Illicit **recall** and **AUC** are reported alongside for context but are **not** pass/fail
inputs (doc-01 §5 fixes F1 as the gate metric). GraphSAGE and GCN both use the ADR-003 frozen
config; GCN inherits the architecture/optimisation (only the convolution differs) so the
comparison is capacity- and compute-matched (doc-00 §8). Seeds: 0, 1, 2.

## Alternatives rejected

- **Fixed absolute margin (e.g. ≥ RF + 0.02 F1).** Rejected: the 0.02 is arbitrary and
  ignores each estimator's seed variance; the bands test is variance-aware and principled.
- **Lenient `mean(GraphSAGE) > mean(RF)`.** Rejected: a mean beat inside overlapping noise
  bands is not "convincing"; it would let a within-noise result pass — the weak claim the
  gate exists to prevent.
- **Decide the threshold after seeing the 35–49 numbers.** Rejected: this is the exact
  motivated-reasoning failure pre-registration exists to remove.

## Consequences

- The Gate-1 verdict becomes a **mechanical check** of two inequalities against
  `experiments/registry.csv` — not a judgment made after the fact. `gates/GATE-ELL1-1.md`
  will show both bands and state pass/fail per this ADR; the researcher still signs the verdict.
- **A miss is a valid, pre-registered result, not a fudge.** Per doc-01 §0 failure semantics,
  "structure did not beat the one-hop-encoding floor under strict inductive protocol" is the
  cheap lesson ELL-1 exists to deliver; it is written into P0 and EDR-1 proceeds regardless.
- Because features 94–164 already encode one hop and the RF floor already sits at the 0.807
  high-water mark that collapses to ~0.12 under strict eval elsewhere, clause 1 is a genuine
  bar, not a formality.

## Revisit when

Never for ELL-1 Gate 1 once dated (the criterion is sacred once the gate is assembled).
Later models pre-register their own gate criteria in their own ADRs.
