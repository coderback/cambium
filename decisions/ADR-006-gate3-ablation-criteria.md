# ADR-006 — Gate 3 pass criteria (the defending ablations)

**Status:** accepted
**Date:** 2026-07-24
**Deciders:** coderback
**Docs affected:** `docs/01-elliptic-embedding-model-BUILD.md` §4 (Phase 3) and §7 (Gate 3 row) —
the phrase "drops to ~baseline" amended per *Decision* clause 1. **Both amendments applied
2026-07-24 on acceptance**, §4 carrying an inline note of the previous wording and why it
changed.

## Disclosure — this is a partial pre-registration, and one clause is not blind

ADR-004 worked because it fixed the bar while **no** test-window number existed. That is not the
situation here, and pretending otherwise would be worse than the gap itself.

A **provisional edge-scramble batch has already been run and seen**: 6 seeds per arm on CUDA,
`git_dirty=true`, run ids `ell1-20260724T202110Z-70e9fab9` … `ell1-20260724T202947Z-f7dc7b7b`.
Observed: real illicit-F1 0.6763 ± 0.0790, scrambled 0.5597 ± 0.0288.

Consequences for the integrity of this ADR:

- **Clause 1 (edge-scramble) is *informed*, not blind.** Its bar is therefore derived from a
  principle — directional sign plus statistical resolvability — and **not** from the observed
  0.117 magnitude. No threshold anywhere in this ADR is a number the provisional result happens
  to clear; there is no absolute-margin bar to tune.
- **Clauses 2 and 3 (random-graph control, GNN-removed) are genuinely blind.** Neither has been
  run. They carry ADR-004's full force.
- The provisional rows stay in the append-only registry, unused for the gate. The authoritative
  batch is re-run per clause 4 and Gate 3 cites only that.

A reader is entitled to discount clause 1 accordingly, and the gate file must repeat this
disclosure.

## Context

Doc-01 §4 Phase 3 requires edge-scramble, a random-graph control, GNN-removed, and
param/compute-matched runs; §7 states Gate 3 as *"Edge-scramble drops to ~baseline; real ≫
random graph; GNN-on-local-only vs RF-on-165 reported."* Doc-00 §8 promotes these ablations to
first-class gates. "Drops to ~baseline" and "≫" are soft, and Gate 3 is the gate that decides
whether ELL-1's structural claim survives — so the bar is fixed here.

Two facts from the Gate-1 work shape the criteria:

- **Run-to-run nondeterminism on CUDA was ≈0.034 illicit-F1** — now fixed by **ADR-005**
  (deterministic kernels, ~5% cost) — and three seeds badly under-sample the real-graph arm:
  6 seeds gave ±0.0790 (range 0.5345–0.7534) where 3 seeds had suggested ±0.0306. Determinism
  removes the run-noise component but not the seed component, so any ablation bar stated as a
  bare mean difference would still be under-powered.
- **The doc's "drops to ~baseline" does not match the geometry of the result.** The RF floor is
  0.806; the GNN with structure destroyed lands *below* it, not at it. Scrambling cannot move
  the GNN "toward" a baseline that sits above it. The ablation's actual question is whether the
  **GNN's own** performance depends on structure — a within-model comparison — not whether it
  converges on RF.

## Decision

**Gate 3 is assembled from three clauses. All numbers: steps 35–49, illicit class, strict
inductive protocol, frozen ADR-003 config, ≥5 seeds per arm.**

Throughout, a difference between two arms is **resolvable** iff

```
mean(A) − mean(B)  >  2 × SE_diff ,    SE_diff = sqrt( s_A²/n_A + s_B²/n_B )
```

i.e. the gap exceeds twice the standard error *of the difference*. Variance-aware like
ADR-004's band test, and it tightens as seeds are added rather than rewarding a lucky run.

1. **Edge-scramble (amended wording).** `mean(real) − mean(scrambled)` is **positive and
   resolvable**, where *scrambled* permutes node identities within each time step
   (`gbe.eval.scramble_edges`), preserving topology, degree sequence and the temporal split.
   Doc-01 §7's "drops to ~baseline" is amended to **"drops materially and resolvably below the
   real-graph run"** — a within-model comparison. *Failure of this clause — scrambling costs
   nothing — is the critical finding doc-01 §4 names: the model would be reading features, not
   structure.*
2. **Random-graph control.** `mean(real) − mean(random)` positive and resolvable, where
   *random* rewires within each time step to a graph of equal edge count but destroyed degree
   sequence. Separates "structure is informative" (clause 1) from "any neighbours help".
3. **GNN-removed / param-matched.** `mean(real) − mean(no_edges)` positive and resolvable,
   where *no_edges* passes an empty edge set through the identical model — message passing
   disabled at identical parameter count and training budget.

**Primary metric is illicit-F1**, unchanged from Gates 0 and 1 (doc-01 §5). **Illicit recall and
AUC are reported alongside but are not pass/fail inputs**, exactly as ADR-004 ruled. This is
deliberate and costly: the provisional batch showed AUC discriminates far better (p≈0.001 vs
p≈0.02), and switching the gate metric to AUC *after observing that* would be the
forking-paths move this ADR exists to prevent. F1 stays primary.

4. **Execution.** Authoritative runs are **deterministic CUDA** (ADR-005: bit-for-bit
   reproducible at ~5% wall-clock cost, ~1.3 min per run) on committed code
   (`git_dirty=false`). Every arm re-runs its own *real* baseline in the same batch — never
   reuse the Gate-1 rows, whose code path and determinism settings differ.

   **Seed count is derived, not assumed.** The provisional σ of 0.0790 on the real arm
   *conflates* seed sensitivity with run nondeterminism, so it cannot size the batch: under
   determinism the residual is true seed variance, which may be smaller. Procedure: run **8
   seeds per arm** (≈42 min for 4 arms — determinism made the old CPU estimate of 6.6 h moot,
   so there is no longer any reason to economise), then compute SE_diff from *those* runs and
   report it in the gate file. If 2×SE_diff exceeds the observed gap, add seeds rather than
   softening the clause. **≥5 per arm is a hard floor** regardless of what the variance shows.

5. **Reported, not gated.** Two diagnostics with **no** pass condition, which must not acquire
   one retroactively:
   - **GNN-on-local-94 vs RF-on-165** (doc-01 §3/§7).
   - **Configuration-model control** — rewire within each time step by double-edge swaps, so
     every node keeps its own degree (and therefore its own features stay paired with its own
     connectivity) and only *who it connects to* is randomised. This is the arm that localises
     where the structural value lives. Clause 2's ER control differs from the real graph in
     three ways at once — wiring, degree sequence, and the node-degree↔feature pairing — so it
     can confirm that the graph matters but not *which* property mattered. `real ≈ config` would
     say the gain is essentially a degree signal, which is pointed given features 94–164 are
     hand-built one-hop aggregates; `real ≫ config` is the strongest structural claim ELL-1 can
     make. Deliberately **not** a gate clause: fixing each arm's role before the runs is what
     prevents passing on whichever arm happens to come out significant.

**Gate 3 passes iff clauses 1–3 all hold.** A partial pass is recorded as a partial pass.

## Alternatives rejected

- **An absolute margin (e.g. "drop ≥ 0.05 F1").** Rejected: with a real-arm σ of ~0.08 any
  fixed margin is arbitrary, and having seen the provisional 0.117 I could not choose one
  without it being contaminated by that number.
- **Switching the gate metric to AUC.** Rejected despite AUC being the better instrument here —
  see clause on metrics. Reconsidering the program's gate metric is legitimate, but it must
  happen in its own ADR, argued on its merits, not mid-gate after seeing which metric wins.
- **Treating "drops to ~baseline" literally as convergence on the RF floor.** Rejected: the
  scrambled GNN sits below RF, so the clause would be unsatisfiable by construction.
- **Reusing the provisional 6-seed GPU batch as the gate numbers.** Rejected: dirty rows,
  non-deterministic device, and clause 1 would then be judged on the very data that informed
  its bar.
- **Keeping 3 seeds for consistency with Gates 0/1.** Rejected: 3 seeds demonstrably
  under-sample this distribution. Consistency with an under-powered precedent is not a virtue.
- **Replacing the ER control with the configuration model.** The configuration model is the
  sharper experiment, but swapping it in would amend a clause that is currently blind, and the
  two answer different questions. Rejected in favour of running both: ER stays the gate input
  (the doc's plain reading of "random graph"), the configuration model enters under clause 5 as
  reported-only. Costs ~11 min of compute and changes no pass criterion.

## Consequences

- Gate 3 becomes a mechanical check of three inequalities against `experiments/registry.csv`,
  with the researcher still signing the verdict.
- Three ablations must be implemented before Gate 3 can be assembled — the random-graph control,
  the GNN-removed arm, and the reported-only configuration-model control (`gbe/eval/ablations.py`
  today provides only `scramble_edges`) — each with guard tests in the same session. Every one
  must assert that its rewiring stays **within a time step**: an ablation that forges an edge
  across the 34/35 cutoff either leaks the future into training or has that edge silently
  dropped by induction, and in both cases stops being comparable to the arm it controls for.
- Wall-clock: ~1.3 min per deterministic-CUDA run, so 5 arms × 8 seeds ≈ **52 min**. The
  pressure to trade seeds for time is gone; do not reintroduce it.
- `gates/GATE-ELL1-3.md` must carry the disclosure at the top of this ADR verbatim.
- If clause 1 fails on the authoritative batch, that outcome is reported as the critical finding
  and **not** re-litigated — the provisional result does not get to overrule it.

## Revisit when

Never for ELL-1 Gate 3 once the gate is assembled and dated. DGF-1/EDR-1 pre-register their own
ablation criteria in their own ADRs, inheriting the resolvability test but re-deriving seed
counts from their own measured variance.
