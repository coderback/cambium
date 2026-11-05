# ADR-006 — Gate 3 pass criteria (the defending ablations)

**Status:** accepted · erratum accepted 2026-10-08 (15:30) by coderback, at the end of this file
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

**`s` is the sample standard deviation (`ddof = 1`).** Pinning this is not pedantry: a criterion
called "mechanical" must not depend on which script evaluates it, and the repo was already
inconsistent — `scripts/run_ell1_gnn.py` and `run_ell1_baselines.py` use numpy's default
`ddof=0`. At n=8 the two differ by √(8/7) ≈ 1.069, ~7% on SE_diff, which is enough to flip a
marginal clause. Gates 0 and 1 were assembled with `ddof=0` and **stand as recorded** — their
scripts are left unchanged so they still reproduce their dated gate files; `ddof=1` binds Gate 3
onward.

The `2 ×` factor approximates a two-sided 95% test. It is mildly liberal at these sample sizes
(the Welch critical value at n=8 per arm is ≈2.14), which is accepted deliberately: the
alternative is a criterion whose threshold moves with n, and a fixed, legible multiplier is
worth more here than the third decimal place of a p-value. The Welch p-value is reported
alongside for context.

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

   **Seed count: a two-stage design, both stages fixed in advance.** The provisional σ of 0.0790
   on the real arm *conflates* seed sensitivity with run nondeterminism, so it cannot size the
   batch: under determinism the residual is true seed variance, which may be smaller.

   - **Stage 1 — n = 8 per arm** (5 arms ≈ 52 min). Compute SE_diff from those runs and report
     it in the gate file.
   - **Stage 2 — n = 20 per arm, at most once.** Triggered iff *any* gated clause is
     unresolvable at stage 1. It extends **every** arm, not the failing one, and **its result is
     final in whichever direction it falls.**

   **There is no stage 3, and no "add a few more seeds until it resolves."** An earlier draft of
   this ADR said exactly that, and it was wrong: topping up repeatedly after seeing a clause miss
   is optional stopping, which inflates the false-positive rate without bound and is the same
   error as re-thresholding after seeing a result — the thing ADR-004 exists to forbid. A
   two-stage rule fixed before the data exists is a legitimate group-sequential design; an
   open-ended one is not. If a clause is still unresolvable at n=20, **it is reported as
   unresolvable** and Gate 3 does not pass on it.

   **≥5 per arm is a hard floor** regardless (enforced in `scripts/run_ell1_ablations.py`, which
   refuses to start below it).

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

Two properties of that conjunction, stated so a reader is not misled by "3/3 resolvable":

- **The gate is strictly harder than any single clause.** Requiring all three biases toward
  failing a genuinely structural model rather than passing a spurious one. That is the intended
  direction for a gate whose job is to stop overclaiming, but it is a real cost and is not
  hidden.
- **The three tests are not independent.** All three are measured against the *same* `real` arm,
  so their errors are correlated through it. Three clauses clearing is therefore weaker evidence
  than three independent confirmations would be, and must not be described as the latter.

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
- **"If unresolvable, add seeds until it resolves."** Present in an earlier draft of this ADR and
  rejected on review: it is optional stopping, and it would have written a licence to p-hack into
  the very document meant to prevent it. Replaced by the fixed two-stage rule in clause 4.
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

## Erratum, 2026-10-08 (accepted 2026-10-08 (15:30) by coderback)

**Source:** research audit 2026-10-07, §2
(`notebooks/audit/2026-10-07/README.md`; finding A2 E-B4).
The text above is not edited, so citations of its lines stay valid. **No clause changes.**

**The pins called "mechanical" (`:69-72`, `:181`) were untested code until 2026-10-07.** `ddof=1`
and `diff > 2·SE_diff` were implemented in scripts that no test imported:
- **ELL-1's copies:** `scripts/run_ell1_ablations.py:174`, `:199`, `:247` and `:286`;
- **DGF-1 Gate 1's copy:** `scripts/assemble_gate_dgf1_1.py:103-123`;
- **Gate 0's sample standard deviation:** `scripts/assemble_gate_dgf1_0.py:82`.

So "mechanical" meant a reviewed reading of code, not a tested function.
- **Since `f3e64fd`,** `tests/test_gate_criteria.py` pins the Gate-1 copy, and an edit to `ddof` or
  to the bound fails it.
- **ELL-1's copy stays untested.** ELL-1 is closed, and its Gate-3 statistics recompute from the
  gate tables (audit A2, *Summary*).
- **Gates pre-registered after ADR-018's acceptance** use its tested shared function instead (plan
  row 2a).
