# ADR-018 — Gate statistics v2: α, power, split-α looks and a Welch criterion

**Status:** accepted
**Date:** proposed 2026-10-07 · **accepted 2026-10-08 (00:43) by coderback**, at draft 3, after two
review rounds (*Draft history*). Tier A, so it is implemented no earlier than 12 hours after
acceptance and not on 2026-10-08: plan row 2a starts 2026-10-09 at the earliest.
**Tier:** A (sets gate criteria; CLAUDE.md *Process*, ADR-017)
**Review rounds:** 2/4. Round 2, a diff-only pass, found nothing blocking; its items are folded in
(*Draft history*).
**Deciders:** coderback
**Amended:** 2026-10-08 by ADR-019 clause 7 (clause 1's absence rule; clause 4.5's forward sentence).
**Replaces, for every gate pre-registered after acceptance:**
- ADR-006's resolvability test (`ADR-006:60-64`), its fixed multiplier (`:77-81`), its unadjusted
  two-stage boundary (`:112-124`), and the inheritance of that test it sets for DGF-1 and EDR-1
  (`:197-199`);
- ADR-012 clause 5's sizing inequality, as the template for later gates;
- ADR-007's inheritance of ADR-006's test for DGF-1 Gate 3 (`ADR-007:124-126`). The metric pair
  stays.

None of those files is edited. **No signed gate is re-evaluated under this ADR** (clause 10).

**Docs affected:**
- `CLAUDE.md`: the seeds bullet in *Integrity of results*, and *Next* item 2;
- `docs/research-plan-UNIFIED-GBE-GDE.md:53`, `:107`, `:151`;
- `docs/00-shared-core-graph-embedding-GUIDE.md:164-165`;
- `docs/02-dgraph-fin-embedding-model-BUILD.md:174`, `:178`, `:264-265`, `:267`, `:269-276`;
- `docs/timeline.md:8`, `:347`.

**Not amended:**
- `docs/01-elliptic-embedding-model-BUILD.md:154` and `docs/02-dgraph-fin-embedding-model-BUILD.md:157-158`:
  ELL-1 and DGF-1 Gate 1 are signed;
- the "≥3" and Gate-2 lines in doc-03, doc-04 and the GDE build docs, and `docs/timeline.md:308`:
  these go to the consistency ADR (clauses 8 and 9);
- `docs/document-amendments-v0.2.md:28`: a historical changelog;
- the timeline's history lines.

**Evidence:**
- `notebooks/measurements/2026-10-07-gate-stats-v2/measure_gate_stats.py` and `output.txt`:
  label-free, sections A–H, cited below as §A–§H;
- the prototype in `notebooks/measurements/2026-10-07-gate-stats-v2/prototype/` (clause 7).

## Context

The audit of 2026-10-07 found unstated error rates and power in the gate statistics
(`notebooks/audit/2026-10-07/README.md:44`; D2 and D4 in
`notebooks/audit/2026-10-07/D-governing-docs.md:46`, `:48`). The researcher decided on the core of
D's proposal (`notebooks/lab/2026-10-07.md:14`). Bootstrap gating and rolling origins are left to
EDR-1's Phase-0 ADRs.

**The rule today:**
- a gap is resolvable iff `mean(A) − mean(B) > 2 × SE_diff`, with `ddof=1` (`ADR-006:60-64`);
- stage 1 runs at 8 seeds, and one stage 2 at 20 is judged against the same 2 (`ADR-006:112-124`);
- stage-1 `n` is the smallest in {8, 12, 16, 20} with `2 × SE_diff(n) ≤ ½·Δ_val` (ADR-012
  clause 5).

**Measured on synthetic normal seed scores** (`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt`):
- **The false-pass rate is well above the nominal 0.0228.** With the SE estimated from seeds, the
  rule passes a zero gap at:
  - 0.040–0.057 with one look at 5 seeds;
  - 0.051–0.063 on the 8-then-20 design;
  - 0.058–0.092 at 3 seeds (§C).
- **There are two causes:**
  - the same 2 at both looks, which costs 0.0394 even with known variance (§A; the audit's Monte
    Carlo gave 0.0397);
  - a normal constant applied to a statistic with 4 to 38 degrees of freedom (§B).
- **Planned power is 0.50** at the sizing rule's own boundary (§D, last line).

These are properties of the rule, not of any verdict.

## Decision

### Clause 1 — α, one-sided, per clause

**Every directional gated clause is tested one-sided, in its pre-registered direction, at
α = 1 − Φ(2) = 0.02275.** That is the one-sided level of "2 × SE" under a normal statistic.
ADR-006:77 calls the factor an approximation to a two-sided 95% test, which is one-sided 0.025;
0.02275 sits just inside that.
- **A gate that requires every clause** is an intersection-union test. No adjustment is needed, and
  the gate's false-pass rate is at most α, by clause 3's bound (Welch is approximate where both arms
  vary).
- **A claim that at least one of k clauses holds** uses α/k in place of α everywhere: the per-look
  levels of clause 3, the power planning of clause 4 and the table of clause 6. It is fixed in the
  pre-registration. That claim's planned power is at least the largest clause's, not clause 4's
  joint bound. A partial pass is still recorded clause by clause.
- **Clauses that share an arm are correlated.** They are never described as independent
  confirmations (as ADR-012 clause 6).
- **Some clauses fall outside clause 2:** those with no pre-registered direction, and those that
  claim absence or equivalence, for example "GNN-removed matches the floor".
  - A non-resolution under clause 2 never establishes absence.
  - **A difference claim with no direction** is tested two-sided, at `ℓ/2` per tail at each look.
  - **A one-sided absence claim** — the gain is below a pre-registered margin `m` — needs one one-sided
    test: clause 2 with the arms swapped and the comparator shifted up by `m`, at the look's `ℓ`. **A
    two-sided absence or equivalence claim** — the gain lies within `±m` — needs two one-sided tests,
    each at the look's `ℓ`. Never at α at both looks. *Amended 2026-10-08 (ADR-019 clause 7), was: "An
    absence or equivalence claim needs two one-sided tests against a pre-registered margin, each at
    the look's ℓ (clause 3), never at α at both looks."*
  - **Clause 4 does not size these clauses,** and its "powered" label does not apply to them. The
    gate's pre-registration fixes, at its acceptance, each such clause's margin, its sizing, what
    "powered" means for it, and whether its non-resolution triggers stage 2.

### Clause 2 — The criterion: a one-sided Welch test

A clause is **resolvable** iff `diff > t_crit × SE_diff`, where:
- **`diff`** is `mean(A) − mean(B)` in the pre-registered direction;
- **`SE_diff`** is `sqrt(s_A²/n_A + s_B²/n_B)`, with `s` the sample standard deviation (`ddof=1`,
  unchanged).
  - **An arm *does not vary* when all its values are bit-identical.** Its `s` is then exactly 0,
    whatever rounding would leave.
- **`df`** is Welch–Satterthwaite, `(v_A + v_B)² / (v_A²/(n_A−1) + v_B²/(n_B−1))` with `v = s²/n`.
  It is not rounded.
  - An arm that does not vary adds zero to both sums, so a seed-deterministic comparator gives
    `df = n − 1`.
- **`t_crit`** is Student's t upper quantile at `df` for clause 3's per-look level `ℓ`.

**Refusals.** The function refuses three cases:
- fewer than 5 seeds in an arm;
- unequal seed counts;
- neither arm varies.

A clause refused at assembly counts as unresolvable for the stage-2 trigger. It is reported as
"refused: <reason>", and it cannot pass. Clause 4 makes these refusals fire at the validation pilot,
before that gate's first test-window run.

**What the gate file prints for each clause and look:**
- `diff`, the realised and planned `SE_diff`, `df`, `ℓ` and `t_crit`;
- the one-sided Welch p-value;
- the verdict.

The inequality decides, and `p < ℓ` holds exactly when it does. A p-value between `ℓ` and α is
reported as not resolvable at that look.

**Arms that share seed ids.** Welch treats arms as independent.
- **Positive correlation makes it conservative:** under no true gap, at a per-seed correlation of
  0.5 the false-pass rate falls to 0.0025 (§H).
- **The price is power.** A pre-registration may choose a paired test instead. The choice is fixed
  at its acceptance, before the pilot is seen.
- **Negative correlation would make Welch liberal,** and
  `notebooks/measurements/2026-10-07-gate-stats-v2/output.txt` §H measured only r of 0, 0.5 and 0.9.
  Where arms share seed ids, clause 6's table prints the pilot's per-seed correlation for each
  clause, so a negative one is disclosed before that gate's first test-window run.

### Clause 3 — Two looks split α

**Inherited unchanged:**
- stage 1 runs at `n1` seeds per arm;
- stage 2 runs at `n2 = 20`, at most once, and extends every arm;
- stage 2 is triggered iff any gated clause is unresolvable or refused at stage 1;
- stage 2's result is final for every clause, in whichever direction it falls (ADR-006 clause 4).

**What changes:**
- **Each look tests at `ℓ = α/2`** when a second look is possible (`n1 < n2`), and at `ℓ = α` when
  it is not (`n1 = n2`). On the z-scale that is 2.2776 against 2.
- **Stage 2 is cumulative.** It judges all `n2` seeds, stage 1's included.
- **Every arm in a comparison runs the same seed count.** ADR-012:235 allowed a seed-deterministic
  floor 3 seeds; under this clause it runs `n` like every arm.

**Why it holds.** A clause passes only by resolving at a look that runs, so its false-pass rate is
at most `α/2 + α/2 = α`. That holds whatever triggers stage 2, whatever the correlation between the
looks, and for any `(n1, n2)`. It is exact wherever each look's test is exact: one-sample t against
a seed-deterministic comparator. Where both arms vary it rests on Welch's approximation.

**Measured false-pass rate** (`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt` §G): **0.0165–0.0215** across stage 1 at 5, 8, 12 or 16 seeds and
comparator spreads from 0 to 2× the arm's. That is below 0.02275 in every cell, with a Monte Carlo SE
of 0.00024. One look of Welch alone measured 0.0201–0.0230
(`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt` §C, the 5- and 20-seed rows).

### Clause 4 — Power and the stage-1 seed count

1. **The pilot** runs on the validation window only. It uses at least 5 seeds per arm, the same count
   in every arm, and is deterministic, on committed code, through `gbe.run` (as `ADR-012:222-226`).
   - Clause 2's refusals apply to the pilot, so a comparison no seed test can judge is found before
     that gate's first test-window run.
   - The pilot never touches a test window.
2. **The planning effect** for each clause is `δ = ½·Δ_val`, the validation gap in the clause's
   direction.
   - The ½ was derived for DGF-1: younger validation users, and arms selected on validation
     (ADR-012 clause 5, from `ADR-012:244`).
   - A pre-registration on another split states, at its acceptance, whether those reasons hold. It
     keeps ½ unless it argues another factor there, before the pilot is seen.
3. **Planned stage-1 power** is `P(T > t_crit)`.
   - `T` is noncentral t, with `df` from clause 2 at `n` seeds per arm and noncentrality
     `δ / SE_diff(n)`.
   - `t_crit` is at stage 1's `ℓ`.
   - The pilot's standard deviations stand in for each arm's.
4. **Stage-1 `n`** is the smallest in {8, 12, 16, 20} whose planned power is at least **0.80** on
   every gated clause. If none qualifies, or if any `Δ_val ≤ 0`, it is 20.
5. **A clause is *powered*** iff its planned stage-1 power at `δ` is at least 0.80. ADR-019 answers
   the GBE half by margin tests instead. "Powered failure" still governs how gated clauses report a
   failure. *Amended 2026-10-08 (ADR-019 clause 7), was: "The failure-semantics ADR (plan row 3) uses
   this definition."*
   - **An under-powered clause** is printed with its planned power. If it does not resolve, the gate
     file says "not resolvable at planned power p", which supports no claim that the effect is absent.
   - **At gate level,** the table prints two bounds at the chosen `n`, both at the planning effects:
     joint stage-1 power is at least `1 − Σ(1 − pᵢ)`, and the chance that stage 2 is triggered is at
     most `Σ(1 − pᵢ)`. They are the same sum.
   - **A clause's failure is a *powered failure*** only if two powers at the pre-registered `δ` both
     reach 0.80: the planned power, and the power recomputed with the realised standard deviations
     at the `n` and `ℓ` of the look that decided the clause. The realised gap is never used. This guards against pilot standard deviations that were too
     small, and it costs no seeds.
   - **A refused clause is never a powered failure.** A gate failure counts as a powered failure only
     through a powered failure of one of its clauses.

**Measured accuracy** (`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt` §G and §D):
- Under the split, planned power 0.7793, 0.7695 and 0.7534 against simulated 0.7795, 0.7696 and
  0.7534. That is for a fixed comparator at 8, 12 and 16 seeds.
- Under the common boundary, planned and simulated power agree within 0.01 when both arms vary
  (§D, 0.7901–0.7973).
- Single-clause power across both stages is 0.88–0.9995 (§G). That is at effects where the split's
  stage-1 power is 0.75–0.78.

**The cost is more seeds than ADR-012's rule.** For example, s = 0.010 at a gap of 0.020 needs 8 seeds
under the old rule (§E) and 16 under this one (§G).

**Not covered:** the pilot's standard deviations come from 5 seeds. Where they are underestimated, the
clause gets less than its planned power. The realised `SE_diff` is printed beside the planned one,
and the powered-failure test above uses it.

**Compute does not weaken this** (as ADR-012 clause 5). If 20 seeds are infeasible, the gate is
postponed and the batch rented (research plan :137).

### Clause 5 — Departures from the default design

A pre-registration may set another candidate set or cap, for example where runs are costly. The
conditions:
- it states a reason, and the pre-registration is accepted before that gate's first test-window run;
- its smallest stage-1 `n` is at least 5, and every arm runs the same count;
- its levels come from clause 3, whose bound needs no new simulation for any `(n1, n2)`;
- it prints clause 6's table.

More than two looks are outside this ADR.

### Clause 6 — The α and power table every pre-registration prints

**For each gated clause:**
- the direction, and α (or α/k);
- the design `n1 → n2`, and each look's `ℓ`;
- each arm's pilot `s`, whether it varied, and `Δ_val` (validation);
- where arms share seed ids, the pilot's per-seed correlation;
- `δ`;
- planned stage-1 power at each candidate `n`;
- the chosen `n`;
- "powered: yes/no".

**For the gate,** at the chosen `n`:
- the joint-power bound and the stage-2-trigger bound (clause 4);
- the design's own basis for α: "per-look levels sum to α (ADR-018 clause 3); measured 0.0165–0.0215
  on §G's two-look designs and 0.0201–0.0230 on §C's single looks; Welch is approximate where both
  arms vary", with "not simulated" added for a design outside both.

**How it is produced:**
- the shared function produces it mechanically from the pilot's registry rows, never by hand;
- the pilot runs after the pre-registration's acceptance (as `ADR-012:220`), so the table cannot be
  in the accepted text. It is committed beside the ADR, and the chosen `n` goes in the ADR's header
  line, as ADR-012 clause 8 did for `**Stage-1 seeds:**`;
- both happen before that gate's first test-window run. Recording this mechanical output is not an
  amendment, because every choice that shapes it was fixed at acceptance.

### Clause 7 — The shared function (implementation: plan row 2a)

**Where it lives:** `scripts/gate_stats.py`, never `gbe/` (`gbe/CLAUDE.md:9-10`), with tests in
`tests/test_gate_stats.py`.

**What it computes:**
- the per-look level;
- clause 2's criterion, with its refusals;
- clause 4's planned power, stage-1 `n` and bounds;
- clause 6's table.

**Who imports it:** every gate assembler written after acceptance imports it and keeps no copy. The
Gate-1 `criterion` (`scripts/assemble_gate_dgf1_1.py:108`) stays as it is, still pinned by
`tests/test_gate_criteria.py`, and is never used for a new gate.

**Prototype evidence (ADR-017 item 3).** The prototype is
`notebooks/measurements/2026-10-07-gate-stats-v2/prototype/` (`gate_stats_proto.py`, 19 tests in
`test_gate_stats_proto.py`, `mutate.py`, `mutation-output.txt`), written 2026-10-07/08 before
acceptance.
- **Where it is kept.** `CLAUDE.md:222-223` says a prototype is scratch code "outside the repo". This
  one is kept in the repo as evidence, by the researcher's decision of 2026-10-08, so the mutation
  results stay checkable. It is not implementation, and pytest never collects it
  (`pyproject.toml:25`). This is an exception for ADR-018 only; it does not amend the rule.
- **Unmutated, it passes. Each of 20 mutations makes it fail:**
  - population variance;
  - a z critical value;
  - pooled df;
  - a two-sided level inside the test;
  - no split across looks;
  - the split applied to a single look;
  - a seed floor of 3;
  - no refusal when neither arm varies;
  - direction ignored;
  - power planned at the full `Δ_val`;
  - a power target of 0.5;
  - "any clause suffices";
  - a z approximation to power;
  - planning at the single-look level;
  - a wrong Welch-df numerator;
  - "varies" judged by computed variance;
  - unequal counts accepted;
  - the pilot not checked before planning;
  - joint power as a product;
  - a two-sided p-value.
- **Anchors:**
  - textbook: Student's t at df 4 for two-sided 95% is 2.776, and the one-sided p at that t is 0.025;
  - hand-computed: a Welch df of 7.2 for two varying arms;
  - the rest are frozen from the measurement's independent copy of the power function (§G) and
    checked there by Monte Carlo.
- **The implementing commit** reruns these mutations on `scripts/gate_stats.py` and cites the output.

### Clause 8 — DGF-1's Gate 2 is pre-registered before Phase 2

**DGF-1's Phase 2 does not run before its Gate-2 pre-registration (Tier A) is accepted.**
- **What that ADR does:** it names Gate 2's metric, comparator, split and clauses, and it applies
  clauses 1–6 to every seed-based comparison it gates.
- **Why it is needed:** today Gate 2 has no metric: "entity embeddings cluster fraud above the
  feature-only baseline" (doc-02:174; audit D9).
- **Where the rule already appears:** this restates `CLAUDE.md:97` and writes it into doc-02.
- **Other models' Gate 2** goes to the consistency ADR. They sit outside the current phase, and some
  of their Gate-2 lines have human-judged samples or "neutral" outcomes
  (`docs/03-edgar-risk-embedding-model-BUILD.md:128`,
  `docs/04-edgar-hiddenlink-embedding-model-BUILD.md:97`,
  `docs/research-discovery-model-BUILD.md:115`).

### Clause 9 — One seed floor: 5

**Every model metric reported for an arm in a gate file, gated or reported-only, comes from at least
5 seeds per arm, with variance** (`ddof=1`). Gated comparisons also use clause 2.

**Exempt:**
- counts;
- measurement figures (the exception at `CLAUDE.md:135-137`);
- bit-for-bit reproduction checks;
- runs labelled as selection runs (for example `ADR-015:456`).

**Why 5:**
- **At 3 seeds the old rule passes a zero gap at 0.058–0.092** (§C).
- **Clause 2 controls the false-pass rate at any n,** so the floor's job is now a usable
  standard-deviation estimate. At df 2, the one-sided `t_crit` at α is 4.53 (§B).
- **It settles D4 for the documents amended here.** The floor was stated as 3, 5 and 8. It is now 5,
  and 8 is the default stage-1 minimum (ADR-012 clause 5's reason), not a floor.

**What it means elsewhere:**
- ADR-015's 5 positioning seeds meet the floor. Its quote of CLAUDE.md's "≥3" (`ADR-015:338`)
  becomes historical, and the file is not edited.
- The "≥3" lines in doc-03, doc-04 and the GDE build docs stay on the consistency ADR's list.

### Clause 10 — Scope, and what this ADR does not decide

- **Applies to:**
  - every gate pre-registered after acceptance, starting with DGF-1 Gate 3;
  - any directional inferential comparison in matched-time's test batch (the audit's "stage 2",
    not clause 3's). Matched-time's go/no-go on informativeness is not a gate clause, and this ADR
    does not govern it.
- **Signed gates:** ELL-1 Gates 0, 1 and 3 and DGF-1 Gates 0 and 1 keep the rules they were decided
  under. No document recomputes them under this ADR.
- **Left to EDR-1's Phase-0 ADRs:**
  - a hierarchical user × seed bootstrap as a gate condition;
  - at least two rolling-origin cutoffs.

  The paired user bootstrap stays reported, not gated (ADR-012 clause 10).
- **Unchanged:**
  - `ddof=1`;
  - the ½ planning factor (with clause 4's condition);
  - the candidate set {8, 12, 16, 20} and the cap of 20;
  - stage 2's rules;
  - "compute does not weaken this".

## Alternatives rejected

- **Keep 2 × SE.**
  - **Why ADR-006:77-81 chose it:** a fixed multiplier is more legible than "the third decimal place
    of a p-value".
  - **Against it:** the measured cost is the second decimal: 0.051–0.063 on 8 → 20 and 0.040–0.057
    with one look at 5 seeds (`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt` §C).
  - **Legibility is kept:** `t_crit` is printed.
- **A Pocock-type common boundary** (2.2315 at 8 → 20; the "≈2.23" of `notebooks/lab/2026-10-07.md:14`).
  - **Against it:**
    - it measured up to 0.0238, above α (`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt` §C);
    - it depends on `corr = √(n1/n2)`, which fails for unequal counts or shared seeds;
    - every new design needs its own simulation;
    - it puts quadrature in the criterion.
  - **The researcher chose the split on 2026-10-08** after §G was measured
    (`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt`).
  - **The split's cost:** 2–5 points of stage-1 power at the same `n` (§G), and 3 of §G's 20 grid
    cells move one step, from 12 to 16.
- **Plan on an upper confidence bound for the pilot's standard deviation.**
  - An 80% bound from 5 seeds is 1.558·s, which can need up to about 2.43× the seeds (§F).
  - On §F's three examples it moved `n` once.
  - Rejected for cost. The realised `SE_diff` is printed, so a shortfall shows.
- **Gate on the user bootstrap now** (D's full proposal). Deferred to EDR-1 Phase 0 by decision.
- **Two-sided tests.** Every directional clause has a pre-registered direction. Absence and
  equivalence clauses get their own test (clause 1).
- **A paired test by default.** Arms do not always share seeds meaningfully. Welch is conservative
  under positive correlation (§H), and a pre-registration may choose paired.
- **A power target of 0.9.** It costs more seeds, and 0.80 is the convention and the decision.

## Consequences

- **Gate 3's pre-registration** (plan row 8) uses clauses 1–6 and imports the shared function.
  ADR-006/012's seed rule does not carry over.
- **Gates cost more seeds** than under ADR-012's rule (§E, §G).
- **Implementation (row 2a)** comes at least 12 hours after acceptance and never the same day. It
  touches nothing in `gbe/`, so no ADR-008 re-run is needed.
- **On acceptance, the listed documents are amended.** Every replacement carries the scope "for
  gates pre-registered after ADR-018", so no signed gate's rule is rewritten:
  - each "≥3 seeds" site becomes "≥5 seeds per arm with variance (ADR-018, for gates pre-registered
    after it)";
  - doc-00:164 and doc-02:178, :265 and :267 add "judged by ADR-018's Welch criterion, split-α looks
    and power rule";
  - doc-00:165 gains "*Amended (ADR-018).* For gates pre-registered after ADR-018 the floor is 5, and
    a count may rise only to the pre-registered stage 2";
  - doc-02:174 and :264 keep Gate 2's question and append "— its metric, comparator and clauses are
    fixed by its own pre-registration ADR before any Phase-2 run (ADR-018 clause 8)";
  - doc-02:269-276 gains "*Amended (ADR-018).* For gates pre-registered after ADR-018, its clauses
    1–6 replace the 2 × SE test, the sizing inequality and the floor above";
  - `docs/` is grepped for "3 seeds" before and after (ADR-012's practice);
  - the consistency ADR's list keeps D4 for the docs not amended here and gains clause 8's other
    models;
  - every change is committed to the governance repository.
- **CLAUDE.md's seeds bullet becomes, on the researcher's instruction:**
  > **≥5 seeds per arm, with variance, on every model metric in a gate file** (ADR-018 clause 9;
  > counts, measurement figures, bit-for-bit reproduction checks and labelled selection runs are
  > exempt). A gate does not pass on a single seed. Never mark a gate passed; gates are decided by me
  > from `gates/GATE-*.md` files that you may help assemble from the registry. A gated comparison is
  > a one-sided Welch test at α = 0.02275 (1 − Φ(2)) per clause, split α/2 across the two looks when
  > stage 2 is possible; its stage-1 seed count is the smallest giving 80% planned power on a
  > validation pilot that is deterministic, on committed code and run through `gbe.run`; the shared
  > function prints its α and power table before the gate's first test-window run. Seeds measure
  > *seed* sensitivity only on a deterministic backend (ADR-005). Each gate's ADR fixes its count
  > rule in advance — never assumed, and never raised beyond the pre-registered stage 2 (anything more
  > is optional stopping; see ADR-006 clause 4). Gates pre-registered before ADR-018's acceptance
  > keep the rules they were pre-registered under.

  *Next* item 2 becomes "accepted (ADR-018); its shared function, plan row 2a, is owed".

## Revisit when

- **EDR-1 Phase 0:** bootstrap gating and rolling origins.
- **A pilot's seed scores look clearly non-normal,** for example bimodal. The pre-registration then
  says so and argues the test, because the rates in §C, §G and §H were measured on normal draws
  (`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt`).
- **A design outside clause 5** is wanted: more than two looks, or fewer than 5 seeds.

## Draft history

- **Draft 1** (2026-10-07). The prototype and measurement (§A–§F) came first. It used a Pocock-type
  common boundary.
- **Round 1** (2026-10-07/08): a full adversarial review by a fresh subagent, whose prompt opened with
  the standing preamble.
  - **Tier A confirmed.**
  - **One blocking finding.** Clause 1's "at most α" was contradicted by the measured 0.0238 (`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt` §C), and the
    "simulated ≤ 0.024" line was unmeasured for designs under clause 5.
  - **Eight should-fix findings,** plus nits.
  - **Checks by the main session:** every citation was range-checked and found in range. The two
    mutants said to survive were reproduced; both survived draft 1's tests.
- **The researcher's decisions, 2026-10-08,** taken after §G and §H were measured
  (`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt`):
  - split α (clause 3);
  - amendments limited to the programme docs and doc-02;
  - the prototype kept under `notebooks/measurements/`.
- **Draft 2** (2026-10-08):
  - **Blocking:** fixed by the split (clause 3).
  - **Should-fix:**
    - joint-power and trigger bounds (clauses 4 and 6);
    - refusals at the pilot, "does not vary" defined as bit-identical, a refusal counted as
      unresolvable, cumulative stage 2, unequal counts refused, df unrounded, and the p-value printed
      beside `ℓ` (clauses 2 and 3);
    - tests extended to 19, with 20 mutants and both survivors now killed, kept in the repo (clause 7);
    - clause 9 narrowed;
    - clause 5's timing;
    - absence and equivalence clauses carved out (clause 1);
    - the deterministic pilot restored (clause 4 and the CLAUDE.md text);
    - clause 8 narrowed to DGF-1.
  - **Nits:**
    - the α wording now quotes ADR-006:77;
    - the excess explanation went with the common boundary;
    - the df range is 4–38;
    - §F's wording;
    - ADR-012:235 "allowed";
    - the *Next* wording;
    - α/k propagation;
    - shared-seed correlation, measured in `notebooks/measurements/2026-10-07-gate-stats-v2/output.txt` §H;
    - the docs list.
  - **Preamble:** test-window locations from ADR-012 and the timeline were added. ADR-007:156 was
    checked, and is the public whole-dataset ratio.
- **Round 2** (2026-10-08): a diff-only pass by a fresh subagent, whose prompt opened with the
  standing preamble. Draft 1 was never committed, so the reviewer checked each round-1 finding
  against draft 2 (`43b566f`) and attacked only new text.
  - **Tier A confirmed.** The blocking finding, SF1, SF2, SF3 and SF8 were judged fixed. SF4, SF5, SF6
    and SF7 were partly fixed.
  - **Nothing blocking.** Six should-fix items and four groups of nits.
  - **Main-session checks:** every citation was range-checked and found in range. Both CLAUDE.md
    test-window figures the reviewer reported are already printed in the signed Gate-1 files.
- **Draft 3** (2026-10-08), folded in under the stopping rule, with no further round:
  - **Should-fix:**
    - a "powered failure" also needs 0.80 power at the pre-registered δ with the realised standard
      deviations, and a refused clause is never one (clause 4);
    - clause 6 prints each design's own α basis, single looks included, and clause 1's "at most α"
      points to clause 3;
    - the absence and equivalence carve-out now sets per-look levels, two-sided tails for undirected
      differences, and leaves sizing, "powered" and the stage-2 trigger to the pre-registration at
      acceptance (clause 1);
    - the table is mechanical post-acceptance output and not an amendment; the planning factor and
      a paired test are fixed at acceptance; "before that gate's first test-window run" throughout
      (clauses 2, 4 and 6; closes SF5);
    - the CLAUDE.md wording now carries clause 9's exemptions, "on committed code", α = 0.02275,
      "never beyond the pre-registered stage 2" and the scope of clause 10 (closes SF4 and SF7);
    - the prototype's location is recorded as an exception to `CLAUDE.md:222-223` (clause 7).
  - **Nits:**
    - doc-00:165 is added to the amendments;
    - every replacement string carries its scope;
    - doc-02:174 keeps Gate 2's question;
    - the accuracy wording now separates the common boundary (§D) from the split (§G);
    - the pilot's per-seed correlation is printed where seeds are shared;
    - the citations of `CLAUDE.md:135-137` and `ADR-012:244` are corrected;
    - the SCM doc is dropped from clause 8's examples;
    - the prototype test comment now gives 3.600 (output unchanged, 20 of 20 killed);
    - "matched-time's test batch";
    - the trigger bound holds at the planning effects;
    - an "at least one" claim is powered by its largest clause.
  - **Preamble:** `CLAUDE.md:36-37` and `:100` were added, as repeats of the signed Gate-1 files.
- **Accepted 2026-10-08** by coderback at draft 3. The amendments under *Consequences* were applied
  the same day and committed to the governance repository. `docs/` was grepped for "3 seeds" before
  and after: the remaining hits are doc-02's v0.2 changelog and ELL-1 precedent note, the timeline's
  history lines, and the GDE Gate 0.5 line left to the consistency ADR.
