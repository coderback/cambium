# ADR-019 — Failure semantics and falsification

**Status:** proposed
**Date:** proposed 2026-10-08 · draft 3 2026-10-08
**Tier:** A. It fixes what gates answer, sets a refutation criterion and amends an accepted Tier-A
ADR (CLAUDE.md *Process*, ADR-017).
**Review rounds:** 2/4
**Deciders:** coderback
**Amends ADR-018, narrowly** (clause 7): clause 1's rule for absence claims, and clause 4.5's
forward sentence. **No seed rule changes**: sizing, the stage-2 trigger and the per-look levels stay
as ADR-018 has them. The restarted wait covers only the changed parts, by the researcher's reading
on 2026-10-08 (`notebooks/lab/2026-10-08.md`). ADR-018's shared function (plan row 2a) computes
neither change, so it may start 2026-10-09. **No verdict changes.**

**Docs affected:**
- `CLAUDE.md`: DGF-1's qualifications (one bullet) and *Next* item 3;
- `decisions/ADR-018-gate-statistics-v2.md:90-91` and `:182-183`, each with an "*Amended … was …*"
  note;
- `docs/research-plan-UNIFIED-GBE-GDE.md:15`, `:79`, `:82` and `:83`;
- `docs/00-shared-core-graph-embedding-GUIDE.md:183` and `:197`;
- `docs/01-elliptic-embedding-model-BUILD.md`: a note under `:6`, plus `:23` and `:88`;
- `docs/02-dgraph-fin-embedding-model-BUILD.md`: `:24` (its failure-semantics sentence only) and
  `:263`;
- `docs/03-edgar-risk-embedding-model-BUILD.md:199`: its failure-semantics sentence only. The
  researcher decided on 2026-10-08 to amend it here.

**Not amended:**
- the gate files and their errata, and ADR-004, ADR-006 and ADR-012;
- ADR-018's other clauses;
- the timeline's history lines (`docs/timeline.md:140-142`);
- `docs/research-plan-UNIFIED-GBE-GDE.md:67` ("DGF-1 (scale gate)" already reads as an engine gate)
  and `:84` (GDE's own failure semantics);
- `docs/document-amendments-v0.2.md:91`, a historical changelog;
- the following, which go to other ADRs:
  - `docs/03-edgar-risk-embedding-model-BUILD.md:8`, EDR-1's start condition (plan row 4);
  - `docs/04-edgar-hiddenlink-embedding-model-BUILD.md:10`, its H1 line (the consistency ADR).

**Evidence:**
- `notebooks/measurements/2026-10-08-margin-tests/measure_margin_tests.py` and `output.txt`
  (label-free);
- ADR-018's measurement, `notebooks/measurements/2026-10-07-gate-stats-v2/output.txt`.

## Context

The audit raised two findings (`notebooks/audit/2026-10-07/D-governing-docs.md:45`, `:49`).

**D1, blocking.** The pre-registered failure semantics tell the write-up to carry, into P0/P1, a
sentence saying that structure did not help on homogeneous, text-free graphs
(`docs/research-plan-UNIFIED-GBE-GDE.md:82`; `docs/01-elliptic-embedding-model-BUILD.md:23` has a
close variant).
- CLAUDE.md forbids that sentence (`CLAUDE.md:124`).
- ELL-1's Gate 3 makes it false: all three ablation clauses resolved (`CLAUDE.md:115-116`;
  `gates/GATE-ELL1-3.md`).

**D5. The thesis has no refutation condition.**
- **The GBE half's wording fits DGF-1:** "structure-aware embeddings beat text/tabular floors on
  temporally held-out financial ground truth" (`docs/research-plan-UNIFIED-GBE-GDE.md:15`). Doc-02's
  Gate 1 also asks "Structure beats features at scale?"
  (`docs/02-dgraph-fin-embedding-model-BUILD.md:263`).
- **Yet ELL-1's and DGF-1's Gate-1 failures are exempt in advance** (`:82`).
- **GDE proceeds even after EDR-1 fails** (`:83`).

So a pass could count for the thesis, a failure never could, and no outcome answers the GBE half "no".

**Decisions.**
- The researcher decided both parts on 2026-10-07 (`notebooks/lab/2026-10-07.md:14`).
- On 2026-10-08, after measuring, the researcher decided (`notebooks/lab/2026-10-08.md`):
  - a margin test for "no";
  - doc-03 amended here;
  - a narrow amendment of ADR-018, with the wait read as covering only the changed parts;
  - a real but sub-margin gain counts as below the bar;
  - a design that cannot answer "no" gets no free pass.

## Decision

### Clause 1 — ELL-1's and DGF-1's Gate 1 are engine gates, in both directions

**What an engine gate tests:** how a GNN on the shared core performs against its floor on a
homogeneous, text-free graph. It neither validates nor indicts the core itself: Gate 0 and ADR-008
do that, and ELL-1's protocol was exonerated (`gates/GATE-ELL1-3.md:166-167`). The regimes the
thesis rests on, node text and typed edges, are absent (`docs/research-plan-UNIFIED-GBE-GDE.md:82`).
**So neither a pass nor a failure of these Gate 1s answers the GBE half.**

**This narrows claims; it changes no verdict.**
- The verdicts stand: ELL-1 Gate 1 FAILED and Gate 3 PASSED; DGF-1 Gate 1 PASSED, always with
  CLAUDE.md's qualifications.
- The failure direction was pre-registered (`docs/01-elliptic-embedding-model-BUILD.md:23`,
  `docs/02-dgraph-fin-embedding-model-BUILD.md:24`).
- The pass direction is new. Adopting it after DGF-1's pass can only remove claims. `CLAUDE.md:107-108`
  already forbids "structure beats features at scale" from that gate.

**The GBE half is answered only by EDR-1** (clause 3). No ELL-1, DGF-1 or EDL-1 result answers it,
and that includes ELL-1's Gate 3 and DGF-1's future Gate 3. Their structure results are evidence
about the homogeneous, text-free regime, reported in P0. `docs/00-shared-core-graph-embedding-GUIDE.md:183`
("the single cleanest test of the whole thesis") is qualified to match.

### Clause 2 — How a Gate-1 failure is written up

- **Never "structure did not help."** A Gate-1 failure is reported decomposed, beside that model's
  Gate 3, with both parts together: whether structure contributed (Gate 3) and whether it was enough
  (Gate 1).
- **For ELL-1 that is CLAUDE.md's two-part statement** (`CLAUDE.md:119-124`):
  - structure demonstrably contributes (`gates/GATE-ELL1-3.md`);
  - it is demonstrably not enough (`gates/GATE-ELL1-1.md`);
  - the P0 lesson: "trees beat neural nets on this tabular data; the graph helps but not by enough".
- **Gate 3 runs after a Gate-1 failure for ELL-1, DGF-1 and EDR-1,** the models whose Gate 3 is a
  structure ablation. Elsewhere Gate 3 is not an ablation: SCM-1's is task concentration
  (`docs/structural-code-security-model-BUILD.md:125`), EDL-1's is cross-sector links
  (`docs/04-edgar-hiddenlink-embedding-model-BUILD.md:101`), and RDM-1's is chains
  (`docs/research-discovery-model-BUILD.md:119`). Those models keep their own failure semantics.

### Clause 3 — The refutation condition: EDR-1's margin tests

**Margins.** Each margin is a positive number, the smallest gain worth having:
- `m₁`, for each metric of EDR-1's Gate 1: graph+text over the matched text-only model;
- `m₃`, for each metric of Gate 3's edge-scramble clause: the real graph over the scrambled one.

The two are argued together, on substance: the smallest gain that would change a decision or carry
P1's claim. They are fixed in EDR-1's Phase-0 ADRs before any model metric on any EDR-1 split
exists, baselines included. Any EDR-1 data statistic they use comes from the training window, through
`gbe.run`.

**Two one-sided Welch tests per metric,** each at ADR-018's per-look level `ℓ`:
- **superiority,** gain > 0, at each look (ADR-018 clause 2);
- **non-superiority,** gain < `m`, at the last look that runs. This is ADR-018 clause 2 with the arms
  swapped and the comparator shifted up by `m`. Both standard deviations, the Welch df and the
  refusals are unchanged.

**The non-superiority tests are *answer clauses*, not gated clauses.**
- **What they decide:** the GBE answer, never a gate verdict.
- **What they don't do:** they don't set stage-1 `n`, and they don't trigger stage 2. ADR-018 clauses
  3 and 4 stay as written. Clause 4 below may raise the cap.

Measured (`notebooks/measurements/2026-10-08-margin-tests/output.txt`):
- **Gating them adds nothing.** "No" requires superiority to be unresolved, and an unresolved
  superiority clause already sends the run to stage 2. So the non-superiority test is judged at the
  cap whenever "no" is possible. With stage 1 at 8 seeds and a true gain of 0, "no" came out 0.796
  as an answer clause and 0.798 when gated.
- **Gating them costs a lot.** Every run then went to 20 seeds, even at twice the margin. The
  gate's own power at a true gain of `m` fell from 0.81 to 0.80.

**Each metric falls into one of four categories:**
- **above:** superiority resolves; non-superiority does not;
- **small:** both resolve, so the gain is real but below `m`;
- **below:** non-superiority resolves; superiority does not;
- **unresolved:** neither resolves.

**Each gate is one of three:**
- **above,** if every metric is above;
- **below the bar,** if every metric is small or below. The write-up says when a gain was real but
  small;
- **unresolved,** otherwise.

**The GBE half's answer, for corporate graph-text on this test window:**

| EDR-1 Gate 1 | Gate 3, edge-scramble clause | answer |
|---|---|---|
| above | above | **yes** |
| above | below the bar | **not supported:** the gain is not attributable to structure |
| below the bar | above | **two-part:** structure contributes but is not enough (the ELL-1 pattern) |
| below the bar | below the bar | **no** |
| any other combination | | **not answered** |

**Further rules.**
- **The verdict and the answer are separate.** Gate 1's verdict line keeps its own criterion
  (PASSED/FAILED). The gate file adds "GBE answer (ADR-019): <answer>". A real but small gain can
  therefore show Gate 1 PASSED with the answer "two-part" or "no".
- **"Yes" shows a gain above zero that was not shown below the bar. It does not show a gain of at least
  `m`.** "Small" needs a measurement precise enough to resolve both tests at once, which the planned
  design almost never gives. Measured (`notebooks/measurements/2026-10-08-margin-tests/output.txt`, section 2): at 20 seeds, a true gain of half the bar reads
  "above" 0.330 of the time and "small" 0.002. So P1's "yes" claims a gain above zero and reports its
  estimate against `m₁` and `m₃`. A rule requiring the gain shown above `m` was rejected (*Alternatives*).
- **The estimand is ADR-018's:** seed variance on one fixed test window and snapshot. A "no" claims
  nothing beyond them. The Phase-0 ADRs decide whether "no" also needs the user bootstrap and the
  rolling-origin cutoffs that ADR-018 left to EDR-1 (clause 6).
- **False "no".** When the true gain equals `m`, a false "no" happens at most α of the time per metric
  (ADR-018 clause 3; Welch is approximate where both arms vary, ADR-018:155). Measured: about 0.011
  with two looks and 0.022 with a single look of 20 seeds
  (`notebooks/measurements/2026-10-08-margin-tests/output.txt`).

### Clause 4 — Power for "no", and when it cannot be had

- **Each "no" test's planned power** (every metric of Gate 1 and of Gate 3's edge-scramble clause) at
  a true gain of 0 is ADR-018 clause 4's formula at `δ = m`, evaluated at the cap, because that is
  where the test is judged.
- **Strength is judged jointly.** "No" needs every one of these tests to resolve, so the design's
  planned chance of "no" when every true gain is 0 is at least `1 − Σ(1 − pᵢ)`, over all of them. The
  design is *able to answer "no"* iff that bound is at least 0.80. With two tests each needs about
  0.90; with four, about 0.95.
- **Why jointly.** Measured (`notebooks/measurements/2026-10-08-margin-tests/output.txt`, section 3): with each test at 0.80 and both gates' true gains 0,
  the answer was "no" 0.633 of the time and "not answered" 0.331. A per-test line would let a false
  thesis escape one time in three with no justification.
- **The table.** EDR-1's assembler prints these powers and the joint bound in its own answer-clause
  table, using ADR-018's shared power function unchanged. ADR-018's table and function are not
  changed.
- **If the bound falls below 0.80,** the pre-registration raises the cap under ADR-018 clause 5,
  renting compute if needed (research plan :137), until it reaches 0.80.
- **No free pass.** If no feasible cap gets there, the pre-registration declares, before EDR-1's first
  test-window run, that the design is unable to answer "no". Then, if the answer is "not answered"
  and Gate 1 is not above, the case where a "no" can hide:
  - P1 states that the GBE half went untested at planned power;
  - GDE's start needs the same written re-justification as after "no" (clause 5).

  Other answers from such a design stand, with its planned chance of "no" disclosed: a "yes",
  "two-part" or "not supported" is protected against false claims whatever the power. Measured
  (`notebooks/measurements/2026-10-08-margin-tests/output.txt` section 3, a design at 0.50): when nothing works, this rule requires the justification 0.971 of
  the time. Where Gate 1 already showed a gain, it requires it 0.011 of the time.

### Clause 5 — What follows each answer

- **yes:** P1 claims a gain above zero over the matched text-only model, attributable to structure,
  and reports its estimates against `m₁` and `m₃`. Gate 3's other clauses (encoder swap, text plus
  serialized graph) license only their own claims, as they pass.
- **not supported:** P1 reports the gain and does not attribute it to structure.
- **two-part:** P1 becomes a negative-result paper on the P0 benchmark, written as clause 2 requires.
- **no:** P1 becomes a negative-result paper on the P0 benchmark, worded as: "On corporate graph-text,
  EDR-1's gain over the matched text-only model was shown below `m₁`, and edge-scrambling cost less
  than `m₃`, on this test window against seed variance." It is never generalised to "structure does
  not help".
- **not answered:** reported as inconclusive at planned power, never as negative.
- **After any answer but yes,** P3's ceiling is recalibrated in writing.
- **GDE's start.** After an EDR-1 "no", or after "not answered" with Gate 1 not above from a design declared unable to answer "no" (ADR-019 clause 4), SCM-1 starts only after a written re-justification accepted as a Tier-A ADR. The same sentence goes into every amended text below. The research plan's H1
  line (`:79`) and doc-00's H1 row (`:197`) are qualified to match; doc-04's H1 line goes to the
  consistency ADR. After any other answer, GDE proceeds, because code graphs are a different
  regime.

### Clause 6 — Obligations on EDR-1's Phase-0 ADRs

- **The margins** `m₁` and `m₃` and their joint argument, under clause 3's timing and inputs.
- **How power will be shown** without test-window label statistics: ADR-018's validation pilot, plus
  whatever the test-window ledger and "peek" ADR (plan row 5) permits.
- **Whether "no" also needs** the hierarchical user × seed bootstrap and at least two rolling-origin
  cutoffs (ADR-018 clause 10), and how they combine with clause 3.

### Clause 7 — The narrow amendment of ADR-018

**Clause 1's bullet at `ADR-018:90-91` becomes:**
> **A one-sided absence claim** — the gain is below a pre-registered margin `m` — needs one one-sided
> test: clause 2 with the arms swapped and the comparator shifted up by `m`, at the look's `ℓ`. **A
> two-sided absence or equivalence claim** — the gain lies within `±m` — needs two one-sided tests,
> each at the look's `ℓ`. Never at α at both looks.

The two-test rule was mis-scoped: right for a two-sided equivalence claim, wrong for a one-sided one.
Measured: with stage 1 at 8 seeds and a true gain
of −2m, two tests return "no" 0.000 of the time and one test 1.000
(`notebooks/measurements/2026-10-08-margin-tests/output.txt`). So the old rule would block "no"
exactly where structure fails most clearly. The same question will come up for DGF-1's Gate 3
(plan row 8).

**Clause 4.5's sentence at `ADR-018:182-183`** ("The failure-semantics ADR (plan row 3) uses this
definition.") **becomes:** "ADR-019 answers the GBE half by margin tests instead. 'Powered failure'
still governs how gated clauses report a failure."

**How it is applied.** Both sentences are amended in place in ADR-018, each with a dated "*Amended
(ADR-019), was …*" note, so a reader of ADR-018 sees the change.

**The wait.** By the researcher's reading, the 12-hour wait restarts only for what the amendment
changes. This reading applies to ADR-019's amendment only and does not amend CLAUDE.md. No
pre-registration may rely on the one-sided rule until 12 hours after ADR-019's
acceptance, and not on that calendar day. ADR-018's shared function (row 2a) computes neither change
and may start 2026-10-09. EDR-1's assembler, when it is built, tests the shift: `diff − m` checked
analytically, plus a hand anchor with unequal standard deviations.

## Alternatives rejected

- **Gate the margin test, changing ADR-018's seed rules.** Measured (`notebooks/measurements/2026-10-08-margin-tests/output.txt`): no gain in "no"
  power, every run goes to 20 seeds, and the gate's own power drops (clause 3).
- **No amendment, calling "below m" a directional claim.** That would leave ADR-018's wording
  contradicting this ADR. The round-1 review rejected it.
- **Two one-sided tests for "no".** Measured (`notebooks/measurements/2026-10-08-margin-tests/output.txt`): they make "no" unreachable when
  graph+text is far worse (clause 7).
- **Powered failure at `m`** (the 2026-10-07 wording). It allows a false "no" up to 0.20 at a true
  gain of `m`, and ADR-018 clause 1 says a non-resolution never establishes absence.
- **Powered failure at ½·Δ_val.** "No" would be unreachable when the validation gap is ≤ 0, which is
  when the thesis is most likely false.
- **A sixth answer, "small".** The researcher chose to count a real gain below the margin as below
  the bar (2026-10-08).
- **"Yes" only when the gain is shown above `m`.** It would be a stronger claim, but at 20 seeds a true
  gain of exactly `m` gives it 0.023 of the time, 1.5m 0.334 and 2m 0.873 (`notebooks/measurements/2026-10-08-margin-tests/output.txt`, section 2). The
  researcher kept "above zero, not shown below the bar" (2026-10-08).
- **Disclose-only for a design that cannot answer "no".** The researcher chose "no free pass"
  (2026-10-08).
- **For a design unable to answer "no", a justification after any "not answered", or after any
  answer but "yes".** Measured (`notebooks/measurements/2026-10-08-margin-tests/output.txt`, section 3, a design at 0.50): where Gate 1 showed a gain but
  structure did not, these rules require a justification 0.488 and 0.981 of the time, although a
  "no" was impossible. The researcher chose clause 4's rule: only where a "no" can hide (2026-10-08).
- **Judging strength per test.** With each test at 0.80, a false thesis escapes as "not answered"
  0.331 of the time (section 3). The researcher chose the joint judgement (2026-10-08).
- **Relabel every ELL-1 and DGF-1 gate.** The decision covers Gate 1. Clause 1's "answered only by
  EDR-1" already keeps their Gate 3s from answering the GBE half.
- **No refutation condition (the status quo).** The GBE half would be unfalsifiable.

## Consequences

**On acceptance, these amendments are applied and committed to the governance repository.** Each
in-place rewrite carries a dated "*Amended (ADR-019)*" note; where the old text quoted the sentence
CLAUDE.md forbids, the note says so without repeating it.

- **`docs/research-plan-UNIFIED-GBE-GDE.md:15`** gains: "Answered by EDR-1's margin tests: Gate 1
  against the matched text-only floor, Gate 3 against edge-scramble. The tabular floor is reported,
  not gated. No ELL-1, DGF-1 or EDL-1 result answers it (ADR-019)."
- **`:79`** gains clause 5's sentence: "After an EDR-1 "no", or after "not answered" with Gate 1 not above from a design declared unable to answer "no" (ADR-019 clause 4), SCM-1 starts only after a written re-justification accepted as a Tier-A ADR."
- **`:82`** becomes:
  > ELL-1/DGF-1 Gate 1 is an **engine gate in both directions** (ADR-019): a failure is not a thesis
  > refutation and a pass is not thesis evidence. They lack text and typed edges — the regimes the
  > thesis actually rests on. A Gate-1 failure is written into P0/P1 decomposed, beside its Gate 3,
  > never as "structure did not help": for ELL-1, structure demonstrably contributes (GATE-ELL1-3)
  > and demonstrably is not enough (GATE-ELL1-1) — trees beat neural nets on that tabular data; the
  > graph helps but not by enough. EDR-1's start condition: plan row 4's ADR.
- **`:83`** becomes:
  > EDR-1 Gates 1 and 3 answer the GBE half by pre-registered margin tests (ADR-019):
  > - **yes:** P1 claims a gain above zero, attributable to structure, with estimates against the
  >   margins;
  > - **not supported:** P1 reports the gain without attributing it to structure;
  > - **two-part:** P1 is a negative-result paper on the P0 benchmark, decomposed;
  > - **no:** P1 is a negative-result paper on the P0 benchmark;
  > - **not answered:** reported as inconclusive, never as negative.
  >
  > After an EDR-1 "no", or after "not answered" with Gate 1 not above from a design declared unable to answer "no" (ADR-019 clause 4), SCM-1 starts only after a written re-justification accepted as a Tier-A ADR. After any answer but yes, P3's ceiling is recalibrated in writing.
- **`docs/00-shared-core-graph-embedding-GUIDE.md`:**
  - `:183` gains: "— for the thesis where it is tested (EDR-1); in ELL-1 and DGF-1 it is evidence
    about the engine regime (ADR-019)";
  - `:197`'s H1 row gains clause 5's sentence: "After an EDR-1 "no", or after "not answered" with Gate 1 not above from a design declared unable to answer "no" (ADR-019 clause 4), SCM-1 starts only after a written re-justification accepted as a Tier-A ADR."
- **`docs/01-elliptic-embedding-model-BUILD.md`:**
  - a dated note under `:6`: "ELL-1's Gate 1 is an engine gate in both directions; no ELL-1 result
    answers the GBE half (ADR-019).";
  - `:23`, from "`EDR-1` proceeds regardless" to the end of the line, becomes: "The decomposed
    result goes into the P0/P1 write-up (ADR-019): structure demonstrably contributes (GATE-ELL1-3)
    and demonstrably is not enough (GATE-ELL1-1) — trees beat neural nets on this tabular data; the
    graph helps but not by enough. EDR-1's start condition: plan row 4's ADR.";
  - `:88`'s "structure is not earning its keep — stop and diagnose" gains: "— reported decomposed
    beside Gate 3, never as structure not helping (ADR-019)".
- **`docs/02-dgraph-fin-embedding-model-BUILD.md`:**
  - `:24`'s first sentence becomes "as with `ELL-1`, Gate 1 is an engine gate in both directions
    (ADR-019): a failure is not a thesis refutation and a pass is not thesis evidence." The
    Expectation-setting sentence and the notes below it stay;
  - `:263`'s question becomes "Does DGF-1 beat the parity floor at scale? (engine gate, ADR-019; the
    earlier wording claimed more than an engine gate can answer)".
- **`docs/03-edgar-risk-embedding-model-BUILD.md:199`'s failure-semantics sentence** becomes:
  > **Failure semantics (pre-registered, ADR-019):**
  > - Gates 1 and 3 carry margins fixed in Phase 0, before any EDR-1 model metric exists.
  > - Gate 3 runs even if Gate 1 fails.
  > - Together they answer the GBE half: yes, not supported, two-part, no, or not answered
  >   (unified plan, *Failure semantics*).
  > - After an EDR-1 "no", or after "not answered" with Gate 1 not above from a design declared unable to answer "no" (ADR-019 clause 4), SCM-1 starts only after a written re-justification accepted as a Tier-A ADR.

  Its "≥3 seeds" sentence stays with the consistency ADR.
- **`CLAUDE.md`:**
  - DGF-1's qualifications gain "**An engine gate, in both directions** (ADR-019): it does not answer
    the GBE half; only EDR-1's margin tests do.";
  - *Next* item 3 becomes "accepted (ADR-019)".
- **ADR-018:** the two in-place amendments of clause 7.
- **Grep** `docs/` and `CLAUDE.md` before and after for "did not help", "beats features at scale",
  "earning its keep", "thesis refutation" and "proceeds regardless". It passes when every remaining
  hit is a history line, a provenance note, or a "never" context.
- **Nothing is implemented in code now.**

## Revisit when

- **EDR-1's Phase-0 ADRs** fix the margins and the bootstrap question (clause 6).
- **A GDE model reaches its own failure semantics.** The SCM-1 fusion line is unchanged here.

## Draft history

- **Draft 1** (2026-10-08, `1e3f08b`).
- **Round 1** (2026-10-08): a full adversarial review by a fresh subagent, whose prompt opened with
  the standing preamble.
  - **Tier A confirmed.**
  - **Five blocking findings:**
    - it silently amended ADR-018;
    - a gain shown below the margin could still be "yes";
    - Gate 3's outcome was undefined;
    - clause 4 and the amendment texts contradicted ADR-018 and each other;
    - "Gate 3 runs after a failure, for every model" was wrong for most models.
  - **Ten should-fix findings and nine nits.** The main session range-checked every citation and
    found them in range.
- **The researcher's decisions, 2026-10-08,** after the margin-test measurement: the narrow
  amendment, the wait read narrowly, "below the bar", and "no free pass" (`notebooks/lab/2026-10-08.md`).
- **Draft 2** (2026-10-08):
  - **Blocking:** B1 by clause 7 and answer clauses; B2 by the four categories and "below the bar";
    B3 by using only the edge-scramble clause, with the other Gate-3 clauses licensing only their
    own claims; B4 by clause 5's per-answer list and "not answered" never being negative; B5 by
    scoping clause 2 to ELL-1, DGF-1 and EDR-1.
  - **Should-fix:**
    - S1: clause 1's scope, the core neither validated nor indicted, and doc-00:183;
    - S2: margins fixed in Phase 0 from label-free inputs;
    - S3: "EDR-1 proceeds" dropped, pointing to row 4;
    - S4: the H1 lines and a Tier-A re-justification;
    - S5: provenance notes, and doc-02:24's other sentence kept;
    - S6: the estimand and clause 6;
    - S7: clause 4;
    - S8: the margin-test measurement;
    - S9: doc-01:88, and research plan :15's tabular floor;
    - S10: the verdict and the answer as separate lines.
  - **Nits:**
    - D1's quote described rather than repeated;
    - ADR-018:155's qualifier;
    - the row-2a test dropped;
    - doc-01:23 keeps the P0 lesson;
    - :67 and the v0.2 changelog listed as not amended;
    - EDL-1 named;
    - sanctioned "no" wording;
    - the 2026-10-08 decisions cited.
  - **Preamble:** `docs/timeline.md:13-14` added. `docs/02-dgraph-fin-embedding-model-BUILD.md:250`
    was checked; it is the public whole-dataset ratio.
- **Round 2** (2026-10-08): a diff-only pass by a fresh subagent, with the standing preamble, over
  `1e3f08b..9a00135`.
  - **Tier A confirmed.** Of round 1's findings, 15 were judged fixed and 5 partly fixed.
  - **Three blocking findings, all in new text:**
    - "more seeds can never turn a sub-margin gain into yes" was false, and "small" almost never
      occurs;
    - the answer-clause power rows were placed in ADR-018's table, another amendment;
    - the scope of "untested" differed across six texts.
  - **Two should-fix findings** (margin-timing inputs; doc-01:23's "proceeds regardless") and ten
    nits. Every quoted number was confirmed in the output.
- **The researcher's decisions, 2026-10-08,** after measuring sections 2 and 3: "yes" means a gain
  above zero, not shown below the bar; the justification is required only where a "no" can hide;
  strength is judged jointly (`notebooks/lab/2026-10-08.md`).
- **Draft 3** (2026-10-08):
  - **Blocking:** what "yes" shows, stated with section 2's numbers (clause 3); the answer-clause
    table moved to EDR-1's assembler (clause 4); one sentence on GDE's start, used in every text
    (clauses 4 and 5, and the Consequences).
  - **Should-fix:** margins fixed before any model metric, with data statistics only from the
    training window through `gbe.run`; doc-01:23 replaced through the end of the line.
  - **Nits:**
    - "don't set stage-1 n";
    - each look vs the last look;
    - "gated clauses";
    - `CLAUDE.md:115-116`;
    - doc-03's note moved out of the quote;
    - the doc-02:263 note no longer repeats the forbidden phrase;
    - "mis-scoped";
    - the wait reading scoped to this ADR;
    - :15 names both margins;
    - doc-04's H1 line deferred;
    - the grep has a pass criterion.
