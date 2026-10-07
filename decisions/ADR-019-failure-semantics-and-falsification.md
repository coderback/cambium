# ADR-019 — Failure semantics and falsification

**Status:** proposed
**Date:** proposed 2026-10-08
**Tier:** A (it fixes what gates answer and sets a refutation criterion; CLAUDE.md *Process*, ADR-017)
**Review rounds:** 0/4
**Deciders:** coderback
**Uses:** ADR-018 for α, the per-look levels, the Welch criterion, power planning, and the rule that
absence claims need a margin test (ADR-018 clause 1). **No verdict changes.**

**Docs affected:**
- `CLAUDE.md`: DGF-1's qualifications block (one bullet) and *Next* item 3;
- `docs/research-plan-UNIFIED-GBE-GDE.md:15` (the GBE half) and `:82-83` (*Failure semantics*);
- `docs/01-elliptic-embedding-model-BUILD.md:6` (a dated note) and `:23`;
- `docs/02-dgraph-fin-embedding-model-BUILD.md:24` and `:263` (Gate 1's question);
- `docs/03-edgar-risk-embedding-model-BUILD.md:199`, its failure-semantics sentence only. The
  researcher decided on 2026-10-08 to amend it here, although EDR-1 is outside the current phase.

**Not amended:**
- the gate files and their errata;
- ADR-004, ADR-006 and ADR-012;
- the timeline's history lines (`docs/timeline.md:140-142`);
- GDE's own failure semantics (`docs/research-plan-UNIFIED-GBE-GDE.md:84`);
- "EDR-1 proceeds" wherever it appears. EDR-1's start condition is plan row 4's ADR.

## Context

The 2026-10-07 audit raised two findings (`notebooks/audit/2026-10-07/D-governing-docs.md:45`, `:49`).

**D1, blocking.** The pre-registered failure semantics tell the write-up to carry, into P0/P1, the
sentence "structure did not help on homogeneous, text-free graphs under strict inductive protocol"
(`docs/research-plan-UNIFIED-GBE-GDE.md:82`; `docs/01-elliptic-embedding-model-BUILD.md:23`).
- CLAUDE.md forbids it: "Never write "structure did not help"" (`CLAUDE.md:124`).
- ELL-1's Gate 3 makes it false: all three ablation clauses resolved, so the gain is structural
  (`gates/GATE-ELL1-3.md`).

**D5. The thesis has no refutation condition.**
- **The GBE half claims "structure-aware embeddings beat text/tabular floors on temporally held-out
  financial ground truth"** (`docs/research-plan-UNIFIED-GBE-GDE.md:15`). DGF-1 fits that wording,
  and doc-02's Gate 1 asks "Structure beats features at scale?" (`docs/02-dgraph-fin-embedding-model-BUILD.md:263`).
- **Yet a failure of ELL-1's or DGF-1's Gate 1 is exempt in advance** (`:82`).
- **GDE proceeds even after EDR-1 fails** (`:83`).

So a pass could count for the thesis, a failure never could, and no outcome anywhere answers the GBE
half "no".

**The researcher decided both parts on 2026-10-07** (`notebooks/lab/2026-10-07.md:14`). On 2026-10-08
the researcher chose a margin test for "no", in place of a powered failure, and decided to amend
doc-03 in this ADR.

## Decision

### Clause 1 — ELL-1 and DGF-1 are engine gates, in both directions

**What they test:** the shared core, the protocol, and how structure behaves on homogeneous,
text-free graphs. The regimes the thesis rests on, node text and typed edges, are absent
(`docs/research-plan-UNIFIED-GBE-GDE.md:82`). So **neither a pass nor a failure of any ELL-1 or DGF-1
gate answers the GBE half.** Their results go into P0 as findings about that regime.

**This narrows claims; it changes no verdict.**
- The verdicts stand: ELL-1 Gate 1 FAILED and Gate 3 PASSED; DGF-1 Gate 1 PASSED, always with
  CLAUDE.md's qualifications.
- The failure direction was pre-registered (`docs/01-elliptic-embedding-model-BUILD.md:23`,
  `docs/02-dgraph-fin-embedding-model-BUILD.md:24`).
- The pass direction is new. Adopting it after DGF-1's pass can only remove claims, so it cannot
  flatter a result. `CLAUDE.md:107-108` already forbids "structure beats features at scale" from
  that gate.

### Clause 2 — How a Gate-1 failure is written up

- **Never "structure did not help."** A Gate-1 failure is reported decomposed, beside that model's
  Gate 3, with both parts together: whether structure contributed (Gate 3) and whether it was enough
  (Gate 1).
- **For ELL-1 that is CLAUDE.md's two-part statement** (`CLAUDE.md:119-124`):
  - structure demonstrably contributes (`gates/GATE-ELL1-3.md`);
  - it is demonstrably not enough (`gates/GATE-ELL1-1.md`);
  - "trees beat neural nets on this tabular data; the graph helps but not by enough".
- **Gate 3 runs after a Gate-1 failure, for every model,** so the decomposition exists. ELL-1's Gate 3
  ran after its Gate 1 failed.

### Clause 3 — The refutation condition: EDR-1's margin tests

**EDR-1 tests the GBE half,** because it has text and typed edges.

**Margins.** EDR-1's Gate-1 and Gate-3 pre-registrations each fix, for every gated metric, the
smallest gain worth having, a positive number:
- `m₁` for graph+text over the matched text-only model (Gate 1);
- `m₃` for the real graph over the edge-scrambled one (Gate 3's structural clause).

Each margin is argued on substance, as the smallest gain that would change a decision or carry
P1's claim. It is fixed at that pre-registration's acceptance, before any EDR-1 validation pilot is
seen, and never derived from data.

**Two one-sided Welch tests per clause, at each look,** each at ADR-018's per-look level `ℓ`:
- **superiority,** gain > 0 (ADR-018 clause 2);
- **non-superiority,** gain < `m`. This is ADR-018 clause 2 with the arms swapped: it tests whether
  the comparator, shifted up by `m`, beats EDR-1's arm. The shift leaves both standard deviations
  and the Welch df unchanged, so ADR-018's measured error rates (`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt`, sections C and G) carry over
  unchanged.

**Each gate's outcome:**
- **pass:** the gate's own pre-registered criterion holds;
- **no:** on every gated clause, superiority is not resolved and non-superiority is;
- **inconclusive:** anything else.

**The claim is one-sided:** the gain is below `m`. A gain far below zero also answers it. So it uses one
of ADR-018 clause 1's two one-sided tests, and claims no equivalence.

**False "no".** When the true gain equals `m`, a false "no" occurs at most α of the time per clause,
by ADR-018 clause 3's bound. The rejected powered-failure rule allowed up to 0.20 there.

**Stage 2.** A clause that neither passes nor resolves below its margin counts as unresolved for
ADR-018's stage-2 trigger.

**Sizing.** Under ADR-018 clause 4 the margin test's planned power at a true gain of 0 equals clause
4's formula at `δ = m`, again by the shift.
- **Stage-1 `n`** is the smallest candidate at which both tests reach 0.80 planned power on every
  gated clause; otherwise it is 20.
- **If the margin test cannot reach 0.80 at 20,** the α and power table says "this design cannot
  answer 'no' at planned power" before EDR-1's first test-window run.

**The GBE half's answer:**

| EDR-1 Gate 1 | EDR-1 Gate 3 (structural clause) | the GBE half, for corporate graph-text |
|---|---|---|
| pass | pass | **yes** |
| pass | no | **not supported:** the gain is not attributable to structure |
| no | pass | **two-part:** structure contributes but is not enough (the ELL-1 pattern) |
| no | no | **no** |
| any other combination | | **not answered**, reported with the margins and planned power |

### Clause 4 — What follows each answer

- **yes:** P1 as planned.
- **not supported:** P1 reports the gain and may not attribute it to structure.
- **two-part, or not answered:** P1 becomes a negative-result paper on the P0 benchmark, still
  citable, written as clause 2 requires.
- **no:** P1 ships as a negative result on the P0 benchmark. **GDE's start (SCM-1 Phase 0) then needs
  a written re-justification, accepted as an ADR.**
- **Any answer but yes:** P3's ceiling is recalibrated in writing
  (`docs/research-plan-UNIFIED-GBE-GDE.md:83`, kept). Apart from the "no" case, GDE proceeds, because
  code graphs are a different regime.

### Clause 5 — Obligations on EDR-1's Phase-0 ADRs

- **The margins and their arguments.** They state `m₁` and `m₃` for every gated metric, with the
  argument for each, before any EDR-1 run.
- **How power will be shown** without test-window label statistics: ADR-018's validation pilot, plus
  whatever the test-window ledger and "peek" ADR (plan row 5) permits.
- **Which "powered" definition applies.** ADR-018 left bootstrap gating to EDR-1's Phase-0 ADRs. If
  they adopt it, clause 3's tests use their definition; ADR-018's is the minimum.

## Alternatives rejected

- **Powered failure at a margin `m`** (the 2026-10-07 wording, with ADR-018 clause 4.5's "powered
  failure").
  - It allows a false "no" up to 0.20 at a true gain of `m`.
  - It conflicts with ADR-018 clause 1, under which a non-resolution never establishes absence.
  - The researcher chose the margin test on 2026-10-08. It needs the same seeds, because its power at
    a true gain of 0 equals the powered-failure rule's power at `m`.
- **Powered failure at ½·Δ_val, ADR-018's default planning effect.** "No" would be unreachable exactly
  when the validation gap is ≤ 0, where ADR-018 clause 4 prints "powered: no". That is when the thesis
  is most likely false.
- **Keep ELL-1 and DGF-1 as thesis tests, in both directions.** ELL-1's Gate-1 failure would become
  partial refutation and DGF-1's pass partial support, both on graphs without text or typed edges.
  The pre-registered exemption already covers failures; making passes count while failures don't is
  the asymmetry D5 found.
- **A two-sided equivalence test for "no".** The claim is one-sided: a large negative gain also
  answers "no".
- **No refutation condition (the status quo).** The GBE half would be unfalsifiable.

## Consequences

**On acceptance, the amendments below are applied and committed to the governance repository.**
- **`docs/research-plan-UNIFIED-GBE-GDE.md:15`** gains, after "financial ground truth": ", tested
  where node text and typed edges exist (EDR-1); ELL-1 and DGF-1 are engine gates in both directions
  (ADR-019)".
- **`:82`** becomes:
  > ELL-1/DGF-1 gates are **engine gates in both directions** (ADR-019): a Gate-1 failure is not a
  > thesis refutation and a pass is not thesis evidence. They lack text and typed edges — the
  > regimes the thesis actually rests on. A Gate-1 failure is written into P0/P1 decomposed, beside
  > its Gate 3, never as "structure did not help": for ELL-1, structure demonstrably contributes
  > (GATE-ELL1-3) and demonstrably is not enough (GATE-ELL1-1). EDR-1 proceeds.
- **`:83`** becomes:
  > EDR-1 Gates 1 and 3 answer the GBE half by pre-registered margin tests (ADR-019): **yes** (both
  > pass), **not supported** (Gate 1 passes; Gate 3's structural gain is shown below its margin),
  > **two-part** (Gate 1 fails; Gate 3 passes), **no** (both shown below their margins), otherwise
  > **not answered**. After any answer but yes, P1 becomes a negative-result paper on the P0
  > benchmark — still citable — and P3's ceiling is recalibrated in writing. After **no**, GDE's
  > start (SCM-1 Phase 0) needs a written re-justification accepted as an ADR; otherwise GDE proceeds
  > (code graphs are a different regime).
- **`docs/01-elliptic-embedding-model-BUILD.md`:**
  - `:23`'s quoted sentence becomes "the decomposed result (ADR-019): structure demonstrably
    contributes (GATE-ELL1-3) and demonstrably is not enough (GATE-ELL1-1)";
  - a dated note under `:6`: "*Amended (ADR-019).* ELL-1's gates are engine gates in both
    directions: no ELL-1 result answers the GBE half."
- **`docs/02-dgraph-fin-embedding-model-BUILD.md`:**
  - `:24` becomes "as with `ELL-1`, Gate 1 is an engine gate in both directions (ADR-019): a failure
    is not a thesis refutation and a pass is not thesis evidence — `EDR-1` proceeds.";
  - `:263`'s question becomes "Does DGF-1 beat the parity floor at scale? (engine gate, ADR-019; was
    "Structure beats features at scale?")".
- **`docs/03-edgar-risk-embedding-model-BUILD.md:199`:** the failure-semantics sentence becomes:
  > **Failure semantics (pre-registered, ADR-019):** Gates 1 and 3 carry margins fixed before any
  > validation pilot is seen, and answer the GBE half yes / not supported / two-part / no / not
  > answered; Gate 3 runs even if Gate 1 fails. After any answer but yes, P1 ships as a
  > negative-result paper on the P0 benchmark — still citable; P0 makes it so. After no, GDE's start
  > needs a written re-justification accepted as an ADR; otherwise GDE proceeds (code graphs are a
  > different regime) with P3's ceiling recalibrated in writing.

  Its "≥3 seeds" sentence stays with the consistency ADR.
- **`CLAUDE.md`:**
  - DGF-1's qualifications gain "**An engine gate, in both directions** (ADR-019): it does not
    answer the GBE half; EDR-1's margin tests do.";
  - *Next* item 3 becomes "accepted (ADR-019)".
- **`docs/` is grepped** for "did not help" and "beats features at scale" before and after the
  amendments.
- **Nothing is implemented in code.** EDR-1's pre-registrations carry the margin tests, and the
  shared function from ADR-018 row 2a computes them by the shift. Row 2a's tests gain one case: the
  non-superiority test equals the superiority test on a comparator shifted by `m`.

## Revisit when

- **EDR-1's Phase-0 ADRs** fix the margins (clause 5).
- **A GDE model reaches its own failure semantics.** The SCM-1 fusion line
  (`docs/research-plan-UNIFIED-GBE-GDE.md:84`) is unchanged here.

## Draft history

- **Draft 1** (2026-10-08). There was nothing to measure first: the margin test is ADR-018's measured
  criterion (`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt`) applied to a comparator shifted by the margin.
