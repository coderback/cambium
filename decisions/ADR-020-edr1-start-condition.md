# ADR-020 — EDR-1's start condition

**Status:** proposed
**Date:** proposed 2026-10-08
**Tier:** B. It sets programme sequencing, touches no held-out set and sets no gate criterion
(CLAUDE.md *Process*, ADR-017). One review, at most one diff pass.
**Deciders:** coderback
**Docs affected:**
- `docs/03-edgar-risk-embedding-model-BUILD.md:8`;
- the "plan row 4" pointers that ADR-019 left at `docs/research-plan-UNIFIED-GBE-GDE.md:82`,
  `docs/01-elliptic-embedding-model-BUILD.md:24` and `docs/02-dgraph-fin-embedding-model-BUILD.md:24`;
- `CLAUDE.md`, *Next* item 5.

## Context

**doc-03's condition can never be met.** It says "Do not start it until `ELL-1` and `DGF-1` have
cleared their gates" (`docs/03-edgar-risk-embedding-model-BUILD.md:8`). But ELL-1's Gate 1 FAILED,
and DGF-1's Gates 2 and 3 are open. The pre-registered failure semantics said "EDR-1 proceeds
regardless" (audit B2, `notebooks/audit/2026-10-07/B-route-review.md:50`). ADR-008 fixed the same
kind of defect for doc-02.

**The researcher decided the condition on 2026-10-07** (`notebooks/lab/2026-10-07.md:15`): EDR-1
starts once ELL-1 is closed and DGF-1 Gate 3 is decided either way, with matched-time resolved.
- **The timing:** the condition is fixed before the next DGF-1 batch, the matched-time validation
  pilot included (`CLAUDE.md:72-73`).
- **The reason:** DGF-1's Gate 3 settles model class, "the DGF-1 result that EDR-1's floor design
  and P0 depend on" (`notebooks/audit/2026-10-07/README.md:119`).

## Decision

**EDR-1 starts** when its first Phase-0 work product is begun: a Phase-0 ADR draft, EDGAR data
acquisition, or EDR-1 code. Reading and literature work are not a start.

**It may start only when all of these hold:**
1. **ELL-1 is closed.** This already holds (`CLAUDE.md:112`): Gate 0 PASSED, Gate 1 FAILED, Gate 3
   PASSED, Gate 2 deferred and not required.
2. **DGF-1's Gate 3 is decided, either way.** `gates/GATE-DGF1-3.md` is assembled under its own
   pre-registration (plan row 8), and the researcher has signed its verdict: PASSED, FAILED or
   partial. A FAILED Gate 3 does not block EDR-1.
3. **Matched-time is resolved.** Its stage-1 go/no-go is recorded. Then either, on a go, its stage-2
   test batch is reported (plan row 7), or it is declined with a dated reason (plan row 6d).
4. **DGF-1's exit decision is written and dated** (plan row 10).
5. **The consistency ADR is accepted** (audit D15; decided 2026-10-07, plan row 12).

**Not required:**
- any gate *passing*. ELL-1's and DGF-1's Gate 1 are engine gates (ADR-019 clause 1);
- DGF-1's Gate 2, which is deferred and needs its own pre-registration (ADR-018 clause 8).

**No silent waiver.** If DGF-1's Gate 3 or matched-time cannot be brought to a decision, this ADR is
amended with the reason before EDR-1 starts. A missed date is not a waiver.

**Conditions 2–5 are checked once,** in a dated lab line that cites each one's file or commit,
before EDR-1's first Phase-0 ADR is drafted.

## Alternatives rejected

- **"Cleared their gates" (doc-03 as written).** It can never hold, because ELL-1's Gate 1 failed.
- **"EDR-1 proceeds regardless."** EDR-1 would start before DGF-1's model-class answer, which its
  floor design needs.
- **Wait for DGF-1's Gate 2.** Gate 2 has no metric and is deferred (ADR-018 clause 8). Waiting for
  it would stall EDR-1 indefinitely.

## Consequences

**On acceptance, these amendments are applied and committed to the governance repository:**
- `docs/03-edgar-risk-embedding-model-BUILD.md:8`'s last sentence becomes "Do not start it until
  ADR-020's start condition holds: ELL-1 closed, DGF-1 Gate 3 decided either way, matched-time
  resolved, DGF-1's exit decision written, and the consistency ADR accepted." It carries the note
  "*Amended (ADR-020); was 'until `ELL-1` and `DGF-1` have cleared their gates', which ELL-1's
  failed Gate 1 made unreachable.*";
- each "plan row 4's ADR" pointer becomes "ADR-020";
- *Next* item 5 becomes "accepted (ADR-020)".

The dated plan is unchanged: row 13, EDR-1's Phase-0 ADRs, already depends on rows 4, 10 and 12.

## Revisit when

- **DGF-1's Gate 3 or matched-time cannot be decided** (see *No silent waiver*).
