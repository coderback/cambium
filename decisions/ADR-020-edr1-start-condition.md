# ADR-020 — EDR-1's start condition

**Status:** proposed
**Date:** proposed 2026-10-08 · draft 2 2026-10-08
**Tier:** B. It sets programme sequencing, touches no held-out set and sets no gate criterion
(CLAUDE.md *Process*, ADR-017). One review (done), at most one diff pass.
**Deciders:** coderback
**Docs affected:**
- `docs/03-edgar-risk-embedding-model-BUILD.md:8`;
- the "plan row 4" pointers that ADR-019 left at `docs/research-plan-UNIFIED-GBE-GDE.md:82`,
  `docs/01-elliptic-embedding-model-BUILD.md:24` and `docs/02-dgraph-fin-embedding-model-BUILD.md:24`;
- `CLAUDE.md`, *Next* item 5;
- `docs/timeline.md`, row 4's "actual" column.

## Context

**doc-03's condition can never be met.** It says "Do not start it until `ELL-1` and `DGF-1` have
cleared their gates" (`docs/03-edgar-risk-embedding-model-BUILD.md:8`). But ELL-1's Gate 1 FAILED,
and DGF-1's Gates 2 and 3 are open. The pre-registered failure semantics said "EDR-1 proceeds
regardless" (audit B2, `notebooks/audit/2026-10-07/B-route-review.md:50`). ADR-008 fixed the same
kind of defect for doc-02.

**Where each condition comes from:**
- **Decided by the researcher on 2026-10-07** (`notebooks/lab/2026-10-07.md:15`; `CLAUDE.md:72-73`):
  - ELL-1 closed;
  - DGF-1 Gate 3 decided either way;
  - matched-time resolved;
  - the condition fixed before the next DGF-1 batch, the matched-time validation pilot included;
  - one consistency ADR before EDR-1 (`CLAUDE.md:88`);
  - the ledger and "peek" ADR before EDR-1's Phase-0 ADRs
    (`notebooks/audit/2026-10-07/README.md:90`).
- **From the dated plan:** DGF-1's exit decision. Row 13 depends on row 10
  (`docs/timeline.md:87`).
- **Proposed here, for the researcher to keep or strike:** the definition of "start" (the
  researcher chose it on 2026-10-08), and the waiver path.

**Why Gate 3 matters:** it settles model class, "the DGF-1 result that EDR-1's floor design and P0
depend on" (`notebooks/audit/2026-10-07/README.md:119`).

**Out of scope:** why ELL-1 was closed and why DGF-1 came before EDR-1 (audit N1). Those are history,
recorded in the ELL-1 gate files and the 2026-07 lab entries.

## Decision

**EDR-1 starts at the first of these:**
- accepting any EDR-1 Phase-0 ADR;
- opening EDGAR filings or EDR-1's label sources (AAER, restatement or delisting lists);
- writing EDR-1 code, including the EXTEND work that adds the text encoder and typed-edge GNN to
  `gbe/` (`docs/00-shared-core-graph-embedding-GUIDE.md:194`).

**Not a start:**
- drafting Phase-0 ADRs. Drafts may begin before the conditions hold, but none is accepted until they
  all do. A draft that depends on DGF-1's model-class answer, such as the floor design, is revised to
  match Gate 3's verdict before acceptance;
- reading papers or documentation;
- the consistency ADR, and the errata that amend doc-03.

**It may start only when all of these hold:**
1. **ELL-1 is closed.** This already holds (`CLAUDE.md:112`): Gate 0 PASSED, Gate 1 FAILED, Gate 3
   PASSED, Gate 2 deferred and not required.
2. **DGF-1's Gate 3 is decided.** `gates/GATE-DGF1-3.md` is assembled under its own pre-registration
   (plan row 8), and the researcher has signed a verdict under it: any verdict that pre-registration
   defines, inconclusive included. A failed Gate 3 does not block EDR-1.
3. **Matched-time is resolved.** One of these holds:
   - it is declined with a dated reason, at any point (plan row 6d included);
   - a stage-1 no-go is recorded (row 6c);
   - a go is recorded, and its stage-2 test batch is reported (row 7).
4. **DGF-1's exit decision is accepted** as a Tier-B ADR (plan row 10). For each remaining DGF-1 item
   (Phase 2 and Gate 2, ADR-015's positioning batch) it states whether that item is closed, deferred
   with its condition, or continuing alongside EDR-1.
5. **The consistency ADR is accepted** (audit D15, plan row 12).
6. **The test-window ledger and "peek" ADR is accepted** (plan row 5).

**Not required:**
- any gate *passing*. ELL-1's and DGF-1's Gate 1 are engine gates (ADR-019 clause 1);
- DGF-1's Gate 2, which is deferred and needs its own pre-registration (ADR-018 clause 8).

**No silent waiver.** If any condition cannot be met, an amendment to this ADR giving the reason is
drafted and reviewed before EDR-1 starts. It is a Tier-B draft with its own review, or Tier A if it
touches held-out access. A missed date is not a waiver.

**The check.** Before EDR-1 starts, a dated lab line cites each condition's file or commit. That line
goes through `scripts/check_evidence.py` like any document brought for a decision.

## Alternatives rejected

- **"Cleared their gates" (doc-03 as written).** It can never hold, because ELL-1's Gate 1 failed.
- **"EDR-1 proceeds regardless."** EDR-1 would start before DGF-1's model-class answer, which its
  floor design needs.
- **Wait for DGF-1's Gate 2.** Gate 2 has no metric and is deferred (ADR-018 clause 8).
- **Count a Phase-0 ADR *draft* as the start.** Row 13 asks for seven Tier-A ADRs accepted by
  2026-12-06, only about ten days after the consistency ADR (11-26). Waiting to begin drafting until
  then would move row 13 to about January. The researcher chose acceptance as the trigger
  (2026-10-08).

## Consequences

**On acceptance, these amendments are applied and committed to the governance repository:**
- **`docs/03-edgar-risk-embedding-model-BUILD.md:8`'s last sentence** becomes "Do not start it until
  ADR-020's start condition holds: ELL-1 closed, DGF-1 Gate 3 decided, matched-time resolved, and
  DGF-1's exit decision, the consistency ADR and the ledger ADR accepted." It carries the note
  "*Amended 2026-10-08 (ADR-020); was 'until `ELL-1` and `DGF-1` have cleared their gates', which
  ELL-1's failed Gate 1 made unreachable.*";
- **the "plan row 4" pointers** change: "plan row 4's ADR" becomes "ADR-020" in the research plan
  :82 and doc-01:24. In doc-02:24, where the pointer sits inside ADR-019's dated note, "(now
  ADR-020)" is appended instead;
- **CLAUDE.md's *Next* item 5** becomes: "**EDR-1's start condition: accepted 2026-10-08 (ADR-020).**
  EDR-1 starts at its first accepted Phase-0 ADR, EDGAR data or EDR-1 code, once ELL-1 is closed,
  DGF-1 Gate 3 is decided, matched-time is resolved, and DGF-1's exit decision, the consistency ADR
  and the ledger ADR are accepted;";
- **timeline row 4** gets its actual date.

The plan's dates are unchanged. Row 13 already depends on rows 4, 10 and 12. Its Phase-0 ADRs may
be drafted earlier and accepted from about 11-26.

## Revisit when

- **Any condition cannot be met** (see *No silent waiver*).

## Draft history

- **Draft 1** (2026-10-08, `de7a3b3`).
- **Review** (2026-10-08): the single Tier-B review, by a fresh subagent with the standing preamble.
  - **Tier B confirmed,** provided "reading" excludes EDGAR data and label sources.
  - **One blocking finding:** condition 3 could not be met on the decline paths.
  - **Six should-fix findings and four nits.** Every citation was range-checked and found in range.
- **The researcher's decision, 2026-10-08:** accepting a Phase-0 ADR, not drafting one, is the start.
- **Draft 2** (2026-10-08):
  - **Blocking:** condition 3 now has three ways to be resolved.
  - **Should-fix:**
    - each condition's source is stated;
    - "start" is keyed to the check, with "reading" narrowed and EXTEND named;
    - any signed Gate-3 verdict counts;
    - the exit decision is a Tier-B ADR with contents;
    - one reviewed waiver path for every condition;
    - exact amendment texts.
  - **Nits:**
    - the doc-02 pointer is appended to, not rewritten;
    - the ledger ADR is condition 6;
    - N1 is out of scope;
    - the check line goes through the evidence checker.
