# ADR-020 — EDR-1's start condition

**Status:** accepted
**Date:** proposed 2026-10-08 · **accepted 2026-10-08 (13:51) by coderback**, at draft 3. Tier B, so implemented
the same day.
**Tier:** B. It sets programme sequencing, touches no held-out set and sets no gate criterion
(CLAUDE.md *Process*, ADR-017). One review and one diff pass, both done. The diff pass's blocking
finding was escalated to the researcher, who accepted its fix on 2026-10-08 (*Draft history*).
**Deciders:** coderback
**Docs affected:**
- `docs/03-edgar-risk-embedding-model-BUILD.md:8`;
- the "plan row 4" pointers that ADR-019 left at `docs/research-plan-UNIFIED-GBE-GDE.md:82`,
  `docs/01-elliptic-embedding-model-BUILD.md:24` and `docs/02-dgraph-fin-embedding-model-BUILD.md:24`;
- `CLAUDE.md`, *Next* item 5;
- `docs/timeline.md`: row 4's "actual" column, and row 10's wording.

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
  - the ledger and "peek" ADR before EDR-1's Phase-0 ADRs. It was recommended at
    `notebooks/audit/2026-10-07/README.md:90` and adopted with §4a (`notebooks/lab/2026-10-07.md:14`;
    `CLAUDE.md:56-58`).
- **From the dated plan:** DGF-1's exit decision. Row 13 depends on row 10
  (`docs/timeline.md:87`).
- **Decided by the researcher on 2026-10-08** (`notebooks/lab/2026-10-08.md`): the definition of
  "start", with accepting a Phase-0 ADR as the trigger rather than drafting one.
- **Proposed here, for the researcher to keep or strike:** the waiver path.

**Why Gate 3 matters:** it settles model class, "the DGF-1 result that EDR-1's floor design and P0
depend on" (`notebooks/audit/2026-10-07/README.md:119`).

**Out of scope:** why ELL-1 was closed and why DGF-1 came before EDR-1. The decisions are recorded
(`notebooks/lab/2026-07-25.md:12`, `notebooks/lab/2026-07-26.md:30`); their reasons are not (audit
N1, `notebooks/audit/2026-10-07/B-route-review.md:59`), and this ADR does not supply them.

## Decision

**EDR-1 starts at the first of these:**
- accepting any EDR-1 Phase-0 ADR;
- opening EDGAR filings, or any label source or dataset doc-03 names
  (`docs/03-edgar-risk-embedding-model-BUILD.md:77-83`): for example AAERs and SEC litigation
  releases, 8-K Item 4.02 restatements, and delisting or bankruptcy data;
- writing EDR-1 code, including the EXTEND work that adds the text encoder and typed-edge GNN to
  `gbe/` (`docs/03-edgar-risk-embedding-model-BUILD.md:31`).

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
   - it is declined with a dated reason, at any point before its stage-2 test batch runs (plan row 6d
     included);
   - a stage-1 no-go is recorded (row 6c);
   - a go is recorded, and its stage-2 test batch is reported (row 7).
4. **DGF-1's exit decision is accepted** as an ADR, tiered under ADR-017 by its review (plan row 10).
   Deciding ADR-015's official-split batch makes it Tier A (`CLAUDE.md:201`). For each remaining DGF-1 item
   (Phase 2 and Gate 2, ADR-015's positioning batch) it states whether that item is closed, deferred
   with its condition, or continuing alongside EDR-1.
5. **The consistency ADR is accepted** (audit D15, plan row 12).
6. **The test-window ledger and "peek" ADR is accepted** (plan row 5).

**Not required:**
- any gate *passing*. ELL-1's and DGF-1's Gate 1 are engine gates (ADR-019 clause 1);
- DGF-1's Gate 2, which is deferred and needs its own pre-registration (ADR-018 clause 8).

**No silent waiver.** If any condition cannot be met, an amendment to this ADR giving the reason is
drafted, reviewed and accepted before EDR-1 starts. It is tiered under ADR-017, and doubt means A.
A missed date is not a waiver.

**The check.** Before EDR-1 starts, a dated lab line cites each condition's file or commit. That line
goes through `scripts/check_evidence.py` like any document brought for a decision.

## Alternatives rejected

- **"Cleared their gates" (doc-03 as written).** It can never hold, because ELL-1's Gate 1 failed.
- **"EDR-1 proceeds regardless."** EDR-1 would start before DGF-1's model-class answer, which its
  floor design needs.
- **Wait for DGF-1's Gate 2.** Gate 2 has no metric and is deferred (ADR-018 clause 8).
- **Count a Phase-0 ADR *draft* as the start.** Row 13 lists seven Phase-0 topics, to be accepted by
  2026-12-06, about ten days after the consistency ADR (11-26) (`docs/timeline.md:87`). Waiting to
  begin drafting until then would push row 13 back. The researcher chose acceptance as the trigger
  (2026-10-08).

## Consequences

**On acceptance, these amendments are applied and committed to the governance repository:**
- **`docs/03-edgar-risk-embedding-model-BUILD.md:8`'s last sentence** becomes "Do not start it until
  ADR-020's start condition holds: ELL-1 closed, DGF-1 Gate 3 decided, matched-time resolved, and
  DGF-1's exit decision, the consistency ADR and the ledger ADR accepted." It carries the note
  "*Amended 2026-10-08 (ADR-020); was 'until `ELL-1` and `DGF-1` have cleared their gates',
  which ELL-1's failed Gate 1 made unreachable.*";
- **the "plan row 4" pointers** change: "plan row 4's ADR" becomes "ADR-020" in the research plan
  :82 and doc-01:24. In doc-02:24, where the pointer sits inside ADR-019's dated note, "(now
  ADR-020)" is appended instead;
- **CLAUDE.md's *Next* item 5** becomes: "**EDR-1's start condition: accepted 2026-10-08
  (ADR-020).** EDR-1 starts at the first of: accepting a Phase-0 ADR, opening EDGAR filings or any
  label source doc-03 names, or writing EDR-1 code (the EXTEND work in `gbe/` included); Phase-0
  ADRs may be drafted earlier. It may start only once ELL-1 is closed, DGF-1 Gate 3 is decided,
  matched-time is resolved, and DGF-1's exit decision, the consistency ADR and the ledger ADR are
  accepted;";
- **timeline row 4** gets its actual date;
- **timeline row 10** becomes "DGF-1 exit decision accepted as an ADR, tiered by its review
  (ADR-020)", with its tier column set to "A or B, by review".

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
    - the check is keyed to the start, with "reading" narrowed and EXTEND named;
    - any signed Gate-3 verdict counts;
    - the exit decision is a Tier-B ADR with contents;
    - one reviewed waiver path for every condition;
    - exact amendment texts.
  - **Nits:**
    - the doc-02 pointer is appended to, not rewritten;
    - the ledger ADR is condition 6;
    - N1 is out of scope;
    - the check line goes through the evidence checker.
- **Diff pass** (2026-10-08): the one Tier-B diff pass, by a fresh subagent with the standing
  preamble, over `de7a3b3..5de1aef`.
  - **Tier B confirmed.**
  - **The blocking finding was fixed;** S1, S2, S3 and two nits were judged fixed.
  - **One new blocking finding:** condition 4 fixed the exit decision's tier as B, although deciding
    ADR-015's official-split batch is Tier A. Five should-fix findings and six nits.
- **Escalated to the researcher (2026-10-08),** as ADR-017 requires for a blocking finding after the
  one diff pass. The researcher accepted the fix (`notebooks/lab/2026-10-08.md`).
- **Draft 3** (2026-10-08), with no further review:
  - **Blocking:** the exit decision is accepted as an ADR tiered by its review, and timeline row
    10's wording is amended.
  - **Should-fix:**
    - CLAUDE.md's text carries the full start definition;
    - a decline must come before the test batch runs;
    - a waiver must be accepted, tiered under ADR-017;
    - the out-of-scope note states that the reasons are unrecorded;
    - the label sources point to doc-03's list.
  - **Nits:**
    - the "start" decision is dated and recorded in the lab;
    - the acceptance date is a placeholder;
    - the check is keyed to the start;
    - EXTEND cites doc-03:31;
    - "seven topics", with no unsourced date;
    - the ledger's adoption is cited.
- **Accepted 2026-10-08** by coderback at draft 3. The amendments under *Consequences* were applied
  the same day and committed to the governance repository.
