# ADR-017 — Process rules after the 2026-10-07 audit

**Status:** accepted
**Tier:** B (process). It creates the tiers, so it is not classified by them; the researcher decided
each item on 2026-10-07. Any later change to items 1–3, or any narrowing of item 4's preamble, is
Tier A.
**Date:** proposed 2026-10-07 · **accepted 2026-10-07 by coderback**, after one review and one
diff pass (*Draft history*). Tier B, so implemented the same day (item 2).
**Deciders:** coderback
**Docs affected:** `CLAUDE.md` (*Integrity of results*, *Architecture*'s inclusion rule, *Process*);
`docs/research-plan-UNIFIED-GBE-GDE.md:128-129`; `docs/00-shared-core-graph-embedding-GUIDE.md:14`
and `:277`; `docs/subagent-preamble.md`; `decisions/ADR-000-template.md`. Everything except the
template is gitignored here and versioned in the private governance repository (local; baseline
commit `33cf50d`).

## Context

The research audit of 2026-10-07 (`notebooks/audit/2026-10-07/README.md` §4b) found process rules
that cost more than they protected, or could not be followed as written:
- **Review depth did not follow stakes.** ADR-014's four drafts concerned a check that then passed in
  572 s (lab 2026-09-15, Session 38). ADR-015 went to a fifteenth draft; it reads a held-out set, so
  it is Tier A under item 1, and one of its rounds found a real leakage defect
  (`notebooks/audit/2026-10-07/C-process-retrospective.md:61`).
- **"Session" is undefined**, and "one build-doc section per session" conflicts with amendments that
  span sections (audit D13).
- **No rule says to measure before writing rules about a feared failure** (D14). On 2026-10-07 a
  label-free timing replaced the matched-time cost projection and changed the §4c decision (lab
  2026-10-07, the §4c line; that measurement predates item 3, and its script was not kept).
- **Subagent prohibitions were rewritten into each prompt by hand**, and a location of
  official-test-mask statistics was missed four times on 2026-10-07 (README §4b item 4, and §6).
- **The lab cap is not kept:** 8 of 9 lab files exceed 10 lines (C-process-retrospective.md:102
  counted 7 of 8, before 2026-10-07's entry).
- **The inclusion rule cannot be checked** while most of the four models are unbuilt (D12).

The researcher decided §4b items 1–7 on 2026-10-07 (lab 2026-10-07).

## Decision

1. **Two ADR tiers**, each stated in the ADR's header.
   - **Tier A:** any ADR that sets gate criteria, splits or metrics, or that sets or exercises access
     to a held-out set (reported-only batches included); and any change to the tier rules, the
     review budget, or the held-out and leakage rules. A full adversarial review, then a diff-only
     pass on every later draft, within **4 review rounds**.
   - **Tier B:** work that touches no held-out set: reported-only work, tooling and process. One
     page, one review, at most one diff pass.
   - **Precedence:** anything not plainly Tier B is Tier A. The reviewer confirms the tier before
     reviewing, and doubt means A.
   - **Stopping rule, both tiers:** when a diff-only pass finds nothing blocking, its should-fix items
     and nits are folded in without another round and listed in the ADR's *Draft history*, and the
     ADR goes to the researcher.
   - **Escalation:** a blocking finding still open when the budget is spent (Tier A), or after the one
     diff pass (Tier B), goes to the researcher, who accepts and records it, moves it into tests, or
     withdraws the ADR. A finding that the ADR belongs in Tier A is never escalated: the ADR is
     re-tiered.
   - **A finding moved into tests** gets a test that fails against a mutation reproducing the defect,
     with the command output cited. It lands with or before the implementing commit, and before any
     batch the ADR governs runs.
2. **A Tier-A ADR waits a night.** It is implemented at least 12 hours after its acceptance, and
   never on the same calendar day. Running a batch the ADR governs counts as implementing it. An
   amendment to an accepted Tier-A ADR is a Tier-A draft: it gets a diff-only pass before it takes
   effect, and it restarts the wait. A Tier-B ADR may be implemented the same day. In CLAUDE.md's
   rules a session is a calendar day; the lab's "Session N" headings still number sittings.
3. **Measure or prototype first.** A feared failure that a short run can test is run before any rule
   is written about it. A guard is prototyped as scratch code with mutation tests, outside the
   repository, before its ADR specifies it; the ADR states the invariant, names the test and cites
   the mutation results.
   - Only a prototype written before the ADR's acceptance is evidence. Code written for a Tier-A ADR
     after its acceptance is implementation, scratch or not.
   - A measurement or prototype uses no dataset labels, reads no held-out set (a test window or the
     official test mask), and computes no model metric. Anything that computes a metric is a run and
     goes through `gbe.run`.
   - Whatever relies on a measurement cites its script and output, kept in `notebooks/measurements/`.
     That folder is tracked in the open release; its contents are label-free by the rule above.
4. **A standing subagent preamble**, `docs/subagent-preamble.md`, prepended verbatim to every
   subagent prompt. It is an operational file, not a governing document. Adding a location or a ban,
   or renumbering a range so the same text stays covered, needs no ADR; removing or narrowing a ban
   or a location is Tier A. Every change is committed to the governance repository. The file was
   drafted on 2026-10-07 for this ADR's own review, before acceptance.
5. **Work is scoped by logical change, not by build-doc section**; a change may span a series of
   commits, each commit one logical step. This replaces "one build-doc section per session".
6. **The daily lab entry** opens with a ≤10-line summary that adds "drafts and review rounds today"
   (C-process-retrospective.md:142); a detailed log may follow it.
7. **The core inclusion rule:** code enters `gbe/` only if it is domain-agnostic, is tested, and has
   at least two consumers: built models, or models whose own build doc names the capability, cited
   by line. doc-00, the shared guide, does not count: it names broad capabilities for all four
   models (doc-00:101), so citing it would admit almost anything.

Two related decisions are recorded here and carried out separately: the evidence-check script
(§4b item 2), whose rule enters `CLAUDE.md` when the script exists, and a dated plan in
`docs/timeline.md` (§4b item 7).

**Amendments, applied on acceptance:**
- `CLAUDE.md` *Process* and the inclusion-rule bullet are rewritten to state items 1–7, and the
  commit bullet reads "one logical step per commit".
- `CLAUDE.md` *Integrity of results*: after "actual run." add "The one exception is a measurement
  figure under ADR-017 item 3 (a label-free timing, memory or size), which comes from its script's
  saved output in `notebooks/measurements/`."
- `research-plan:128`: "**Daily lab entry (≤10 lines):** what ran, what broke, what surprised,
  tomorrow's first move." becomes "**Daily lab entry:** a ≤10-line summary (what ran, what broke,
  what surprised, drafts and review rounds today, tomorrow's first move), optionally followed by a
  detailed log." The rest of the line is unchanged.
- `research-plan:129`: "One page: context → decision → alternatives rejected → revisit-when." becomes
  "Structure: context → decision → alternatives rejected → revisit-when; one page for a Tier-B ADR
  (CLAUDE.md *Process*)."
- `doc-00:14`: "only if **all four models would implement it identically**" becomes "only if it is
  **domain-agnostic, tested, and has at least two consumers (built models, or models whose own build
  doc names the capability, cited by line)**". The rest of the line is unchanged.
- `doc-00:277`: "*the core is whatever all four models do identically;" becomes "*the core is what
  is domain-agnostic, tested and has at least two consumers;". The rest is unchanged.
- `docs/subagent-preamble.md` is committed to the governance repository.
- `decisions/ADR-000-template.md`: `**Tier:** <A | B>` and, for Tier A, `**Review rounds:** <n>/4`
  are added after `**Date:**`.

## Alternatives rejected

- **Keep the rules as written:** the costs above continue.
- **A budget of 3, or none:** the researcher chose 4.
- **Session = one conversation:** a fresh conversation the same day would satisfy it, so a decision
  that binds gates or held-out access could be built while its author is still invested in it.
- **Retire the same-session rule entirely:** the same reason.

## Consequences

- From ADR-018 on, every ADR states its tier, and a Tier-A header records `Review rounds: n/4`
  (C-process-retrospective.md:116).
- **Changed meanings.** CLAUDE.md's leakage rule ("write its leakage/integrity tests in the same
  session") now means the same calendar day, as do `docs/02-dgraph-fin-embedding-model-BUILD.md:125`
  and `docs/timeline.md:300`.
- **ADR-016 was implemented on the day it was accepted** (`8060cf2` at 15:42, `1f4dbe7` at 18:20,
  2026-10-07), in a separate conversation; its header's "Implemented in a later session" holds only
  under the older reading. Under item 2 it would have waited. This is recorded, not reversed: the
  implementation passed its suite and mutation checks (lab 2026-10-07).
- **No code moves under item 7.** The rule applies the next time code is proposed for `gbe/`.
- Lab entries written before acceptance are not rewritten.

## Revisit when

- The Tier-A budget is spent twice running: the budget, or the tier boundary, is wrong.
- A defect that a Tier-B review missed reaches a gated number: the tier boundary is wrong.

## Draft history

- **Draft 1, Tier-B review.** Four blocking findings, all fixed in draft 2:
  - the tiers overlapped and had no precedence rule;
  - the prototype carve-out had no time limit;
  - "label-free" had been dropped from item 3;
  - "a later calendar day" could be met by crossing midnight.

  Should-fix items folded in: an amendment restarts the wait; "session" scoped, with the changed
  meanings listed; a Tier-B escalation path; a non-empty consumer test; the preamble declared
  operational; the governance repository cited; research-plan:129 and the template amended. Nits
  folded in: sources cited for added wording; the lab count updated; items 1–7; the ADR-016 example
  replaced by the rationale; the commit wording reconciled.
- **Draft 2, the one diff pass. Nothing blocking**, so under the stopping rule these were folded in
  without another round:
  - **Should-fix:**
    - Tier A's "exercises" now applies only to held-out access, so reported-only work on validation
      stays Tier B;
    - narrowing the preamble is Tier A, and the preamble is committed on acceptance;
    - a tests-moved finding needs a failing mutation test that lands with the implementation, and a
      tier finding forces re-tiering;
    - an amendment to an accepted Tier-A ADR gets a diff pass;
    - CLAUDE.md's "every reported number" rule gains the measurement exception;
    - doc-00 does not count as a consumer's build doc.
  - **Nits:** one logical step per commit; "in this file's rules" in CLAUDE.md; the 2026-10-07 timing
    cited as predating item 3; `notebooks/measurements/` declared tracked; research-plan:129's
    structure kept for both tiers; Tier B's "no held-out set" covers tooling too.
  - **Not adopted:** a commit-free gap of at least 6 hours as proof of a night (N6). Commit gaps
    measure commits, not rest, and the 12-hour, next-day rule is already mechanical.
  - **Open:** at about 1,800 words this exceeds one page, and "one page" is not defined.
