# ADR-022 — An order of work and working-day budgets, in place of calendar deadlines

**Status:** accepted
**Date:** proposed 2026-11-04 · **accepted 2026-11-04 (16:21) by coderback**, at draft 3, after two review rounds
**Tier:** A, re-tiered from B by its round-1 review (ADR-017). The plan's dates also hold held-out looks
in order, and this ADR sets that order for rows 7, 8 and 11. It sets no gate criterion, split or metric,
and changes no tier rule or review budget.
**Review rounds:** 2/4
**Deciders:** coderback
**Implemented by:** one governance commit, at least 12 hours after acceptance and not on the same
calendar day. Until then `docs/timeline.md:50` governs, matched-time's 2026-11-24 limit included.
**Docs affected** (each edit stays on its own lines, so no cited line moves; clause 3 and *Wording* give
the text):
- `docs/timeline.md:28`, `:40`, `:46-60`, its plan table, and `:278-279`;
- `docs/research-plan-UNIFIED-GBE-GDE.md:93` and `:138`;
- `docs/03-edgar-risk-embedding-model-BUILD.md:43` and `:213`;
- `CLAUDE.md:62`, `:79`, `:89` and `:283`.

## Context

**The dates exist to signal slippage.** The audit found "nothing can signal slippage"
(`notebooks/audit/2026-10-07/D-governing-docs.md:60`), and the researcher set calendar dates on
2026-10-07 (`docs/timeline.md:46`). ADR-017:89-90 records that decision. This ADR replaces it, and
ADR-017 stays as the record.

**The calendar counted absence as well as effort.** Nothing was committed between 2026-10-08 and
2026-11-04, and matched-time's timebox ran out with no work done on it. Moving every open row 27 days
(governance `5807e72`) repaired this once. Every later break would need the same repair.

**Nothing external needs a date.** This is an independent project with no venue or submission date
chosen (the researcher, 2026-11-04). The research plan names standing targets and says their
deadlines move (`docs/research-plan-UNIFIED-GBE-GDE.md:56`).

**But limits do real work.** ADR-015 took fifteen drafts (`decisions/ADR-015-dgf1-official-split-positioning.md:4`),
and doc-03's scope rule and "No 'one more experiment'" both exist to stop work expanding
(`docs/03-edgar-risk-embedding-model-BUILD.md:43`, `docs/research-plan-UNIFIED-GBE-GDE.md:93`).

**And the dates hold an order.** Only its date keeps ADR-015's official-split batch (row 11) after
Gate 3's batch, and only theirs orders matched-time's test batch (row 7) against Gate 3's
pre-registration (row 8). Both orders involve a scores look, so removing the dates has to set them.

## Decision

This ADR authorises no look, and no choice here uses a held-out label or score.

1. **The plan is an order of work.** Each row's `after` column says what must come first, and its
   calendar dates go. They stay in the governance repository at `5807e72`. Row 6 is done at row 6c or
   row 6d. These rows change or are confirmed:

   | Row | `after` | Why |
   |---|---|---|
   | 6a | 2, 4, 5a | The backfill lands before it is accepted (ADR-021 clause 9) |
   | 7 | 6c, 8 | Gate 3's pre-registration is written without another set of test-window scores in hand. It could use them only to remove claims (ADR-021 clause 4) |
   | 8 | 2, 5, 5a, 6 | Matched-time's pre-registration is accepted or declined first (ADR-013:324-325). The backfill lands first on a decline too, because Gate 3's disclosure comes from the ledger (ADR-021:255-260) |
   | 9 | 7 on a go, 8b | Matched-time's batch runs before Gate 3's (ADR-013:326). On a no-go or a decline there is no row 7 to wait for. A no-go, dated in both documents, is the decline ADR-013:327's guard reads |
   | 10 | 9, and 11 if chosen | Row 11 runs before it, or row 10's decision records its status |
   | 11 | 5a, 9 | It needs row 5a's accessors (ADR-021:20), and its look counts as a look at both sets (ADR-021:118-119), so it stays behind Gate 3's batch |
   | 12 | 3 | Unchanged. Row 13 holds it before EDR-1, and no rule needs more |
   | 17 | 15, once row 16's writing has started | Row 16's writing is the next thing worked on after Gate 0 |

2. **A limit is a budget in working days.** A working day on an item is a calendar day on which any
   work on it is done: drafting in or outside the repo, review, code, or a batch running. That day's lab
   entry carries a line `**Budget:** <item> n/N`, and a day found later without one is charged when
   found. Other work and breaks spend nothing.
   - **A budget does not grow.** Where an item's scope can shrink, it shrinks before the budget runs out.
   - **Matched-time stage 1: 13 working days,** its 20-day timebox divided by the ×1.5 stretch the dated
     plan added for non-research days, rounded down. 2026-11-04, when its pre-registration was first
     drafted, is day 1. Its required items cannot shrink (ADR-013:321). When the budget runs out, it is
     declined with a dated reason recorded as ADR-013:324-325 requires. Once all its validation pilot's
     rows exist, its go/no-go rule decides, and running out of budget cannot decline it. The day the
     rule runs still gets its line, even if n passes N. *Note 2026-11-05 (ADR-023 clause 5): a void attempt's rows are not "the pilot's rows" here. Only ten clean rows are, whether their reproduction has passed or is still owed, so the budget can still run out while no attempt has reproduced.*
   - **EDR-1's Phase 0 for its initial universe (`docs/03-edgar-risk-embedding-model-BUILD.md:43`): 30
     working days,** six five-day weeks, which the dated plan did not stretch (`docs/timeline.md:58`).
     It counts every day of Phase-0 work from row 13's acceptance: ingestion, baselines and leakage
     tests. Drafting the Phase-0 ADRs spends nothing. The number is provisional until its Phase-0 ADR
     fixes it, as `docs/03-edgar-risk-embedding-model-BUILD.md:213` requires, and it is fixed from then.
     When the budget runs out, Phase-0 work stops and row 15, Gate 0, judges what was built.
   - **Every other row** has no budget. An ADR stays bounded by its review budget.
3. **The ship rule loses its clock.** In `docs/research-plan-UNIFIED-GBE-GDE.md:93`, "writing starts
   within two weeks of the gate clearing" becomes "writing is the next thing worked on after the gate
   clears", and the rest of the line stays. `:138` opens "**20-minute review, every fifth lab entry:**
   re-read those entries," and the rest of the line stays.
4. **A chosen venue brings back its deadline.** That date enters the plan, and the plan works back from
   it. Choosing a venue needs no new ADR.
5. **Unchanged:**
   - the overnight wait for Tier-A ADRs (`decisions/ADR-017-process-rules-after-audit.md:58`), which
     schedules nothing, and which ADR-021 I3 builds into the ledger's authority check;
   - timestamps: acceptance dates, ledger rows, the `actual` column, and a decline's date;
   - every order rule in an accepted ADR, such as those at ADR-013:324-327;
   - ADR-015's two calendar stops. After a batch's first official score, a lapsed update pause
     (`decisions/ADR-015-dgf1-official-split-positioning.md:752`), or five weeks with no dated report
     (`decisions/ADR-015-dgf1-official-split-positioning.md:754`), ends the batch for good.

## Wording

**`docs/timeline.md`:**
- `:28`: "the *Dated plan* below" becomes "the *Plan* below".
- `:40`: "starts within 2 weeks of the gate clearing" becomes "is the next thing worked on after the gate
  clears".
- `:278-279`: "ETL timeboxed —" becomes "ETL budgeted —", and "scope shrinks, timebox doesn't move"
  becomes "scope shrinks, budget doesn't grow".
- `:46-60`, line for line, with `:52`, `:59` and `:60` unchanged:

  ```
  46  ## Plan: an order of work (ADR-022)
  47
  48  Adopted under ADR-022, replacing the dated plan of 2026-10-07, whose dates stay in the governance repository at `5807e72`. Each row's `after` column says what comes first, and each row gets its actual date when it completes. Row 6 is done at row 6c or row 6d.
  49  - **A budget does not grow.** Where an item's scope can shrink, it shrinks before the budget runs out.
  50    When matched-time's runs out, it is declined as ADR-013 clause 5.1 requires, unless all its pilot's rows exist, when its go/no-go rule decides. When the ETL's runs out, Phase-0 work stops and Gate 0 judges what was built.
  51  - **A working day** on an item is a calendar day on which any work on it is done, a batch running included. That day's lab entry carries `**Budget:** <item> n/N`, and a day found later without one is charged when found.
  52  - **Tier-A rows include ADR-017's one-night wait** between acceptance and implementation.
  53
  54  **How the budgets and compute were set:**
  55  - matched-time's 13 is its 20-day timebox divided by the ×1.5 stretch the dated plan added for
  56    non-research days, rounded down, and 2026-11-04 is its day 1;
  57  - the ETL's 30 is six five-day weeks of Phase-0 work from row 13's acceptance, which the dated plan
  58    did not stretch, provisional until its Phase-0 ADR fixes it;
  59  - DGF-1's remaining batches run locally, on the stack the trainer is certified on;
  60  - EDR-1's training runs on RunPod, as the research plan's §7 already says, under an environment pin
  ```

- **The plan's header** becomes `| # | milestone | tier | after | actual |`, its separator row gets five
  cells, every row loses its two date cells, and clause 1's `after` values apply. These milestones change:
  - row 6: "**matched-time stage 1: budget 13 working days**";
  - row 6d: "declined, at any point or when the budget runs out, with the dated reason";
  - row 11: "ADR-015 official-split positioning: run before row 10 only if the researcher chooses to;
    otherwise row 10's decision records its status";
  - row 13: "ETL timebox N" becomes "ETL budget N";
  - row 14: "**EDR-1 ETL budget: N = 30 working days of Phase-0 work, until its Phase-0 ADR fixes N**";
  - row 16: "SHIP P0: writing is the next thing worked on after Gate 0";
  - row 17: "EDR-1 Gate 1, then Gate 3, then P1 submission (H1)".

**`docs/03-edgar-risk-embedding-model-BUILD.md`:**
- `:213`: "ETL timebox N" becomes "ETL budget N".
- `:43`: the bold sentence reads "**ETL budget: Phase 0 is capped at N working days for the initial
  universe (fill N when scheduling, in the ADR); when the budget is threatened, the *scope shrinks* —
  the budget does not grow.**"
- `:5` stays, because it is v0.2's changelog.

**`CLAUDE.md`:**
- `:62` reads "`docs/timeline.md`, since replaced by an order of work with working-day budgets
  (ADR-022), the numbers set by me;". `:61` stays, because the dated plan was done on that date.
- `:79`: "runs only on a go." becomes "runs only on a go, after plan row 8."
- `:89`: "**ADR-015** when cheap:" becomes "**ADR-015** when cheap, after Gate 3's batch (plan row 11):".
- `:283`: "it; I edit." becomes "it; I edit. A day's work on a budgeted item adds its `**Budget:**` line
  (ADR-022)."

## Alternatives rejected

- **Keep dates and re-date after each break.** Each break adds another column and another exception, and
  the rule that a timebox does not move stops meaning anything.
- **An order of work with no limits.** It loses the guard against one question taking over the project.
- **Charging every working day to every budget.** Work on row 5a would spend matched-time's budget.
- **Counting only days that carry a budget line.** Leaving the line out would then avoid the count.
- **Matched-time's test batch before Gate 3's pre-registration,** the order the dates gave. Gate 3's
  document would be written with another set of test-window scores in hand. It could use them only to
  remove claims, so the order gains little.
- **Declining matched-time now.** It has had one working day, so the decline would record a break, not
  a limit reached.
- **Declining EDR-1's ETL when its budget runs out.** EDR-1 is the only route to the GBE half
  (`docs/research-plan-UNIFIED-GBE-GDE.md:15`), so this would abandon that question over a scheduling
  limit.
- **Dropping the overnight wait as well.** It is not a schedule, and changing it is Tier A.

## Consequences

- **A break costs nothing,** and it no longer forces a decision. The exception is an ADR-015 batch, whose
  update pause has a five-week ceiling (`decisions/ADR-015-dgf1-official-split-positioning.md:671`). A
  lapse before the first official score costs a restart at `certify`
  (`decisions/ADR-015-dgf1-official-split-positioning.md:672-673`). After it, the batch stops.
- **What is lost:** the plan no longer warns when the project stalls, and nothing external pushes it to
  finish. Rows with no budget have no limit, including the DGF-1 exit decision, which the audit asked to
  have dated (`notebooks/audit/2026-10-07/README.md:103`). The researcher accepts this for an independent
  project.
- **Matched-time's budget is looser than its old timebox,** since the 20 days also held other work and
  the 13 count only days on it.
- **Matched-time's test batch waits** for Gate 3's pre-registration to be accepted. A decline after a go
  and after that acceptance needs a Tier-A amendment to Gate 3's ADR, because the dated reason is
  recorded in both documents (ADR-013:324-325).
- **An ETL that runs out may fail Gate 0.** That is a recorded result, not a decline. What follows a
  failed Gate 0 is a new ADR's question.
- **Each budgeted day costs one lab line.**

## Revisit when

- A venue is chosen and its deadline does not fit the remaining order of work.
- A budget is exceeded by re-labelling work, or budgets prove too loose to stop an item spreading.

## Draft history

- **Draft 1** (2026-11-04, `eb0204a`), as Tier B.
- **Round 1, full review** (2026-11-04). It re-tiered the ADR to A and found 4 blocking items, 8 should-fix
  and nits:
  - row 11 lost its place behind Gate 3's batch;
  - rows 7 and 8 were ordered only by dates, and ADR-013's rule was misquoted;
  - a budget that ran out had no defined outcome;
  - the working-day definition could be avoided by leaving out the line.
- **Draft 2** (2026-11-04). All should-fix items and most nits are folded in. The researcher set three
  choices on 2026-11-04:
  - row 8 before row 7, where the review's fix had followed the dates;
  - the ETL stops at its budget and Gate 0 judges what was built;
  - 2026-11-04 counts as matched-time's day 1.
- **Round 2, diff-only pass** (2026-11-04). Nothing blocking, 9 should-fix, 12 nits.
- **Draft 3** (2026-11-04), folded in under the stopping rule with no further round:
  - row 8 gains `after` 5a, so the backfill lands first on a decline too;
  - a no-go counts as the decline ADR-013:327's guard reads;
  - "all" the pilot's rows, so a part-run pilot leaves one outcome;
  - the ETL budget counts all Phase-0 work from row 13's acceptance, and all of it stops at the limit;
  - a failed Gate 0 goes to a new ADR;
  - ADR-015's two calendar stops named;
  - ADR-021 clause 4's "only remove claims" stated correctly;
  - the plan's instruction to record actual dates kept;
  - `CLAUDE.md:79` and `:89` added;
  - nits: the separator row, row 6d's wider decline, row 11's status wording, row 17's `after`, the
    `:54` heading, and `docs/03-edgar-risk-embedding-model-BUILD.md:5` left alone as a changelog.
- **Accepted** 2026-11-04 (16:21) by coderback, at draft 3. Draft 2 was reviewed from the working tree
  against `eb0204a` and was never committed on its own.
