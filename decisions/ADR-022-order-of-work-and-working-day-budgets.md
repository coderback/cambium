# ADR-022 — An order of work and working-day budgets, in place of calendar deadlines

**Status:** proposed
**Date:** proposed 2026-11-04 · draft 1 2026-11-04
**Tier:** B. It changes how the plan limits work. It touches no held-out set, gate criterion, tier rule,
review budget, or held-out or leakage rule (CLAUDE.md *Process*, ADR-017). One review, at most one
diff pass.
**Deciders:** coderback
**Docs affected** (edited in place, so no cited line moves):
- `docs/timeline.md:46-60` and its plan rows;
- `docs/research-plan-UNIFIED-GBE-GDE.md:93` and `:138`;
- `docs/03-edgar-risk-embedding-model-BUILD.md:43`;
- `CLAUDE.md:61-62`.

## Context

**The dates exist to signal slippage.** The audit found "nothing can signal slippage"
(`notebooks/audit/2026-10-07/D-governing-docs.md:60`), and the researcher set calendar dates on
2026-10-07 (`docs/timeline.md:46`).

**The calendar counted absence as well as effort.** Nothing was committed between 2026-10-08 and
2026-11-04, and matched-time's timebox ran out with no work done on it. Moving every open row 27 days
(governance `5807e72`) repaired this once. Every later break would need the same repair.

**Nothing external needs a date.** This is an independent project with no venue or submission date
chosen (the researcher, 2026-11-04). The research plan names standing targets and says their
deadlines move (`docs/research-plan-UNIFIED-GBE-GDE.md:56`).

**But limits do real work.** ADR-015 took fifteen drafts (`decisions/ADR-015-dgf1-official-split-positioning.md:4`),
and doc-03's scope rule and "No 'one more experiment'" both exist to stop work expanding
(`docs/03-edgar-risk-embedding-model-BUILD.md:43`, `docs/research-plan-UNIFIED-GBE-GDE.md:93`).

## Decision

1. **The plan is an order of work.** Each row's `after` column says what must come first, and its
   calendar dates go. They stay in the governance repository at `5807e72`. Row 6a gains `after`:
   2, 4, 5a (ADR-021 clause 9). Row 11 reads "run before row 10 only if the researcher chooses to;
   otherwise deferred in row 10's decision".
2. **A limit is a budget in working days.** A working day on an item is a calendar day whose lab entry
   records work on that item, with a line `**Budget:** <item> n/N`. Other work and breaks spend nothing.
   - **A budget is fixed.** When it runs out, the scope shrinks or the item is declined in writing with
     the date. The budget does not grow.
   - **Matched-time stage 1: 13 working days,** from zero. That is the 20-day timebox less the plan's
     ×1.5 stretch for non-research days, rounded down.
   - **EDR-1's ETL: 30 working days,** six five-day weeks. Its Phase-0 ADR fixes the number, as
     `docs/03-edgar-risk-embedding-model-BUILD.md:213` already requires.
   - **Every other row** has no budget. Each ADR stays bounded by its review budget.
3. **The ship rule loses its clock.** `docs/research-plan-UNIFIED-GBE-GDE.md:93` reads: "at each
   `[SHIP]`, writing is the next thing worked on after the gate clears, using the gate tables as-is.
   No 'one more experiment.'" The weekly review at `:138` becomes "every fifth working day".
4. **A chosen venue brings back one date.** Its deadline becomes the plan's only calendar date, and
   the plan works back from it. Choosing a venue needs no new ADR.
5. **Unchanged:**
   - the overnight wait for Tier-A ADRs (`decisions/ADR-017-process-rules-after-audit.md:58`), which
     schedules nothing and is tested into the ledger's authority check (ADR-021 I3);
   - timestamps: acceptance dates, ledger rows, the `actual` column, and a decline's date;
   - every order rule in an accepted ADR, such as matched-time resolved before Gate 3's
     pre-registration is accepted.

**Wording.**
- `CLAUDE.md:61-62` reads: "a plan in `docs/timeline.md`: an order of work, with matched-time's
  working-day budget, the numbers set by me (ADR-022);".
- Row 6 reads "budget 13 working days", row 6d "if the budget runs out first", row 14 "N = 30 working
  days", and row 16 "writing is the next thing worked on after Gate 0".
- doc-03:43 says "N working days" and "budget" where it says "N weeks" and "timebox".
- The plan rows drop the two date columns, and the exception added at `5807e72` goes.

## Alternatives rejected

- **Keep dates and re-date after each break.** Slip would be reset each time, so the dates would signal
  nothing, and the rule that a timebox does not move would stop meaning anything.
- **An order of work with no limits.** It loses the guard against one question taking over the project.
- **Charging every working day to every budget.** Work on row 5a would spend matched-time's budget.
- **Declining matched-time now.** No work on it was tried, so the decline would record a break, not a
  limit reached.
- **Dropping the overnight wait as well.** It is not a schedule, and changing it is Tier A.

## Consequences

- **A break costs nothing,** and it no longer forces a decision.
- **What is lost:** the plan no longer warns when the project stalls, and nothing external pushes it to
  finish. The researcher accepts this for an independent project.
- **Matched-time's budget is looser than its old timebox,** since the 20 days also held other work and
  the 13 count only days on it.
- **Each budgeted day costs one lab line.**

## Revisit when

- A venue is chosen and its deadline does not fit the remaining order of work.
- A budget is exceeded by re-labelling work, or budgets prove too loose to stop an item spreading.

## Draft history

- **Draft 1** (2026-11-04).
