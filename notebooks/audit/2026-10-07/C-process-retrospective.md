# Audit C: process retrospective (subagent output, saved by the main session, 2026-10-07)

> Main-session spot-check notes:
> - **Confirmed against git:**
>   - 104 commits on 13 commit-days;
>   - 7 of 8 lab files exceed 10 lines;
>   - `git log -S extra_metrics` over gbe/adapters/scripts/tests returns nothing.
>   - **Commits touching each ADR file, all time:** ADR-004 1, ADR-006 2, ADR-011 4, ADR-012 7, ADR-013 8, ADR-014 4, ADR-015 15.
> - **Commits after `202c537`:** `git rev-list --count 202c537..HEAD` = 35, where C says 34. The difference is minor and unresolved.
> - **"Converged within 3 commits"** counts commits up to acceptance. The all-time counts above include later amendments.
> - **A fourth location of official-mask statistics** that the prompts missed: lab 2026-07-27.md:18 gives official-split node-time medians per mask (label-free). Line 17 has dataset-level totals only. C met it; B (which read the July lab) probably did too, and did not mention it. D was told to skip it.
> - **C read one home-directory memory file** (the stopping-rule memory) and disclosed it.

## 1. Summary

Git shows 104 commits on 13 days, with two stretches of no commits: 46 days and 20 days. Runs and code cluster around ELL-1's gates (07-24/25) and DGF-1's Gate 1 (09-12 to 09-14). Since Gate 0 was signed (`202c537`, 09-14 22:50) there have been 34 commits: 28 docs-only, 3 code and 1 run (572 s). Those 34 commits account for 64% of all lines ever added under `decisions/`.

ADR-015 is a reported, ungated protocol. It took 15 drafts and grew from 206 to 1,277 lines. By contrast, the gated pre-registrations (ADR-004, -006, -011, -012) each settled within 3 commits. Of ADR-015's 13 rejected drafts, 7 were rejected for defects in text the previous fix had added, and 3 for citation errors. ADR-014 spent four drafts on a feared failure of a check that then ran once, in 572 s, and passed.

Fifteen rounds of prose review missed two defects that grep would find. Meanwhile the trainer's leakage wiring is not pinned by any test.

The assistant's habits contributed directly:
- claims made without evidence in the same turn;
- answering findings by adding mechanism;
- writing subagent prohibitions from scratch each time.

There are five proposals:
1. a review budget matched to stakes, with a standing stopping rule;
2. a mechanical evidence check;
3. prototype guards before specifying them;
4. a standing subagent preamble;
5. lab discipline.

## 2. Findings

### Q1. Where the time went

- **F1. The timeline.** CONFIRMED (`git log`). 104 commits from 2026-07-21 19:51 to 10-07 11:46, on 13 calendar days. There are two stretches with no commits: 07-27 03:23 → 09-11 13:04, and 09-15 21:19 → 10-06 13:42. These are gaps, and nothing is inferred about them.
- **F2. Where runs and documents happened.** CONFIRMED.
  - Phase B (ELL-1 gates) and phase E (DGF-1 Gate 1) hold 10 of the 16 commits that touch the registry.
  - After `202c537`: 34 commits, of which 28 are docs-only. There are 3 code commits (`ccb986f`, `cd4912d`, `9d1fceb`) and 1 run (`e32a503`, the re-certification).
  - Those 34 commits added 4,727 of the 7,374 lines ever added under `decisions/`.
  - Phase G (ADR-015 v4–v15) has 0 code, 0 test and 0 run commits.
- **F3. ADR-015's span.** CONFIRMED. It is reported-only and ungated (ADR-015:10-12).
  - Drafts 1–2 are `5f7f388` and `30fbf92` (09-15). Draft 3 was never committed (ADR-015:6).
  - Drafts 4–15 plus acceptance are 13 commits, from 10-06 13:42 to 10-07 11:46. That 22h04m window includes an 8h27m overnight gap with no commits (`b98fac3` 23:03 → `d86b55e` 07:30).
- **F4. Corrections to the assistant's account.** CONFIRMED.
  - `becb85c` is not a draft. It committed a 206-line copy of draft 1 as the root file `xaa`, under the message "introduce ADR-015". It was removed at `54b8367`.
  - ADR-013's pre-acceptance window holds 5 commits (two drafts, each revised once, then acceptance), not "about six".
  - "15 drafts over 22 days" reported a calendar span as effort. "22 hours of drafting" also claims more than git shows.

### Q2. Why drafts kept failing

- **F5. Effort ran inverse to stakes.** CONFIRMED.
  - The gated pre-registrations converged in at most 3 commits: ADR-004 in 1, ADR-006 in 2, ADR-011 in 3, and ADR-012 in 3 (`d5c2916` → `c0f2053`). Their triggers were type (i) or (iii), with none of type (ii).
  - The reported ADR-015 took 15 drafts.
- **F6. ADR-015's 13 rejections** (ADR-015:64-160). CONFIRMED.
  - 7 were (ii), a defect introduced by the previous fix: the state machine in d5–d7, and d10–d13, where four consecutive blocking findings sat in newly added text.
  - 3 were (iv), cited sources that lacked the proposition (d1–d3).
  - 2 were (iii), completeness gaps (d8–d9).
  - 1 was (i), a real leakage problem: d4's hyperparameters had seen official-test labels.
  - 1 reviewer claim was (v), a false positive: torch.save (:98).
- **F7. What the scope growth was for.** Mixed.
  - CONFIRMED real: d4's retune, and d8's finding that deleting the gitignored score files re-pinned the result.
  - Most of the remaining growth handled non-reproduction: environment-variable families, an update pause, credential hashing, and stop/lapse rules. It did so on a path where all five recorded GNN-path reproductions matched bit-for-bit (lab 2026-09-11 Sessions 29–31; 2026-09-15 Session 38).
  - Clause 5 grew from 24 to 339 lines, and clause 7 from 22 to 193.
  - The ADR itself concedes that read points, fixtures and certify mechanics are checked "mechanically, which prose review cannot" (:185).
  - PLAUSIBLE: prototype code with mutation tests would have closed most of this more cheaply.
- **F8. ADR-014 was the same pattern.** CONFIRMED.
  - Four drafts in 5h10m (`bd422c3` 11:02 → `399dd7f` 16:12). Each invented a check that "can never pass" (ADR-014:27-31).
  - Most rejections were (iv): claims about code that running it refuted.
  - The check then ran in 572.1 s and passed (lab 2026-09-15:102).
  - Its *Revisit when* says the rule should be written "by someone who has done it once" (:82-83). ADR-015 did not apply that.
- **F9. What the reviews missed.** CONFIRMED.
  - **Inside the reviews' scope:** ADR-015:793-797 relies on ADR-014:60-62's "Verified" (audit A S5), and ADR-013:297-298's stale "until ADR-014 is accepted" sits in a paragraph `8296f4b` edited (S4). Grep would catch both.
  - **Outside the reviews' scope, on the gate path:** trainer leakage mutations leave the suite green (A-B1), guard messages overclaim (S1, S6), a "computed" footer sits over hard-coded constants (S2), and a citation has been misquoted since 07-21 (B-S2).
  - PLAUSIBLE: review effort went to reported-only prose while the gate-path wiring stayed unpinned.

### Q3. What should change

**The assistant's habits.** The assistant is at fault in each.
- **H1. Stating state or verification without evidence in the same turn.** CONFIRMED.
  - Examples: `xaa` reported as "gone"; "238 → 251"; the "Verified" channel and its reuse in ADR-015; a false import claim (ADR-014:34-36); a timing misdiagnosis (lab 2026-09-11:314); "22 days"; the assembler footer.
  - **Replace with:** no "verified", "gone", "passes" or "measured" without a command output, commit or run id in the same message.
- **H2. Answering a finding by adding mechanism.** CONFIRMED (ADR-014 d1–d4; ADR-015 d5–d7, d10–d13).
  - **Replace with:** first delete or narrow. Otherwise, state the invariant and name its test.
- **H3. Writing rules for a failure before running the cheap check.** CONFIRMED for ADR-014; PLAUSIBLE for ADR-015 clause 5.
- **H4. Forecasting, and offering to save effort.**
  - It forecast that v8 would be shorter; it went from 724 to 756 lines.
  - It suggested skipping v10's review, and that review found the read-point defect (:100-113).
- **H5. Writing subagent prohibitions from scratch each time.** CONFIRMED, four incidents:
  1. held-out labels were computed (lab 2026-09-11:12);
  2. `pypdf` was installed (:168);
  3. today's prompts left official-mask tables unfiltered;
  4. this retrospective's prompt omitted lab 2026-07-27 (official-split statistics).

**The process.**
- **The stopping rule is not a budget.** It ended the loop once. Under the v10–v13 pattern it would never have stopped.
- **The rule lives in the wrong place.** It exists only in the assistant's memory and ADR-015. CLAUDE.md is gitignored, and its "Next 1" is stale.
- **Untracked `docs/` silently defeats grep checks** (ADR-013:14-15).
- **The lab rule is not followed.**
  - 7 of 8 lab files exceed 10 lines.
  - `2026-09-11.md` is 457 lines and covers 09-11 to 09-15.
  - There is no entry for 10-06/07, which hold 13 commits.
  - ADR-015 nonetheless treats the cap as a design constraint (:166).

### Proposals

**P1. Match review depth to stakes, with a round budget.**
- **What changes (CLAUDE.md Process, plus a memory rule):**
  - Gated pre-registrations: a full review, then diff-only passes, with a budget of 4 rounds.
  - Reported-only or procedural ADRs: one full review plus at most one diff pass.
  - The stopping rule becomes standing.
  - When the budget is spent with a blocking finding still open, escalate to the researcher with three options: accept and record it, move it to tests, or withdraw.
- **Cost:** some defects reach implementation, where P2 and P3 catch them.
- **Check on Gate 3's ADR:** its header records `Review rounds: n/4`, and at most one round's blocking finding is in text the previous round added.

**P2. A mechanical evidence check before any document goes for decision.**
- **What changes:** a read-only script that verifies:
  - **quotes:** each block quote matches its cited `path:line` at HEAD;
  - **commits:** each cited commit hash exists;
  - **verification claims:** every "verified/measured" sentence carries a commit or run id, and any identifier it cites exists in a commit (`git log -S`);
  - **statuses:** every ADR status phrase matches that ADR's Status line;
  - **docs coverage:** it runs over `docs/` with `--no-ignore`.
- **Cost:** one script and its tests; the researcher decides whether it needs an ADR.
- **Check:** at today's HEAD it must flag ADR-015:793-797 and ADR-013:297-298. On Gate 3's ADR, reviewers raise no (iv) findings.

**P3. Prototype a guard before specifying it, and run the cheap check first.**
- **What changes (a CLAUDE.md carve-out):**
  - A mechanism clause is prototyped as scratch code and tests outside the repo, and the prototype counts as evidence, not implementation.
  - The ADR states the invariant, names the test, and cites the mutation results.
  - Any feared failure that a short run can test is run before rules are written about it.
- **Cost:** some prototypes are thrown away, and the "not in the same session" rule needs explicit wording for this case.
- **Check:** Gate 3's mechanism clauses each cite a prototype test, section sizes stay flat across drafts, and matched-time's projected 1.2–2.9 h per seed (ADR-013:431) is measured on one seed before drafting.

**P4. A standing subagent preamble.**
- **What changes:** a memory rule plus a file outside the repo. It lists the banned actions and every known location of official-mask statistics, is prepended verbatim to every subagent prompt, and is extended whenever a new location is found.
- **Cost:** maintaining the inventory.
- **Check:** every Gate-3 review prompt contains the preamble, and no reviewer reports incidental exposure.

**P5. Make the lab rule and practice agree.**
- **What changes:** one file of at most 10 lines per commit-day, with a "drafts/rounds today" line and lessons promoted to rules. Alternatively, amend the cap.
- **Cost:** less narrative in the lab entries.
- **Check:** during Gate 3 drafting, every commit-day has a file of at most 10 lines.

## 3. Evidence appendix

### A. Phase timeline

Times are local (+01:00).

| Phase (hash range) | Span | Commit-days | Commits: total; docs/code/tests/registry (docs-only) | Added: decisions/code/tests | Produced |
|---|---|---|---|---|---|
| A. Scaffold, ELL-1 Phase 0 (`4b90aae`–`2270bbe`) | 07-21 19:51 – 07-22 11:46 | 2 | 11; 5/6/5/2 (2) | 135/1,187/487 | ADR-001, -002; Gate-0 floor runs |
| B. ELL-1 Phases 1–3 (`7c3f763`–`b065779`) | 07-24 09:32 – 07-25 18:42 | 2 | 13; 12/9/4/4 (4) | 584/2,213/491 | Gate 0 verdict recorded; Gate 1 FAILED; Gate 3 PASSED; ADR-003–006 |
| C. Close-out, EXTRACT prep (`d00779a`–`d505c36`) | 07-26 03:36 – 07-27 03:23 | 2 | 6; 6/5/2/1 (1) | 569/907/482 | ADR-007, -008, -009; GCN reference run |
| *Gap* | 07-27 03:23 – 09-11 13:04 | 0 | — | — | — |
| D. DGF-1 Phase 0 (`0b73eb5`–`2e17089`) | 09-11 13:04 – 09-12 01:04 | 2 | 17; 12/8/5/2 (6) | 886/1,466/864 | ADR-010, -011; DataSource, floor, sampler |
| E. Gate-1 pre-registration to Gates 0/1 (`d5c2916`–`9859036`) | 09-12 19:55 – 09-14 22:52 | 3 | 23; 21/9/8/6 (12) | 473/1,924/753 | ADR-012; retune, pilot, gate runs; Gate 1 and Gate 0 PASSED |
| F. Post-gate (`70f61a4`–`becb85c`) | 09-14 23:02 – 09-15 21:19 | 2 | 20; 15/3/3/1 (15) | 2,209/331/483 | ADR-013 accepted; ADR-014 withdrawn; erratum; repro-check PASS; ADR-015 d1–2; `xaa` |
| *Gap* | 09-15 21:19 – 10-06 13:42 | 0 | — | — | — |
| G. ADR-015 v4–v15 (`8296f4b`–`15ae058`) | 10-06 13:42 – 10-07 11:46 | 2 | 14; 13/0/0/0 (13) | 2,518/0/0 | ADR-015 accepted; no runs |

**ADR-015 section growth, from v4 to v15, in lines:**

| Section | v4 | v15 |
|---|---|---|
| clause 5 | 24 | 339 |
| clause 7 | 22 | 193 |
| draft history | 17 | 130 |
| clause 3 | 22 | 120 |
| **total** | **289** | **1,277** |

### B. Revision triggers

Key: (i) design flaw · (ii) defect introduced by the previous fix · (iii) scope growth · (iv) factual or citation error · (v) reviewer false positive.

| ADR | Revisions | Triggers |
|---|---|---|
| ADR-012 (gated) | `d5c2916`, `b2a93b3` → accepted `c0f2053` | mostly (i) (seven design flaws), plus (iii) |
| ADR-013 | `70f61a4` → `4667dfe` → `593442b` → `f29cce7` → `5104f78` accepted; `8296f4b` | (iv), (i), then (i)×3 + (iii)×2 + (ii)×1 + (iv)×4, then (iv); later a stale (iv) |
| ADR-014 | `bd422c3`, `d7c453e`, `3a21d3b`, `6bbfbe1`, withdrawn `399dd7f` | (i)+(iv); (ii)+(iv); (i)+(iv); (ii)+(iv); root cause (iii), a speculative deadlock, and the check passed on its first run |
| ADR-015 (reported) | d1–d15 | (iv) d1–d3; (i)+(iii) d4; (ii) d5–d7; (iii)+(i) d8; (iii)+(v) d9; (ii) d10–d13; d14 clean; d15 accepted 8 minutes later |

**ADR-015 tally:** (ii) 7, (iv) 3, (iii) 2, (i) 1, plus one (v).

### C. Compliance notes

- No registry, data, scores or execution, and no repo writes.
- One home-directory memory file read (the stopping-rule memory).
- Metadata-level touches of ADR-010/011: `wc -l` and a Status grep.
- Official-split statistics met in lab 2026-07-27 (Session 11), and dataset-level label figures in lab 2026-07-26. Nothing was reproduced or used.

## Citation note (main session, 2026-10-07)

Mechanical range check: all 16 resolvable citations in range. In-range citations were not each content-checked.
