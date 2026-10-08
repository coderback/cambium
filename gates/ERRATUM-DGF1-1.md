# ERRATUM-DGF1-1 — withdrawn passages in GATE-DGF1-1

**Dated:** 2026-09-15
**Decision:** `decisions/ADR-013-dgf1-sensitivity-view-inputs.md` (accepted 2026-09-15), clause 2
**Concerns:** `gates/GATE-DGF1-1.md`, signed PASSED 2026-09-14. That file is **not edited**; dated
gate files never are.

**This is not a gate file and carries no verdict.** Gate 1's gated result is unaffected: every gated
number, the clause-6 check, the paired bootstrap and the verdict's PASS stand exactly as recorded.

## What is withdrawn, and why

GATE-DGF1-1 reports DGF-1 under two sensitivity views, window-only and first appearance (ADR-011
clause 4). The code built DGF-1's node inputs under **both** views from the graph as of step 821:
the edge-type histogram, degree and recency. That breaks ADR-011 clause 3, which requires every
edge-derived feature to come from the edge set of the view it feeds. The numbers are exactly what
the code computed, but they do not measure what their labels say.

| GATE-DGF1-1 passage | lines | status |
|---|---|---|
| *Scoring-view sensitivity*: the window-only and first-appearance rows, and the implementation-gap note beneath them | 133–141 | **withdrawn**: cite them for no inference |
| *Verdict*, qualification 3: the first-appearance and gated AUPRC figures it compares | 209–213 | **the figures are withdrawn**; the qualification's conclusion, that the question is unsettled, stands; its closing "that pre-registered comparison … is owed" is **superseded** — the comparison moves to ADR-013 clause 5, and no run is pending under ADR-011 |
| *Verdict*, claims table: "the gain does not depend on activity after users appear — **not established**" | 224 | **the status stands**; its stated reason ("floor unscored under the first-appearance view") is superseded by: no valid row bears on the claim |

The registry keys behind these passages are withdrawn too: `window_only_*` and `first_appearance_*`
in all 24 DGF-1 GNN rows (retune, pilot and gate). The rows stay, because the registry is
append-only.

## How to state the claim now

> **Post-appearance activity: not established.** The first-appearance figure cited in the verdict is
> withdrawn (ADR-013); no valid row bears on this claim.

That holds until the matched-time pre-registration in ADR-013 clause 5 produces rows. Any write-up
drawing on GATE-DGF1-1 reads this notice with it.

## Addendum, 2026-10-08 (proposed): how GATE-DGF1-1 describes its own assembly

**Source:** research audit 2026-10-07, §2 (`notebooks/audit/2026-10-07/README.md`; findings A S1–S3
and N4). Nothing above changes, and no verdict changes: every gated number, the clause-6 check, the
paired bootstrap and the PASS stand.

| GATE-DGF1-1 passage | lines | status |
|---|---|---|
| the ordering sentence: "this assembly refuses to run if that ever stops being true" | 8 | **overstated.** The check compares the first test row with the commit those rows themselves record (`scripts/assemble_gate_dgf1_1.py:165`, `:209-216`). A row cannot predate its own commit, and nothing checks that this commit introduced the seed line. **The ordering itself is true,** shown by git: `09c6e72^` carries the placeholder; `09c6e72`, committed 00:37:15Z, is the only commit that introduced `**Stage-1 seeds:** 8` (`git log -S`); the first test-window row started at 00:37:47Z. |
| the disclosure of what had been seen before the batch | 11 | **incomplete.** The full list is below. |
| the clause-7 facts: the validation prevalence, the age ranges, edge type 8's count and dates, the sentinel's share | 145–148 | **typed, not computed.** They are string literals in the assembler (`scripts/assemble_gate_dgf1_1.py:386-395`), taken from ADR-011's window table and lab 2026-09-11 Sessions 14–15. Only line 145's test-window prevalence and counts are computed from the rows. |
| the sentinel "lies strictly below every observed value" | 148 | **profiled on training-window users only** (lab 2026-09-11:64-65), and stated here for all users. |
| "Scoring-view sensitivity … reported above" | 149 | **points at withdrawn rows** (this notice, above). The paired bootstrap it also points at stands. |
| "All numbers computed by `scripts/assemble_gate_dgf1_1.py` from `experiments/registry.csv` …" | 180 | **not true of the typed facts** on lines 145–148. |

**Seen before the Gate-1 batch, in full.** No DGF-1 model score on 482–821 existed before the batch.
All of the following did, and line 11 names only the first two:
1. **Validation:** both retune grids and both pilots, on 370–481, on which both arms were selected.
2. **Test-window label statistics, from ADR-011's review:** a red-team subagent's fraud-vs-normal
   degree (ADR-012:75-79).
3. **Test-window label statistics, from ADR-011's own measurement:** the window's support, fraud
   count and prevalence (`scripts/measure_dgf1_temporal_split.py`; disclosed at ADR-011:78-84). They
   are pinned and re-checked by `scripts/audit_dgf1_datasource.py`. ADR-011 records that prevalence
   was not an input to the cutoffs (:83-84).
4. **Label-free test-window facts that ADR-011 or ADR-012 disclose:**
   - the train view matched against the test view, for role-matching (lab 2026-09-11:13;
     ADR-011:111-112);
   - edge type 8 occurring only in the test window, with its count and dates (lab 2026-09-11:68;
     ADR-012:72-73);
   - the age ranges of training, validation and test users at scoring (lab 2026-09-11:57), which
     ADR-012:74 discloses as "the val-window age profile".
5. **Label-free test-window facts that neither ADR discloses:**
   - the largest standardised input over test targets, under the inherited clamp and under the fix,
     and how many test targets carry a type-8 edge (lab 2026-09-11:69). This look led to the
     zero-variance column rule in `gbe/features/scaling.py`, which applies to both arms;
   - the test window's edge volume relative to validation, used for a timing projection (lab
     2026-09-11:272).

None of these is a model score, and none set a threshold or seed count. ADR-012 fixed its criteria
blind (ADR-012:68-70), and the seed count came from the validation pilots. The test-window ledger
(plan row 5a) is backfilled from this list.

**One passage in GATE-DGF1-0 rests on withdrawn keys.** Lines 58–59 identify two pre-echo rows by
"all 18 of its float metrics", and 12 of those 18 are the `window_only_*` and `first_appearance_*`
keys withdrawn above. On 2026-10-08 the three validation rows were re-checked with no value printed:
the 6 remaining floats (five metrics and the prevalence) also match exactly, so the identification
stands. The function behind it, `identical_twin` (`scripts/assemble_gate_dgf1_0.py:157-164`),
still compares the withdrawn keys, which ADR-013 clause 2 forbids from 2026-09-15 on. Any
re-assembly of GATE-DGF1-0 must exclude them first.
