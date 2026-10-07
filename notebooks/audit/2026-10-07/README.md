# Research audit, 2026-10-07

**Status:** draft for coderback to edit. Nothing here changes a signed gate file, the registry, CLAUDE.md or `docs/`. Each item is an erratum to draft, a test to write, or a proposal for your decision.

**Why it ran:** you asked for it after ADR-015 was accepted (`15ae058`) and before any ADR-015 implementation. It was widened twice at your request: to the whole route back to ELL-1, then to the governing docs and the lightly-reviewed early ADRs.

**Method:** six parts, each run by a fresh read-only agent under the held-out prohibitions (see §6), then checked by the main session.

| part | object | report |
|---|---|---|
| A | DGF-1 evidence chain: ADR-010–014, GATE-DGF1-0/1, ERRATUM-DGF1-1, trainer, runners, assemblers, guards | [A-dgf1-integrity.md](A-dgf1-integrity.md) |
| A2 | early ADRs, ADR-001–009, live ones first (005, 007, 008) | [A2-early-adrs.md](A2-early-adrs.md) |
| B | route: ELL-1 → EXTRACT → DGF-1 → EDR-1/EDL-1/GDE | [B-route-review.md](B-route-review.md) |
| C | process retrospective from git, lab entries and ADR draft histories | [C-process-retrospective.md](C-process-retrospective.md) |
| D | the governing docs themselves: design, statistics, consistency, rule cost, outside standards | [D-governing-docs.md](D-governing-docs.md) |

**What the main session verified:**
- **Mutation tests:** the 15 leakage or integrity edits were run on a scratch `git archive` copy of HEAD, which has no `data/`. Two control mutations were caught, so the harness works.
- **Statistics:** D's arithmetic was reproduced by an independent Monte Carlo.
- **Citations:** every `path:line` citation in the reports was range-checked by script. 13 were out of range (9 in A, 2 in B, 2 in D) and are corrected at the foot of each report. In-range citations were **not** each content-checked, so treat them as pointers.
- **This file** cites only lines the main session read itself.

## Bottom line

1. **Nothing found overturns a signed verdict or number.** These all hold:
   - ELL-1 Gate 1 FAILED and Gate 3 PASSED;
   - DGF-1 Gates 0 and 1 PASSED;
   - the seed-count derivation, the reproduction claims and the gate arithmetic, all recomputed.
2. **The guide's integrity core is sound:** pre-registration enforced in code, determinism, the parity floor, the ROC-AUC + AUPRC pair, edge-dated training graphs, and the append-only registry (D, "What is sound").
3. **The guards are weaker than the rule "untested guards don't exist" assumes.** 15 edits that break a guard leave the test suite green. The code is currently correct: the DGF-1 trainer's wiring lines are unchanged since the Gate 1 runs (`git diff 09c6e72 HEAD`).
4. **The statistical rules need fixing before Gate 3.** Gate 1 is unaffected.
5. **Process: effort after Gate 1 went into prose, not evidence.** Since `202c537` (Gate 0 signed) there have been 35 commits and one run. ADR-015 took 15 drafts. Part of the contrast with earlier ADRs is that review intensity rose over time (§5).

## 1. Before any further gated run

These are the prerequisites for Gate 3's pre-registration, and for re-using the trainer in ADR-015 or matched-time.

| # | item | source | evidence the main session checked |
|---|---|---|---|
| 1 | **Pin the trainer and runner wiring with tests that fail under mutation.** All of these leave the suite green: (a) training on every labelled user; (b) training inputs from the graph as of the test window; (c) scoring with every edge; ELL-1 eval as a full-graph forward; ELL-1 training on all edges; the HPO objective scoring the test window. | A-B1, A2 E-S6/E-S7 | `adapters/dgf1/train_gnn.py:75-81`, `:121`; `tests/test_dgf1_trainer.py:46-63`; mutation runs (118-test DGF-1 subset for A; full 244-test suite for A2) |
| 2 | **Fix the EXTRACT checker before it certifies any further `gbe/` change.** It takes the latest row per `(arm, seed)` from any commit and exits 0 when there's nothing to compare. | A2 E-B3 | `scripts/check_extract_regression.py:231-246`, `:305-310`; M4 not caught |
| 3 | **Record determinism truthfully.** The row's `deterministic` field is read right after it's forced on, so it is constant True. Strict mode (not `warn_only`) is neither tested nor recorded. | A2 E-B1/E-B2 | `gbe/run/session.py:56-61`; M1 and M2 not caught |
| 4 | **One shared, tested resolvability function** (`ddof=1`, `diff > 2·SE_diff`). There are four untested copies today, and Gate 3's assembler must import the shared one. | A2 E-B4 | M5a/b/c not caught |
| 5 | **Gate statistics before Gate 3:**<br>• state α and power;<br>• a two-look boundary of about 2.23 instead of 2 at both looks;<br>• a seed floor of 5 with the Welch critical value. | D2/D4, D-P1 | Monte Carlo: z>2 at both looks gives α 0.0397 vs 0.0228 for one look, and a common boundary of 2.23 restores 0.023. With 3 seeds the false-positive rate is 0.058–0.092. |
| 6 | **Gate 3's content:**<br>• compare GNN vs GNN-removed (message passing) **and** GNN-removed vs floor (model class), in one batch;<br>• define scramble and random for dated edges;<br>• name the function that evaluates the criteria. | B-B1, D10, A2 | `docs/02-*:177-178`, `:265` vs `gates/GATE-DGF1-1.md:205-207`, `:222` |
| 7 | **The shared pre-registration guard's identity fix** (already owed). | ADR-015 | — |
| 8 | Smaller test gaps:<br>• the `git_dirty()` wrapper;<br>• optimizer-state restore on checkpoint;<br>• `CUBLAS_WORKSPACE_CONFIG` set at import;<br>• the BatchNorm lint covers only `gbe/gnn`;<br>• the domain-import lint checks top-level names only. | A2 E-S8/E-N8, A S7/S8 | M8, M9, M3 not caught |

## 2. Errata to draft (each sits beside its file; no verdict changes)

| erratum | content | source |
|---|---|---|
| ERRATUM-ELL1-1 | **Gate 1's stated cause** (`GATE-ELL1-1:69-71`) is superseded by the local-94 result. **Correct B's draft** to "71 of the 72 aggregates were removed; one remained", per the ADR-001 column slip. | B-S3, A2 E-S1 |
| ERRATUM-DGF1-1 addendum | **The ordering guard checks the wrong commit.** It compares rows with their own commit (`scripts/assemble_gate_dgf1_1.py:165`, `:209-216`), while the ordering fact itself is true via `09c6e72`. **The clause-7 facts are hard-coded** under a footer that says "computed". **The sentinel claim** rests on training users only. **The disclosure of prior looks is incomplete.** | A S1–S3, N4 |
| ADR-001 | **The local/aggregate split is off by one.** `ADR-001:21-22` says Weber's 94 local features include `time_step`; `:57-58` takes columns 2..95 with `time_step` excluded, so they include one aggregate. | A2 E-S1 |
| ADR-005 | **Four errors:**<br>• the determinism field is constant;<br>• strictness is unrecorded;<br>• its seed-count example (ADR-006) wasn't derived from a deterministic batch;<br>• "σ ≈ 0.034" is a range (sd 0.017). | A2 |
| ADR-006/007 | **The criterion pins are untested** (see §1 item 4). **"AUPRC has no published DGraph comparator" is false:** GADBench reports it. **Davis & Goadrich is misattributed.** | A2 E-S4/E-S5 |
| ADR-008 | **The checker doesn't implement the no-retry or commit binding,** the driver pin isn't checked, and the "run noise" context is unsupported. | A2 |
| ADR-011 | **`:282` "the same bar ELL-1's RF floor set" is not true.** ELL-1's tree had neighbour-feature aggregates; DGF-1's has node-level counts only. | B-S1 |
| ADR-013 | **`:295-298` still says "until ADR-014 is accepted",** but ADR-014 was withdrawn. | A-S4 |
| ADR-015 | **`:791-797` relies on ADR-014:60-62's "Verified" unhashed channel,** which was checked on uncommitted code and exists in no commit (`git log -S extra_metrics` is empty). The implementation must establish it by test. | A-S5 |
| Elliptic leakage citation | **The "0.807→≈0.12" figure (doc-00, doc-01, ADR-004) may misread arXiv:2604.19514.** Check it against the paper before correcting. | B-S2 |

## 3. Amendments to the governing docs (only on your instruction)

- **BLOCKING:**
  - **Failure semantics.** `docs/research-plan-UNIFIED-GBE-GDE.md:82` and `docs/01-*:23` tell the write-up to carry the sentence about structure that CLAUDE.md forbids, and ELL-1's Gate 3 makes that sentence false (D1).
  - **EDR-1's start condition** (`docs/03-*:8`, "cleared their gates") contradicts "EDR-1 proceeds regardless" (`docs/01-*:23`; research plan `:82`) (B-B2).
- **One consistency ADR before EDR-1** covering D15's list:
  - doc-03's encoder default;
  - the SIC universe vs the sector slice;
  - EDR-1's Phase-2 ordering;
  - SCM-1's two Gate 1 statements;
  - missing GDE documents;
  - doc-00's stale "166" and GCN wording;
  - "toward baseline" wording;
  - stale statuses (A-N6);
  - CLAUDE.md's missing Gate-1 qualification 4 (D7);
  - one seed floor (D4).
- **Version `docs/` and CLAUDE.md.** Both are gitignored today, so no amendment has a history. Raised independently by A (S10), B (S7) and D.

## 4. Decisions for you

### 4a. Research design: each would be its own ADR

| proposal | what it fixes | source | my view |
|---|---|---|---|
| Gate statistics v2 | §1 item 5; seed-only variance on one window (D3) | D-P1 | Accept before Gate 3's ADR. |
| Failure semantics and falsification | D1; the thesis has no refutation condition (D5) | D-P3 | Accept. It also resolves the D1 contradiction. |
| Leakage model sheet + test-window ledger + a definition of "peek" | Kapoor types missing for EDR-1: serial fraud, pre-event text, encoder lookahead, duplicates (D6–D8) | D-P2 | Accept before EDR-1's Phase-0 ADRs. The ledger part also fixes A-S3's incomplete disclosures. |
| Gate-2 ADR before Phase 2 | Gate 2 has no metric and no pre-registration (D9) | D | Accept. It is needed only when Gate 2 is scheduled. |

### 4b. Process: C and D reconciled (they converge independently on 1–5)

| # | proposal | sources | my view |
|---|---|---|---|
| 1 | **Two ADR tiers.**<br>• **Tier A:** criteria, splits, metrics, held-out access. Full review, then diff passes, with a round budget (C suggests 4).<br>• **Tier B:** reported-only work and tooling. One page, one review.<br>• The stopping rule becomes standing, and an exhausted budget escalates to you. | C-P1, D-P4d | Accept. The budget number is yours. |
| 2 | **A mechanical evidence check before any document goes to you:**<br>• quotes match their cited lines;<br>• cited commits exist;<br>• every "verified" names a commit or run id;<br>• cited identifiers exist in git;<br>• statuses match. | C-P2 | Accept. Today's range check alone found 13 bad citations in the agents' reports. |
| 3 | **Measure or prototype first:** a cheap label-free check, or a scratch prototype with mutation tests, before writing rules about a feared failure. CLAUDE.md needs a carve-out so a scratch prototype doesn't count as same-session implementation. | C-P3, D-P4c, D14 | Accept. ADR-014's four drafts concerned a check that then passed in 572 s. |
| 4 | **A standing subagent preamble:** the banned actions plus an inventory of every place official-mask statistics appear, kept current. | C-P4, D-P2 | Accept. I missed a location four times today (§6). |
| 5 | **Lab rule:** a ≤10-line daily summary plus an optional log, or amend the cap. 7 of 8 lab files exceed 10 lines, and there are no entries for 10-06 or 10-07. | C-P5, D-P4e | Your call. |
| 6 | **Retire or redefine the ceremony rules:**<br>• "one section per session" becomes one logical change per commit series;<br>• define "session";<br>• the core inclusion rule becomes "domain-agnostic, at least 2 named consumers, tested". | D-P4a/b/f, D12–D13 | Accept the definitions. Retiring the rules is your call. |
| 7 | **A dated plan:** planned dates per remaining gate, a timebox per pre-registration, EDR-1's ETL timebox N filled in, and a dated DGF-1 exit decision. | D-P5, D16 | Accept. Nothing can currently signal slippage. |

### 4c. Direction (B's options)

1. As planned.
2. Narrow DGF-1.
3. Close DGF-1 at Gate 1 and start EDR-1.
4. Re-attempt ELL-1, labelled post-hoc.

**My recommendation is option 2:**
1. the §1 prerequisites;
2. Gate 3 with the decomposition;
3. decline matched-time with a dated reason;
4. ADR-015 when cheap;
5. fix EDR-1's start condition, then start EDR-1 Phase 0 under the leakage sheet.

The model-class answer is the DGF-1 result that EDR-1's floor design and P0 depend on. B rates option 4 low-value, because option 2 tests neural vs tree on DGraph without touching Elliptic again.

## 5. Corrections and caveats

- **C's "effort ran inverse to stakes" is confounded.**
  - **Review intensity rose over time:**
    - ADR-001, 002, 003, 004, 008, 009 and 010 have no review on record. For ADR-004, lab 07-24:9 records only a pause "for review" before the Gate 1 runs;
    - ADR-005 and 006 had one adversarial self-review by the assistant, in session (lab 07-24:52), and ADR-007 was "drafted, reviewed and accepted" in one session (lab 07-26:3);
    - ADR-011 and 012 had two subagent rounds;
    - ADR-013 to 015 had many.
  - **Early reviews happened inside a single commit,** while ADR-015 committed every draft.
  - **What survives:** ADR-011/012 were reviewed independently and still converged in two rounds. ADR-015's 7 fix-introduced defects are a real pattern. You raised this; A2 was run because of it.
- **My errors today:**
  - I said "15 drafts over 22 days". Git shows drafts 4–15 between 10-06 13:42 and 10-07 11:46, with no commits 09-16 to 10-05.
  - I named `becb85c` as a draft; it is the stray `xaa` copy.
  - I drafted the ADR-015 clause that rests on ADR-014's uncommitted check.
- **Agent errors:**
  - 13 out-of-range citations (corrected in each report);
  - B's GATE-ELL1-1 line numbers;
  - B's pace finding assumed three weeks of drafting;
  - C's count of commits since `202c537` (34; git says 35).
- **Literature:** several sources were read through a summarising fetch tool (Weber, Maganti, GADBench, Kapoor, and arXiv:2604.19514). Re-check against the PDFs before anything is cited in a paper.

## 6. Exposure ledger for this audit

- **No data, score files or held-out labels were read.** No official-test statistic was computed. A read 31 DGF-1 validation-window rows through a filter script; nobody else read the registry.
- **Label-free official-mask material was seen incidentally,** because my prompts' filters missed it. Nothing was used or reproduced:

  | agent | what it saw |
  |---|---|
  | B | doc-02 ~69-77, node-time quantiles and sizes |
  | A | doc-02:87 (size-match percentage); ADR-015:202 (labelled counts per mask) |
  | C | lab 2026-07-27:18 (node-time medians) |
  | D | `wc` passed ADR-010/011 through, with totals only displayed |

- **Known locations,** now in the standing inventory: ADR-010:35-40, ADR-011:197 and 202-203, doc-02:60-90, lab 2026-07-27:18, ADR-015:202.
- **Dataset-level totals** (whole-graph fraud count and prevalence) and **published Gate-1 test-window figures** appear in many docs and were read as normal.
- **Mutation runs used a scratch export without `data/`.** No repo file was modified.

## 7. Keep

Pre-registration enforced in code; determinism by default; ADR-008's equality bar (once its checker is fixed); ADR-007's metric pair; the parity floor with equal tuning budget; edge-dated training graphs; within-model ablations; errata instead of edits; the researcher signing every gate. D gives the evidence for each.
