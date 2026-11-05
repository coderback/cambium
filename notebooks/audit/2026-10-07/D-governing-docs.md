# Audit D: governing documents (subagent output, saved by the main session, 2026-10-07)

> Main-session verification notes:
> - **D1 CONFIRMED.** The research plan:82 failure semantics tell the write-up to carry, into P0/P1, the very sentence CLAUDE.md forbids about structure (doc-01:23 likewise).
> - **D2 and D4 arithmetic CONFIRMED** by an independent Monte Carlo (2M draws, ρ = √(8/20)):
>   - single-look α at z>2 = 0.0228;
>   - two looks at z>2 = 0.0397;
>   - a common boundary of 2.23 restores α = 0.023;
>   - Welch t sf(2): df=4 → 0.0581, df=2 → 0.0918;
>   - power at effect = 2SE is 0.5.
> - **Nuance on D2.** "50% stage-1 power" is the rule's worst case, at the boundary where 2·SE_diff = ½Δ_val. DGF-1 Gate 1's realised n=8 sat well inside it (audit A: 2·SE_diff(8) = 0.00582 vs ½Δ_val = 0.01847 ROC-AUC; 0.00126 vs 0.00402 AUPRC), and Gate 1 needed no stage 2. So D2 is a rule-design issue for Gate 3 onwards, not a defect in Gate 1's verdict.
> - **D's proposals converge with C's:** tiered ADRs ≈ C-P1; measure first ≈ C-P3; the held-out ban in every subagent prompt plus a ledger ≈ C-P4; lab ≈ C-P5. These need reconciling in the combined report.
> - **D's own disclosures:** `wc` pipelines passed ADR-010/011 content through (only totals displayed); it saw temporal test-window counts in gate files and the timeline (not reproduced).

## Summary

**Q1. Research design: sound at the gate level, weaker at the programme level.**
- **Sound:** blind pre-registration enforced in code, determinism, the parity floor with equal tuning budget, and the jointly gated metric pair.
- **The thesis has no refutation condition.** ELL-1's and DGF-1's Gate-1 questions are framed as thesis tests, yet a failure on them is declared in advance not to count against the thesis.
- **Gate statistics:** they model seed variance only, on one window and one cutoff, and the seed-count and two-stage rules have unstated error rates and power.
- **Leakage rules:** strong on edges and preprocessing, but three Kapoor types are missing, and those gaps bear on EDR-1.

**Q2. Consistency.**
- One binding contradiction: the failure semantics tell you to write the sentence CLAUDE.md forbids.
- About a dozen inconsistencies, mostly in docs not yet executed (EDR-1, its Phase-2 doc, SCM-1, GDE references) and in the seed floors.

**Q3. Process rules.**
- **Keep:** the integrity rules (determinism, append-only registry, pre-registration, the researcher signing every gate, ADR-008's equality bar).
- **Retire, replace or tier:** the ceremony rules (one section per session, the ≤10-line lab entry, the same-session ban, an ADR for everything). They are routinely unmet or only nominally met, and they were the main cost after Gate 1.

**Q4. Against outside standards.**
- **Stricter than the field:** determinism and pre-registration.
- **Laxer:** sources of variance, temporal robustness, duplicate and entity leakage, and lookahead bias from pretrained models.

**Q5. Feasibility.** No planned dates exist, so miscalibration cannot be measured.
- Build work was fast: ELL-1 took 4 commit-days, and DGF-1 through Gate 1 took 4.
- In the 23 days since Gate 1 there was no gate batch.

**Counts:** 1 BLOCKING, 15 SHOULD-FIX, 5 NOTE.

## Findings

| id | sev | status | claim | evidence | remedy |
|---|---|---|---|---|---|
| D1 | **BLOCKING** | CONFIRMED | **The failure semantics tell you to write, into P0/P1, the sentence CLAUDE.md forbids.** GATE-ELL1-3's within-model result also makes that sentence false. | research plan:82; doc-01:23; CLAUDE.md:71-75 | Replace both with CLAUDE.md's two-part wording. |
| D2 | SHOULD-FIX | CONFIRMED | **The seed-count rule and two-stage design have unstated error rates and power.**<br>• At the planned effect ½Δ_val, stage-1 power is 50% (about 0.89 with the 8→20 stage 2).<br>• Using the same z=2 at both looks gives one-sided α ≈ 0.040, against 0.023 for a single look.<br>• ADR-006 calls this "group-sequential", yet the boundary is never adjusted.<br>• Gate 3 inherits all of this. | ADR-012:240-246, :457-458; ADR-006:118-124; doc-02:275-276 | Gate 3's ADR states α and power before any run. Use a common boundary of ≈2.23, or a Pocock-type boundary. Size n for 80% power (2·SE ≤ 0.35·Δ_val). |
| D3 | SHOULD-FIX | CONFIRMED | **Gates ignore every source of variance except seeds.**<br>• Seeds only, on one window and one cutoff; the user bootstrap is reported, not gated.<br>• The reason given (comparability with ELL-1) does not hold, because ELL-1 already used two different statistics.<br>• This becomes load-bearing at EDR-1, where positive counts will be small. | ADR-012:300-307, :386-398; ADR-004:80-84; ADR-006:60-63; Bouthillier 2021 (arXiv:2103.03098) | P1 |
| D4 | SHOULD-FIX | CONFIRMED | **The seed floor is stated four ways, and the constitutional ≥3 is unsafe.**<br>• ≥3 in CLAUDE.md and doc-00; ≥5 in doc-02:269 and ADR-006; 8 in doc-02:276 and ADR-012.<br>• At n=3 the 2×SE rule's false-positive rate is 0.058–0.092 against 0.023 nominal.<br>• ADR-006 rejects 3, yet ADR-015:338 cites 3 as its authority. | CLAUDE.md:87-94; doc-00:164; doc-02:267-276; ADR-006:126, :168-169; ADR-015:338 | Floor of 5 with the Welch critical value. |
| D5 | SHOULD-FIX | CONFIRMED | **The thesis has no refutation condition.** ELL-1/DGF-1 Gate 1 are framed as thesis tests but exempt in advance. GDE proceeds after an EDR-1 failure. Only EDR-1 Gate 1 can answer the GBE half "no". | research plan:13-15, :81-84; doc-01:6, :14 vs :23; doc-02:24 | P3 |
| D6 | SHOULD-FIX (BLOCKING at EDR-1 Phase 0) | CONFIRMED (absence) | **The leakage rules miss four Kapoor types:**<br>• duplicates and near-duplicates;<br>• serial fraud and firm persistence; Bao's code handles serial fraud and leaves a one-year gap;<br>• pre-cutoff text that discloses an investigation; doc-05 excludes pre-event market and concealment features, but doc-03 has nothing for text;<br>• no lookahead guard for pretrained encoders in EDR-1 or RDM-1, though SCM-1 has one. | doc-03:63, :76-89; doc-05:46, :67; SCM-1:82; RDM-1:105; Kapoor & Narayanan (arXiv:2207.07048); github.com/JarFraud/FraudDetection; arXiv:2502.21206 | P2 |
| D7 | SHOULD-FIX | CONFIRMED | **CLAUDE.md's leakage-test rule cannot be met for undated labels or snapshot features (DGF-1).** CLAUDE.md also omits the gate's qualification 4 (seed-only estimand; snapshot features). | CLAUDE.md:121-123, :50-57; GATE-DGF1-1:215-217 | Reword to "a test, or a documented limitation carried into every gate file". Add qualification 4. |
| D8 | SHOULD-FIX | CONFIRMED | **"Peek" is undefined, and the rule learned from a real breach never reached the constitution.** The post-look rule and the subagent ban live only in ADR-011 and the lab notes. | CLAUDE.md:102; lab 09-11:12, :15 | P2 |
| D9 | SHOULD-FIX | CONFIRMED | **Gate 2 is unblocked but has no metric and no pre-registration requirement.** | CLAUDE.md:48; doc-02:174, :264; ADR-007:104-106; ADR-012:400-401 | A Gate-2 ADR before Phase 2 runs. |
| D10 | SHOULD-FIX | CONFIRMED (gap); PLAUSIBLE (impact) | **Gate 3's scramble and random arms are defined for ELL-1's geometry.** Rewiring is "within each time step". DGF-1 has dated edges, and its view-derived degree, type and recency inputs reach both arms. Whether scrambled arms recompute those inputs decides what Gate 3 tests, and doc-02 is silent. | ADR-006:83-95; doc-00:155-157; doc-02:176-184 | P6 |
| D11 | SHOULD-FIX | CONFIRMED | **The official-split track is oversold.** doc-02 promises a comparison with fraud-specialised GNNs; ADR-015 runs three arms under its own definition. GADBench tunes each model by random search over 10 trials. | doc-02:196-199, :220; ADR-015:27, :281; arXiv:2306.12251 | Reword doc-02:220 to "reported beside published numbers, labelled cross-protocol". |
| D12 | SHOULD-FIX | CONFIRMED | **The core inclusion rule cannot be satisfied as written.** It needs all four models, most not yet built. doc-00 itself places `gbe.serve` and degree correction in the core, and the GDE models import `gbe/`. | doc-00:14, :106, :166; CLAUDE.md:106 | P4(f) |
| D13 | SHOULD-FIX | CONFIRMED | **The session-scoped rules conflict or are only nominal.**<br>• "One section per session" conflicts with multi-section amendments being applied on acceptance.<br>• "Session" is undefined: ADR-012 was drafted, accepted and implemented on one day.<br>• The lab entries run long because the lab carries the changelog for gitignored docs. | CLAUDE.md:138, :141; ADR-000:9-11; lab 09-11 | P4 |
| D14 | SHOULD-FIX | CONFIRMED | **Doc-first has no carve-out for measuring first.** ADR-005's CPU draft, and ADR-014's four drafts about a check nobody had run; it took about 10 minutes. | ADR-005:11-17; ADR-014:29-30; lab 09-15:83-110 | P4(c) |
| D15 | SHOULD-FIX | CONFIRMED | **Docs not yet executed contradict themselves:**<br>(a) doc-03's encoder default;<br>(b) a universe of one or two SIC groups vs the sector-held-out slice;<br>(c) EDR-1's Phase-2 ordering, between doc-05 and doc-03/timeline;<br>(d) SCM-1's Gate 1 stated two ways;<br>(e) a missing "GDE shared-core guide" and fusion-decision document;<br>(f) gates in `gbe.eval` vs gbe/CLAUDE.md;<br>(g) doc-00's 166 features vs ADR-001, and GCN allowed vs not;<br>(h) "toward baseline/floor" in doc-00:176 and doc-01:128, the source doc-03 and SCM-1 inherit. | as cited in the subagent message | One consistency ADR before EDR-1 starts. |
| D16 | SHOULD-FIX | CONFIRMED | **The plan has no dates.** EDR-1's ETL timebox N is unfilled, and nothing can signal slippage. | timeline.md; doc-00:190; doc-03:43; research plan:138 | P5 |
| D17 | NOTE | CONFIRMED | **Paper claims outrun the gates** (P2's "Pareto … competitive with SOTA"; P0's protocol shipping before any link task). | SCM-1:26; research plan:35 | Gate or narrow each claim. |
| D18 | NOTE | CONFIRMED | **Unblinded self-judgement of candidates.** | doc-04:97; RDM-1:115 | Blind the rating, and interleave baseline candidates. |
| D19 | NOTE | PLAUSIBLE | **Hard negatives shift the test distribution** (Kapoor L3.3). | doc-00:159; doc-03:89 | Report beside the natural distribution, with an independent selector. |
| D20 | NOTE | CONFIRMED | **The templates lag practice.**<br>• The gate template lacks a pre-registration reference, prior-look disclosure, estimand, deviations section and leakage sheet.<br>• The ADR template lacks Disclosure and Supersedes fields and any length limit. | GATE-TEMPLATE:10-14; ADR-000 | P2 |
| D21 | NOTE | CONFIRMED | **Nothing ties the signed verdict to the mechanical check.** | GATE-TEMPLATE:26-27; ADR-004:105-107 | Any departure from the mechanical result requires an ADR. |

## What is sound

- **Pre-registration enforced in code** (seed count committed 33 s before the first test row).
- **Determinism by default.**
  - **Catch:** three repeats spanned 0.034 F1.
  - **Cost:** about 5%.
  - **What it enabled:** ADR-008's 49/49 check and the repro-check.
- **ADR-008's equality bar.**
- **ADR-007's metric pair:** decided blind, and it matches GADBench.
- **The parity floor and equal tuning budget.**
- **Edge-dated training graphs:** a node-time check would have missed 399,366 post-cutoff edges.
- **Within-model ablations,** with the correlation disclosed.
- **The estimand is stated.**
- **Declining the matched-time study cannot upgrade a claim.**
- **Snapshot verifiers run before any code.**
- **Overclaims were caught before signing,** and errata are used instead of edits.

## Proposals

**P1. ADR "Gate statistics v2", accepted before Gate 3's ADR.**
- **What changes:**
  - gate on a Welch-critical seed test **and** a hierarchical user×seed bootstrap CI that excludes zero;
  - adjusted two-look boundaries (≈2.23);
  - n sized for 80% stage-1 power;
  - a floor of 5;
  - for EDR-1, at least two rolling-origin cutoffs.
- **Cost:** more seeds.
- **How to tell it worked:** every pre-registration prints an α and power table.

**P2. A leakage model sheet and a test-window ledger.**
- **What changes:**
  - each DataSource answers Kapoor's eight types, with a test or a documented limitation;
  - "peek" gets three tiers (structure, label statistics, scores);
  - an append-only ledger is written by the pre-registration guard, and by hand for subagent looks;
  - the held-out ban goes into every subagent prompt;
  - EDR-1's Phase-0 ADRs cover serial fraud and a train/test gap, pre-event text exclusion, firm non-independence, and encoder lookahead.
- **How to tell it worked:** the next gate's disclosure is generated from the ledger.

**P3. ADR "Failure semantics and falsification".**
- **What changes:**
  - fix D1;
  - relabel ELL-1/DGF-1 Gate 1 as engine/replication gates;
  - add a refutation clause: a powered EDR-1 Gate 1 and Gate 3 failure answers the GBE half "no" for corporate graph-text, and GDE's start must then be re-justified in writing;
  - EDR-1's Phase-0 ADR shows the test window is powered.

**P4. Right-size the process section of CLAUDE.md.**
- **What changes:**
  - (a) retire "one section per session" in favour of one logical change per commit series;
  - (b) session = calendar day, with an overnight gap between acceptance and implementation for Tier-A ADRs only;
  - (c) measure first: run a label-free validation check of 30 min or less before drafting an ADR that depends on its outcome;
  - (d) two ADR tiers. Tier A covers criteria, splits, metrics and held-out access, with full review and the stopping rule. Tier B covers reported-not-gated work and tooling: one page and one review;
  - (e) a ≤10-line daily summary plus an optional log, and version `docs/`;
  - (f) a new inclusion rule: domain-agnostic, at least 2 current or doc-named consumers, tested.
- **How to tell it worked:** drafts per Tier-B ADR, and days from a gate to the next batch.

**P5. A dated plan in `timeline.md`.**
- **What changes:** planned dates for each remaining gate, a fixed timebox for each pre-registration (scope shrinks, the timebox doesn't move), EDR-1's N filled in, and a dated DGF-1 exit decision (audit B's options).
- **How to tell it worked:** slip is measured per milestone.

**P6. Content requirements for Gate 3 and Gate 2.**
- **What changes:**
  - define scramble and random for dated edges: keep the edge dates, and run both recomputed and frozen derived inputs, gating on one named in advance;
  - include audit B's B1 decomposition;
  - a Gate-2 ADR comes before Phase 2.

## Could not check

- **Prohibition breach (disclosure).** `wc` passed ADR-010/011 content through, but only totals were displayed.
- **Not read:** doc-02:60-90, ADR-015:190-239, ADR-010 and ADR-011; lab 07-27:18 was skipped.
- **The statistical figures are rule arithmetic, not data.**
- **Sources only partly verified:** Bao (README only), Sarkar & Vafa (search summaries), GADBench and Kapoor (summarising fetch), Grinsztajn and Agarwal (search results).
- **`docs/` is untracked,** so its history cannot be audited.

## Citation corrections (main session, 2026-10-07)

- `ADR-004:80-84` and `:105-107` are out of range (file has 62 lines). The band test is at `ADR-004:22-27`, and the mechanical-check clause at `:49-51`.
In-range citations were not each content-checked; treat them as pointers.
