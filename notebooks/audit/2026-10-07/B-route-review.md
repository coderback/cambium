# Audit B: route review (subagent output, saved by the main session, 2026-10-07)

> Main-session spot-check notes (added when saving):
> - **B1 CONFIRMED.**
>   - doc-02:177 lists GNN-removed as a run.
>   - Gate 3's pass condition (doc-02:178, :265) names only edge-scramble and random-graph.
>   - GATE-DGF1-1:205-207 and :222 say Gate 3's GNN-removed arm separates message passing from model class.
> - **B2 CONFIRMED.** doc-03:8 says "Do not start it until ELL-1 and DGF-1 have cleared their gates". doc-01:23 and research plan:82 say "EDR-1 proceeds regardless".
> - **S1's text CONFIRMED** at ADR-011:281-283 ("the same bar ELL-1's RF floor set").
> - **B's GATE-ELL1-1 line numbers are wrong.** The file has 77 lines. The causal clause is at GATE-ELL1-1:69-71, and "proceeds regardless" is at :71-72 (B cited :115-117 and :106). Every citation needs a mechanical check before anything is drafted from it.
> - **S8's premise is partly wrong.** Git shows no commits at all from 2026-09-16 to 10-05. ADR-015 v4–v15 were drafted from 2026-10-06 13:42 to 10-07 11:46. The three weeks were a gap, not drafting.
> - **S2 is unverified against the paper.** It rests on a summarising web fetch of arXiv:2604.19514.
> - **Incidental exposure.** B saw doc-02 ~69-77 (label-free official-mask node-time quantiles and sizes) because the prompt's filter did not cover it. It did not use or reproduce the table.

## Verdicts

**Q1. Did each move-on follow from evidence and the doc's rules?** Mostly yes.
- **Gate 0 → Gate 1:** followed the doc.
- **Failed Gate 1 → Gate 3:** matches doc-01 §4's "stop and diagnose"; the ablations are the diagnosis.
- **Gate 3's redefinition (ADR-006):** came after a provisional batch had been seen, but it was disclosed, and the original wording could never have been met.
- **Gate 2 deferral:** follows doc-01 §6.
- **EXTRACT → DGF-1:** doc-00 §9's build order.
- **Protocol choices:** none made the comparison harder than in the literature, and ELL-1's numbers match the literature.
- **Gaps:**
  - ELL-1 was closed without the failure-semantics ADR the research plan asks for.
  - The choice of DGF-1 over EDR-1 is recorded only as "direction set".
  - The next move, to EDR-1, is blocked by a start condition that cannot be met (B2).

**Q2. Was ELL-1's lesson drawn correctly and carried into DGF-1?** The FAIL verdict is robust. The lesson's direction is supported, but its wording is broader than the evidence.
- **Ruled out:** a protocol-specific reading.
- **Still open:** budget-, feature-path- and threshold-specific readings. ADR-005 closed them as "unmeasurable, not refuted", and they were never re-measured.
- **Learned well by DGF-1:** XGBoost tuned on validation, the parity floor, ROC-AUC + AUPRC, and seed counts fixed in advance.
- **Under-learned:** DGF-1's floor lacks the one-hop neighbour-feature aggregates that made ELL-1's tree hard to beat, yet ADR-011 calls it "the same bar".
- **Gate 3 as written cannot answer ELL-1's open question.** The GNN-removed arm is not in Gate 3's pass condition, and no comparison against the floor is specified (B1).

**Q3. Does EDR-1 → EDL-1 → GDE still hold?** The order holds, but several assumptions are weaker than stated.
- **Strengthened:** shared-core reuse (EXTRACT reproduces ELL-1 bit-for-bit).
- **Weakened:**
  - "validate against published baselines", because DGF-1's official split is random;
  - the ELL-1 HPO region as a basis for transfer;
  - ELL-1's lessons have not reached doc-03 or the SCM-1 doc.
- **Order as stated is slightly wrong:** under H1, EDL-1 is optional and GDE starts at P1 submission.
- **Biggest route risk:** pace, not order.

## Findings

| id | sev | status | claim | evidence | remedy / option |
|---|---|---|---|---|---|
| B1 | BLOCKING | CONFIRMED | **DGF-1's Gate 3, as specified, cannot decide "message passing vs model class", though the docs say it will.** It is also DGF-1's only chance to test ELL-1's open question, neural vs tree on identical features. GNN-removed is absent from Gate 3's pass condition, and no doc requires the parity floor to be re-run in the same batch. | doc-02:177 vs :178, :265; GATE-DGF1-1:205-207, :222; timeline:224; ADR-006 clause 4 | **Proposed ADR, in Gate 3's pre-registration: "Gate 3 decomposition".** Run GNN-removed (the identical model on the 30 parity inputs, with no edges) and the parity floor in the same batch. Pre-register two comparisons: GNN vs GNN-removed (message passing) and GNN-removed vs floor (model class). Also fix what each outcome licenses. |
| B2 | BLOCKING | CONFIRMED | **EDR-1's start condition cannot be met.** doc-03 says "cleared their gates", but ELL-1 Gate 1 FAILED and the failure semantics say "EDR-1 proceeds regardless". DGF-1's Gates 2 and 3 are open, so its half is ambiguous. This is the defect class ADR-008 fixed for doc-02. | doc-03:8; GATE-ELL1-1:71-72 (B cited :106); doc-01:23; research plan:82; document-amendments-v0.2:91 | **Proposed ADR: "EDR-1 start condition".** ELL-1 closed with dated verdicts, plus a named list of required DGF-1 gates, chosen before the next DGF-1 batch. |
| S1 | SHOULD-FIX | CONFIRMED | **The parity floor is not "the same bar" as ELL-1's.** ELL-1's RF had one-hop aggregates of neighbour features; DGF-1's tree has node-level counts only (type histogram, degree, recency). GADBench's RF-Graph and XGB-Graph tree-plus-neighbourhood baselines were known to the programme but never considered. | ADR-011:281-283 vs doc-01:51 and ADR-011:417; GATE-DGF1-1:227-230; notebooks/literature/2026-09-12…:91; https://arxiv.org/abs/2306.12251 | **Option:** a reported-only "XGBoost + one-hop neighbour-feature aggregates" arm in Gate 3's batch. **Correction:** a note on ADR-011:282. |
| S2 | SHOULD-FIX | CONFIRMED in the repo; paper via fetched HTML only | **The leakage citation is misquoted.** The docs say a GraphSAGE+RF hybrid "collapses 0.807→≈0.12 F1". The paper's HTML, read through a summarising fetch, says the hybrid falls from 0.807 to 0.699 ± 0.015, and 0.124 is its gap to raw-feature RF (0.823). ADR-004 also attaches the collapse to the RF floor. This citation motivated the strict inductive protocol. | doc-01:115, :175; ADR-004:55-57; lab 2026-07-21:30; timeline:85; doc-00:76; document-amendments-v0.2:42; https://arxiv.org/html/2604.19514 | **Proposed ADR: "Correct the Elliptic leakage citation"** in doc-00 §3.2 and doc-01 §5/§9, before P0's related work is written. |
| S3 | SHOULD-FIX | CONFIRMED | **Gate 1's stated cause is superseded.** The gate file says the GNN fails "because features 94–164 already encode one hop". Later evidence contradicts this: removing all 71 aggregates cost the GNN nothing measurable, and it stayed about 0.21 below the tree. RF on the 94 local features was never run. The timeline repeats the cause. | GATE-ELL1-1:69-71 (B cited :115-117); lab 2026-07-25:6, :11; GATE-ELL1-3:165-171; timeline:91-93 | Erratum (drafted below). |
| S4 | SHOULD-FIX | CONFIRMED that they were left open; PLAUSIBLE that it mattered | **The lesson's wording is broader than its evidence.** "Trees beat neural nets" rests on one frozen neural configuration: class-weighted cross-entropy, scored at argmax, on standardised features.<br>• **Open handicaps:** three, never measured.<br>• **HPO:** 4 of 50 trials completed, and validation used steps 30–34 at 16.8% prevalence vs 6.5% on test.<br>• **AUC gap is small** (0.918 vs 0.939), but precision at matched recall (0.534 vs 0.913) shows a real ranking gap.<br>• **"The graph helps" is shown only inside the neural model.**<br>• **The cited paper's edge-shuffle result contradicts Gate 3.** | ADR-005:112-113; lab 2026-07-24:28-33; ADR-003:98-99; adapters/ell1/train_gnn.py:63-69, :158, :173; ADR-007:36-41; Weber (https://ar5iv.labs.arxiv.org/html/1908.02591); https://arxiv.org/abs/2604.19514 | **Reword:** "a tree beat our frozen-config neural model on these features; message passing helps that model, not enough to close the gap". **Write-ups:** list the open readings in P0's limitations. The direction is consistent with Grinsztajn et al. 2022 (https://arxiv.org/abs/2207.08815). |
| S5 | SHOULD-FIX | CONFIRMED | **The ELL-1 HPO region is a thin basis for programme-wide transfer.**<br>• Depth 3 rests on an Elliptic multi-hop story, which DGF-1 inherits with no DGraph story (CLAUDE.md requires one).<br>• EDR-1's R-GCN was never searched.<br>• Both DGF-1 retune winners sit on the grid edge. | ADR-003:124-128; ADR-012:176-178; GATE-DGF1-0:45; ADR-012:165 | **Proposed ADR:** record DGF-1's depth story, or the deviation, in Gate 3's pre-registration. Require EDR-1's gate ADR to argue its depth and transfer basis. |
| S6 | SHOULD-FIX | CONFIRMED | **ELL-1's lessons have not reached later docs.**<br>• EDR-1's Gate 1 gates only against text-only, and it assumes "text-only > tabular". The bar ELL-1's lesson predicts is an XGBoost-on-financials floor (Bao et al.).<br>• doc-03's Gate 3 and the SCM-1 doc still say "toward baseline", the wording ADR-006 found could never be met. | doc-03:124, :172, :195, :137; structural-code-security:115; ADR-006:49-53 | **In EDR-1's gate ADR:** gate or position the tree floor, with parity, and carry ADR-006's within-model wording. |
| S7 | SHOULD-FIX | CONFIRMED | **The governing docs are not version-controlled.**<br>• /docs/ and /CLAUDE.md are gitignored.<br>• Every doc amendment and every quoted pass condition is therefore unauditable. Example: ADR-006 rewrote doc-01's Gate-3 wording after a provisional batch had been seen.<br>• ADR-010 is dated 07-27 but was committed 09-11. | .gitignore; ADR-012:15; 0b73eb5 | **Proposed ADR: "Track docs/".** |
| S8 | SHOULD-FIX | PLAUSIBLE | **Pace threatens the plan's stated failure mode, "an unfinished one".**<br>• **Since 09-14:** ADR-013 (several drafts), ADR-014 (4 drafts, withdrawn) and ADR-015 (15th draft, 1,277 lines), all ungated.<br>• **The CLAUDE.md "Next" list:** none of its four items is a dependency of P0 or P1.<br>• **Urgency:** doc-03 asks for an early P0 preprint.<br>[Main session: there were no commits from 09-16 to 10-05.] | git log; research plan:35-36, :86, :138; doc-03:141 | See the options. |
| N1 | NOTE | CONFIRMED | **Two route decisions lack a written reason.** ELL-1 was closed with no ADR, though the plan asks for one per failure-semantics invocation. EXTRACT→DGF-1 was chosen over EDR-1 with no reason beyond doc order. | research plan:129; lab 2026-07-25:12; lab 2026-07-26 | Record both in B2's ADR. |
| N2 | NOTE | CONFIRMED | **Gate 3's PASS answers a narrower question than doc-01 §7 asks** ("is the gain structural?"). There was no gain over the floor, and the gate file says so. | GATE-ELL1-3:153-163 | None. |
| N3 | NOTE | CONFIRMED | **The config-arm inference is confounded.** "Message passing reads what the aggregates lack" is drawn from it, but "misleading neighbours hurt" explains it equally well. The clean support is real vs no_edges (+0.070). | GATE-ELL1-3:173-184 | Cite real vs no_edges. |
| N4 | NOTE | CONFIRMED | **Clean:**<br>• **Gate 1 FAIL is robust:** all 3 Gate-1 seeds are ≤0.7131 and all 8 deterministic seeds ≤0.7624, against the RF band floor of 0.8034.<br>• **Strict inductive is a no-op for edges.**<br>• **ADR-001's 165 features:** deliberate and neutral.<br>• **Numbers match the literature:** RF 0.806 vs Weber 0.788 and Maganti 0.821; GraphSAGE 0.679 vs Maganti 0.689; GCN 0.633 vs Weber 0.628. | GATE-ELL1-1, -3; lab 2026-07-24:27; ADR-001 | No finding. |
| N5 | NOTE | CONFIRMED | **DGF-1 Gate 1's question and title are worded "Structure beats features at scale?"**, the claim CLAUDE.md forbids. | doc-02:263; GATE-DGF1-1:1 | Reword in future docs. |
| N6 | NOTE | CONFIRMED | **CLAUDE.md's order omits H1.** EDL-1 is optional, and GDE starts at P1 submission. | research plan:79; doc-00:196 | None. |

## Direction options

1. **As planned.** ADR-013 re-certification → official-split positioning → matched-time pre-registration → Gate 3 → Gate 2 → EDR-1.
   - **Cost:** building the matched-time views is projected at 1.2–2.9 h per seed (lab 2026-09-11, session 35), plus Gate 3. Likely weeks at the current pace.
   - **Establishes:** every DGF-1 qualification.
2. **Narrow DGF-1.**
   - **Steps:**
     - do the ADR-013 re-certification, which is required before the trainer is reused;
     - decline the matched-time design with a dated reason (ADR-013 permits this);
     - run Gate 3 with B1's decomposition, plus S1's reported tree arm;
     - run ADR-015 only if cheap, and defer Gate 2;
     - then write B2's ADR and start EDR-1 Phase 0.
   - **Cost:** "post-appearance activity: not established" becomes permanent.
   - **Establishes:** the model-class attribution, which is the one DGF-1 result that bears on EDR-1's floor design and on P0.
3. **Close DGF-1 at Gate 1 and start EDR-1 Phase 0 now** (ADRs first: universe, labels, timebox).
   - **Cost:** Gate-1 qualifications 2 and 3 become permanent.
   - **Establishes:** fastest progress toward P0.
4. **Re-attempt ELL-1, labelled post-hoc** (a tuned MLP, a quantile transform, a threshold sweep).
   - **Requirements:** a new ADR, a new pre-registration, and either a fresh held-out set (Elliptic has none; Elliptic++ has a different schema) or an explicit post-hoc label, because steps 35–49 already informed decisions.
   - **Cost:** days.
   - **Value:** low for the thesis. Option 2's decomposition tests neural vs tree on DGraph without touching Elliptic again.

## Drafted erratum (to sit beside GATE-ELL1-1; the gate file is not edited)

> **ERRATUM-ELL1-1 (draft).** **Status:** GATE-ELL1-1's verdict (FAILED, 2026-07-24), its tables and its criterion check stand unchanged.
>
> **Superseded passage:** the causal clause "because Elliptic's features 94–164 already encode one hop and RF on them is a hard 0.806 bar" [GATE-ELL1-1:69-71].
>
> **Why:**
> - **Local-94 diagnostic (2026-07-25, reported-not-gated):** removing all 71 aggregates did not measurably change the GNN, which stayed about 0.21 F1 below RF-165. RF on the 94 local features was never run.
> - **GATE-ELL1-3 (2026-07-25)** locates the deficit in the neural feature path.
>
> **State the cause as:** a tree beat our frozen-config neural model on identical features; message passing helps that model but not enough. The protocol and the absence of structure are both excluded.
>
> **Also note:** the "0.807→0.12" figure cited around this gate misreads arXiv:2604.19514 (see S2).

## Could not check under the prohibitions

- **The registry.** No number was verified against registry.csv. Local-94 figures come from lab entries and ADR-007 (which states git_dirty=true). "RF on the 94 local features was never run" is inferred from the text.
- **Implementation claims** (strict inductive, Standardizer) were not run or tested. Only ELL-1's class-weight and argmax lines were read.
- **Whether ELL-1's handicaps were ever re-measured.** Checked against docs and lab text only.
- **Doc history.** docs/ is untracked, so its amendment history could not be audited.
- **Incidental exposure.** doc-02 §2.3 (~69-77) shows official-mask node-time quantiles and counts. They were seen but not used.
- **Literature.** The Weber and Maganti figures came through a summarising fetch and should be re-checked against the PDFs. GADBench's per-model DGraph-Fin numbers were extracted unreliably and are not quoted.

## Citation corrections (main session, 2026-10-07)

- `ADR-003:98-99` and `:124-128` are out of range (file has 95 lines). The claims are at `ADR-003:34` ("50 trials, 4 completed") and `:45` (`num_layers` 3).
- GATE-ELL1-1 corrections are in the notes at the top.
In-range citations were not each content-checked; treat them as pointers.
