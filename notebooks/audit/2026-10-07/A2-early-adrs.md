# Audit A2: early ADRs, ADR-001 to ADR-009 (subagent output, saved by the main session, 2026-10-07)

> Main-session verification notes:
> - **Mutation results, executed.** Method: scratch `git archive` export of HEAD (no `data/`), FULL suite (244 tests, baseline 244 passed, `CUBLAS_WORKSPACE_CONFIG` unset). Script: `scratchpad/mut_run2.py`.
>   - **All 12 mutations M1–M9 leave the suite GREEN:** M1, M2, M3, M4, M5a, M5b, M5c, M6, M7, M7b, M8, M9.
>   - **Both controls are CAUGHT,** so the harness detects real failures:
>     - PC1 by `test_edge_scramble.py::test_scramble_never_crosses_the_temporal_cutoff` and `::test_scramble_preserves_degree_sequence`;
>     - PC2 by `test_ell1_leakage.py::test_loader_parses_and_maps_labels`.
> - **E-B1 CONFIRMED in code.** `session.py:56-61` calls `determinism_state()` immediately after `seed_everything`.
> - **E-B3 CONFIRMED in code.**
>   - `check_extract_regression.py:231-246`: "Later rows win", no commit, dirty or error filter.
>   - `:305-310`: returns 0 when there is nothing to compare.
> - **E-S1 CONFIRMED in text.** ADR-001:21-22 says Weber's 94 local features include `time_step`. ADR-001:57-58 then sets local = columns 2..95 (94 columns) with `time_step` excluded, so "local-94" includes one aggregate. Whether the CSV column order matches the paper's is PLAUSIBLE, and would need a label-free data check.
> - **Correction to audit A.** A2 notes that A cited `seeding.py:206-243`, but the file has 88 lines.
> - **A2 calls itself "Audit E" in its text.** It is A2.

## Summary

**No signed verdict changes.** ELL-1 Gates 0/1/3 and DGF-1 Gates 0/1 rest on evidence that holds up. Gate 1's and Gate 3's statistics recompute from the gate tables, including the Welch p-values. The defects are in the mechanisms of the live ADRs (005, 006/007, 008) and in the tests that are supposed to pin them.

**Counts:** 4 BLOCKING, 9 SHOULD-FIX, 12 NOTE.

**Verdict per question:**
- **Q1 (claims trace to code):** mostly yes. Exceptions: ADR-001's local/aggregate column mapping, ADR-005's seed-count precedent, the run-noise claims in ADR-005/008, and ADR-007's claim that AUPRC has no published comparator.
- **Q2 (guards):** about 7 of 24 guards have no test that fails when the guard is weakened.
- **Q3 (citations):**
  - Weber and the PyG claim are confirmed, as is the PyTorch determinism behaviour (v2.5 docs).
  - Davis & Goadrich is half-misattributed in ADR-007.
  - GADBench contradicts ADR-007.
- **Q4 (live status):** ADR-002, 003 (config), 005, 006 (criterion), 007, 008 and 009 still bind. ADR-007's and ADR-009's premises were later corrected, and neither carries a forward pointer.

## Findings

| id | sev | status | ADR | claim | evidence | remedy |
|---|---|---|---|---|---|---|
| E-B1 | BLOCKING | CONFIRMED (+ M1 executed) | 005 cl.3 | **The `deterministic` field is constant `True` by construction.** It is read right after `RunSession` forces determinism on, and no caller passes `False`. The test asserts only True, and every "clean + deterministic" filter downstream can't fail. Signed reproducibility rests on separate bit-for-bit evidence (ADR-008; GATE-DGF1-0:26-28). | `session.py:57-61`; `seeding.py:58-61,64,87`; `test_determinism_guard.py:59-82`; `assemble_gate_dgf1_0.py:89-91`; `assemble_gate_dgf1_1.py:92`; `freeze_extract_reference.py:124` | **Test:** a mid-run `use_deterministic_algorithms(False)` must record False. **Code:** record the state at exit, and add `RunSession(deterministic=…)`. |
| E-B2 | BLOCKING | CONFIRMED (+ M2 executed) | 005 cl.2 | **"Strict, not `warn_only`" is neither tested nor recorded.** | `seeding.py:58-61,87`; `test_determinism_guard.py:30-35`; PyTorch 2.5 docs | **Test:** assert `not torch.is_deterministic_algorithms_warn_only_enabled()`, and record `warn_only` in the row. |
| E-B3 | BLOCKING | CONFIRMED (+ M4 executed) | 008 cl.2 | **The EXTRACT checker can pass on rows that don't certify HEAD.** It takes the latest row per `(arm, seed)` from any commit, with no filter on dirty or errored rows, and exits 0 on an empty set; `main()` is untested. This contradicts "a mismatch is never retried". GATE-DGF1-0 itself is sound: it was assembled per commit. | `check_extract_regression.py:231-246,306-310,327` | **Code:** require exactly 49 clean, deterministic, non-errored rows at HEAD (or `--commit`), fail on a retried identity, and fail on an empty set. **Tests** for each. |
| E-B4 | BLOCKING | CONFIRMED (+ M5a/b/c executed) | 006, 007/012 | **The resolvability rule (`ddof=1`, `diff > 2·SE_diff`) has four untested copies.** No signed verdict is affected; the verdicts recompute. | `run_ell1_ablations.py:174,196-199,247,286`; `assemble_gate_dgf1_1.py:103-123`; `assemble_gate_dgf1_0.py:80-82` | **One shared, tested function in the core,** with fixtures where ddof and the boundary flip the verdict. Gate 3's assembler imports it. |
| E-S1 | SHOULD-FIX | CONFIRMED (text) / PLAUSIBLE (CSV order) | 001 | **The local/aggregate split is off by one against ADR-001's own source.** It should be 93 local + 72 aggregate: "local-94" keeps one aggregate, and "94–164" spans 71 columns, not 72. The same slip recurs at ADR-003:61, ADR-004:55, `ablations.py:112`, GATE-ELL1-1:70, GATE-ELL1-3:28,176 and lab 07-25:6. **B's draft ERRATUM-ELL1-1 inherits it.** | ADR-001:21-22 vs :57-58; Weber et al. | Erratum beside ADR-001. Correct B's draft to "71 of the 72 aggregates; one remained". Optional label-free check on column 95. |
| E-S2 | SHOULD-FIX | CONFIRMED | 005 cl.4 | **ADR-005's only example contradicts its seed-count rule.** ADR-006's 8→20 was fixed a priori, and no deterministic batch existed yet. | ADR-005:78-83; ADR-006:108-116 | Erratum line. |
| E-S3 | SHOULD-FIX | CONFIRMED | 005, 008, 007 | **"Gate 1's ±0.031 is largely run noise" is unsupported.** A 3-repeat range (0.0344) is labelled σ, though its sd is 0.0174. The deterministic seed sd is 0.0702 at n=8, and ddof=0 and ddof=1 sds are compared. Not load-bearing. | ADR-005:35-36,52; ADR-008:94-95; lab 07-25:9 | Erratum notes. |
| E-S4 | SHOULD-FIX | CONFIRMED | 007 | **"AUPRC has no published DGraph comparator" is false.** GADBench reports AUPRC for DGraph-Fin. This is relevant to ADR-015's sources. | ADR-007:112,162-163; arXiv:2306.12251 | Erratum. The researcher decides whether ADR-015's positioning lists GADBench AUPRC. |
| E-S5 | SHOULD-FIX | CONFIRMED | 007 | **Davis & Goadrich is misattributed.** Curve dominance is equivalent in ROC and PR space; only area rankings can disagree. | ADR-007:47-48; `metrics.py:9-10` | Reword in an erratum. |
| E-S6 | SHOULD-FIX | CONFIRMED (+ M6 executed) | 003 | **"Enforced by `test_hpo_no_eval_peek.py`" overclaims.** The test pins only a constant. | `test_hpo_no_eval_peek.py:16-27`; `hpo.py:70-83` | A spy test on `evaluate` and `train_model` inside `make_objective`. |
| E-S7 | SHOULD-FIX | CONFIRMED (+ M7, M7b executed) | 004 | **ELL-1's strict-inductive call sites are unpinned** (same class as A-B1). On Elliptic both edits are no-ops, so the regression checker can't catch them either. | `test_gnn_strict_inductive.py:76-83`; `train_gnn.py:163-171,237-244` | A spy on `model.forward` in `predict_window`. |
| E-S8 | SHOULD-FIX | CONFIRMED (+ M8 executed) | 002 | **The `git_dirty()` subprocess wrapper is untested.** | `config.py:76-96` | A test in a temporary git repo. |
| E-S9 | SHOULD-FIX | CONFIRMED (code) / PLAUSIBLE (torch-scatter) | 008 cl.4 | **The environment check misses things ADR-008 pins.** The driver version is never compared, it checks the checking process rather than the producing runs, and the presence of torch-scatter is unpinned. | `check_extract_regression.py:76,155-177`; ADR-008:141 | Record an environment fingerprint per batch, including the driver. |
| E-N1 | NOTE | CONFIRMED | 006 | GATE-ELL1-3's "Reproduced verbatim" disclosure is paraphrased. | ADR-006:13-14 vs GATE-ELL1-3:12-13 | — |
| E-N2 | NOTE | CONFIRMED | 007, 009 | **Corrected premises carry no forward pointers.** The degree premise was corrected (lab 07-27:21-22), and ADR-007's revisit trigger fired (ADR-010). | ADR-007:68-69,221-224; ADR-009:44-49 | One-line pointers. |
| E-N3 | NOTE | CONFIRMED | 003 | The top-2 HPO gap was decided under CUDA noise later measured at sd ≈ 0.017. | ADR-003:57-58; ADR-005:28-33 | — |
| E-N4 | NOTE | PLAUSIBLE | 004 | A 3-epoch smoke run on the real data (lab 07-23:8) may have printed a 35–49 metric. It cannot have informed an RF-anchored bar. | ADR-004:18 | Confirm from the session log. |
| E-N5 | NOTE | CONFIRMED | 006 | The ≥5-seed floor is enforced only on the run path; `--assemble-only` skips it. | `run_ell1_ablations.py:77-90` | — |
| E-N6 | NOTE | CONFIRMED | 006 | The random graph has "equal edge count" only up to collisions. | `ablations.py:76-77,84` | — |
| E-N7 | NOTE | CONFIRMED | 001 | The `data_snapshot_id` label says Kaggle, but the data came from the PyG mirror. A stale "166" sits in the docstring. | `config.yaml:6`; `datasource_elliptic.py:88` | — |
| E-N8 | NOTE | CONFIRMED (+ M9 executed) | — | The checkpoint test can't catch a dropped optimizer-state restore. No gate run uses checkpoints. | `test_checkpoint_resume.py:30-33` | — |
| E-N9 | NOTE | CONFIRMED | 007 | No generator is committed for the synthetic table. An analytic check agrees. | ADR-007:54-59 | — |
| E-N10 | NOTE | CONFIRMED | 005 | `PYTHONHASHSEED` is set at runtime, so it has no effect on the current process. | `seeding.py:79` | — |
| E-N11 | NOTE | CONFIRMED | — | Audit A's `seeding.py:206-243` citation is out of range. The file has 88 lines, and the state code is at :50-61. | — | — |
| E-N12 | NOTE | CONFIRMED | 008 | `--manifest` accepts any manifest; the 49-row shape check exists only in the tests. | `check_extract_regression.py:258,269-275` | — |

**Clean:**
- **ADR-001:** 165 features, `time_step` excluded (tested), and the PyG `loc[:, 2:]` claim.
- **ADR-002:** implemented as written.
- **ADR-003:** `config.yaml` is unchanged since `2306fa4`, and Gates 1 and 3 used it.
- **ADR-004:** pre-registered (`e8f338c`, 11:38Z) before the first row (18:32Z). The arms, seeds, band test and argmax F1 are as written, and Gate 1's arithmetic recomputes.
- **ADR-006:** the final text (`48ee87b`, 09:54:51Z) predates the batch (09:57:27Z), and the clauses match the code.
- **ADR-007:** AUPRC = `average_precision_score`, with tests that fail on mutation.
- **ADR-008:** `compare()` uses exact equality and reports missing rows.
- **ADR-009:** applied.

## Guard table

| guard | enforced at | test | fails if removed or weakened? |
|---|---|---|---|
| `time_step` excluded from x | `datasource_elliptic.py:49-50` | `test_ell1_leakage.py:158-168` | yes (PC2 executed) |
| `EXPECTED_FEATURES=165` strict | `:117-137` | none | no |
| ELL-1 train graph ≤ cutoff (call site) | `train_gnn.py:237-244` | helpers only | **no (M7b executed)** |
| ELL-1 eval on the test-induced subgraph | `train_gnn.py:163-171` | re-derived | **no (M7 executed)** |
| HPO never sees 35–49 | `hpo.py:37,70-83` | constant only | **no (M6 executed)** |
| deterministic algorithms on | `seeding.py:87` | `test_determinism_guard.py:30-35` | removal: yes; **warn_only: no (M2 executed)** |
| CUBLAS set at import | `seeding.py:32` | `:38-44` | **moved into `seed_everything`: no (M3 executed, full suite)** |
| determinism recorded in the row | `session.py:57-61` | `:59-82` | **no (M1 executed)** |
| checker equality / missing rows | `check_extract_regression.py:95-125` | `:35-89` | yes for `compare()`; **verdict ignoring missing rows: no (M4 executed)** |
| checker row selection and commit binding | `:231-246` | none | absent |
| environment drift | `:155-177` | `:105-128` | packages yes; driver not covered |
| AUPRC = average precision | `metrics.py:80` | `test_metrics.py:16-61` | yes |
| scramble/random stay within step | `ablations.py` | `test_edge_scramble.py` | yes (PC1 executed) |
| Gate 3 ≥5-seed floor | `run_ell1_ablations.py:85-90` | none | no |
| resolvability (`ddof=1`, 2×SE) | 4 copies | none | **no (M5a/b/c executed)** |
| `git_dirty` wrapper | `config.py:86-96` | none | **no (M8 executed)** |
| checkpoint optimizer restore | `checkpoint.py` | `test_checkpoint_resume.py` | **no (M9 executed)** |

## Drafted errata (each sits beside its ADR; no verdict changes)

**ERRATUM-ADR-005 (draft)**
> **Clause 3:** the row's `deterministic` field is read right after `RunSession` forces determinism on, so it is True for every row by construction. It is not evidence of how a run executed. Reproducibility rests on bit-for-bit re-runs (ADR-008; GATE-DGF1-0 batches 1–3).
>
> **Clause 2:** strictness (not `warn_only`) is neither recorded nor tested.
>
> **Clause 4:** ADR-006's 8→20 design was fixed in advance; no deterministic batch existed. ADR-012 is the first count derived as the rule requires.
>
> **Context table:** "σ ≈ 0.034" is a 3-repeat range; the sample sd is 0.017.

**ERRATUM-ADR-008 (draft)**
> **Clause 2's no-retry rule, and its binding to one commit, are not implemented by `check_extract_regression.py`.** It pairs the latest tagged row per `(arm, seed)` from any commit, including dirty and errored rows, and exits 0 with nothing to compare.
>
> EXTRACT's certification stands, because it was verified per commit (lab 2026-09-11; GATE-DGF1-0 §1). Until the checker is fixed, a `--compare-only` PASS does not certify HEAD.
>
> **Clause 4's driver pin is not checked.**
>
> **Context:** "Gate 1's ±0.031 is largely run noise" is unsupported. The deterministic seed sd is 0.070 at n=8.

**ERRATUM-ADR-006/007 (draft)**
> The `ddof=1` and 2×SE_diff pins are implemented in four untested scripts. "Mechanical" here means a reviewed reading, not a tested function. Until a shared, tested implementation exists, a Gate 3 pre-registration should name the function that evaluates its criteria.

## Could not check

- **Registry-backed claims:** ADR-003's trial table, ADR-007's local-94 table, and ADR-008's measured values and pin contents.
- **The smoke run (E-N4):** whether it printed a 35–49 metric.
- **CSV column order versus Weber's (E-S1):** needs the data.
- **The two 0.26% figures** (ADR-009:48; lab 07-27:22): a coincidence or a copy error?
- **PyTorch docs version:** the determinism claims were checked against v2.5, not 2.13.
- **Incidental exposure:** dataset-level DGraph prevalence and label counts appear at several places (ADR-007:44, ADR-010:26, doc-02:52 and :250, timeline.md:126, lab 07-26:5, lab 07-27:17). None are official-mask tables, and none were reproduced or used.

## Citation note (main session, 2026-10-07)

Mechanical range check: all in range except the two `seeding.py:206-243` mentions, which quote audit A error (E-N11). In-range citations were not each content-checked.
