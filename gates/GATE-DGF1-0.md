# GATE-DGF1-0 — does DGF-1 run at scale, with its baseline logged, on a core that still reproduces ELL-1?

**Date assembled:** 2026-09-14
**Phase / doc:** DGF-1 Phase 0 — docs/02-dgraph-fin-embedding-model-BUILD.md §4 (Phase 0, Gate 0), §7, §8  ·  **Criteria:** doc-02 §7 Gate-0 row (metrics ADR-007, reproduction ADR-008, splits ADR-010)
**Certifying commit (ADR-008):** d614671528cec8101b64a75e67609622fefbd3e7  ·  **Data snapshot:** dgraphfin-zip-150476320  ·  **ELL-1 reference manifest frozen at:** 55b595b62ca1790bdb097a2048ad9b785af65b5b
**Assembled from registry rows:** ADR-008 `experiment=extract_check` batches at `26774a0` (49 rows), `d9787e9` (49 rows), `d614671` (49 rows); DGF-1 GNN on 370-481: dgf1-20260913T000325Z-1bb64a27, dgf1-20260913T051059Z-61b566ac, dgf1-20260913T144417Z-b3622ad8, dgf1-20260913T151753Z-e068f931, dgf1-20260913T153524Z-ac72170d, dgf1-20260913T154506Z-3a4f7f61, dgf1-20260913T161731Z-ca343140, dgf1-20260913T163504Z-b6f953be, dgf1-20260913T164450Z-67f68d85, dgf1-20260913T171743Z-6f7fd7c4, dgf1-20260913T173548Z-4c1f39eb, dgf1-20260913T232506Z-1acd5067, dgf1-20260913T233439Z-388ea9b8, dgf1-20260913T234459Z-05cf9e3f, dgf1-20260913T235530Z-26af2632, dgf1-20260914T000520Z-ed807866; parity floor on 370-481: dgf1-20260912T235158Z-22be1e66, dgf1-20260912T235214Z-9498a09c, dgf1-20260912T235232Z-d2c3617b, dgf1-20260912T235251Z-fa47a397, dgf1-20260912T235306Z-93f167c5, dgf1-20260912T235326Z-7d5f6dc0, dgf1-20260912T235347Z-fda2936a, dgf1-20260912T235406Z-7e30aa90, dgf1-20260912T235431Z-f66f503b, dgf1-20260913T232205Z-c2af9760, dgf1-20260913T232228Z-7d7c21ad, dgf1-20260913T232250Z-8cde556f, dgf1-20260913T232310Z-85857ccf, dgf1-20260913T232331Z-856b67ab.

> **Disclosure — this gate is assembled after Gate 1.** Gate 1 was run and signed (**PASSED 2026-09-14**, `gates/GATE-DGF1-1.md`) before this file existed. doc-02 places Gate 0 before Phase 1, and ADR-008 clause 5 says DGF-1's Gate 0 "cannot be signed until this check is green" and that the result "is recorded in `gates/GATE-DGF1-0.md`". The check was green before any Gate-1 run, but no gate file recorded it until now. **Every row this file relies on predates the first test-window row** (latest: `dgf1-20260914T000520Z-ed807866` at 2026-09-14 00:05:20Z; first test row: `dgf1-20260914T003747Z-1b2af0e8` at 2026-09-14 00:37:47Z — checked in code). The rows are uninformed by test-window results; the assembly is not, and a reader should weigh that.

## Question
Does sampled DGF-1 training run at DGraph-Fin scale without running out of memory, is its tabular baseline logged on ROC-AUC **and** AUPRC, and does the refactored `gbe/` core still reproduce ELL-1's recorded numbers bit-for-bit?

## Pass condition (doc-02 — quoted)
> **Gate 0:** sampled training runs without OOM; baseline **ROC-AUC + AUPRC** logged; the refactored core **reproduces `ELL-1`'s recorded numbers bit-for-bit** over ADR-008's reference set, in an unchanged environment. (§4, Phase 0)
>
> Gate 0 — *Scales + baselines + ELL-1 reproduces on shared core?* — Pass: No OOM; baseline **ROC-AUC + AUPRC** logged (ADR-007); ELL-1's reference set reproduces **bit-for-bit** in an unchanged environment (**ADR-008** — not "passes"; Gate 1 failed and is not re-earned) (§7)

Doc-02 names no split in either statement. The Phase-0 task list and §8 step 3 do — see section 4.

## 1. ELL-1's reference set reproduces bit-for-bit on the refactored core (ADR-008)

_Reference set frozen pre-refactor at `55b595b`: 49 rows (rf×3, lr×3, real×8, scrambled×8, random×8, no_edges×8, config×8, gcn×3), 199 metric values. Bar: exact float equality, paired on `(arm, seed)`, never retried (clause 2). **Each batch is compared against its own commit's rows only**, not the checker's latest-row-wins view, so no batch can borrow another's pass._

| batch | commit | rows written (UTC) | rows | clean + deterministic | metrics compared | mismatches | missing | result |
|---|---|---|---|---|---|---|---|---|
| 1 | `26774a0` | 2026-09-11 15:16–16:04Z | 49 | 49/49 | 199 | 0 | 0 | **PASS** |
| 2 | `d9787e9` | 2026-09-11 17:02–17:46Z | 49 | 49/49 | 199 | 0 | 0 | **PASS** |
| 3 | `d614671` | 2026-09-12 20:30–21:11Z | 49 | 49/49 | 199 | 0 | 0 | **PASS** |

**The certificate is batch 3, at `d614671`** — the commit that moved the training loop into `gbe.gnn.train`, the last extraction step.
- Commits touching `gbe`, `adapters/ell1`, `scripts/check_extract_regression.py`, `experiments/extract_reference_manifest.json`, `experiments/extract_reference_env.txt` after `d614671` and up to the Gate-1 commit `09c6e72`: **none**.
- The same, up to the assembly-time `HEAD`: **none**.
- **Environment (clause 4).** The pin `experiments/extract_reference_env.txt` is present in every certifying commit's tree: yes. At assembly the live stack **matches the pin** for `torch`, `torch-geometric`, `pyg-lib`, `numpy`, `scikit-learn` and the GPU/CUDA runtime. The registry records no package versions per row, so the environment **at run time** rests on the checker's console output for each batch ("environment matches the pin"), which is not persisted in the repo. Installed outside the pinned five: `xgboost 3.4.1` (pin: absent), `pypdf 6.18.1` (pin: absent).

## 2. Sampled training at scale runs without running out of memory

_DGF-1 trains GraphSAGE through PyG `NeighborLoader` on the graph as of step 369 (858,702 training seeds) and scores 183,430 validation users per run. Every DGF-1 GNN row on the validation window 370-481:_

| experiment | seed | lr | batch_size | epochs | commit | wall-clock (min) | ROC-AUC | AUPRC | run_id |
|---|---|---|---|---|---|---|---|---|---|
| retune | 0 | not echoed | not echoed | not echoed | `b9c4c4a` | 32.5 | 0.7721 | 0.0375 | dgf1-20260913T000325Z-1bb64a27 |
| retune | 0 | not echoed | not echoed | not echoed | `b9c4c4a` | 34.3 | 0.7721 | 0.0375 | dgf1-20260913T051059Z-61b566ac |
| retune | 0 | 0.0003318 | 512 | 40 | `2bd88e0` | 33.6 | 0.7721 | 0.0375 | dgf1-20260913T144417Z-b3622ad8 |
| retune | 0 | 0.0003318 | 1024 | 40 | `2bd88e0` | 17.5 | 0.7756 | 0.0399 | dgf1-20260913T151753Z-e068f931 |
| retune | 0 | 0.0003318 | 2048 | 40 | `2bd88e0` | 9.7 | 0.7810 | 0.0413 | dgf1-20260913T153524Z-ac72170d |
| retune | 0 | 0.0006637 | 512 | 40 | `2bd88e0` | 32.4 | 0.7681 | 0.0359 | dgf1-20260913T154506Z-3a4f7f61 |
| retune | 0 | 0.0006637 | 1024 | 40 | `2bd88e0` | 17.5 | 0.7758 | 0.0406 | dgf1-20260913T161731Z-ca343140 |
| retune | 0 | 0.0006637 | 2048 | 40 | `2bd88e0` | 9.8 | 0.7791 | 0.0406 | dgf1-20260913T163504Z-b6f953be |
| retune | 0 | 0.001327 | 512 | 40 | `2bd88e0` | 32.9 | 0.7640 | 0.0360 | dgf1-20260913T164450Z-67f68d85 |
| retune | 0 | 0.001327 | 1024 | 40 | `2bd88e0` | 18.1 | 0.7717 | 0.0388 | dgf1-20260913T171743Z-6f7fd7c4 |
| retune | 0 | 0.001327 | 2048 | 40 | `2bd88e0` | 10.0 | 0.7729 | 0.0385 | dgf1-20260913T173548Z-4c1f39eb |
| pilot | 0 | 0.0003318 | 2048 | 40 | `2c8f484` | 9.6 | 0.7810 | 0.0413 | dgf1-20260913T232506Z-1acd5067 |
| pilot | 1 | 0.0003318 | 2048 | 40 | `2c8f484` | 10.3 | 0.7840 | 0.0408 | dgf1-20260913T233439Z-388ea9b8 |
| pilot | 2 | 0.0003318 | 2048 | 40 | `2c8f484` | 10.5 | 0.7662 | 0.0372 | dgf1-20260913T234459Z-05cf9e3f |
| pilot | 3 | 0.0003318 | 2048 | 40 | `2c8f484` | 9.9 | 0.7789 | 0.0415 | dgf1-20260913T235530Z-26af2632 |
| pilot | 4 | 0.0003318 | 2048 | 40 | `2c8f484` | 10.4 | 0.7686 | 0.0401 | dgf1-20260914T000520Z-ed807866 |

- `dgf1-20260913T000325Z-1bb64a27` predates the knob echo; all 18 of its float metrics are identical to `dgf1-20260913T144417Z-b3622ad8` (`lr=0.0003318`, `batch_size=512`), so it is that configuration, reproduced bit-for-bit.
- `dgf1-20260913T051059Z-61b566ac` predates the knob echo; all 18 of its float metrics are identical to `dgf1-20260913T144417Z-b3622ad8` (`lr=0.0003318`, `batch_size=512`), so it is that configuration, reproduced bit-for-bit.
- **Completed:** 16 runs on 370-481, 3 with echoed `batch_size=512`, 3 with echoed `batch_size=1024`, 8 with echoed `batch_size=2048`; plus 8 on 482-821 (GATE-DGF1-1). Unclean or non-deterministic among the rows above: 0. DGF-1 rows marked errored: 0.

> **Disclosure — runs that wrote no row.** Two retune batches were stopped by the Claude Code harness for low **host** memory, each during configuration 2 (lab Session 29). Killed runs write no row, so nothing above comes from them. The diagnosis there was measured before any code changed — no per-epoch growth, and a flat footprint across the three scoring views — and the grid then completed in the researcher's own terminal. **The registry records no memory figure**, so this condition rests on completion: every run allowed to finish did finish. Whether a harness kill on host memory counts against "no OOM" is the verdict's to decide.

## 3. Baseline ROC-AUC + AUPRC logged (ADR-007)

_Validation window 370-481: 183,430 scored users, 2,475 fraud, **prevalence 1.3493%** (the AUPRC chance level). Never compared with test-window or ELL-1 AUPRCs._

### Parity floor — validation pilot at the selected configuration

_XGBoost on the 30 parity inputs, `max_depth=8`, `subsample=0.8` (ADR-012 clause 2). Sample std, `ddof=1`._

| seed | ROC-AUC | AUPRC | recall @argmax | precision @argmax | run_id |
|------|---------|-------|----------------|-------------------|--------|
| 0 | 0.7399 | 0.0323 | 0.6586 | 0.0286 | dgf1-20260913T232205Z-c2af9760 |
| 1 | 0.7404 | 0.0318 | 0.6598 | 0.0287 | dgf1-20260913T232228Z-7d7c21ad |
| 2 | 0.7380 | 0.0325 | 0.6432 | 0.0291 | dgf1-20260913T232250Z-8cde556f |
| 3 | 0.7408 | 0.0323 | 0.6420 | 0.0292 | dgf1-20260913T232310Z-85857ccf |
| 4 | 0.7350 | 0.0318 | 0.6424 | 0.0288 | dgf1-20260913T232331Z-856b67ab |
| **mean ± std** | **0.7388 ± 0.0024** | **0.0321 ± 0.0003** | **0.6492 ± 0.0091** | **0.0289 ± 0.0003** | |

### Parity floor — validation retune grid (selection runs, one seed each)

| configuration | commit | ROC-AUC | AUPRC | run_id |
|---|---|---|---|---|
| not echoed | `e58ccb6` | 0.7358 | 0.0305 | dgf1-20260912T235158Z-22be1e66 |
| not echoed | `e58ccb6` | 0.7377 | 0.0310 | dgf1-20260912T235214Z-9498a09c |
| not echoed | `e58ccb6` | 0.7373 | 0.0305 | dgf1-20260912T235232Z-d2c3617b |
| not echoed | `e58ccb6` | 0.7386 | 0.0320 | dgf1-20260912T235251Z-fa47a397 |
| not echoed | `e58ccb6` | 0.7370 | 0.0309 | dgf1-20260912T235306Z-93f167c5 |
| not echoed | `e58ccb6` | 0.7382 | 0.0314 | dgf1-20260912T235326Z-7d5f6dc0 |
| not echoed | `e58ccb6` | 0.7383 | 0.0320 | dgf1-20260912T235347Z-fda2936a |
| not echoed — ADR-012 clause 2's winner, `max_depth=8`, `subsample=0.8` | `e58ccb6` | 0.7399 | 0.0323 | dgf1-20260912T235406Z-7e30aa90 |
| not echoed | `e58ccb6` | 0.7386 | 0.0318 | dgf1-20260912T235431Z-f66f503b |

- **Both metrics on every floor row:** 30/30 DGF-1 XGBoost rows carry `fraud_auc` and `fraud_auprc`, so Gate 1's AUPRC clause had a comparator from the first floor row on.
- **Logged before the first DGF-1 GNN row:** first floor row 2026-09-12 23:51Z; first GNN row 2026-09-13 00:03Z. (Session 27's timing run scored the GNN without writing a row, so this orders rows, not every number ever computed.)
- The retune rows predate the config echo; their configurations were recovered 9/9 by config-hash rebuild (lab Session 28), and the winner is recorded in ADR-012.
- **The raw-17 floor was never logged on validation**: its rows exist only on 482-821 (8 rows), reported in GATE-DGF1-1.

## 4. The official (random) split — not run

| experiment | arm | window | rows |
|---|---|---|---|
| gate | dgf1-parity | 482-821 | 8 |
| gate | xgboost-parity | 482-821 | 8 |
| gate | xgboost-raw17 | 482-821 | 8 |
| pilot | dgf1-parity | 370-481 | 5 |
| pilot | xgboost-parity | 370-481 | 5 |
| retune | dgf1-parity | 370-481 | 11 |
| retune | xgboost-parity | 370-481 | 9 |

**DGF-1 rows on any split other than the temporal windows 370-481 / 482-821: 0.** No official-split number exists for either arm. doc-02's Gate-0 statements (quoted above) do not name a split, but its Phase-0 task list does, and §8 step 3 reads:

> Run **XGBoost** on **both** splits. On the temporal split the gated comparator is the **parity floor** (raw 17 features plus view-derived node statistics, **ADR-011**), and the raw-17 floor is reported. *(Amended 2026-09-11, was "Run **XGBoost on node features**".)* Throughout, logging **ROC-AUC and AUPRC** for each (both metrics — ADR-007; AUPRC cannot be added to a row later). **The numbers to beat are the TEMPORAL-split ones** (ADR-010); the official-split numbers are leaderboard positioning only.

ADR-010 clause 1 makes those numbers **reported, never gated** — "They acquire no pass condition and may not acquire one retroactively." They are owed either way; whether their absence bears on this gate is the verdict's to decide.

_All numbers computed by `scripts/assemble_gate_dgf1_0.py` from `experiments/registry.csv`, the frozen ELL-1 manifest and `git`. No cell is filled by estimate, extrapolation, or smoothing._

## Verdict

**PASSED** — 2026-09-14, coderback.

All three conditions of doc-02 §7 hold as worded. **ELL-1's reference set reproduces bit-for-bit
on the refactored core** (ADR-008): three independent batches, each compared against its own
commit's rows only — 49/49 rows, 199 metric values, 0 mismatches, 0 missing, all clean and
deterministic. The certificate is `d614671`, the last extraction commit, and nothing it certifies
changed before the Gate-1 commit or since. **Sampled training completes at full scale:** all 16
GNN runs on the validation window finished — 858,702 training seeds, 183,430 users scored per
run, at batch sizes 512, 1024 and 2048 — clean, deterministic, none errored. **The baseline logs
both metrics:** 30/30 floor rows carry ROC-AUC and AUPRC, the first logged before any GNN row; the
selected parity floor reads 0.7388 ± 0.0024 ROC-AUC and 0.0321 ± 0.0003 AUPRC on validation, at
prevalence 1.3493%.

**Three qualifications are part of this verdict, not footnotes.**

**1. This verdict is recorded late, and only the record is late.** Gate 1 ran and was signed before
this file existed, although doc-02 places Gate 0 first and ADR-008 clause 5 ties Gate 0's signature
to the check. The evidence was not late: the reproduction check was green at `d614671` before any
Gate-1 run, no certified path changed before Gate 1's commit, and every row this gate relies on
predates the first test-window row. So GATE-DGF1-1 did not run on an uncertified core. The
lesson is procedural: a gate file is assembled when its evidence completes, not when the next gate
needs it.

**2. "No OOM" rests on completion, not on a measured margin.** The registry records no memory
figure. Two retune batches were stopped by the Claude Code harness for low host memory and wrote
no rows; the process was measured healthy, and the grid completed in the researcher's own terminal.
That is an environment constraint, not a model failure — but it binds: long DGF-1 batches run
outside the harness.

**3. The official-split positioning numbers are owed.** No DGF-1 row exists on the official
(random) split. Gate 0's wording names no split, so their absence does not fail it, but doc-02's
Phase-0 tasks and §8 step 3 require them. ADR-010 clause 1 makes them reported and never gated, so
producing them later cannot move this verdict or Gate 1's.

| claim | status |
|---|---|
| the refactored core reproduces ELL-1 bit-for-bit | **supported** — 3 batches × 49/49, 0 of 199 values moved; covers the Gate-1 commit |
| sampled DGF-1 training completes at full scale | **supported** — 16/16 validation runs finished (8 more on test, GATE-DGF1-1) |
| memory headroom is measured | **not recorded** — no memory figure in any row |
| the floor logs ROC-AUC and AUPRC | **supported** — 30/30 rows |
| official-split positioning logged | **not run** — owed; reported, never gated |
| Gate 0 recorded before Gate 1 | **no** — assembled afterwards; the evidence predates Gate 1 |

*DGF-1 runs at scale on a core that still computes exactly what ELL-1 computed. The gate was
earned on time and recorded late.*

---
_Verdict, seeds, and table are sacred once dated. Papers are assembled from gate files; nothing is reported that is not in one._
