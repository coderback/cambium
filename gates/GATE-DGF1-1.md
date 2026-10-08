# GATE-DGF1-1 — does structure beat features at scale? (temporal holdout)

**Date assembled:** 2026-09-14
**Phase / doc:** DGF-1 Phase 1 — docs/02-dgraph-fin-embedding-model-BUILD.md §4 (Gate 1), §5, §7
**Git commit:** 09c6e7261d5bd602f6e99a3b25d38ce95cbeb3aa  ·  **Data snapshot:** dgraphfin-zip-150476320  ·  **Criteria:** ADR-012 (metrics ADR-007, split and protocol ADR-011, determinism ADR-005)
**Determinism:** all rows `deterministic=true`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` — bit-for-bit reproducible on the recorded environment, not anywhere.
**Batch:** stage 1 of ADR-012 clause 5 — **8 seeds per arm**, the count recorded as `**Stage-1 seeds:** 8` in ADR-012 at commit `09c6e72`.
**Ordering (all UTC):** the seed count was committed at `2026-09-14 00:37:15Z`; the first-ever row on the test window started at `2026-09-14 00:37:47Z`, **33 s later**. No test-window number existed before the seed count was fixed — and this assembly refuses to run if that ever stops being true.
**Assembled from registry rows:** gated GNN dgf1-20260914T004714Z-a0861f53, dgf1-20260914T005653Z-70ce66c4, dgf1-20260914T010640Z-0975c3d8, dgf1-20260914T011616Z-2140ee56, dgf1-20260914T012559Z-86cdf43f, dgf1-20260914T013537Z-52386170, dgf1-20260914T014513Z-c1947a8e, dgf1-20260914T015504Z-927aca45; gated floor dgf1-20260914T003747Z-1b2af0e8, dgf1-20260914T003810Z-9e97d531, dgf1-20260914T003831Z-489de2e8, dgf1-20260914T003852Z-f249d0d7, dgf1-20260914T003913Z-e3565b70, dgf1-20260914T003935Z-40c2fab9, dgf1-20260914T003956Z-75c1702c, dgf1-20260914T004017Z-ce0dc211; reported raw-17 floor dgf1-20260914T004039Z-d7c96ee4, dgf1-20260914T004058Z-9ae28997, dgf1-20260914T004116Z-2aefabd3, dgf1-20260914T004134Z-de05665a, dgf1-20260914T004153Z-a6922b62, dgf1-20260914T004213Z-0017dfbb, dgf1-20260914T004233Z-20780d4c, dgf1-20260914T004253Z-bbe7459f.

> **Disclosure — what had been seen before this batch.** No DGF-1 test-window number of any kind existed before it. What *had* been seen: both 9-configuration retune grids and both 5-seed pilots, **all on the validation window 370–481**, on which both arms were selected; and ADR-011's review look at test-window **label** statistics, reproduced verbatim under *Facts*. The seed count was derived mechanically from the validation pilots by ADR-012 clause 5 and recorded before the first test row.

## Question
On a temporal holdout of users who first appear after the training cutoff, does the sampled GraphSAGE model beat a tabular floor that sees the same node-level information — on ROC-AUC **and** on AUPRC?

## Pass condition (ADR-012 clause 6 — quoted)
> **Gate 1 passes iff both clauses hold**, on the test window 482–821, under ADR-011's protocol, deterministic, on committed code, against the parity floor **re-run in the same batch**:
> 1. **ROC-AUC:** `mean(DGF-1) − mean(floor) > 2 × SE_diff`, positive and resolvable;
> 2. **AUPRC:** the same.
>
> `s` is the sample standard deviation (`ddof=1`, ADR-006). The Welch p-value is reported alongside for context and is not the criterion.

**Estimand (ADR-012 clause 6):** DGF-1's expected performance over seeds exceeds the floor's, **on this fixed test window**. Only seed variability is modelled; uncertainty from which users form the window is reported below as a paired bootstrap, not gated.

## Results — test window 482-821, 183,469 scored users, 2,717 fraud (**prevalence 1.4809%**, the AUPRC chance level)

### DGF-1 — gated

_GraphSAGE, ADR-003 region; `lr=0.0003318335548548489`, `batch_size=2048`, `epochs=40` (ADR-012 clause 3 amendment, retune run `dgf1-20260913T153524Z-ac72170d`). Trained on the graph as of 369; scored on the graph as of 821._

| seed | ROC-AUC | AUPRC | recall @argmax | precision @argmax | run_id |
|------|---------|-------|----------------|-------------------|--------|
| 0 | 0.7542 | 0.0393 | 0.7460 | 0.0308 | dgf1-20260914T004714Z-a0861f53 |
| 1 | 0.7531 | 0.0392 | 0.7979 | 0.0281 | dgf1-20260914T005653Z-70ce66c4 |
| 2 | 0.7472 | 0.0381 | 0.8403 | 0.0257 | dgf1-20260914T010640Z-0975c3d8 |
| 3 | 0.7551 | 0.0390 | 0.7736 | 0.0298 | dgf1-20260914T011616Z-2140ee56 |
| 4 | 0.7509 | 0.0389 | 0.7762 | 0.0283 | dgf1-20260914T012559Z-86cdf43f |
| 5 | 0.7558 | 0.0383 | 0.7181 | 0.0320 | dgf1-20260914T013537Z-52386170 |
| 6 | 0.7507 | 0.0385 | 0.7589 | 0.0295 | dgf1-20260914T014513Z-c1947a8e |
| 7 | 0.7593 | 0.0395 | 0.8557 | 0.0264 | dgf1-20260914T015504Z-927aca45 |
| **mean ± std** | **0.7533 ± 0.0037** | **0.0388 ± 0.0005** | **0.7834 ± 0.0464** | **0.0288 ± 0.0021** | |

### Parity floor — gated

_XGBoost on the same 30 inputs, `max_depth=8`, `subsample=0.8` (clause 2 amendment, retune run `dgf1-20260912T235406Z-7e30aa90`)._

| seed | ROC-AUC | AUPRC | recall @argmax | precision @argmax | run_id |
|------|---------|-------|----------------|-------------------|--------|
| 0 | 0.7292 | 0.0322 | 0.6993 | 0.0293 | dgf1-20260914T003747Z-1b2af0e8 |
| 1 | 0.7288 | 0.0317 | 0.6956 | 0.0289 | dgf1-20260914T003810Z-9e97d531 |
| 2 | 0.7278 | 0.0318 | 0.6971 | 0.0295 | dgf1-20260914T003831Z-489de2e8 |
| 3 | 0.7287 | 0.0321 | 0.6941 | 0.0294 | dgf1-20260914T003852Z-f249d0d7 |
| 4 | 0.7279 | 0.0317 | 0.7041 | 0.0295 | dgf1-20260914T003913Z-e3565b70 |
| 5 | 0.7281 | 0.0318 | 0.6993 | 0.0293 | dgf1-20260914T003935Z-40c2fab9 |
| 6 | 0.7280 | 0.0319 | 0.6894 | 0.0294 | dgf1-20260914T003956Z-75c1702c |
| 7 | 0.7267 | 0.0316 | 0.7008 | 0.0291 | dgf1-20260914T004017Z-ce0dc211 |
| **mean ± std** | **0.7282 ± 0.0008** | **0.0318 ± 0.0002** | **0.6975 ± 0.0045** | **0.0293 ± 0.0002** | |

### Raw-17 floor — reported, not gated

_XGBoost on the raw 17 features only, same configuration._

| seed | ROC-AUC | AUPRC | recall @argmax | precision @argmax | run_id |
|------|---------|-------|----------------|-------------------|--------|
| 0 | 0.7057 | 0.0276 | 0.7917 | 0.0252 | dgf1-20260914T004039Z-d7c96ee4 |
| 1 | 0.7037 | 0.0273 | 0.7869 | 0.0251 | dgf1-20260914T004058Z-9ae28997 |
| 2 | 0.7056 | 0.0276 | 0.7913 | 0.0252 | dgf1-20260914T004116Z-2aefabd3 |
| 3 | 0.7053 | 0.0276 | 0.7921 | 0.0253 | dgf1-20260914T004134Z-de05665a |
| 4 | 0.7038 | 0.0274 | 0.7917 | 0.0252 | dgf1-20260914T004153Z-a6922b62 |
| 5 | 0.7052 | 0.0275 | 0.7921 | 0.0253 | dgf1-20260914T004213Z-0017dfbb |
| 6 | 0.7037 | 0.0274 | 0.7876 | 0.0252 | dgf1-20260914T004233Z-20780d4c |
| 7 | 0.7056 | 0.0277 | 0.7898 | 0.0253 | dgf1-20260914T004253Z-bbe7459f |
| **mean ± std** | **0.7048 ± 0.0009** | **0.0275 ± 0.0001** | **0.7904 ± 0.0021** | **0.0252 ± 0.0001** | |

## ADR-012 clause 6 check — mechanical, not the verdict

| clause | DGF-1 | parity floor | difference | 2×SE_diff | resolvable | Welch p | exact Welch multiplier | resolvable at exact multiplier |
|---|---|---|---|---|---|---|---|---|
| 1. ROC-AUC | 0.7533 ± 0.0037 | 0.7282 ± 0.0008 | +0.0251 | 0.0027 | **yes** | 1.16e-07 | 2.33 (df 7.6) | yes |
| 2. AUPRC | 0.0388 ± 0.0005 | 0.0318 ± 0.0002 | +0.0070 | 0.0004 | **yes** | 1.84e-11 | 2.25 (df 9.4) | yes |

**Both gated clauses: PASS** (2/2 resolvable). This is the pre-registered mechanical result; the verdict below is the researcher's.

**Stage 2 is not triggered** (ADR-012 clause 5: a top-up to n=20 runs only if a gated clause is unresolvable at stage 1).

The two clauses share one floor arm, so they are **correlated, not independent**, and must not be described as two independent confirmations (clause 6).

## Paired bootstrap over test users — reported, not gated (ADR-012 clause 10)

_Per seed pair (k, k): users resampled once per replicate, **stratified by label**, both arms scored on the identical resample; 1000 replicates, bootstrap seed 0 (pinned). Difference is DGF-1 − floor. ADR-012 does not fix how eight per-seed intervals aggregate, so all eight are shown and any that straddle zero are counted._

| seed | ROC-AUC diff (95% CI) | AUPRC diff (95% CI) |
|------|------------------------|---------------------|
| 0 | +0.0249 [+0.0187, +0.0313] | +0.0071 [+0.0049, +0.0092] |
| 1 | +0.0244 [+0.0181, +0.0309] | +0.0075 [+0.0055, +0.0097] |
| 2 | +0.0194 [+0.0132, +0.0265] | +0.0063 [+0.0042, +0.0087] |
| 3 | +0.0263 [+0.0202, +0.0328] | +0.0070 [+0.0049, +0.0091] |
| 4 | +0.0230 [+0.0160, +0.0298] | +0.0071 [+0.0050, +0.0095] |
| 5 | +0.0276 [+0.0209, +0.0344] | +0.0065 [+0.0046, +0.0087] |
| 6 | +0.0227 [+0.0164, +0.0293] | +0.0067 [+0.0046, +0.0089] |
| 7 | +0.0324 [+0.0267, +0.0385] | +0.0079 [+0.0056, +0.0100] |

**Intervals straddling zero:** ROC-AUC 0/8 · AUPRC 0/8.

_Boyd, Eng & Page (2013) logit 95% intervals for AUPRC, per arm and seed (n_pos = 2,717):_

| seed | DGF-1 AUPRC [95% CI] | floor AUPRC [95% CI] |
|------|----------------------|----------------------|
| 0 | 0.0393 [0.0326, 0.0473] | 0.0322 [0.0262, 0.0396] |
| 1 | 0.0392 [0.0325, 0.0471] | 0.0317 [0.0258, 0.0390] |
| 2 | 0.0381 [0.0315, 0.0460] | 0.0318 [0.0258, 0.0391] |
| 3 | 0.0390 [0.0324, 0.0470] | 0.0321 [0.0261, 0.0394] |
| 4 | 0.0389 [0.0322, 0.0468] | 0.0317 [0.0257, 0.0390] |
| 5 | 0.0383 [0.0317, 0.0462] | 0.0318 [0.0258, 0.0391] |
| 6 | 0.0385 [0.0319, 0.0464] | 0.0319 [0.0259, 0.0392] |
| 7 | 0.0395 [0.0327, 0.0475] | 0.0316 [0.0256, 0.0389] |

## Precision at matched recall — reported, not gated (ADR-012 clause 1)

_Per seed: the recall each arm reaches at its own argmax, then the **other** arm's precision at the highest threshold that still reaches that recall._

| seed | floor recall @argmax | floor precision | DGF-1 precision at that recall (recall achieved) | DGF-1 recall @argmax | DGF-1 precision | floor precision at that recall (recall achieved) |
|---|---|---|---|---|---|---|
| 0 | 0.6993 | 0.0293 | 0.0324 (0.6993) | 0.7460 | 0.0308 | 0.0280 (0.7460) |
| 1 | 0.6956 | 0.0289 | 0.0326 (0.6956) | 0.7979 | 0.0281 | 0.0265 (0.7979) |
| 2 | 0.6971 | 0.0295 | 0.0310 (0.6971) | 0.8403 | 0.0257 | 0.0248 (0.8403) |
| 3 | 0.6941 | 0.0294 | 0.0332 (0.6941) | 0.7736 | 0.0298 | 0.0276 (0.7736) |
| 4 | 0.7041 | 0.0295 | 0.0313 (0.7041) | 0.7762 | 0.0283 | 0.0274 (0.7762) |
| 5 | 0.6993 | 0.0293 | 0.0325 (0.6993) | 0.7181 | 0.0320 | 0.0288 (0.7181) |
| 6 | 0.6894 | 0.0294 | 0.0319 (0.6894) | 0.7589 | 0.0295 | 0.0278 (0.7589) |
| 7 | 0.7008 | 0.0291 | 0.0332 (0.7008) | 0.8557 | 0.0264 | 0.0243 (0.8557) |

## Scoring-view sensitivity — reported, not gated (ADR-011 clause 4)

| DGF-1 scored under | ROC-AUC | AUPRC |
|---|---|---|
| gated view (graph as of 821) | 0.7533 ± 0.0037 | 0.0388 ± 0.0005 |
| window-only (edges inside 482–821) | 0.7437 ± 0.0045 | 0.0350 ± 0.0007 |
| first appearance (edges up to each user's own node time) | 0.7479 ± 0.0038 | 0.0382 ± 0.0005 |

> **Implementation gap, stated rather than left unnoticed.** ADR-011 clause 4 pre-registered these sensitivity rows as GNN-**vs-floor** comparisons, with the floor's view-derived features recomputed under each view. **The floor was only ever scored under the gated view** — its gate rows carry no window-only or first-appearance metrics — so the rows above show how DGF-1's *own* score moves with the view, not how the GNN-floor gap moves. They are reported, not gated, so the clause-6 check above is unaffected; producing the floor under both views is owed as follow-up work, and is not attempted in this assembly.

## Facts this gate file must carry (ADR-012 clause 7)

- **Every AUPRC here is on the test window at prevalence 1.4809% (2,717 of 183,469).** AUPRC's chance level is that prevalence; these numbers are never compared with validation AUPRCs (1.3493%) or with ELL-1.
- **The validation window is short.** Validation users were scored at ages 0–111 steps where training users span 0–368 and test users 0–339, so the pilot that sized this batch measured variance on younger users than the gate scores.
- **Edge type 8 exists only in the test window** (all 84,338 such edges are dated 529–821). Its histogram column is constant in training and reads 0 for **both** arms, while the GNN still passes messages over those edges.
- **49.95% of raw feature values are the −1 sentinel**, standardised as a value; it lies strictly below every observed value, so it stays separable.
- **Scoring-view sensitivity and the paired bootstrap** are reported above, beside the gated numbers.
- **Tuning selected:** DGF-1 `lr=0.0003318335548548489`, `batch_size=2048`; parity floor `max_depth=8`, `subsample=0.8` — the floor subsamples, so its seed variance is real and it runs at the GNN's seed count (clauses 2, 5).
- **ADR-011's test-window label look, repeated verbatim.** Clause numbers *inside* the quote refer to ADR-011's own clauses, not to ADR-012's pass criteria above:

  > - **Test-window label statistics were seen during review (2026-09-11). Recorded verbatim; a reader
  >   should weigh this.** A red-team subagent reviewing this ADR computed fraud-vs-normal degree
  >   statistics **on the test window (482–821)**. Its prompt did not forbid test-window labels, and that
  >   omission is the reviewer's error, not the agent's. What was seen:
  >   - mean degree, normal vs fraud: **1.90 vs 1.42** under the as-of-821 view, and **1.38 vs 1.19**
  >     at each node's first appearance;
  >   - share at degree 1, normal vs fraud: **0.50 vs 0.75** as of 821, and **0.74 vs 0.88** at first
  >     appearance.
  >
  >   So degree separates the classes more under clause 4's view than under the first-appearance view.
  >   No model was trained or scored. The agent also reported statistics for the ≤ 481 view (mean
  >   degree 2.32 vs 1.77). Those use pre-test (train + val) labels, not the held-out test window. **Ordering:** clause 4's
  >   as-of-window-end rule was drafted *before* this look.
  >
  >   **Rule from here on:** any change to clauses 2–4 made after this point must either make Gate 1
  >   harder for the GNN, or rest entirely on label-free evidence. (When first written in review, the
  >   rule read "only harder". It is refined here because a direction-neutral change justified without
  >   labels cannot exploit the look either. **The researcher accepted the refinement on 2026-09-11,
  >   together with this ADR.**) The three post-look decisions satisfy the rule as follows:
  >   - **Clause 2 (role-matching)** is justified entirely by label-free evidence: pilot/gate identity,
  >     the controlled comparison, and the section-5 view match.
  >   - **Clause 4's floor parity** makes Gate 1 strictly harder for the GNN.
  >   - **Clause 6** only strengthens a leakage guard.
  >   - **The sensitivity rows** are reported, not gated.
  >
  >   Future subagent prompts must forbid held-out labels explicitly.

_All numbers computed by `scripts/assemble_gate_dgf1_1.py` from `experiments/registry.csv` and the score files those rows reference. No cell is filled by estimate, extrapolation, or smoothing._

## Verdict

**PASSED** — 2026-09-14, coderback.

Both pre-registered clauses of ADR-012 hold on the temporal holdout (482–821; 183,469 users,
2,717 fraud, prevalence 1.4809%), 8 seeds per arm, deterministic, against the parity floor re-run
in the same batch. **ROC-AUC:** DGF-1 0.7533 ± 0.0037 vs floor 0.7282 ± 0.0008, a difference of
+0.0251 against 2×SE_diff 0.0027 (Welch p 1.2e-7). **AUPRC:** 0.0388 ± 0.0005 vs 0.0318 ± 0.0002,
+0.0070 against 0.0004 (Welch p 1.8e-11). Both survive the exact Welch multipliers (2.33, 2.25), so
neither rests on the 2× approximation, and stage 2 was not triggered. The uncertainty the criterion
does not model agrees with it: paired bootstraps over test users exclude zero for all 8 seed pairs
on both metrics (smallest lower bounds +0.0132 ROC-AUC, +0.0042 AUPRC), and DGF-1's precision at
matched recall is higher on every seed, in both directions.

**Four qualifications are part of this verdict, not footnotes.**

**1. The gain is resolvable, and it is small.** AUPRC 0.0388 is about 2.6× the chance level of
0.0148; the floor sits at about 2.1×. doc-02 §0 predicted a small structural delta, and that is what
was measured. Neither arm is near usable precision: at its own operating point DGF-1 flags about 34
licit users for every fraud case it catches.

**2. What the floor saw bounds what this licenses.** The floor received every node-level statistic
DGF-1's inputs contain (ADR-011 clause 4), so the gain is not degree, edge-type histograms or recency.
It is not yet attributable to message passing either: DGF-1 differs from XGBoost in model class as
well as in structure, and separating the two is Gate 3's GNN-removed arm, which needs its own
pre-registration before it runs.

**3. Activity after users appear is only partly ruled out.** The gated view scores a user who joined
at step 490 with edges up to 821 — a channel only the GNN can read. DGF-1's own AUPRC barely moves
when restricted to edges up to each user's first appearance (0.0382 vs 0.0388). But the floor was
never scored under that view, so the GNN-vs-floor gap under it is unmeasured; that pre-registered
comparison (ADR-011 clause 4) is owed.

**4. Two clauses are not two confirmations.** They share one floor arm, so their errors are
correlated. The estimand is seed variability on this fixed window, and undated labels and snapshot
features make both arms' absolute numbers optimistic.

| claim | status |
|---|---|
| sampled GraphSAGE beats a tabular floor given identical node-level information, on this temporal holdout | **supported** — +0.0251 ROC-AUC, +0.0070 AUPRC, both resolvable |
| the gain comes from message passing rather than model class | **not established** — Gate 3's GNN-removed arm |
| the gain survives destroying the graph's wiring | **not established** — Gate 3 |
| the gain does not depend on activity after users appear | **not established** — floor unscored under the first-appearance view |
| the model is operationally useful for fraud detection | **not established** — AUPRC ≈ 2.6× chance |

**Against ELL-1, stated without re-litigating it:** there, a tree beat the neural model on identical
features, and the graph helped but not by enough. Here the neural model beats a tree given the same
node-level information. The datasets, floors and scales all differ, so this is a contrast worth
reporting, not a reversal of ELL-1's lesson.

*At 3.7M users on a temporal holdout, the GNN beats a tree given the same node-level information —
by a margin that is small, resolvable on every check we ran, and far from operationally useful.*

---
_Verdict, seeds, and table are sacred once dated. Papers are assembled from gate files; nothing is reported that is not in one._
