# ADR-012 — DGF-1 Gate 1, pre-registered: arms, floor, seed-count rule, and what is reported

**Status:** accepted
**Date:** 2026-09-12 · **accepted** 2026-09-12
**Deciders:** coderback
**Stage-1 seeds:** _not yet derived — recorded here after the validation pilot (clause 5). Until
this line carries an integer, `scripts/run_dgf1_floor.py` and the DGF-1 GNN runner refuse the test
window (clause 8)._
**Inherits, does not re-open:** the metric pair and resolvability test (**ADR-007**), the split,
views, parity floor and leakage protocol (**ADR-011**), determinism (**ADR-005**), `ddof=1` and the
two-stage seed design (**ADR-006**), the frozen hyperparameter region (**ADR-003**) and the
transfer rule (**ADR-009**).
**Docs affected — all amendments applied 2026-09-12 on acceptance. Sites enumerated by grepping
`docs/` first, and the grep re-run after applying to confirm no live contradiction remains
(ADR-009 practice; `docs/` is gitignored, so ripgrep skips it):**
- `docs/02-dgraph-fin-embedding-model-BUILD.md`: §4 Phase 1 (the retune line gains its grid and
  selection metric), §4 Phase 1 Gate 1 (the arms and the seed-count rule), §5 (the reporting
  template gains the pilot and retune rows), §7 Gate-1 row, and §7's ADR-007 seed-count note (which
  says the count "goes in DGF-1's Gate-1 pre-registration ADR" — this is that ADR).
- `docs/timeline.md`: the DGF-1 Gate-1 row and the EXTRACT pending line.

> **Review note (2026-09-12, pre-acceptance): the floor's determinism was re-decided.** The first
> draft pre-committed the floor to `subsample=1.0` and called the resulting determinism "a genuine
> trade". A review with a literature agent and a red-team subagent found that framing wrong in three
> ways, and the clauses below are the repair.
> - **Manufacturing floor variance would break clause 5, not merely raise the bar.** Measured
>   arithmetic: with the floor at 3 seeds and `s_floor = s_GNN`, the bar is **1.91×** the
>   deterministic one at n=8 and **2.77×** at n=20 — and 8→20 seeds moves it only 0.677→0.619·`s_GNN`,
>   because `s_floor/√3` dominates. Buying GNN seeds would stop buying resolution.
> - **"A stronger comparator" was an overclaim, and plausibly backwards.** Subsampling is
>   Friedman's regulariser; the major tabular benchmarks **tune** it (Shwartz-Ziv & Armon 2022,
>   arXiv:2106.03253; Grinsztajn et al. 2022, arXiv:2207.08815). Fixing it at 1.0 risks an
>   *undertuned* floor, which biases toward the GNN.
> - **The variance the gate ignores is the same order as the one it models.** Bouthillier et al.
>   (MLSys 2021, arXiv:2103.03098) measure data sampling as the dominant source, with init under
>   half of it, and seed-only randomisation worth ~2 ideal runs. At this window's support the paired
>   bootstrap SE of the AUPRC difference is ≈0.005–0.006 (two independent synthetic simulations)
>   against a seeds-only SE of ≈0.007 at `s_GNN`=0.02, n=8. The fix is to **report** the missing
>   term (clause 10), not to inflate the modelled one.
>
> Also repaired: the estimand is now named (clause 6), `n_jobs` is pinned because `hist` determinism
> depends on thread count (clause 2), and clause 5 gained the ADR-007 carve-out it needed.

> **Review note 2 (2026-09-12, pre-acceptance): full adversarial pass.** The review above covered
> only the floor question; this pass attacked the whole ADR. One clause was **strengthened by
> measurement** and seven defects were fixed. Nothing about the criterion changed.
> - **Clause 3's exact scoring was an untested assumption, and is now measured.** The suspicion was
>   hub blow-up at max degree 882; the structural probe refutes it (median 8,088 nodes/batch,
>   max 9,203, ~46 MB — against 7,009/7,254 sampled). Recorded in the clause, with the ~18 GB
>   full-graph figure that justifies loader-based scoring at all.
> - **The runtime escape valve was gameable:** "epochs may be reduced" set no value, so the budget
>   could have been chosen after seeing runtimes. Now a fixed single halving, 40 → 20.
> - **"Precision at matched recall" was reported but never defined** — now specified in both
>   directions, from ELL-1's diagnostic.
> - **Official-split numbers were listed as reported with no clause producing them.** They are a
>   separate positioning batch, outside clause 5's counts, never sharing a row with a gated number.
> - **The retune selects among 9 configurations on one seed**, which can pick seed-luck. ELL-1's
>   precedent, now stated with its weakness and with the pilot as the check.
> - **Clause 10 stored scores without node ids**, so the paired bootstrap could not have aligned two
>   arms except by trusting an undocumented ordering. Ids are now stored and asserted equal.
> - **The bootstrap's seed and replicate count were unpinned**, so its CI would not have been
>   reproducible from the stored files. Both pinned (1,000 replicates, seed 0).
> - **Clause 8's guard line had no format**, leaving the regex to guesswork. Fixed as
>   `**Stage-1 seeds:** <integer>`, matching the shape the guard already parses.

## Disclosure — what had been seen when this was written

- **No DGF-1 score of any kind exists.** The floor has never been run, on any window; no DGF-1 GNN
  exists; no registry row is `model=dgf1`. Every threshold and count below is therefore fixed
  blind in ADR-004's full sense.
- **Structural and distributional facts about the data have been seen** and are stated where they
  bear on a clause: window supports and prevalences (ADR-011), the −1 sentinel at 49.95%, edge
  type 8 occurring only in the test window, `edge_type_8` constant in the training view, the
  val-window age profile, and the sampler's timings.
- **Test-window label statistics were seen once, during ADR-011's review, and are re-disclosed
  here because they bear on this ADR's floor.** A red-team subagent computed fraud-vs-normal degree
  on 482–821: mean degree 1.90 vs 1.42 as of 821, and share at degree 1 of 0.50 vs 0.75. No model
  was trained or scored. That look argued **for** floor parity, which makes Gate 1 **harder** for
  the GNN — the conservative direction, and the direction ADR-011's post-look rule requires.
- **The pilot has not run.** This ADR fixes the *rule* that turns the pilot's measured variance
  into a seed count, so the number is mechanical once the pilot exists (clause 5).

## Context

ADR-007 fixed DGF-1's metric pair and deferred the seed count to "DGF-1's Gate-1 pre-registration
ADR", requiring it to be derived in advance from a pilot on the validation split, with a floor of
5 per arm and ADR-006's fixed two-stage design. ADR-011 fixed the windows, the training graph, the
scoring views and the parity floor, and left five items to this ADR. EXTRACT's code is now in
place — DataSource, transform, floor machinery, temporal sampler — and all ten of ADR-011's
clause-5 tests pass, so what remains before a test-window run is the specification of the runs
themselves.

Two measured facts make parts of this specification load-bearing rather than clerical:

* **XGBoost's seed is inert without subsampling** (measured 2026-09-11 on synthetic data: identical
  predictions at different seeds). A floor's "seed variance" would then be zero by construction,
  and `SE_diff` would come entirely from the GNN. That must be decided, not discovered.
* **doc-02 §0 predicts a small structural delta** on this graph. A small true effect with an
  under-powered batch is how a gate returns "unresolvable" for reasons of budget rather than
  biology, which is exactly what the seed-count rule exists to prevent.

## Decision

### Clause 1 — Arms

**Gated (two arms, same batch, same protocol):**

| arm | inputs | notes |
|---|---|---|
| **DGF-1** | the 30 `node_inputs` columns, standardised (ADR-011 clause 3) | sampled GraphSAGE, clause 3 below |
| **parity floor** | the *same* 30 columns, untransformed | XGBoost, clause 2 below |

**Reported, never gated** (no pass condition, now or retroactively): the raw-17 floor; the
window-only and first-appearance sensitivity rows (ADR-011 clause 4, both permitted — the sampler
passed its determinism check on 2026-09-11); official random-split numbers, always labelled; and
precision at matched recall, the diagnostic that made ADR-007 necessary.

Two definitions, so the reported rows are not left to later judgement:

* **Precision at matched recall** is computed by taking the recall each **floor** seed reaches at
  its own argmax threshold, then reporting each GNN seed's precision at the threshold where the GNN
  reaches that same recall (and the reverse direction alongside). This is ELL-1's diagnostic —
  0.534 vs 0.913 precision at Δrecall 0.0013 — which is what ADR-007 was written from.
* **Official random-split numbers come from a separate positioning batch**, not from the gate
  batch: they require training on the official train mask, they are not part of clause 5's seed
  counts, and they are labelled *random-split, leaderboard-comparable* wherever they appear
  (ADR-010 clause 1). No gated number and no official-split number may share a table row.

### Clause 2 — The floor is tuned on validation, with the same budget as the GNN

**Fixed:** XGBoost 3.4.1, `tree_method="hist"`, `n_estimators=300`, `learning_rate=0.1`,
`min_child_weight=1`, `reg_lambda=1.0`, `colsample_bytree=1.0`,
`scale_pos_weight = n_negative / n_positive` on the training rows (the imbalance handling doc-02
§2.2.5 requires, and the analogue of ELL-1's `class_weight="balanced"`), `random_state=seed`, no
early stopping (a fixed budget, so nothing selects on the scored window), and **`n_jobs=8`, a fixed
integer, never `-1`** — `hist` is deterministic given identical data order *and* thread count, so a
machine-dependent thread count would make reproducibility machine-dependent (ADR-005).

**Tuned on the validation window, 9 configurations, one seed each, selected on AUPRC** — the same
budget, window and selection metric as the GNN's retune (clause 3):

> `max_depth ∈ {4, 6, 8}` × `subsample ∈ {1.0, 0.8, 0.6}`

* **Determinism is an outcome here, not a pre-commitment.** If the winner has `subsample = 1.0` the
  floor is deterministic, its seed is inert, and `s_floor = 0`; the pilot **verifies** that its rows
  are identical rather than assuming it. If the winner subsamples, the measured `s_floor` is used.
* **Why tuned rather than fixed.** Subsampling is a regulariser (Friedman 2002), and the major
  tabular benchmarks tune it rather than fix it, so pre-committing to 1.0 risks an **undertuned
  floor — a bias in the GNN's favour**, and the weak-baseline failure doc-00 §10 calls the first
  cause of desk rejection. Tuning both arms the same way removes the author's prior from the
  comparator's strength.
* **Seed count.** If the floor is deterministic it is run at **3 seeds**, to evidence the identity.
  If it subsamples it is run at **the same count as the GNN** — see clause 5, which explains why a
  3-seed stochastic floor would stall the seed-count rule.
* **A cost stated plainly:** tuning both arms on validation makes `Δ_val` optimistic in both
  directions (a winner's curse on each arm). Clause 5's ½ factor absorbs this; that is now one of
  its two stated jobs, not a coincidence.

> **Amendment 2026-09-13 — the floor's tuning is resolved.** Clause 2's grid ran on the validation
> window (9 configs × 1 seed, `experiment=retune`, 9 rows, all `git_dirty=false` and deterministic).
> **Winner: `max_depth=8, subsample=0.8`, val AUPRC 0.0323** (ROC-AUC 0.7399), run
> `dgf1-20260912T235406Z-7e30aa90`. Spread across the grid was AUPRC 0.0305–0.0323, so the floor is
> insensitive to its own tuning.
>
> **The winner subsamples, so the branch this clause left open resolves the second way:**
> `s_floor ≠ 0`. The floor's pilot therefore runs **5 seeds**, not 3, and under clause 5 the floor
> runs at **the same seed count as the GNN** in the gate batch. Determinism was a measured outcome,
> and the measurement went against the pre-commitment I nearly wrote.

### Clause 3 — The DGF-1 arm is specified here, in full

* **Architecture and budget inherited from ADR-003** unchanged: GraphSAGE, 3 layers, hidden 128,
  mean aggregation, dropout 0.2, 2-layer encoder MLP, LayerNorm, weight decay 5e-4, 40 epochs,
  fan-out [25, 10] extended to [25, 10, 10] as ELL-1 extends it.
* **Retune exactly two knobs (ADR-009): `lr` and `batch_size`.** The grid is fixed here:
  `lr ∈ {0.5×, 1×, 2×}` ADR-003's `6.636671097096978e-4`, `batch_size ∈ {512, 1024, 2048}`; nine
  configurations, one seed each, **scored on the validation window only**, selected on **AUPRC**
  (the more sensitive of the two gated metrics). The winning pair is frozen before any test-window
  run and recorded here as an amendment.
  **One seed per configuration is ELL-1's precedent (ADR-003's sweep used a fixed trial seed), and
  its weakness is stated:** with GNN seed variance of ELL-1's order, a 9-way selection on one seed
  can pick seed-luck rather than a better configuration. The pilot's 5 seeds at the winner are the
  check — if its pilot mean falls well short of its retune value, that is recorded in the gate file
  as a caveat on the configuration, and the configuration still stands. Re-selecting after seeing
  the pilot would be tuning on the same data twice.
* **Scoring uses exact neighbourhoods** (`num_neighbors = [-1, -1, -1]`) over the window's scoring
  view, for every arm and every reported row. Verified 2026-09-11: exact mode involves no sampling
  RNG at all, so no score depends on an evaluation seed.
  **Feasibility measured, not assumed (2026-09-12, val view, structural probe).** The worry was
  hub blow-up at max degree 882. It does not materialise: on the gated view (5,408,862 edges) exact
  3-hop neighbourhoods at batch 1024 give a median of **8,088 nodes per batch, max 9,203**
  (~46 MB of activations), against **7,009 / 7,254** for the sampled fan-out — the graph is too
  sparse for exact scoring to cost much. A full-graph forward, by contrast, would need ~18 GB, which
  is why scoring goes through a loader at all. The test view carries ~1.6× the edges of the val
  view, so these figures are a lower bound there, with ample headroom either way.
* **Training budget escape valve, on runtime only, and with a fixed fallback.** If a single
  training run at the winning configuration exceeds **2 hours** (measured in the pilot, on this
  machine, wall-clock), epochs are **halved once, 40 → 20**, applied identically to every arm and
  recorded as a dated amendment **before any test-window run**. The fallback value is fixed here so
  the budget cannot be tuned after seeing runtimes; if 20 epochs is still infeasible, the gate is
  postponed and the batch rented (clause 5), never shortened further. Conditioned on wall-clock,
  never on a result.

> **Amendment 2026-09-14 — the GNN retune is resolved.** Clause 3's grid ran on the validation
> window on 2026-09-13 at commit `2bd88e0` (9 configs × 1 seed, `experiment=retune`, all clean and
> deterministic, every row stating its own `lr`/`batch_size`/`epochs`). **Winner:
> `lr=3.318335548548489e-4` (0.5× ADR-003), `batch_size=2048`, val AUPRC 0.0413** (ROC-AUC 0.7810),
> run `dgf1-20260913T153524Z-ac72170d`. Every batch-512 configuration ranked in the bottom three,
> which is ADR-009's named second knob binding on measurement. The top two differ by 0.0007 AUPRC
> on one seed each — within plausible seed noise, as this clause anticipated; the pilot is the check.
> The winning configuration runs in ~10 min, well inside this clause's 2-hour rule, so the 40-epoch
> budget stands.

### Clause 4 — The pilot (validation window only)

After acceptance, and after both tuning grids (clause 2's 9 floor configurations and clause 3's 9
GNN configurations, each one seed, scored on validation, selected on AUPRC):

**5 seeds** of the DGF-1 arm, and of the floor **3 seeds if its winning configuration has
`subsample = 1.0`** (enough to evidence that its rows are identical) **or 5 if it subsamples**
(clause 2), at their winning configurations, trained on `≤ 369`, scored on **370–481**,
deterministic, on committed code. It records, per gated metric:
each arm's mean and sample standard deviation (`ddof=1`), the val gap
`Δ_val = mean(DGF-1) − mean(floor)`, **whether the floor's rows are identical** (clause 2), the
paired bootstrap SE of the difference on the validation window (clause 10), and the wall-clock per
run. **The pilot never touches 482–821**, and its numbers are never reported as gate numbers.

### Clause 5 — The seed-count rule, fixed now and mechanical afterwards

Let `s_G` be the pilot's GNN standard deviation and `s_F` the floor's, per metric. Let `n_F` be the
floor's seed count under clause 2: **3 if the floor is deterministic** (then `s_F = 0` and its term
vanishes anyway), **`n` if it subsamples**. For a candidate `n`,

> `SE_diff(n) = sqrt(s_G²/n + s_F²/n_F)`.

> **Stage-1 `n` is the smallest `n` in {8, 12, 16, 20} for which
> `2 × SE_diff(n) ≤ ½ · Δ_val` holds for *both* gated metrics.**
> If no `n` in that set qualifies, or if `Δ_val ≤ 0` for either metric, **`n = 20`**.

* The half is a deliberate safety factor, doing **two** jobs: a val-window effect is an optimistic
  estimate of a test-window effect (val users are younger, clause 7), **and** both arms were
  selected on val, so `Δ_val` carries a winner's curse on each side (clause 2).
* **`Δ_val` is a power-planning input, never a comparison.** ADR-007 forbids comparing AUPRC across
  windows of differing prevalence, and val (1.3493%) differs from test (1.4809%). Using the val gap
  to *size a batch* makes no claim about the test AUPRC and reports no cross-window comparison; the
  carve-out is stated here so the inheritance is not self-contradictory.
* **The rule is per-metric and scale-free** — each metric's gap is compared with its own `SE_diff` —
  so the binding metric is whichever has the larger `s/Δ_val` ratio, which is an empirical outcome
  of the pilot and not a preference. The nearest precedent says not to assume ROC-AUC binds: on
  ELL-1's Gate 3 the scrambled-vs-real ratios were **0.69 for illicit-F1** (0.0702/0.1018) against
  **0.32 for ROC-AUC** (0.0097/0.0301), i.e. the threshold-based metric was the harder one.
* **If clause 2's tuned floor subsamples, it is run at the same seed count as the GNN.** With a
  3-seed stochastic floor the `s_floor/√3` term dominates and the rule stalls: at `s_floor = s_GNN`
  the bar only moves from 0.677 to 0.619·`s_GNN` between n=8 and n=20, so more GNN seeds would stop
  buying resolution.
* **8 is the floor** (above ADR-007's 5, because ELL-1 measured 3 seeds to be badly under-powered:
  ±0.031 at 3 seeds vs ±0.070 at 8 on the same arm), and **20 is the cap** (ADR-006's stage 2).
* **Stage 2 — n = 20, at most once**, triggered only if a gated clause is unresolvable at stage 1
  and stage-1 `n < 20`. It extends **both** arms and **its result is final in whichever direction
  it falls**. There is no stage 3 and no open-ended top-up: that is optional stopping (ADR-006).
* If a clause is still unresolvable at 20, **it is reported as unresolvable and Gate 1 does not
  pass on it.**
* **Compute does not weaken this.** If 20 seeds are infeasible on this machine, the gate is
  postponed and the batch rented (research plan §7), never run under-powered.

**The rule is satisfiable, and was checked before acceptance.** This is arithmetic on the formula
above, not a measurement, and it predicts nothing about DGF-1's variance — it exists so the rule
cannot turn out to be vacuous (always 20) or trivial (always 8) after the pilot has been run.
Selected `n`, with `s_floor = 0`:

| `s_GNN` \ `Δ_val` | 0.010 | 0.020 | 0.050 | 0.100 |
|---|---|---|---|---|
| 0.002 | 8 | 8 | 8 | 8 |
| 0.005 | 8 | 8 | 8 | 8 |
| 0.010 | 16 | 8 | 8 | 8 |
| 0.020 | 20 | 16 | 8 | 8 |
| 0.050 | 20 | 20 | 16 | 8 |

The cap binds exactly when `s_GNN > ½·Δ_val·√20 / 2` — i.e. 0.0112 at a val gap of 0.01, 0.0224 at
0.02, 0.0559 at 0.05. In that corner the batch runs at 20 and any clause that stays unresolvable is
**reported as unresolvable**, which is the intended outcome, not a failure of the rule.

### Clause 6 — Pass criteria

**Gate 1 passes iff both clauses hold**, on the test window 482–821, under ADR-011's protocol,
deterministic, on committed code, against the parity floor **re-run in the same batch**:

1. **ROC-AUC:** `mean(DGF-1) − mean(floor) > 2 × SE_diff`, positive and resolvable;
2. **AUPRC:** the same.

`s` is the sample standard deviation (`ddof=1`, ADR-006). The Welch p-value is reported alongside
for context and is not the criterion. A partial pass is recorded as a partial pass. The two clauses
share one floor arm, so they are **correlated, not independent**, and must not be described as two
independent confirmations.

**The estimand, stated so the criterion cannot be over-read.** This tests whether **DGF-1's expected
performance over seeds exceeds the floor's, on this fixed test window**. It is conditional on that
window: the only variability modelled is seed variability, so a deterministic floor contributes zero
**seed** variance — which is arithmetically right and is *not* a claim that the floor carries no
uncertainty. Uncertainty from which 183,469 users happen to constitute the window is real, is of the
same order (clause 10), and is **reported** rather than gated. Gate 1's verdict is therefore a
statement about reliability across seeds, never on its own a statement about which method
generalises better.

### Clause 7 — Facts the gate file must carry

Not footnotes; part of the verdict:

* every AUPRC with its window's **prevalence and positive count** — test 1.4809%, 2,717 fraud of
  183,469 (ADR-007), and never compared across windows or against ELL-1;
* **the val window is short**: val users are scored at ages 0–111 where training users span 0–368
  and test users 0–339, so the pilot's variance is measured on younger users than the gate scores;
* **edge type 8 exists only in the test window** (all 84,338 edges dated 529–821), so its histogram
  column is constant in training and reads 0 for **both** arms, while the GNN still passes messages
  over those edges;
* **49.95% of raw feature values are the −1 sentinel**, standardised as a value;
* the **window-only and first-appearance** sensitivity rows, which show how much of any gap depends
  on the scoring view (ADR-011's disclosure);
* **the paired bootstrap CI of the difference** (clause 10) beside every gated number, and the
  explicit statement if the two disagree;
* **which configuration each arm's tuning selected**, including whether the floor subsamples;
* ADR-011's **test-window label look**, repeated verbatim.

### Clause 8 — Execution order, enforced in code

1. Accept this ADR. 2. Run the retune (val). 3. Run the pilot (val). 4. Record the winning
`lr`/`batch_size` and the stage-1 seed count — in the exact line format fixed below — here as dated
amendments. 5. Only then the single test
batch. 6. Assemble `gates/GATE-DGF1-1.md`; the researcher signs the verdict, never a script.

`scripts/run_dgf1_floor.py` already refuses `--window test` without an accepted pre-registration.
**It must be extended to also require a stage-1 seed count in that ADR**, so the test window stays
closed until the count exists — and the same guard must cover the GNN runner when it is written.
Encoding the rule, not remembering it.

The line the guard looks for is fixed here, so the check is not left to guess a format. It is
written into this ADR's header block on amendment, exactly as:

> `**Stage-1 seeds:** <integer>`

matched as `^\*\*Stage-1 seeds:\*\*\s*(\d+)` — one line, one integer, the same shape as the
`**Status:**` line the guard already parses.

### Clause 9 — Ratifications carried over from ADR-011

* **The two recency features** (`steps_since_last_edge`, `steps_since_first_edge`, both measured
  from the view's own cutoff) **stand as defined**. They are part of the parity set, so both arms
  receive them.
* **Floor parity is binding** (ADR-011 clause 4): no gated floor may lack a node statistic the GNN
  receives.
* **The first-appearance row is permitted**: the sampler passed its determinism check on the real
  graph (both modes deterministic, every sampled edge within its seed's time).

### Clause 10 — Per-node scores are persisted, and uncertainty beyond seeds is reported

**Every pilot and gate run writes its per-node score vector** for the window it scored: float32,
one file per run under `experiments/scores/<run_id>.npz` (gitignored — regenerable from a
deterministic config), with the path and its **SHA-256** recorded in the run's registry row. Cost is
~0.7 MB per run, ~17 MB for the whole gate.

**Each file stores the scored node ids alongside the scores**, not scores alone. Without the ids
the paired bootstrap cannot align two arms except by trusting an undocumented ordering, which is
precisely the kind of implicit contract that breaks silently; the bootstrap asserts the two arms'
id vectors are equal before resampling. **ELL-1's exact regret drives this**: AUPRC could not be
computed retroactively because only summary metrics were stored (ADR-007), and without score vectors
the same thing happens to every uncertainty question, every calibration curve and every
precision-at-k, each one costing a full re-run.

**Reported, not gated: a paired bootstrap over test users.** Resample the scored users **once per
replicate, stratified by label** (so prevalence is preserved), score **both arms on the identical
resample**, and report the CI of the *difference* per metric; **1,000 replicates at bootstrap seed
0, both pinned here** so the interval is reproducible from the stored score files rather than
re-derived; and Boyd et al. (2013) logit intervals per arm alongside, since they find bootstrap
slightly biased for AUPRC at skew. Pairing is what makes this the right instrument: it cancels the shared "which users are hard"
component, and measured on synthetic data at this window's support it is ~1.5–2× tighter than the
unpaired form (≈0.005 vs ≈0.009 on the AUPRC difference).

**Option (a) is chosen: the criterion stays seeds-only, and the bootstrap is reported beside it.**
If the two disagree — the seed test resolving while the bootstrap CI straddles zero — **the gate
file must say so in the verdict**, and the claim is qualified accordingly. Gating on the bootstrap
as well (option (b)) is strictly harder and was considered; it is rejected only because changing the
estimand for one gate would break comparability with ELL-1's dated gates, which is a programme-level
decision.

**This is P0 material, and the limitation belongs in the protocol paper.** Bouthillier et al. (2021)
measure data sampling as the dominant variance source and seed-only randomisation as worth ~2 ideal
runs; a protocol paper that reports seed bands alone, without saying what they exclude, would be
open to exactly that criticism. Moving the programme to a multi-source variance model — or to a
P(A>B) ≥ 0.75 criterion — needs its own ADR and a re-statement of every dated gate, so it is named
here and not smuggled in.

**Out of scope:** Gate 3's ablation criteria and seed count, which need their own ADR before those
runs (ADR-007), and Gate 2 (retrieval), which has its own metrics.

## Alternatives rejected

- **Fix the seed count now as a bare number.** Rejected: ADR-005 requires it to come from measured
  variance on a deterministic batch, and DGF-1's variance cannot be known before DGF-1 runs. The
  rule is fixed instead, which is what makes the number mechanical rather than chosen.
- **Run the pilot first, then write this ADR.** Rejected: every other clause here — arms, floor
  specification, pass criteria, reported rows — would then be written with val numbers in view.
  The split is deliberate: the rule is blind, the input is measured.
- **Power the batch for the full val gap rather than half of it.** Rejected: a val effect is an
  optimistic estimate of the test effect, and the val window scores younger users (clause 7).
- **A subsampled floor chosen *for* its seed variance.** Rejected: it would add noise to a
  comparator purely to make `SE_diff` larger — inventing uncertainty to make a test harder — and
  measured arithmetic shows it would also stall clause 5 (the bar barely improves from n=8 to n=20).
  Subsampling may still win clause 2's grid on its merits; that is a different thing.
- **Pre-committing the floor to `subsample = 1.0`** (the first draft). Rejected: benchmarks tune it,
  so fixing it risks an undertuned floor — a bias toward the GNN — and the draft's claim that it
  made the floor "stronger" was unsupported and plausibly backwards.
- **Gating on the paired bootstrap as well (option (b)).** Strictly harder and arguably better
  statistics, but it changes the estimand for one gate only and breaks comparability with ELL-1's
  dated gates. Rejected here, named in clause 10, and revisitable programme-wide in its own ADR.
- **Adopting a multi-source variance model now** (Bouthillier-style data resampling, or P(A>B)).
  Rejected for the same reason, and because it would re-open ADR-006 mid-gate.
- **An absolute margin (e.g. "beat the floor by ≥ 0.02 AUPRC").** Rejected for ADR-006's reason: no
  principled width exists, and the resolvability test is already variance-aware.
- **Select the retune on ROC-AUC.** Rejected: AUPRC is the sensitive metric at 1.48% prevalence
  (ADR-007), so selecting on ROC-AUC would tune for the instrument that hides precision collapse.
- **Early stopping on the validation window for the gate runs.** Rejected: a fixed budget keeps the
  arms comparable and keeps every decision off the scored window.
- **Reduce epochs if a run is slow, decided later.** Rejected as stated; permitted only under
  clause 3's runtime-only condition, applied to every arm, recorded before any test run.
- **Gate the first-appearance row as well.** Rejected: it answers a cold-start question, not
  Gate 1's (ADR-011). It is reported.

## Consequences

- **Nothing may run on 482–821 until clause 8's amendments exist.** The floor runner's guard is
  extended to check for the seed-count line, and the DGF-1 GNN runner inherits the same guard.
- **The DGF-1 trainer is the next build**, and it is constrained by clause 3: inherited ADR-003
  config, exact-neighbourhood scoring, the three sensitivity views from one set of weights.
- **The pilot costs 8–10 runs** (5 GNN + 3 or 5 floor) on the validation window, plus **two** 9-config
  tuning grids (GNN and floor). All are registry rows like any other, tagged as retune/pilot, and
  none is a gate number. The floor's grid is cheap; the GNN's dominates the budget.
- **A new artifact directory, `experiments/scores/`** (gitignored), plus the path and SHA-256 in
  each row, and a bootstrap utility in `gbe.eval` — it is domain-agnostic and every later model
  needs it, so it belongs in the core, not the adapter.
- **`FLOOR_MODELS` gains XGBoost with clause 2's fixed parameters**, and the tuned axes
  (`max_depth`, `subsample`) become explicit run config, hashed like everything else.
- **A Gate-1 failure remains an engine lesson, not a thesis refutation** (doc-02 §0): EDR-1
  proceeds, and the lesson goes to P0 — as ELL-1's did.

## Revisit when

- **Never for DGF-1 Gate 1 once any 482–821 number exists.** From that point, changing arms, floor,
  criteria or seed counts is re-thresholding after seeing a result.
- **Before Gate 3's ADR**, which inherits this structure but re-derives its own counts from its own
  measured variance.
- **If the pilot cannot run** — an op without a deterministic CUDA kernel at 3.7M nodes (ADR-005
  flagged this as blocking for DGF-1), or OOM that no batch size fixes. That is a finding about the
  engine at scale and belongs in its own ADR, not in a weakened Gate 1.
