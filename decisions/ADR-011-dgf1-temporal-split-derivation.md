# ADR-011 — DGF-1 temporal split: node time from the earliest edge, training graph by edge date

**Status:** accepted
**Date:** 2026-09-11 · **accepted** 2026-09-11
**Deciders:** coderback
**Data snapshot:** `DGraphFin.zip` (150,476,320 bytes), verified by `scripts/verify_dgraph_snapshot.py`
(ADR-010). Every number below is from `scripts/measure_dgf1_temporal_split.py`, which is read-only:
it trains nothing, scores nothing and writes no registry row. **The one exception** is the set of
test-window class statistics in *Disclosure*, which a review subagent computed and which are
recorded there for that reason.
**Docs affected — all amendments applied 2026-09-11 on acceptance. Sites found by grepping
`docs/` and `CLAUDE.md` before drafting (ADR-009 practice); `docs/` is gitignored, so ripgrep skips
it and the grep has to be run explicitly. The grep was re-run after applying, to confirm no live
contradiction remains:**
- `docs/02-dgraph-fin-embedding-model-BUILD.md`:
  - §2.3: the "train on the graph as of ≤ cutoff `T`" bullet and the "split derivation is not yet
    fixed" bullet;
  - §4 Phase 0: the split + leakage-test bullet, and the baselines bullet (floor parity, clause 4);
  - §4 Phase 1 Gate 1: what "the tabular floor" means (clause 4);
  - §5: the comparator table's floor row, and the reporting template (the parity floor, plus two
    reported-only sensitivity rows);
  - §7: the Gate-1 row;
  - §8 step 3: "XGBoost on node features" becomes the parity floor.
- `docs/00-shared-core-graph-embedding-GUIDE.md` §7: the "Time-split, not random-split" bullet gains
  one clarifying sentence on what "as of" means for dated edges.
- `docs/timeline.md`: the DGF-1 Phase-0 split row, and the DGF-1 Gate-1 row (parity floor). The
  Gate-1 row was added on application: the timeline mirrors doc-02 §7, which this ADR amends.
- `CLAUDE.md` *Leakage discipline*, strict-inductive line: a DGF-1 pointer. It is the constitution,
  so it was applied on the researcher's explicit instruction at acceptance.

> **Enumeration corrected on application (2026-09-11).** The grep re-run after applying the
> amendments found two more live references to "the tabular floor" that the list above missed:
> doc-00 §9's build-order row and doc-02 §5's "story" line. Neither contradicted this ADR, but both
> now say "parity" explicitly, so the loose wording cannot drift. The only other match, doc-02
> §2.3's "not yet fixed", sits inside the dated quote of the previous wording and is intentional.
> ADR-007, ADR-008 and ADR-009 all undercounted their sites; this one did too, by two.

> **Review note 1 (2026-09-11, pre-acceptance).** An adversarial pass on the first draft, in the
> manner of the ADR-005/006 review, found eleven defects. Three were serious:
> (1) **clause 4 was chosen with the isolation statistic in view, and was said to only help the
> GNN**, but the disclosure did not say so;
> (2) **clause 6 made the stronger leakage guard opt-in** (`edge_time=None`);
> (3) **no clause fixed feature-normalisation statistics to the training window.**
> Five were overclaims:
> (4) "only the minimum" is prefix-determined;
> (5) "size-matched" was said of the whole track;
> (6) two sentences asserted a third party's protocol;
> (7) core placement was justified by future models;
> (8) "strict subset" was stated as a law.
> Three were omissions:
> (9) pytest vs snapshot audit;
> (10) look-ahead stated only inside clause 4;
> (11) the pilot/gate training mismatch.
> All eleven were fixed. (11) was first deferred; it is now resolved by review note 2's clause-2
> decision.
>
> **Review note 2 (2026-09-11): decisions on the three open questions.** These were taken with a
> literature review and a red-team subagent. Each decision below records its evidence.
> - **Clause 2:** gate runs train on `≤ 369`, role-matched to the official split. Previously they
>   trained on `≤ 481`, the ELL-1-as-deployment reading.
> - **Clause 4:** the view is kept, but it gains **floor parity** plus two reported-only
>   sensitivity rows.
> - **Clause 6:** `edge_time` becomes **required, with no default**. ELL-1 derives its edge dates in
>   the adapter.
>
> The red-team also found four defects, all fixed here:
> - "never harder" was an overclaim;
> - the post-appearance outcome channel was unnamed;
> - clause 4's fairness depended on a floor question the ADR left open;
> - the train/test view shift was unmeasured.
>
> One **process failure** is recorded in *Disclosure*: the red-team read test-window labels.

## Disclosure — what had been seen when this was written

- **No DGF-1 model has been trained or scored, on any split.** `adapters/dgf1/` is still a one-line
  scaffold, and every one of the 103 registry rows is `model=ell1`.
- **Data statistics have been seen:** ADR-010's shape/label/degree verification, plus this ADR's
  measurement of support, prevalence, edge counts and neighbourhood growth per window. **The
  windowing rule (clause 2's 70/85) was fixed before any measurement ran.** The two rule constants,
  `TRAIN_FRAC = 0.70` and `VAL_FRAC = 0.85`, were never changed. Later runs of the script only
  *added* reporting.
- Prevalence per window is reported because AUPRC's chance level travels with it (ADR-007). It was
  **not** an input to the cutoffs.
- **Clause 4 is informed.** The as-of-window-end rule was chosen with the 36.66% window-only
  isolation figure in view.
  - Compared with window-only eval it is **expected** to favour the GNN. The direction is **not
    verified**: more edges can hurt as well as help, and GATE-ELL1-3 measured misleading neighbours
    doing worse than none.
  - **Floor parity** (clause 4) hands the node-level part of that advantage to the floor as well.
  - The two reported-only sensitivity rows expose how much the result depends on the view.
- **Test-window label statistics were seen during review (2026-09-11). Recorded verbatim; a reader
  should weigh this.** A red-team subagent reviewing this ADR computed fraud-vs-normal degree
  statistics **on the test window (482–821)**. Its prompt did not forbid test-window labels, and that
  omission is the reviewer's error, not the agent's. What was seen:
  - mean degree, normal vs fraud: **1.90 vs 1.42** under the as-of-821 view, and **1.38 vs 1.19**
    at each node's first appearance;
  - share at degree 1, normal vs fraud: **0.50 vs 0.75** as of 821, and **0.74 vs 0.88** at first
    appearance.

  So degree separates the classes more under clause 4's view than under the first-appearance view.
  No model was trained or scored. The agent also reported statistics for the ≤ 481 view (mean
  degree 2.32 vs 1.77). Those use pre-test (train + val) labels, not the held-out test window. **Ordering:** clause 4's
  as-of-window-end rule was drafted *before* this look.

  **Rule from here on:** any change to clauses 2–4 made after this point must either make Gate 1
  harder for the GNN, or rest entirely on label-free evidence. (When first written in review, the
  rule read "only harder". It is refined here because a direction-neutral change justified without
  labels cannot exploit the look either. **The researcher accepted the refinement on 2026-09-11,
  together with this ADR.**) The three post-look decisions satisfy the rule as follows:
  - **Clause 2 (role-matching)** is justified entirely by label-free evidence: pilot/gate identity,
    the controlled comparison, and the section-5 view match.
  - **Clause 4's floor parity** makes Gate 1 strictly harder for the GNN.
  - **Clause 6** only strengthens a leakage guard.
  - **The sensitivity rows** are reported, not gated.

  Future subagent prompts must forbid held-out labels explicitly.

## Context

ADR-010 found that DGraph's official split is a random mask. DGF-1's gates therefore run on a
temporal split we build ourselves, from `edge_time` (steps 1–821). ADR-010 deferred how that split
is derived, with one constraint: *a node's assigned time may not depend on any edge after the
cutoff.* doc-02 §4 Phase 0 requires the split and its leakage test to ship in the same session.

Answering "how does a node get a time?" is only half the problem. The existing core's guards were
written for Elliptic, where **every edge lies inside one time step** (all 468,710 of them). So
"both endpoints are pre-cutoff" implied "the edge is pre-cutoff", and filtering the graph by node
time was enough. `gbe.eval.induced_train_subgraph` and `gbe.eval.assert_no_temporal_leakage` both
reason **only about node times** as a result. DGraph breaks that implication: users persist, and two
users who both appeared before the cutoff can gain an edge between them long after it.

**Measured (directed edges, before reverse edges are added):**

| training cutoff | edge-dated graph (`t_e ≤ T`) | node-induced graph (both endpoints `≤ T`) | post-cutoff edges admitted by the node-induced graph | labelled in-window nodes touched |
|---|---|---|---|---|
| **`T = 369` (the training cutoff, clause 2)** | 1,743,128 | 2,142,494 | **399,366 (18.64%)** | 324,249 of 858,702 (37.76%) |
| `T = 481` (rejected alternative, for scale) | 2,704,431 | 3,021,063 | 316,632 (10.48%) | 254,217 of 1,042,132 (24.39%) |

If ELL-1's core were used unchanged on DGraph, **hundreds of thousands of future edges would enter
training, and the leakage assertion would stay silent**, because it checks endpoints, not edges.
That is the failure constitution-level leakage tests exist to stop. ELL-1 never exercised the
difference: zero of its 468,710 edges crossed the cutoff (GATE-ELL1-3).

The eval graph has the same problem in reverse. ELL-1 scored the test window using **only edges
inside the window**. On Elliptic that choice cost nothing. On DGraph, new users mostly connect to
users who already existed:

| test window 482–821 | value |
|---|---|
| edges incident to a test node | 1,279,936 |
| … with both endpoints inside the window | 411,021 (**32.11%**) |
| labelled test nodes left **isolated** by window-only eval | **67,266 of 183,469 (36.66%)** |

**Neighbourhoods keep growing after users appear, so the view matters.** Of 8,601,998 edge
incidences, **46.27%** arrive after the endpoint's first appearance, and **33.40%** arrive more than
100 steps after it. So the training view and the test view have to be compared, not assumed alike.
Label-free, labelled nodes only:

| view | node age at scoring (median / mean steps) | degree (median / mean) |
|---|---|---|
| **train `≤ 369`, graph as of 369** | 219 / 213.47 | 2 / 1.87 |
| train `≤ 481`, graph as of 481 (rejected) | 298 / 279.12 | 2 / 2.31 |
| **test 482–821, graph as of 821** | 202 / 188.40 | 1 / 1.89 |
| test 482–821, graph at first appearance | — | 1 / 1.38 |

## Decision

### Clause 1 — A node's time is the earliest time of any edge touching it

`t_node(v) = min { t_e : e is incident to v }`. Both endpoints count, because being named as
someone's emergency contact is evidence that the node exists. ADR-010 measured **zero isolated
nodes**, so every node gets a time. The implementation asserts this rather than assuming it.

**Why the minimum satisfies ADR-010's constraint.** For any boundary `B`, `t_node(v) ≤ B` holds
exactly when `v` has an incident edge dated `≤ B`. So membership of every window, at every
boundary, is decided **only by edges dated at or before that boundary**. Call this
*prefix-determined*. A max- or mean-based time lacks the property: a later edge can move a node
across the boundary.

The minimum is **not** the only prefix-determined proxy: the time of a node's k-th earliest edge
has the property too. It is the only one that **gives every node a time**, because for k ≥ 2 a node
with fewer than k edges has none, and 38.9% of nodes sit at total degree ≤ 1 (ADR-010). It is also
the least assumption-laden reading of "first appearance" (`verify_dgraph_snapshot.py`). Those two
reasons choose it among the proxies the constraint admits.

### Clause 2 — The windows come from a fixed rule, and their roles match the official split

**The rule.** `train_max` is the first step at which the cumulative **labelled** count reaches
**70%**, and `val_max` the first step at which it reaches **85%**. On the snapshot:

| window | steps | role | labelled | fraud | prevalence | background | all nodes |
|---|---|---|---|---|---|---|---|
| train | 1–369 | **the one training set**, for every run | 858,702 | 10,317 | 1.2015% | 1,206,715 | 2,065,417 |
| val | 370–481 | the ADR-009 retune and the ADR-007 seed pilot are **scored** here; nothing trains on it | 183,430 | 2,475 | 1.3493% | 496,064 | 679,494 |
| test | 482–821 | gate scoring only | 183,469 | 2,717 | 1.4809% | 772,170 | 955,639 |
| *official (random)* | — | train / val / test | 857,899 / 183,862 / 183,840 | 10,857 / 2,326 / 2,326 | 1.2655% / 1.2651% / 1.2652% | 0 | — |

**The roles are matched to the official split, not only the proportions.** Every run trains on
`≤ 369`: the `lr` + `batch_size` retune, the seed pilot, and every gate run of both the floor and
the GNN. Val is used only to score the retune and the pilot. Test is scored only by gate runs. All
three windows match the official masks in size to within 0.3%: train 858,702 vs 857,899, val
183,430 vs 183,862, test 183,469 vs 183,840. Four reasons, three of them label-free measurements or
properties:

1. **The pilot measures the configuration it sizes.** ADR-007's seed pilot and the gate runs share
   one training set, so the seed variance the pilot measures is the gate's own, not a proxy
   extrapolated from a smaller training set.
2. **Random-vs-temporal becomes a controlled comparison.** Sizes and roles are identical; only how
   the split is drawn differs. That is the stated reason for mirroring the official proportions.
   The literature review found **no citable precedent** for mirroring a dataset's official
   proportions as such (TGB uses 70/15/15 of *edges*; OGB and RelBench cut on calendar time), so
   the justification has to be the comparison it enables, and it is.
3. **The model is scored on the kind of view it was trained on.** As of 369, training nodes are
   median age 219 with mean degree 1.87. As of 821, test nodes are median age 202 with mean degree
   1.89 (*Context*). Training on `≤ 481` would break that: median age 298, mean degree 2.31.
4. **It is the correct reading of ELL-1's precedent.** ELL-1's split *was* the published Weber et
   al. split, so "follow ELL-1" means "match the published split's roles". ELL-1's retrain-on-all
   pattern was an artefact of having no official validation window.

**Cost:** 183,430 labelled val nodes never enter a gate run's training set, and there are 112 steps
between training's end and test's start. Both apply identically to the floor and the GNN.

The concrete numbers 369 / 481 / 821 live in `adapters/dgf1/config.yaml`, never in `gbe/`
(gbe rule 3), so they appear in every row's config hash. The *rule* lives here.

**Scored targets:** labelled nodes whose `t_node` falls inside the window. A labelled user who
appeared before the cutoff is never a test target, even though users persist. DGF-1's temporal
claim is exactly this: *detect fraud among users who first appear after the cutoff.* Write-ups must
use that sentence, not a looser one.

### Clause 3 — The training graph is filtered by edge date, not by node time

The training graph is **exactly the edges with `t_e ≤ 369`**. Reverse edges (doc-02 §2.2.4, on by
default) are added **after** filtering and keep the forward edge's date. Because of clause 1, every
edge dated `≤ T` already has both endpoints at time `≤ T`, so the edge-dated graph is always a
subset of the node-induced one. On this snapshot it is a strict subset: this clause removes the
399,366 edges in *Context*.

Every feature derived from edges (doc-02 §2.2.4b's 11-wide edge-type histogram and recency
features) is computed from **the same edge set as the graph or view it feeds**. **Recency is
measured from the view's own cutoff** (369 when training; the window's last step when scoring), never
from step 821. Measuring from 821 would bake the dataset's end date into every training feature.

**Every fitted feature statistic is fitted on nodes with `t_node ≤ 369` only.** That covers scalers,
quantile transforms, and any normalisation of the raw 17 features or the edge-derived ones, for the
floor and the GNN alike. This is doc-01 §2.2.5's fit-on-train rule carried over. The feature tensor
covers all 3.7M nodes, so a statistic computed over it would include test-window users.

### Clause 4 — The eval graph is the graph as of the window's end, and the floor sees what the GNN sees

**The view.** Scoring a window uses the frozen `≤ 369`-trained model on **all edges dated at or
before the window's last step**: edges `≤ 481` to score val, edges `≤ 821` to score test.
- Pre-window nodes take part **as neighbours only**. Their features enter message passing, their
  labels never do, and no parameter or normalisation statistic is updated. LayerNorm is per-node,
  and BatchNorm stays banned.
- This is strict-inductive where it matters: the model is trained only on the clause-3 graph, and
  no post-cutoff edge or label reaches a parameter.
- It departs **explicitly** from ELL-1's window-only eval, because of the measured 36.66% isolation.

**Why this view.** Training scores users with everything up to 369, and test scores users with
everything up to 821. Both are the same task, *detection from a snapshot taken at the window's end*,
and *Context* measures the two views as matched in node age and degree. It honours the rule that
TGB (arXiv:2307.01026) and RelBench (arXiv:2407.20060) apply: a prediction may use only history up
to its prediction time. Here the prediction time is the window's end. That the two benchmarks'
rule is satisfied is this ADR's inference; it claims nothing else about any third party's protocol
(ADR-007). Labels are undated, so the data cannot settle *when* the prediction should happen
relative to the outcome. That is why the first-appearance alternative is reported below rather than
dismissed.

**Floor parity (binding on DGF-1's Gate-1 pre-registration).** The **gated** tabular floor receives
**every node-level statistic derived from the graph view that the GNN's input features contain**:
- at minimum, doc-02 §2.2.4b's 11-wide edge-type histogram (its row sum is the node's degree) and
  the recency features;
- computed from the same view as the GNN's: the `≤ 369` graph for training, the window-end graph
  for scoring.

The raw-17-feature floor is **reported, not gated**. Neighbourhoods grow after appearance (*Context*),
and degree correlates with the label in the pre-test windows (*Disclosure*), so without parity the
GNN alone would read post-appearance activity through degree. With parity, the node-level part of
that channel reaches both arms, and Gate 1 tests **message passing beyond local counts**. That is
the same bar ELL-1's RF floor set by already containing one-hop aggregates, and the conservative
direction: it makes Gate 1 strictly harder for the GNN.

**Reported, not gated: two sensitivity rows, both scored from the same trained weights.** Floor and
GNN are both scored under each view, with the floor's view-derived features recomputed from that
view, so each row stays a GNN-vs-floor comparison. Neither row has a pass condition, and neither may
acquire one retroactively.
1. **Window-only view:** only edges with both endpoints inside the window, which is ELL-1's rule.
   One extra forward pass.
2. **First-appearance view:** each scored node sees only edges dated at or before its own `t_node`.
   This uses PyG `NeighborLoader` temporal sampling, with `time_attr` set to the edge time and
   `input_time = t_node`. The literature review read the pyg-lib source to confirm the details:
   - the cutoff is inclusive (`≤`);
   - it is enforced at **every hop** against the seed's time;
   - it is supported by the installed pyg-lib 0.8.0 / PyG 2.8.0.post1.

   **Produced only if** the temporal sampler passes a determinism test under ADR-005's strict
   setting. If it fails, the row is not produced, that fact is recorded, and there is no
   non-deterministic fallback.

   A model *trained* on first-appearance views is a different experiment and is not pre-registered
   here.

If the GNN beats the floor on the gated view but not on the first-appearance view, a reader knows
the gain depends on structure that forms after users appear. That is a result, not an embarrassment,
and it is reported either way.

> **Amended 2026-09-15 (ADR-013): the two sensitivity rows were not produced in this clause's form.**
> - **Why:** the GNN's implementation built its node inputs from the window-end graph under every
>   view, breaking clause 3. Its reported-view numbers are **withdrawn** from all inference (ADR-013
>   clauses 1–2; `gates/ERRATUM-DGF1-1.md`).
> - **Correctly built, the rows cannot answer their question.** They score at-arrival inputs with
>   snapshot-trained models, so they cannot separate post-appearance dependence from worse transfer.
> - **Where the question goes:** a matched-time pre-registration, with these rows run beside it as
>   exploratory (ADR-013 clause 5).
>
> The clause above is kept as accepted.

### Clause 5 — Leakage tests, written in the implementation session (untested guards don't exist)

**pytest, on synthetic fixtures.** This is the repo's practice (`tests/test_ell1_leakage.py`,
`tests/test_dgraph_verification.py`): the snapshot is gitignored, so the suite must not need it.
Each fixture includes at least one edge between two pre-cutoff nodes dated after the cutoff, the
case ELL-1's fixtures never contained.

1. **Exact equality, not just a bound:** the adapter's training edge set equals
   `edges_as_of(…, t_max=369)` after reverse-edge doubling. That catches leaked edges and wrongly
   dropped ones.
2. Every training seed (labelled node used in the loss) has `t_node ≤ 369`.
3. **The guard has teeth:** the node-induced graph *fails* the edge-date assertion. Calling the
   assertion or the filter **without** `edge_time` raises `TypeError`.
4. **Prefix-determinism:** node times recomputed from only the edges dated `≤ B` give identical
   window membership at every boundary `B`.
5. An eval view built for a window contains no edge dated after that window's last step.
6. Edge-derived features computed for training equal features recomputed from the `≤ 369` edge set
   alone, with recency measured from 369.
7. **Fit-on-train:** perturbing the features of nodes with `t_node > 369` leaves every fitted
   feature statistic unchanged.
8. **Floor parity:** for the same nodes and view, the floor's view-derived columns equal the
   view-derived columns in the GNN's input.
9. **ELL-1 derivation:** the adapter's derived edge date equals the shared step. A fixture edge
   joining two different steps makes the derivation raise.
10. **Temporal sampler** (only if the first-appearance row is produced): every sampled edge, at
    every hop, has `t_e ≤` its seed's `t_node`, and two runs at one seed give identical batches.

**Snapshot audit, in a script, not pytest** (the `verify_dgraph_snapshot.py` pattern). It runs once
when the DGF-1 DataSource first loads the real graph, and exits non-zero on any mismatch:
- cutoffs 369 / 481;
- clause 2's per-window support;
- the 399,366 node-induced-minus-edge-dated count at 369;
- zero isolated nodes;
- invariants 1, 4 and 5 re-checked on the real graph.

A mismatch is a doc-vs-data discrepancy and is escalated, never absorbed (ADR-001).

### Clause 6 — Edge dates become a required input to the core; node-time derivation stays in the adapter

- **Core (`gbe.eval.temporal`):** `edges_as_of(edge_index, *, edge_time, t_max)` and
  `assert_no_temporal_leakage(edge_index, time_step, split, *, edge_time)`.
  - **`edge_time` is keyword-only and required, with no default.** Forgetting it raises
    immediately, instead of silently falling back to the node-only check. That is ADR-005's
    loud-over-silent principle.
  - Why the core:
    - doc-00 §7 makes leakage checks harness-level.
    - An edge date is not domain knowledge.
    - The check is defined identically for **both models that exist today**: Elliptic's edges
      carry a date too, the step both endpoints share.
    - P0 packages `gbe.eval` as a protocol for *dated graphs*. A harness whose guard is correct only
      for within-step edges would be a defect in the published artifact.
- **ELL-1 derives its edge dates in its adapter, never through a core default.**
  - It sets `edge_time` to the shared step of each edge's endpoints, and asserts that every edge lies
    within one step.
  - A core default of "the later endpoint's time" *is* the node-induced graph, which is exactly the
    leak in *Context*, so no such default exists.
  - Measured on the real Elliptic data: all 468,710 edges lie within one step. The edge-date filter
    reproduces the node-induced train graph **exactly, with the same edges in the same order**
    (`torch.equal`), at both ELL-1 cutoffs, 29 and 34. The refactor is therefore bit-for-bit safe
    by construction. `scripts/check_extract_regression.py` (ADR-008) confirms it after the refactor
    rather than by argument.
- **`induced_train_subgraph` is retired from every call site** (`adapters/ell1/train_gnn.py`,
  `adapters/ell1/hpo.py`, the tests) and replaced by `edges_as_of` with ELL-1's derived edge dates.
  That is part of EXTRACT's refactor of ELL-1 onto the core.
- **Adapter (`adapters/dgf1/`):** the earliest-edge node time. ELL-1 has a native node time, and no
  second model derives node time this way, so it stays out of the core: the inclusion rule, "when
  unsure, keep it out". Promoting it later is cheap.

### Scope boundary

This ADR fixes the **split and protocol only**. **DGF-1's Gate-1 pre-registration** is a separate,
later ADR:
- the seed count, derived from a validation pilot on this split's val window;
- a floor of ≥ 5 seeds per arm;
- ADR-006's two-stage design.

ADR-010 anticipated one "Phase-0/Gate-1 ADR". The two are separated deliberately, because the pilot
cannot run until this split exists. Together they discharge ADR-010's requirement. The first draft
left the Gate-1 ADR an open question about pilot/gate training mismatch; clause 2 removes it. The
Gate-1 ADR is **bound by clause 4's floor parity**: it may add node-level view statistics to both
arms, but may not gate against a floor that lacks any the GNN receives.

The official-split track trains and scores on the official masks over the full graph as distributed.
That is our own definition, not a claim about any third party's protocol. It is always labelled
*random-split, leaderboard-comparable*, and by design carries none of this ADR's guarantees
(ADR-010 clause 1).

## Limitations recorded, not fixable from this snapshot

- **Labels carry no date.** A training user's fraud label may reflect behaviour observed after the
  cutoff. This affects both arms equally, so the *comparison* is fair, but absolute numbers are
  optimistic relative to a real as-of-`T` deployment. Every write-up carries this caveat.
- **Node features are a profile snapshot taken at release.** They cannot be verified as pre-cutoff.
  This also affects both arms equally.
- **Activity after appearance, including possibly after default, is in the scoring view (clause
  4).** A test user who joined at step 490 is scored with its edges up to 821. Floor parity gives
  the node-level part of this (degree, type histogram, recency) to **both** arms. What remains
  **GNN-only** is *who* a user's later neighbours are and what their features say. The
  first-appearance sensitivity row measures how much the result depends on it. A reviewer may call
  that remainder post-outcome leakage; the ADR does not claim it is not.
  *Amended 2026-09-15 (ADR-013): no valid first-appearance row exists. The GNN figure once reported
  under that name carried post-appearance node statistics and is withdrawn. This dependence stays
  **not established** until ADR-013 clause 5's pre-registration produces rows.*
- **Prevalence drifts upward** across the windows: 1.20% → 1.35% → 1.48%. The val pilot's AUPRC and
  the test AUPRC have different chance levels. Per ADR-007 they are never compared, and each is
  reported with its own prevalence and positive count.

## Alternatives rejected

- **Max- or mean-based node time.** Violates ADR-010's constraint: a post-cutoff edge can move a
  node's time across the boundary.
- **Source-only node time (only out-edges count).** Leaves any node with only incoming edges
  without a time, and throws away evidence of when a node existed.
- **Reuse ELL-1's node-induced training graph unchanged.** Measured to admit 399,366 post-cutoff
  edges, touching 37.76% of labelled training nodes, which the current guard cannot see. This is
  the alternative this ADR most exists to reject.
- **`edge_time` optional, defaulting to `None`** (the first draft). That makes the stronger guard
  opt-in, so a forgotten argument silently reproduces the leak.
- **A core default of "the later endpoint's time"**, for graphs without dates. It *is* the
  node-induced graph, which is the leak. Undated edges must have their date derived and asserted
  by the adapter that knows why the derivation is valid.
- **Keep the edge-date check in the adapter.** Leaves a guard in the core that is known to be
  insufficient for dated graphs, while P0 packages that core as a dated-graph protocol.
- **Train gate runs on all pre-test time (`≤ 481`)**, the first draft's reading of the ELL-1 pattern.
  It breaks three things:
  - the pilot would size a configuration it doesn't measure;
  - the random-vs-temporal comparison would no longer be role- and size-controlled;
  - the training view (median age 298, mean degree 2.31) would no longer match the test view (202,
    1.89).

  ELL-1's precedent, read correctly, is to match the published split's roles.
- **Cutoffs at fixed fractions of calendar time.** This has precedent (OGB, RelBench), but labelled
  support per window would depend on arrival rates. The controlled comparison with the official
  split, which is the reason to prefer 70/85, would be lost.
- **ELL-1's window-only view as the gated view.** Isolates 36.66% of labelled test nodes and hands
  the floor an advantage created by the protocol. It is kept as a reported row.
- **Gate on first-appearance views (Δ = 0), for scoring or for training and scoring** (the
  red-team's recommendation).
  - A Gate-1 failure would then be uninterpretable: the test graph is barely formed at first
    appearance (mean degree 1.38), so "structure doesn't help at scale" and "the structure hadn't
    formed yet" would be indistinguishable. Gate 1 exists to answer the first question.
  - The sampler's determinism is also unverified.
  - It is kept as a reported row, conditional on that verification. If write-ups ever need a
    deployment claim ("detect at sign-up") rather than a detection-at-snapshot claim, this is the
    decision to revisit, in its own ADR.
- **A fixed observation window Δ > 0.** Standard in credit scoring, but Δ is a free parameter with
  no anchor in this data. Choosing it after the review look would also sit badly with *Disclosure*'s
  rule.
- **A raw-17-feature floor as the gated floor.** It leaves the node-level post-appearance channel
  GNN-only. It is kept as a reported row.
- **An edge-level holdout.** DGF-1 is node classification, and labels live on nodes.
- **Put the node-time derivation in the core.** Premature: no second model derives node time this
  way (doc-00 §1).
- **Fold this into the Gate-1 pre-registration ADR.** That ADR needs a validation pilot, and the
  pilot needs this split.
- **Implement it in this session.** Doc-first: an idea is proposed in the session it was
  conceived, and built only after acceptance.

## Consequences

- **Core, in `gbe/eval/temporal.py`:**
  - `edges_as_of` and the edge-date leakage assertion, both with a required, keyword-only
    `edge_time`;
  - `induced_train_subgraph` retired from every call site;
  - ELL-1's call sites (`train_gnn.py`, `hpo.py`) and `tests/test_ell1_leakage.py` move to the new
    API, passing edge dates derived in the adapter.
  - The ADR-008 checker must report every one of the 49 reference rows bit-for-bit. The `torch.equal`
    measurement above is why it should.
- **`adapters/dgf1/`:**
  - the DataSource adds node time, `labelled_mask = (y == 0) | (y == 1)` (ADR-010), and edge-derived
    features built per view;
  - its config pins 369 / 481 / 821 and `reverse_edges: true`.
- **The gated tabular floor** is the parity floor:
  - raw 17 features plus the view-derived node statistics;
  - trained on labelled nodes `≤ 369`, with every fitted statistic fitted on that window;
  - scored on labelled nodes in 482–821;
  - both metrics routed through `gbe.eval.classification_metrics` (ADR-007).

  The raw-17 floor is a reported row.
- **Every gated row gains two reported sensitivity scores**, window-only and first-appearance (the
  latter conditional on the determinism test), logged in the same registry row under distinct keys,
  so they can never be confused with the gated number or reported apart from it.
  *Amended 2026-09-15 (ADR-013): these keys, as logged in all 24 DGF-1 GNN rows, are withdrawn, and
  `run_dgf1` no longer computes them. Any future reported-view scores live in their own rows,
  carrying the reproduced gated metrics beside them (ADR-013 clauses 2, 3 and 5).*
- **DGF-1 Gate 3 inherits a problem ELL-1 never had.** ELL-1's ablations kept cutoff integrity by
  rewiring within a time step. DGraph edges don't live in steps, so any rewiring has to preserve
  each edge's date, or it forges future edges into the training graph. That belongs to Gate 3's own
  ADR, but it is recorded now so it isn't rediscovered then.
- `scripts/measure_dgf1_temporal_split.py` is committed as provenance for every number here except
  the disclosed review statistics.

## Revisit when

- **DGraphFin-2** turns out to replace the original (ADR-010's *Revisit when*).
- **Label or feature timestamps become available.** The limitations above would then become fixable
  and should be fixed. The first-appearance or observation-window question would then have data to
  settle it.
- **A write-up needs a deployment claim** ("detect at sign-up"). Decide gated first-appearance views
  in their own ADR, **before** any DGF-1 score exists, and never after.
- **Never for DGF-1 once any DGF-1 score exists on 370–481 or 482–821.** From that point, changing
  the split, the view or the floor is re-thresholding after seeing a result.
