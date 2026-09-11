# ADR-011 — DGF-1 temporal split: node time from the earliest edge, training graph by edge date

**Status:** proposed
**Date:** 2026-09-11
**Deciders:** coderback
**Data snapshot:** `DGraphFin.zip` (150,476,320 bytes), verified by `scripts/verify_dgraph_snapshot.py`
(ADR-010). Every number below is from `scripts/measure_dgf1_temporal_split.py`, which is read-only:
it trains nothing, scores nothing and writes no registry row.
**Docs affected (amendments applied on acceptance, not before). Sites found by grepping `docs/`
and `CLAUDE.md` before drafting (ADR-009 practice); `docs/` is gitignored, so ripgrep skips it and
the grep has to be run explicitly:**
`docs/02-dgraph-fin-embedding-model-BUILD.md` §2.3 (the "train on the graph as of ≤ cutoff `T`"
bullet and the "split derivation is not yet fixed" bullet), §4 Phase 0 (the split + leakage-test
bullet), §5 (one reported-only row: the window-only eval sensitivity, clause 4);
`docs/00-shared-core-graph-embedding-GUIDE.md` §7 (the "Time-split, not random-split"
bullet: one clarifying sentence on what "as of" means for dated edges); `docs/timeline.md` DGF-1
Phase-0 split row; `CLAUDE.md` *Leakage discipline*, strict-inductive line (a DGF-1 pointer, which is
the researcher's call because it is the constitution).

> **Review note (2026-09-11, pre-acceptance).** An adversarial pass on the first draft, in the
> manner of the ADR-005/006 review, found eleven defects. Three were serious:
> (1) **clause 4 was chosen with the isolation statistic in view, and can only help the GNN** against
> a floor that uses no graph, yet the disclosure did not say so;
> (2) **clause 6 made the stronger leakage guard opt-in** (`edge_time=None`), so a DGF-1 call site
> that forgot the argument would reproduce the exact silent failure this ADR exists to prevent;
> (3) **no clause fixed feature-normalisation statistics to the training window**, the classic leak
> doc-01 §2.2.5 closed for ELL-1.
> Five were overclaims:
> (4) "only the minimum" is prefix-determined (false: every k-th-earliest edge time is);
> (5) the temporal track was called size-matched to the official one (true of the windows, false of
> the gate training set);
> (6) two sentences asserted what a third party's protocol does, which ADR-007 says we cannot verify;
> (7) clause 6 justified core placement by future models (the speculative pushing doc-00 §1
> forbids);
> (8) "strict subset" was stated as a law when it is a property of this snapshot.
> Three were omissions:
> (9) clause 5 did not say which checks are pytest (synthetic fixtures, the repo's practice) and
> which are real-snapshot audits;
> (10) look-ahead within a window was a limitation stated only inside clause 4;
> (11) the seed pilot trains on `≤ 369` while gate runs train on `≤ 481`.
> All eleven are fixed below. No rule constant, window boundary or measured number changed.

## Disclosure — what had been seen when this was written

- **No DGF-1 model has been trained or scored, on any split.** `adapters/dgf1/` is still a one-line
  scaffold, and every one of the 103 registry rows is `model=ell1`. No threshold here can have been
  tuned to a result, because no result exists.
- **Data statistics have been seen:** ADR-010's shape/label/degree verification, plus this ADR's
  measurement of support, prevalence and edge counts per window. **The windowing rule (clause 2) was
  fixed before that measurement ran.** The script was run twice: the second run only *added
  reporting* (the final-window row and the cutoff-481 edge counts). The two rule constants,
  `TRAIN_FRAC = 0.70` and `VAL_FRAC = 0.85`, were never changed.
- Prevalence per window is reported because AUPRC's chance level travels with it (ADR-007). It was
  **not** an input to the cutoffs.
- **Clause 4 is informed, and its direction favours the GNN; a reader is entitled to weigh that.**
  The eval-graph rule was chosen with the 36.66% window-only isolation figure in view. The tabular
  floor uses no graph, so the eval-graph rule moves only the GNN arm. Compared with ELL-1's
  window-only rule, this clause can make Gate 1 **easier** to pass, never harder. That is the
  opposite of the asymmetry that defended ADR-007. The defence here is a principle, not an
  asymmetry: window-only eval deletes edges that existed at scoring time, which is the artificial
  choice. doc-00 §7's "as of" reading is the natural one. **Mitigation:** the window-only score is
  computed from the same trained models and reported beside the gated one (clause 4), so how much
  the result depends on this choice is visible, not hidden.

## Context

ADR-010 found that DGraph's official split is a random mask. DGF-1's gates therefore run on a
temporal split we build ourselves, from `edge_time` (steps 1–821). ADR-010 deferred how that split
is derived, with one constraint: *a node's assigned time may not depend on any edge after the
cutoff.* doc-02 §4 Phase 0 requires the split and its leakage test to ship in the same session.

Answering "how does a node get a time?" is only half the problem. The existing core's guards were
written for Elliptic, where **every edge lies inside one time step**. So "both endpoints are
pre-cutoff" implied "the edge is pre-cutoff", and filtering the graph by node time was enough.
`gbe.eval.induced_train_subgraph` and `gbe.eval.assert_no_temporal_leakage` both reason **only
about node times** as a result. DGraph breaks that implication: users persist, and two users who
both appeared before the cutoff can gain an edge between them long after it.

**Measured, at the cutoffs this ADR proposes (directed edges, before reverse edges are added):**

| training cutoff | edge-dated graph (`t_e ≤ T`) | node-induced graph (both endpoints `≤ T`) | post-cutoff edges admitted by the node-induced graph | labelled in-window nodes touched |
|---|---|---|---|---|
| inner, `T = 369` | 1,743,128 | 2,142,494 | **399,366 (18.64%)** | 324,249 of 858,702 (37.76%) |
| final, `T = 481` | 2,704,431 | 3,021,063 | **316,632 (10.48%)** | 254,217 of 1,042,132 (24.39%) |

If ELL-1's core were used unchanged on DGraph, **hundreds of thousands of future edges would enter
training, and the leakage assertion would stay silent**, because it checks endpoints, not edges.
That is the failure constitution-level leakage tests exist to stop. It is invisible here because
ELL-1 never exercised the difference: zero of its 468,710 edges crossed the cutoff (GATE-ELL1-3).

The eval graph has the same problem in reverse. ELL-1 scored the test window using **only edges
inside the window**, with no forward pass over any other part of the graph. On Elliptic that choice
cost nothing. On DGraph, new users mostly connect to users who already existed:

| test window 482–821 | value |
|---|---|
| edges incident to a test node | 1,279,936 |
| … with both endpoints inside the window | 411,021 (**32.11%**) |
| labelled test nodes left **isolated** by window-only eval | **67,266 of 183,469 (36.66%)** |

Copying ELL-1's eval rule would silently switch off message passing for over a third of the scored
nodes. That would hand the no-graph floor an advantage the protocol created, not the data.

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

### Clause 2 — The windows come from a fixed rule over labelled node time

- `train_max` is the first step at which the cumulative **labelled** count reaches **70%**, and
  `val_max` the first step at which it reaches **85%**. This mirrors the official split's 70/15/15
  over labelled nodes, so the **scored windows** of the two tracks match in size to within 0.3%:
  val 183,430 vs 183,862, test 183,469 vs 183,840. The training sets deliberately do **not** match.
  The gate model trains on `≤ 481` (1,042,132 labelled) where the official train mask holds 857,899.
  That follows ELL-1's pattern (next bullet), and comparisons across tracks must say so.
- On the snapshot this gives:

| window | steps | labelled | fraud | prevalence | background | all nodes |
|---|---|---|---|---|---|---|
| inner train | 1–369 | 858,702 | 10,317 | 1.2015% | 1,206,715 | 2,065,417 |
| val | 370–481 | 183,430 | 2,475 | 1.3493% | 496,064 | 679,494 |
| test | 482–821 | 183,469 | 2,717 | 1.4809% | 772,170 | 955,639 |
| **final train** (inner train + val) | 1–481 | 1,042,132 | 12,792 | 1.2275% | 1,702,779 | 2,744,911 |
| *official (random), for contrast* | — | 857,899 / 183,862 / 183,840 | — | 1.2655% / 1.2651% / 1.2652% | 0 | — |

- **This follows ELL-1's pattern (ADR-003):** tune on the inner split (train `≤ 369`, validate on
  370–481), then train the gate model on all pre-test time (`≤ 481`) and score 482–821. The val
  window is where ADR-009's `lr` + `batch_size` retune and ADR-007's seed-count pilot run.
  **Neither may touch 482–821.**
- The concrete numbers 369 / 481 / 821 live in `adapters/dgf1/config.yaml`, never in `gbe/`
  (gbe rule 3). That puts them in every row's config hash. The *rule* lives here.
- **Scored targets:** labelled nodes whose `t_node` falls inside the window. A labelled user who
  appeared before the cutoff is never a test target, even though users persist. DGF-1's temporal
  claim is exactly this: *classify users who first appear after the cutoff.* Write-ups must use that
  sentence, not a looser one.

### Clause 3 — The training graph is filtered by edge date, not by node time

The training graph at cutoff `T` is **exactly the edges with `t_e ≤ T`**. Reverse edges (doc-02
§2.2.4, on by default) are added **after** filtering and keep the forward edge's date. Because of
clause 1, every edge dated `≤ T` already has both endpoints at time `≤ T`, so the edge-dated graph is
always a subset of the node-induced one. On this snapshot it is a strict subset: the difference is
exactly the 399,366 / 316,632 edges in *Context*, and this clause removes them.

Every feature derived from edges (doc-02 §2.2.4b's 11-wide edge-type histogram and recency
features) is computed from **the same edge set as the graph it feeds**. A training-time feature may
not count an edge the training graph excludes. **Recency is measured from the view's own cutoff**
(`T` when training, the window's last step when scoring), never from step 821. Measuring from 821
would bake the dataset's end date into every training feature.

**Every fitted feature statistic is fitted on nodes with `t_node ≤ T` only.** That covers scalers,
quantile transforms, and any normalisation of the raw 17 features or the edge-derived ones, at each
cutoff (`≤ 369` for tuning and the pilot, `≤ 481` for gate runs). It applies to the tabular floor
and the GNN alike. This is doc-01 §2.2.5's fit-on-train rule carried over. The feature tensor
covers all 3.7M nodes, so a statistic computed over it would include test-window users.

### Clause 4 — The eval graph is the graph as it stood at the end of the window

Scoring a window uses a **frozen** model on **all edges dated at or before the window's last step**:
edges `≤ 481` for validation (inner-trained model), edges `≤ 821` for test (final-trained model).
Pre-window nodes take part **as neighbours only**: their features enter message passing, their
labels never do, and no parameter or normalisation statistic is updated. LayerNorm is per-node, and
BatchNorm stays banned.

This departs **explicitly** from ELL-1's window-only eval. The reason is the measured 36.66%
isolation, not a preference. It is still strict-inductive where it matters: the encoder is trained
only on the clause-3 graph, and no post-cutoff edge or label reaches a parameter. The cost: a node
early in a window can be scored with edges from later in the same window (see *Limitations*). The
dataset ships as one graph with no per-edge filtering. This ADR claims nothing about how any third
party evaluated on it (ADR-007).

**Reported, not gated: the window-only sensitivity.** Each trained GNN is also scored under ELL-1's
window-only rule, with edges restricted to both endpoints inside the window. This is a second
forward pass from the same weights, costing no training. It is reported beside the gated number,
labelled as a protocol sensitivity. It has no pass condition and may not acquire one retroactively.
It exists so a reader can see how much of any GNN–floor gap the eval-graph choice accounts for (see
*Disclosure*).

### Clause 5 — Leakage tests, written in the implementation session (untested guards don't exist)

**pytest, on synthetic fixtures.** This is the repo's practice (`tests/test_ell1_leakage.py`,
`tests/test_dgraph_verification.py`): the snapshot is gitignored, so the suite must not need it.
Each fixture includes at least one edge between two pre-cutoff nodes dated after the cutoff, the
case ELL-1's fixtures never contained.

1. **Exact equality, not just a bound:** the adapter's training edge set equals `edges_as_of(…, T)`,
   after reverse-edge doubling, at both cutoffs. That catches leaked edges and wrongly dropped ones.
2. Every training seed (labelled node used in the loss) has `t_node ≤ train_max`.
3. **The guard has teeth:** the node-induced graph *fails* the edge-date assertion, while the
   node-only form of the assertion passes it. That pins the exact failure this ADR fixes.
4. **Prefix-determinism:** node times recomputed from only the edges dated `≤ B` give identical
   window membership at every boundary `B`.
5. An eval graph built for a window contains no edge dated after that window's last step.
6. Edge-derived features computed for training equal features recomputed from the `≤ T` edge set
   alone, with recency measured from `T`.
7. **Fit-on-train:** perturbing the features of nodes with `t_node > T` leaves every fitted feature
   statistic unchanged.

**Snapshot audit, in a script, not pytest** (the `verify_dgraph_snapshot.py` pattern). It runs once
when the DGF-1 DataSource first loads the real graph, and exits non-zero on any mismatch:
- cutoffs 369 / 481;
- the per-window support in clause 2;
- node-induced-minus-edge-dated counts of 399,366 / 316,632;
- zero isolated nodes;
- invariants 1, 4 and 5 re-checked on the real graph.

A mismatch is a doc-vs-data discrepancy and is escalated, never absorbed (ADR-001).

### Clause 6 — Where the code goes: edge-date filtering joins the core, node-time derivation does not

- **Core (`gbe.eval.temporal`):** an edge-date filter, e.g. `edges_as_of(edge_index, edge_time,
  t_max)`, used for both the training graph (clause 3) and the eval graph (clause 4). Also an
  optional `edge_time` argument to `assert_no_temporal_leakage` that, when given, asserts every
  edge's own date is pre-cutoff. Reasons:
  - doc-00 §7 makes leakage checks harness-level.
  - An edge date is not domain knowledge.
  - The check is defined identically for **both models that exist today**. Elliptic's edges carry a
    date too, the step both endpoints share, so the check applies to ELL-1 unchanged and never
    fires there. The argument rests on the two current models, not on speculation about later ones
    (doc-00 §1).
- **ELL-1 stays bit-for-bit (ADR-008).** The new argument defaults to `None`, and ELL-1's call sites
  stay as they are. `scripts/check_extract_regression.py` is what verifies this, not argument.
- **The default makes the stronger guard opt-in, and that is a hazard, not a detail.** A DGF-1
  call site that omits `edge_time` gets the node-only check and silently reproduces the failure in
  *Context*. DGF-1 therefore does **not** rely on remembering the argument. Coverage comes from
  clause 5's exact-equality test on the adapter's actual training edge set. The DGF-1 adapter must
  not call `induced_train_subgraph`. That function's docstring gains its validity condition: correct
  only when every edge lies within one time value.
- **Adapter (`adapters/dgf1/`):** the earliest-edge node time. ELL-1 has a native node time, and
  EDR-1's company nodes will carry their own dates, so not all four models would derive it the same
  way. The inclusion rule applies: when unsure, keep it out.

**This is the clause to review first.** Moving the edge-date check into the core rests on two
things: the harness-level leakage argument, and the check being defined identically for ELL-1 and
DGF-1. It does not rest on ELL-1 ever calling it; ELL-1 won't, under ADR-008.

### Scope boundary

This ADR fixes the **split and protocol only**. **DGF-1's Gate-1 pre-registration** is a separate,
later ADR: seed count derived from a validation pilot, floor ≥ 5 per arm, ADR-006's two-stage
design. ADR-010 anticipated one "Phase-0/Gate-1 ADR". The two are separated deliberately, because
that pilot cannot run until this split exists. Together they discharge ADR-010's requirement.
That ADR inherits one open question from here: the pilot trains on `≤ 369` while gate runs train on
`≤ 481`. It must argue why seed variance measured on the smaller training set sizes the larger one,
or pilot at the gate's training size on a window that stays clear of 482–821.

The official-split track trains and scores on the official masks over the full graph as distributed.
That is our own definition, not a claim about any third party's protocol. It is always labelled
*random-split, leaderboard-comparable*, and by design carries none of this ADR's guarantees
(ADR-010 clause 1).

## Limitations recorded, not fixable from this snapshot

- **Labels carry no date.** A training user's fraud label may reflect behaviour observed after the
  cutoff, since default is observed later. This is inherent to the dataset. It affects the floor and
  the GNN equally, so the *comparison* is fair, but absolute numbers are optimistic relative to a
  real as-of-`T` deployment. Every write-up carries this caveat.
- **Node features are a profile snapshot taken at release.** They cannot be verified as pre-cutoff.
  This also affects both arms equally.
- **Look-ahead within a window (clause 4).** A test user who joined at step 490 is scored with its
  edges up to step 821. Unlike the two limitations above, this affects **only the GNN**, since the
  floor uses no graph. It is why the window-only sensitivity is reported beside every gated GNN
  number.
- **Prevalence drifts upward** across the windows: 1.20% → 1.35% → 1.48%. The val pilot's AUPRC and
  the test AUPRC have different chance levels. Per ADR-007 they are never compared, and each is
  reported with its own prevalence and positive count.

## Alternatives rejected

- **Max- or mean-based node time.** Violates ADR-010's constraint: a post-cutoff edge can move a
  node's time across the boundary.
- **Source-only node time (only out-edges count).** Leaves any node with only incoming edges
  without a time, and throws away evidence of when a node existed.
- **Reuse ELL-1's node-induced training graph unchanged.** Measured to admit 399,366 post-cutoff
  edges at the inner cutoff, touching 37.76% of labelled training nodes, which the current guard
  cannot see. This is the alternative this ADR most exists to reject.
- **ELL-1's window-only eval graph.** Isolates 36.66% of labelled test nodes and hands the no-graph
  floor an advantage created by the protocol.
- **Score each node using only edges up to its own node time.** Under clause 1 a test node would
  then see only the edges of its first active step. That is a cold-start question, not the one
  Gate 1 asks. It could be a later reported-only sensitivity row, and is not decided here.
- **Cutoffs at fixed fractions of calendar time (steps 1–821).** Labelled support per window would
  depend on arrival rates rather than being fixed, and the scored windows would no longer match the
  official split's val/test windows in size.
- **Train the gate model on the inner window only (`≤ 369`).** Discards 183,430 labelled nodes,
  widens the gap between training and test time, and departs from ELL-1's precedent for no gain.
- **An edge-level holdout.** DGF-1 is node classification, and labels live on nodes.
- **Put the node-time derivation in the core.** Premature: no second model derives node time this
  way (doc-00 §1).
- **Fold this into the Gate-1 pre-registration ADR.** That ADR needs a validation pilot, and the
  pilot needs this split. Bundling them would force the seed count to be chosen before its input
  exists.
- **Implement it in this session.** Doc-first: an idea is proposed in the session it was
  conceived, and built only after acceptance.

## Consequences

- **Core change, small and guarded:** the edge-date filter plus an optional leakage-check argument
  in `gbe/eval/temporal.py`, with the clause-5 tests. ADR-008's checker must stay green.
- **`adapters/dgf1/`:** the DataSource adds node time, `labelled_mask = (y == 0) | (y == 1)`
  (ADR-010) and edge-derived features built per graph view. Its config pins 369 / 481 / 821 and
  `reverse_edges: true`.
- **Tabular floor:** trained on labelled nodes `≤ 481`, with any scaler fitted on that window only
  (clause 3), scored on labelled nodes in 482–821, with both metrics routed through
  `gbe.eval.classification_metrics` (ADR-007). Whether the floor also gets the edge-derived features
  is a Gate-1 pre-registration question, left open here.
- **Every gated GNN row gains a sibling window-only score** (clause 4). It is logged in the same
  registry row under distinct keys, so the two can never be confused or reported apart.
- **`induced_train_subgraph`'s docstring** gains its validity condition (clause 6). The docstring
  changes; the behaviour doesn't, so ELL-1's numbers are untouched.
- **DGF-1 Gate 3 inherits a problem ELL-1 never had.** ELL-1's ablations kept cutoff integrity by
  rewiring within a time step. DGraph edges don't live in steps, so any rewiring has to preserve
  each edge's date, or it forges future edges into the training graph. That belongs to Gate 3's own
  ADR, but it is recorded now so it isn't rediscovered then.
- **The eval-graph choice drives the result.** At 36.66% isolation the two eval rules would produce
  materially different GNN numbers. Fixing the rule now, before any score exists, stops it from
  being chosen afterwards for whichever reads better.
- `scripts/measure_dgf1_temporal_split.py` is committed as provenance for every number here.

## Revisit when

- **DGraphFin-2** turns out to replace the original (ADR-010's *Revisit when*).
- **Label or feature timestamps become available.** The limitations above would then become fixable
  and should be fixed.
- **Never for DGF-1 once any DGF-1 score exists on 370–481 or 482–821.** From that point, changing
  the split or the eval graph is re-thresholding after seeing a result.
