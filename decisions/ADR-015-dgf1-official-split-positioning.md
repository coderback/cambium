# ADR-015 — DGF-1's official-split positioning batch: the protocol for a number that is reported and never gated

**Status:** proposed
**Date:** 2026-09-15 · **second draft**, replacing the first (commit `5f7f388`), rejected before
acceptance. See *Draft history*.
**Deciders:** coderback
**Positioning seeds:** 5
**Inherits, does not re-open:** the official split is random and its numbers are *reported, never
gated* (**ADR-010** clause 1); ROC-AUC + AUPRC together, each AUPRC with its prevalence and positive
count, and the **hard floor of 5 seeds per arm** (**ADR-007**); official numbers come from a separate
batch and may not share a table row with a gated number (**ADR-012**); the frozen architecture and
the two-knob budget (**ADR-003**, **ADR-009**).
**Changes no gated quantity and touches no gate.** Both signed gates stand exactly as recorded.
**Opens no gated window:** this document deliberately carries **no** `Stage-1 seeds:` line — see
clause 6, which is a correction to the first draft rather than a refinement of it.
**Decides a protocol, not a result.** No metric value appears in this ADR.

**Docs affected — to apply on acceptance:**
- `docs/02-dgraph-fin-embedding-model-BUILD.md` §5's reporting template: the official-split rows gain
  a pointer here; §4 Phase 0's "both splits" line names this ADR as the protocol.
- `CLAUDE.md`'s *Leakage discipline* section gains clause 2's invariant, alongside the temporal rule.
  ADR-011 set the precedent of amending that section.
- `CLAUDE.md` Next item 2 and `docs/timeline.md` gain the run ids once they exist. **Both are
  gitignored**, so this ADR and `notebooks/lab/` are the versioned account.
- `gates/GATE-DGF1-0.md` is **not edited** — signed and dated. Its §4 "not run" was true when signed.

## Draft history — why the first draft was rejected

Reviewed at `5f7f388` and rejected on four independent grounds, each verified against the repo:

1. **It would have opened the temporal test window.** Clause 6 reused ADR-012's `**Stage-1 seeds:**`
   line and the shared guard. `require_accepted_preregistration` authenticates nothing about *which*
   document it is handed — it regexes `Status:` and the seed line out of any path. Measured: with the
   status flipped to accepted, the guard returns **3**, so
   `--mode gate --preregistration <this ADR>` would have run the temporal gate batch at 3 seeds in
   place of ADR-012's pre-registered 8.
2. **It named the wrong leakage discipline.** It called the transform fit "the one discipline that
   still binds" and never constrained the **training loss mask**. A run seeding on every labelled
   node would train directly on official test labels, produce an excellent number, and pass all six
   tests the draft listed.
3. **Its purpose contradicted its own prohibition.** Clause 3 justified the floor arms as making "our
   temporal gap interpretable as a split effect rather than a model effect" — a claim about the
   temporal gap, which its own clause 8 and ADR-010 clause 1 forbid.
4. **It cited ADR-009 backwards.** It called ADR-009 "the ADR that exists because this programme does
   not assume transfer". ADR-009's frame is the opposite: the frozen region *is* transferred, and it
   fixes only *which second knob* is retuned per graph. Retuning here also adds a second difference
   between the tracks, replacing ADR-011 clause 2's control (identical sizes and roles, "only how the
   split is drawn differs") with a split-versus-configuration confound.

Three factual errors are corrected below: the cost figure quoted the winner's runtime (9.6 min) as
representative when the retune grid averaged **22.6 min** and totalled **4.1 h**; the claim that
`assemble_gate_dgf1_1.py` could absorb an official row is **false** (`load_gate_rows` raises on any
test-window row not tagged `gate`, and demands exactly seeds 0–7, clean, deterministic,
single-commit); and `split` was proposed as a row field although it is already a hashed config key
holding the temporal window block.

## Disclosure — what had been seen when this was written

- **The official masks' label statistics are already on record**, in ADR-010 Finding 1 and ADR-011
  clause 2's table: per-mask sizes, fraud counts and prevalences, including the test mask, computed
  for ADR-010's split-verification task. This protocol is therefore **not** pre-registered against an
  unseen mask, and the first draft's "those figures come from the run; none is stated here" implied a
  freshness that does not exist. No figure is restated here, and none was recomputed for this draft.
- **No model has been run on any official mask.** The masks are loaded
  (`adapters/dgf1/datasource_dgraph.py:140–143`) and consumed nowhere in `adapters/`, `gbe/`,
  `scripts/` or `tests/`. No registry row carries an official-split result.
- **Temporal test-window figures have been seen**, from the signed `gates/GATE-DGF1-1.md`.

## Context

**Fixed already, and not re-decided.** ADR-010 clause 1 established the official split is a random
node mask, so it cannot carry DGF-1's structural claim; its numbers are reported for positioning,
labelled *random-split, leaderboard-comparable*, and "may not acquire a pass condition
retroactively". ADR-012 added that they come from a separate batch. doc-02 §5 carries the template.

**Not fixed, and why this ADR exists.** Every accepted document governs how the number is *reported*;
none governs how it is *produced*. DGF-1's trainer is built entirely on `TemporalSplit` — `train_dgf1`
filters edges via `graph_view(data, split.train_max)` and takes seeds from `train_seed_mask`, and
`score_view` takes targets from `window_target_mask`. A random node mask has no cutoff and no window.

## Decision

### Clause 1 — What is produced, and what it may never become

- **One positioning batch**, separate from every gate batch (ADR-012), producing on
  `official_test_mask`: **ROC-AUC and AUPRC** (ADR-007) per arm, at clause 5's seed count, reported as
  **mean ± SE over seeds**, each AUPRC with the mask's prevalence and positive count.
- **Reported, never gated.** No pass condition now or retroactively. These numbers may never appear in
  a `GATE-*` decision table, never be cited for or against the structural claim, and never share a
  table row with a gated number.
- **The claim available is narrow, and narrower than the first draft implied.** Clause 3's arms
  contain **no fraud GNN**, so this batch cannot by itself support doc-02 §5's "competitive with
  strong fraud GNNs". What it supports is: *DGF-1's ROC-AUC and AUPRC on the official random split,
  beside its own tabular floors on that same split, for positioning.*
- **Published leaderboard numbers may be quoted beside it, never differenced against it**, and only
  with source and access date, labelled as **unverified third-party figures under a protocol this
  programme has not reproduced** (ADR-007's original reason for demoting the comparison). No
  arithmetic may combine one of ours with one of theirs.

### Clause 2 — The training protocol, its departure, and the invariant that replaces the temporal one

- **Transductive by construction.** The split is a random node mask, so the whole graph and every edge
  is visible; there is no cutoff to filter on. That is what the leaderboard's protocol implies, and it
  is the only way the number is comparable to it.
- **This departs from CLAUDE.md's "temporal splits only", and is named as one.** It is confined to
  this reported track under ADR-010 clause 1's two-track design. No gated number is produced this way,
  and **no temporal leakage test is weakened**: the temporal assertions are inapplicable here, not
  relaxed.
- **The replacement invariant, which is what CLAUDE.md's DataSource rule becomes on this split:**

  > **No tensor entering the loss may be indexed by an `official_val_mask` or `official_test_mask`
  > node.** Training seeds are exactly the labelled nodes of `official_train_mask`.

  This is the discipline the first draft omitted, and it is the one that matters: node *features* may
  legitimately aggregate over neighbours in other masks (that is what transductive means, and what the
  published baselines do), but no held-out **label** may reach the objective.
- **The input transform is fitted on `official_train_mask` nodes only.** Node statistics come from the
  full graph, which is inherent here; the standardiser has no such excuse.
- **Recency is measured from step 821**, the full graph's last step, because that is this view's own
  cutoff. ADR-011 clause 3 forbids measuring recency from 821 *for a temporal view*, since it would
  bake the dataset's end date into a training feature; here there is no earlier cutoff to measure
  from, so the feature means something different than it does on the temporal track. **Stated so the
  two are never read as the same feature.**
- **`reverse_edges` stays on** (the default, and part of the hashed provenance), so the official run
  trains on the full 4.3M edges doubled — strictly more expensive than the ≤369 subgraph.
- **Scoring** uses exact neighbourhoods (`num_neighbors = [-1] * layers`), as the gated path does.
- **The code path lives in `adapters/dgf1/`.** Nothing about official masks enters `gbe/`.

### Clause 3 — The arms, and why the floors are here

Three arms:

1. **DGF-1** — GraphSAGE, ADR-003's frozen architecture unchanged, on parity inputs: the raw 17
   features plus the view-derived statistics (the 11-wide edge-type histogram, whose row sum *is* the
   degree, and the two recency columns — 30 columns in all), computed from the full graph.
2. **XGBoost, parity features** — the same inputs.
3. **XGBoost, raw 17 features.**

**Why the floors are in this batch, stated without reference to the temporal gap:** doc-02 §4 Phase 0
requires baselines "on **both** splits", and §5's template has a row for each. That obligation is the
justification. The first draft instead argued the floors make the temporal gap interpretable as a
split effect, which is a claim about the temporal gap and is forbidden by clause 8 — the arms stay,
the argument for them does not.

### Clause 4 — Hyperparameters are ADR-012's winner, carried over and disclosed

- **The GNN uses ADR-012 clause 3's winner** (`lr` 0.5×, `batch_size` 2048); the floor uses ADR-012
  clause 2's winner. Neither is retuned on the official masks.
- **Why, and it reverses the first draft.** ADR-011 clause 2 built the temporal split so that sizes and
  roles match the official masks and "only how the split is drawn differs". Retuning here would add a
  second difference, so any gap between the tracks would confound split with configuration. Carrying
  the configuration over preserves the one-variable design.
- **It is a transfer, and is disclosed as one.** A configuration selected on temporal validation may be
  suboptimal on a random split, so DGF-1's official number may understate what a split-specific tune
  would reach. **Every report of this batch says so.** The batch is positioning, not a best-effort
  leaderboard entry, and it is not presented as one.
- **This does not invoke ADR-009**, which governs *which second knob* is retuned per graph, not
  transfer across splits of one graph. The first draft cited it backwards.
- **If a split-specific tune is ever wanted**, it is its own experiment with its own ADR, and its cost
  is stated in advance: the temporal GNN retune grid took **4.1 h** (11 rows, mean 22.6 min, max 34.3),
  and the official grid would be slower, since it trains on the full graph.

### Clause 5 — Seeds: 5, the floor ADR-007 already fixed

- **5 seeds per arm**, reported as mean ± SE.
- **ADR-007 fixes "a hard floor of 5 per arm" for DGF-1**, and that floor is not scoped to gates. The
  first draft argued 3 without naming it.
- **ELL-1 is the precedent ADR-007 cites:** 3 seeds gave ±0.031 where 8 gave ±0.070. A 3-seed band
  understated the spread by more than half on this programme's own arm, so "3 with variance" would
  have reported a variance already measured to be wrong.
- **Not 8:** 8 was derived by ADR-012 clause 5 to power a resolvable gate clause. Nothing here
  resolves, so there is no effect size to power for.
- **The floor arms take 5 seeds too.** ADR-012's floor winner subsamples (`subsample = 0.8`), so its
  rows are not identical and its spread must be measured rather than assumed.

### Clause 6 — The held-out mask, and a guard that cannot be borrowed

- **`official_test_mask` is a held-out eval set** (CLAUDE.md). It is scored **once**, after clause 4
  and clause 5 are fixed and recorded. Every selection decision uses `official_val_mask` only.
- **This document carries no `**Stage-1 seeds:**` line, deliberately.** Its count is on a differently
  named line, so `scripts/preregistration.py`'s existing regex cannot parse this ADR even when it is
  accepted — which closes the first draft's temporal-window unlock without waiting for a code change.
- **The shared guard must bind document identity, and that is a deliverable of this ADR.**
  `require_accepted_preregistration` currently accepts any path whose text matches two regexes. It
  must take the batch kind it is authorising and refuse a document that does not declare the same
  kind, with a test. Until that lands, the naming above is the whole defence, and it is recorded as
  such rather than presented as sufficient.
- **The official runner is a separate entry point** with its own guard reading this ADR's own line. It
  cannot select a temporal window; the temporal runners cannot select a mask.
- **A dirty tree is refused**, as for every batch.

### Clause 7 — Tests, written in the implementation session

1. **The loss never sees a held-out label.** With a synthetic fixture, every node contributing to the
   objective is in `official_train_mask`. **Mutation:** seeding on all labelled nodes must fail this
   test. This is clause 2's invariant and the first draft's omission.
2. **The transform is fitted on `official_train_mask` nodes only.** Mutation: fitting on all labelled
   nodes must fail — the shape of check that caught the frozen-transform hole in lab Session 25.
3. **The two paths cannot be confused**, concretely: the official trainer's edge set equals the full
   edge set (no cutoff applied), and the temporal entry points raise if handed an official mask.
4. **Official rows are unmistakable in the registry, by a mechanism that already exists.** They carry
   `arm ∈ {dgf1-official-parity, xgboost-official-parity, xgboost-official-raw17}` and
   `window = "official-test"`. Both keys already reach `metrics_json` through `PROVENANCE_KEYS`, so
   **no new channel, no change to `dgf1_base_config()`, and no change to `PROVENANCE_KEYS`** — and
   therefore no change to any config hash and no effect on ADR-013 clause 3's re-certification. The
   first draft proposed a `split` field, which collides with an existing hashed config key and would
   not have reached a row at all.
5. **Gate assemblers are unaffected, and a test pins that.** `assemble_gate_dgf1_1.py` already refuses
   any test-window row not tagged `gate` and requires exact arm names and seeds 0–7; the distinct arm
   names above cannot match. The test records the property rather than fixing a hole — the first
   draft claimed a vulnerability that does not exist.
6. **Scored ids align to `official_test_mask`**, the id-alignment property the gated path tests.

### Clause 8 — What this batch may never be used for

- To support or undermine **"structure beats features at scale"**, or any claim about the temporal
  gap. A random split cannot evidence a temporal claim (ADR-010).
- As a **comparator for Gate 3's ablations**, which run on the temporal split.
- To **revisit Gate 1**, which is signed and dated.
- As a **fallback** if a temporal number disappoints, and equally **not as a promotion** if the
  official number is the better one. If official beats temporal, that is a property of random versus
  temporal evaluation and is reported as such — it is not evidence that DGF-1 is better than the gate
  found, and doc-02 §2.3's warning about mixing the two applies in both directions.
- In any **cross-split AUPRC comparison**. The masks differ in prevalence, and ADR-007 forbids
  comparing AUPRC across windows of differing prevalence.

## Alternatives rejected

- **Retune on the official validation masks** (the first draft's clause 4). Rejected: it adds a second
  difference between the tracks and destroys ADR-011 clause 2's one-variable control, for ~4.1 h of
  compute on a number that carries no claim. The transfer is instead disclosed.
- **3 seeds** (the first draft's clause 5). Rejected: below ADR-007's hard floor of 5, and the
  variance it would report is the one ELL-1 measured to be understated.
- **Reuse ADR-012's `Stage-1 seeds:` line and the shared guard.** Rejected: it opens the temporal test
  window at this document's seed count.
- **Gate on the official split for comparability.** Rejected by ADR-010 clause 1, not reopened.
- **Skip the official split.** Rejected: doc-02 Phase 0 and §8 step 3 require it, ADR-010 rejected
  dropping it, and `gates/GATE-DGF1-0.md` §4 records it as owed.
- **Put the mask path in `gbe/`.** Rejected: official masks are a DGraph artifact; `gbe/` takes only
  what all four GBE models implement identically.
- **Reuse `TemporalSplit` with sentinel values** to force the existing trainer through the masks.
  Rejected: it would make a temporal-looking object that is not temporal.
- **A `split` column on every row.** Rejected: `split` is already a hashed config key, and `base_cfg`
  keys reach `metrics_json` only via `PROVENANCE_KEYS`. Arm naming achieves the same end with no hash
  change.

## Consequences

- **A second training path in `adapters/dgf1/`,** additive, with its own tests in the same session
  (CLAUDE.md). The gated path is untouched, so ADR-013 clause 3's re-certification stands — and
  clause 7 item 4 is what keeps that true.
- **Cost, honestly:** no retune grid. 5 seeds × 3 arms, the GNN arm dominating. Training on the full
  graph with reverse edges is more expensive per run than the temporal ≤369 subgraph, so the per-run
  figure will exceed the temporal winner's 9.6 min; the batch runs in the researcher's own terminal.
- **The transductive departure is on the record** in a versioned document, rather than being an
  unexamined consequence of running a random-split benchmark.
- **The positioning claim is weaker than the leaderboard's framing invites**, and now doubly so: the
  configuration is transferred rather than tuned for this split, and no fraud GNN is in the batch.
- **`gates/GATE-DGF1-0.md` §4's "not run" becomes historically true rather than currently true.** The
  gate file is not edited.

## Revisit when

- **The shared guard binds document identity.** Clause 6's naming defence is then belt-and-braces
  rather than the whole mechanism.
- **A split-specific tune is wanted.** Its own ADR, with the 4.1 h cost stated in advance.
- **The official leaderboard adopts a temporal split.** ADR-010's *Revisit when* anticipates this; the
  two tracks would converge and this ADR's separation would need rethinking.
- **Never, to attach a pass condition** to any number this batch produces, or to move an official-split
  figure into a gate table.
