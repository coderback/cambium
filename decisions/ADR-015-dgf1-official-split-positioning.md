# ADR-015 — DGF-1's official-split positioning batch

**Status:** proposed
**Date:** 2026-09-15 · **fourth draft**, replacing `5f7f388`, `30fbf92` and the uncommitted third.
See *Draft history*.
**Deciders:** coderback
**Positioning seeds:** 5
**Opens no gated window.** This document carries no `Stage-1 seeds:` line, so
`scripts/preregistration.py`'s regex cannot parse it even when accepted (clause 5, test 4).
**Decides a protocol, not a result.** No metric value appears here.

**Every external claim below carries its source verbatim, with file and line.** Three drafts were
rejected for citing documents for propositions they do not contain; quoting in place is the
mechanism that replaces intending to be accurate.

## Inherits, does not re-open

> **Reported, not gated:** official-split numbers, labelled explicitly as *random-split,
> leaderboard-comparable*, for positioning against published baselines. They acquire no pass
> condition and may not acquire one retroactively.
> — `decisions/ADR-010-dgraph-snapshot-reconciliation.md:77-79`

> **Official random-split numbers come from a separate positioning batch**, not from the gate
> batch: they require training on the official train mask, they are not part of clause 5's seed
> counts, and they are labelled *random-split, leaderboard-comparable* wherever they appear
> (ADR-010 clause 1). No gated number and no official-split number may share a table row.
> — `decisions/ADR-012-dgf1-gate1-preregistration.md:128-131`

ROC-AUC + AUPRC together, each AUPRC with its prevalence and positive count (**ADR-007**); the
frozen architecture (**ADR-003**) under **ADR-009**'s transfer rule.

## Draft history

Three rejections, all for the same class of fault. Draft 1 (`5f7f388`) reused ADR-012's
`**Stage-1 seeds:**` line, so accepting it would have opened the temporal test window at 3 seeds,
and justified its arms by a temporal-gap claim its own clause forbade. Draft 2 (`30fbf92`) relocated
that forbidden argument into another clause, asserted the leaderboard's protocol while elsewhere
calling it unverified, and claimed "gate assemblers are unaffected" having checked one of two.
Draft 3 claimed to be written "verify-first" and was not: it stated the official masks were
"consumed nowhere" (two scripts read them under unprefixed names), mis-stated ADR-011 clause 3's
recency rule, and claimed 3 seeds was "below every count this programme has used with variance"
when `gates/GATE-ELL1-1.md:42` reports `0.6790 ± 0.0306` at n=3.

**The meta-error worth recording:** draft 3's failure was a false claim about my own process. I
greped this ADR's vocabulary (`official_*`) rather than the code's (`data.train_mask`), and
summarised four citations from memory while believing I had read them. Verbatim quotation above is
the structural fix; it is not a promise to be more careful.

## Disclosure

- **No model has been trained or scored on any official mask, and no registry row carries an
  official-split result.** The masks are attached at
  `adapters/dgf1/datasource_dgraph.py:140-143` and no trainer or scorer reads them.
- **Two scripts do read them**, which draft 3 wrongly denied:
  `scripts/measure_dgf1_temporal_split.py:108-110` (`_window("o.trn", data.train_mask, y)`, `o.val`,
  `o.tst` — per-mask fraud counts and prevalences) and
  `scripts/verify_dgraph_snapshot.py:179-193` (sizes and node-time quartiles). They read the raw PyG
  attributes, which are unprefixed.
- **The per-mask fraud counts and prevalences are therefore already on record**, in ADR-011 clause
  2's table, produced by `measure_dgf1_temporal_split.py` for **ADR-011's** window-derivation task.
  ADR-010 Finding 1's table carries sizes and node-time quartiles only. This protocol is **not**
  pre-registered against an unseen mask. No figure is restated here and none was recomputed.

## Decision

### Clause 1 — The protocol

- **Transductive, by choice.** The whole graph and every edge is visible in training. This does
  **not** follow from the split — a random node mask permits an inductive protocol that hides edges
  incident to held-out nodes. It is chosen because the mask supplies no principled edge-exclusion
  rule, and the cost is stated: the number is not an inductive one and may not be described as such.
  This ADR asserts nothing about any third party's protocol.
- **The leakage invariant**, replacing the temporal assertions, which are inapplicable here:

  > **No label of an `official_val_mask` or `official_test_mask` node may enter any fitted
  > quantity.** Training seeds are exactly the labelled nodes of `official_train_mask`.

  **It binds both arms.** For the GNN that is the loss and the class weights
  (`gbe/gnn/train.py:149`, `balanced_class_weights(y, train_seed_mask, …)`); for the floor it is
  ADR-012 clause 2's `scale_pos_weight = n_negative / n_positive`, which must be computed on
  training rows only. Node *features* may aggregate over neighbours in any mask — that is what
  transductive means. Only labels are withheld. No feature may be label-derived.
- **The input transform is fitted on `official_train_mask` nodes only**, for both arms.
- **Recency.** ADR-011 clause 3 says:

  > **Recency is measured from the view's own cutoff** (369 when training; the window's last step
  > when scoring), never from step 821. Measuring from 821 would bake the dataset's end date into
  > every training feature.
  > — `decisions/ADR-011-dgf1-temporal-split-derivation.md:241-243`

  The prohibition is on the **training** view. This batch's training view is the full graph, whose
  own cutoff is 821, so measuring from 821 here is the training case clause 3 warns about and is
  accepted as a consequence of a transductive protocol — stated, not hidden. At **scoring** time
  clause 3 already requires the window's last step, which for the temporal test window is also 821,
  so the two tracks' scoring features do not differ on this account. Draft 3 claimed the opposite.
- **Hyperparameters, pinned literally here** rather than by reference, because
  `adapters/dgf1/config.yaml:43,48` carries ADR-003's `lr: 0.0006636671097096978` and
  `batch_size: 1024`, **not** ADR-012's winners:
  - GNN: `lr = 3.318335548548489e-4`, `batch_size = 2048`
    (`decisions/ADR-012-...:210-211`), `epochs = 40`, `reverse_edges: true`
    (`adapters/dgf1/config.yaml:47,23`).
  - Floor: `max_depth = 8, subsample = 0.8` (`decisions/ADR-012-...:165`), with ADR-012 clause 2's
    other fixed parameters unchanged.
- **No runtime escape valve is pre-registered.** If a run proves infeasible the batch is postponed,
  never shortened.
- **Scoring** uses exact neighbourhoods (`num_neighbors = [-1] * layers`, `train_gnn.py:127`).
- **The code path lives in `adapters/dgf1/`.** Nothing about official masks enters `gbe/`.

### Clause 2 — Arms, and why hyperparameters are not retuned

- **Three arms:** DGF-1 (GraphSAGE, ADR-003's frozen architecture) on parity inputs; XGBoost on the
  same parity inputs; XGBoost on the raw 17. Parity inputs are the raw 17 plus the 11-wide edge-type
  histogram (its row sum is the degree) and the two recency columns.
- **What obliges each arm.** The floors:

  > Stand up baselines (§5) on **both** splits; log **ROC-AUC and AUPRC** for each, labelled by split
  > — `docs/02-dgraph-fin-embedding-model-BUILD.md:126`

  The DGF-1 arm, which that line does **not** cover:

  > Comparators are graph + tabular methods, judged by **ROC-AUC and AUPRC** on **both splits,
  > reported separately** — `docs/02-dgraph-fin-embedding-model-BUILD.md:190`

  > **3. The official-split positioning numbers are owed.** No DGF-1 row exists on the official
  > — `gates/GATE-DGF1-0.md:151`

  No argument about the temporal gap is made or relied on anywhere in this ADR.
- **Hyperparameters are ADR-012's winners, not retuned**, for one reason that references no
  comparison: retuning costs at least the temporal grid's **4.1 h** (11 rows, mean 22.6 min, max
  34.3, from `experiments/registry.csv`) for a number that can never acquire a pass condition. The
  programme's transfer rule (ADR-009, superseding ADR-003's transfer clause) already carries a
  frozen region across models and graphs; carrying two knobs across splits of one graph is a smaller
  step. **It is a transfer, and may understate what a split-specific tune would reach.**
- **Seeds: 5**, reported as **mean ± sample standard deviation (`ddof=1`)** — ADR-006's convention,
  used by both DGF-1 assemblers.
  - **ADR-007's "hard floor of 5 per arm" is gate-scoped** and is **not** authority here:

    > **Scope:** DGF-1's binary-classification gates — **Gate 1** (structure vs the tabular floor) and
    > **Gate 3** (defending ablations). — `decisions/ADR-007-dgf1-gate-metric.md:104-105`
  - 5 is the count both validation pilots used. The argument against 3 is measurement, not
    precedent-by-assertion: `gates/GATE-ELL1-1.md:42` reports `0.6790 ± 0.0306` at n=3 and
    `gates/GATE-ELL1-3.md:51` reports `0.6629 ± 0.0702` at n=8 on the same arm — a 3-seed band
    understated the 8-seed band by more than half.
  - Not 8: 8 was derived to power a resolvable gate clause, and nothing here resolves.

### Clause 3 — What may be claimed

- **The claim available:** *DGF-1's ROC-AUC and AUPRC on the official random split, beside its own
  tabular floors on that same split.* It carries **no gated claim** and can never acquire one.
- **Every appearance is labelled *random-split, leaderboard-comparable*** (ADR-010 clause 1,
  ADR-012, both quoted above).
- **Published leaderboard numbers may be quoted for context**, with source and access date, labelled
  as figures this programme has not reproduced. They may not be differenced against ours, and no
  ranking claim may rest on them.
- **Two disclosed weaknesses, reported together wherever this batch is cited:** the configuration is
  transferred from the temporal track rather than tuned for this split; and `official_val_mask` is
  used for nothing — there is no retuning, no early stopping (`train_dgf1` passes no `on_epoch` hook;
  `adapters/dgf1/train_gnn.py:80-81`, `gbe/gnn/train.py:126,162`) and no model selection, so the
  **labels** of ~15% of labelled nodes go unused. Those nodes still participate as structure and
  features.
- **The official and temporal numbers are never differenced, ranked, or explained against each
  other**, in either direction and whichever is larger. Draft 3 said "not a promotion either" and
  then explained the case, which is the explanation this rule forbids.
- **Never** to support or undermine "structure beats features at scale" or any claim about the
  temporal gap; never as a comparator for Gate 3; never to revisit Gate 1; never in a cross-split
  AUPRC comparison (the masks differ in prevalence; ADR-007 forbids it).

### Clause 4 — How official rows are distinguishable

- **Row identity:** `arm ∈ {official-dgf1-parity, official-xgboost-parity, official-xgboost-raw17}`,
  `experiment = "official"`, `window = "official-test"`, `phase = "P0"`, and the base config's
  `split` value becomes `"official-random-mask"` in place of the temporal block
  `dgf1_base_config()` returns — so no official row asserts a split it did not apply.
- **These cannot be set through `base_cfg`.** Both entry points write `arm` and `window` *after*
  `**base_cfg`, so anything passed in is overwritten:
  - `adapters/dgf1/train_gnn.py:166-168` — `{**base_cfg, …, "window": …, "arm": f"dgf1-{feature_set}"}`
  - `adapters/dgf1/baselines_tabular.py:219-224` — `{**base_cfg, …, "window": window, "arm": f"{model}-{feature_set}"}`

  So the official path needs its own config builder for **both** arms, not one GNN-shaped path.
- **Checked against every consumer of `arm` and `experiment`:**
  - `assemble_gate_dgf1_1.py:83` filters `window == "482-821"` first, then `:86-87` matches the three
    gate arms by equality — official rows are invisible to it.
  - `assemble_gate_dgf1_0.py:206` is `m(r)["arm"].startswith("xgboost")`, which the `official-`
    prefix does not match. That is why the prefix leads.
  - **But it is not unaffected.** `:219` counts `ERRORED` across all DGF-1 rows, `:221` counts rows
    whose window is neither temporal window, and `:222` tabulates every `(experiment, arm, window)`.
    Official rows would appear in all three — under a heading reading "The official (random) split —
    not run". Today the only thing preventing that is `refuse_to_overwrite_a_signed_gate()`
    (`:97-104`), which is a verdict guard, not a filter.
  - **Therefore:** any future Gate-0-style assembly must filter `experiment != "official"` at `:206`,
    `:219`, `:221` and `:222`. Recorded here rather than assumed.

### Clause 5 — The held-out mask, and the guard

- **`official_test_mask` is a held-out eval set** (CLAUDE.md). It is scored **once**, after this ADR
  is accepted. The `experiment = "official"` tag is what makes a second batch visible in an
  append-only registry.
- **Interruption is the documented failure mode here, not a bug:**

  > Two retune batches were stopped by the Claude Code harness for low host memory and wrote
  > no rows — `gates/GATE-DGF1-0.md:146-147`

  **If the batch is interrupted**, the completed rows stand, the remaining seeds are run at the same
  commit to finish the batch, and the interruption is recorded in `notebooks/lab/`. **If a bug forces
  a re-run**, the whole batch re-runs and both sets of rows are cited. Neither is a re-score chosen
  after seeing a result.
- **The official runner is a separate entry point** reading this ADR's `**Positioning seeds:**` line.
  It cannot select a temporal window; the temporal runners cannot select a mask.
- **The shared guard must bind document identity — a deliverable with an ordering condition.**
  `require_accepted_preregistration` checks `^\*\*Status:\*\*` and `^\*\*Stage-1 seeds:\*\*` on any
  path handed to it and binds no identity. **This fix lands before Gate 3's pre-registration ADR is
  accepted**, because that document will carry its own `Stage-1 seeds:` line and would otherwise open
  the **Gate-1** test window at Gate 3's count. **Nothing enforces that ordering today** — it is a
  procedural commitment, and clause 6 test 4 pins only the half that is already true.
- **A dirty tree is refused**, as for every batch.

### Clause 6 — Tests and audit, written in the implementation session

1. **No held-out label enters any fitted quantity**, for **both** arms — the GNN's loss and class
   weights, and the floor's `scale_pos_weight`. Mutation: computing either over all labelled nodes
   must fail it.
2. **The transform is fitted on `official_train_mask` nodes only**, both arms. Mutation: fitting on
   all labelled nodes must fail it.
3. **The two paths cannot be confused:** the official trainer applies no cutoff (its edge set equals
   the full edge set), and the temporal entry points raise if handed an official mask.
4. **The guard refuses this ADR** even with the status set to accepted.
5. **Row identity** is as clause 4 fixes it, with no gate arm name.
6. **Scored ids align to `official_test_mask`.**
7. **A snapshot audit in a script, not pytest** — ADR-011 clause 5's pattern
   (`decisions/ADR-011-...:347`, "**Snapshot audit, in a script, not pytest**"). On the real snapshot:
   the three official masks are boolean, disjoint, and together exactly the labelled set. Tests 1–6
   run on fixtures whose masks the author constructs and cannot catch a real-snapshot overlap.
8. **The gated path is byte-identical after this work.** `scripts/run_dgf1_gnn.py --mode repro-check`
   is re-run after implementation and must still reproduce
   `dgf1-20260913T232506Z-1acd5067`. ADR-013 clause 3 blocks later runs until the check passes, and
   clause 4's shared config builders sit close enough to the gated path that "additive" must be
   demonstrated rather than asserted.

## Docs affected — to apply on acceptance

- **doc-02 §5 line 220**, the story sentence, currently:

  > …is competitive with strong fraud GNNs on the official random split (reported, not gated)…

  This batch has no fraud-GNN arm, so it cannot discharge that claim — but the claim is doc-02's
  target for §5 as a whole, and its evidence is the GCN/GAT/GTAN rows §5's template still marks
  "_your run_". **It is therefore not deleted**; it gains a marker that it remains **owed** and is
  not supported by this batch. Replacement wording is the researcher's.
- **doc-02 §5's reporting template** (lines 203-213): one row per model, four columns
  (`Test ROC-AUC | Test AUPRC | Val ROC-AUC | Val AUPRC`), **no split dimension** — so as it stands
  it cannot satisfy ADR-012's no-shared-row rule or ADR-010 clause 1's "Every reported number states
  **which split produced it**". It gains a split column or a second table.
- **CLAUDE.md**'s *Leakage discipline* first bullet ("**Temporal splits only**") is **amended**, not
  supplemented, to record this non-temporal exception under ADR-010 clause 1; its *Integrity of
  results* held-out rule gains clause 5's once-only scoring. **`CLAUDE.md` and `docs/` are
  gitignored**, so this ADR and `notebooks/lab/` are the versioned account.
- `gates/GATE-DGF1-0.md` is **not edited** — signed and dated.

## Alternatives rejected

- **Retune on the official masks** (draft 1): ≥4.1 h for a number that can never acquire a claim.
- **3 seeds** (draft 1): the ELL-1 measurement above.
- **Reuse ADR-012's `Stage-1 seeds:` line** (draft 1): it opens the temporal test window.
- **A `split` row field** (draft 1): `split` is already a hashed config key holding the temporal block.
- **An inductive official protocol.** Rejected: no principled edge-exclusion rule follows from a
  random node mask, and inventing one would make the number comparable to nothing.
- **Gate on the official split.** Rejected by ADR-010 clause 1, not reopened.
- **Skip it.** Rejected: doc-02:190 and `gates/GATE-DGF1-0.md:151` above.
- **Put the mask path in `gbe/`.** Rejected: a DGraph artifact; `gbe/` takes only what all four
  models implement identically.

## Consequences

- **A second config builder and training path for both arms** in `adapters/dgf1/`, with its own
  tests. Clause 6 test 8 is what keeps "the gated path is untouched" a demonstrated claim.
- **Cost:** no retune grid; 5 seeds × 3 arms. Training on the full graph with reverse edges is more
  expensive per run than the temporal ≤369 subgraph, so the per-run figure will exceed the temporal
  winner's 9.7 min. The batch runs in the researcher's own terminal.
- **The non-temporal departure is recorded in the constitution**, not left implicit in a script.
- **The positioning claim is weak by construction** — transferred configuration, unused validation
  labels, no fraud-GNN comparator, transductive by choice — and every citation of it says so.

## Revisit when

- **The shared guard binds document identity** — required before Gate 3's ADR is accepted.
- **A split-specific tune is wanted.** Its own ADR, with the ≥4.1 h cost stated in advance.
- **The official leaderboard adopts a temporal split** (ADR-010's *Revisit when* anticipates this).
- **Never, to attach a pass condition** to any number this batch produces, or to move an
  official-split figure into a gate table.
