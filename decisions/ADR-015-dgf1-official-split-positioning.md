# ADR-015 — DGF-1's official-split positioning batch: the protocol for a number that is reported and never gated

**Status:** proposed
**Date:** 2026-09-15
**Deciders:** coderback
**Stage-1 seeds:** 3
**Inherits, does not re-open:** the official split is random and its numbers are *reported, never
gated* (**ADR-010** clause 1); ROC-AUC + AUPRC together, each AUPRC with its prevalence and positive
count (**ADR-007**); official numbers come from a separate batch and may not share a table row with
a gated number (**ADR-012**); the frozen architecture and the two-knob tuning budget (**ADR-003**,
**ADR-009**).
**Changes no gated quantity and touches no gate.** Both signed gates stand exactly as recorded. This
ADR produces the leaderboard-positioning numbers `gates/GATE-DGF1-0.md` §4 records as **not run**,
and doc-02 Phase 0 and Gate 1 require.
**Decides a protocol, not a result.** No number in this ADR; every figure comes from the batch.

**Docs affected — to apply on acceptance:**
- `docs/02-dgraph-fin-embedding-model-BUILD.md` §5's reporting template: the official-split rows
  gain a pointer here, and §4 Phase 0's "both splits" line names this ADR as the protocol.
- `CLAUDE.md` Next item 2 and `docs/timeline.md` gain the run id once it exists. **Both are
  gitignored**, so this ADR and `notebooks/lab/` are the versioned account.
- `gates/GATE-DGF1-0.md` is **not edited** — it is a signed, dated gate file. Its §4 "not run"
  statement was true when signed and stays as written.

## Context

**What is already fixed, and is not re-decided here.** ADR-010 clause 1 established that the
official DGraphFin split is a **random node mask**, not temporal, so it cannot carry DGF-1's
structural claim; its numbers are reported for positioning against published baselines, "labelled
explicitly as *random-split, leaderboard-comparable*", and "may not acquire a pass condition
retroactively". ADR-012 added that they come from a separate batch and may never share a table row
with a gated number. doc-02 §5 carries the reporting template.

**What is not fixed, and is why this ADR exists.** Every accepted document governs how the number is
*reported*. None governs how it is *produced*, and five things must be decided before a run:

1. **There is no code path for it.** DGF-1's trainer is built entirely around `TemporalSplit`:
   `train_dgf1` calls `graph_view(data, split.train_max)` to filter edges by date, `train_seed_mask`
   for its seeds, and `window_target_mask` for its targets. A random node mask has no cutoff and no
   window. The masks are loaded — `adapters/dgf1/datasource_dgraph.py:140–143` carries
   `official_train_mask`, `official_val_mask` and `official_test_mask`, already flagged "the official
   RANDOM split: reported track only, never gated (ADR-010)" — but nothing consumes them.
2. **The parity floor is defined relative to a view**, and a random split has no view.
3. **The hyperparameters are unsettled:** ADR-012's winner was tuned on the *temporal* validation
   window.
4. **The seed count is unsettled:** ADR-005's ≥3-with-variance rule is written for gate numbers, and
   this is not one.
5. **`official_test_mask` is a held-out eval set** under CLAUDE.md's "never train on, tune against,
   or peek at" rule, and nothing currently says which mask tuning may use.

## Decision

### Clause 1 — What is produced, and what it may never become

- **One positioning batch**, separate from every gate batch (ADR-012), producing, on
  `official_test_mask`: **ROC-AUC and AUPRC** (ADR-007) for each arm in clause 3, at the seed count
  in clause 5, reported as **mean ± SE over seeds**.
- **Every AUPRC carries the prevalence and positive count of the mask it scores** (ADR-007). Those
  figures come from the run; none is stated here.
- **Reported, never gated.** No pass condition now or retroactively (ADR-010 clause 1). These numbers
  may never appear in a `GATE-*` file's decision table, never be cited as evidence for or against the
  structural claim, and never share a table row with a gated number.
- **The only shape of claim available** is the one doc-02 §5 states: *competitive with strong fraud
  GNNs on the official random split, reported not gated* — and only with its variance attached.

### Clause 2 — The training protocol, and the departure it requires

- **Transductive by construction.** The split is a random node mask, so the whole graph and every
  edge is visible to training; there is no cutoff to filter on. This is what the published
  leaderboard measures, and it is the only way the number is comparable to it.
- **This is a departure from CLAUDE.md's "temporal splits only", and it is named as one.** It is
  confined to this reported track, authorised by ADR-010 clause 1's two-track design. **No gated
  number is produced this way, and no temporal leakage test is weakened to accommodate it** — the
  temporal assertions are not applicable here, not relaxed, and clause 7 requires a test that the
  two paths cannot be confused.
- **The one discipline that still binds, and is enforced:** the input transform is **fitted on
  `official_train_mask` nodes only**, never on all nodes. Node statistics come from the full graph,
  which is inherent to a transductive benchmark and is what the leaderboard's baselines do; the
  *standardiser* has no such excuse, and fitting it across val/test nodes would be ordinary leakage.
- **Scoring** uses exact neighbourhoods (`num_neighbors = [-1] * layers`), as the gated path does, so
  no score depends on an evaluation seed.
- **The code path is new, and lives in `adapters/dgf1/`.** Nothing about official masks enters
  `gbe/`: it is domain-specific, and the core inclusion rule (CLAUDE.md) excludes it.

### Clause 3 — The arms

Three, matching doc-02 §5's template rows:

1. **DGF-1** — GraphSAGE on parity inputs: the raw 17 features plus the view-derived node statistics
   (11-wide edge-type histogram, degree, recency), computed from the **full graph**, which is this
   split's only available view.
2. **XGBoost, parity features** — the same inputs as arm 1, which is what makes it the comparator.
3. **XGBoost, raw 17 features** — reported, as doc-02 §5 has it.

**Why the floors come too.** A DGF-1 number alone positions nothing: the published leaderboard is a
table of models on one split, and a GNN-vs-tabular gap on *this* split is the only thing that makes
our own temporal gap interpretable as a split effect rather than a model effect. Both floors are
cheap relative to the GNN.

### Clause 4 — Hyperparameters are retuned on the official validation mask

- **The same 9-configuration budget as the temporal track** (ADR-012 clause 3): `lr` × `batch_size`,
  one seed each, **selected on AUPRC on `official_val_mask`**. The floor retunes on its own
  9-configuration grid, as ADR-012 clause 2 did.
- **Why not reuse ADR-012's winner.** It was selected on the temporal validation window. Carrying it
  to a different split is an untested transfer claim, and ADR-009 is the ADR that exists because this
  programme does not assume transfer — it requires `lr` plus a graph-appropriate second knob to be
  retuned, with `batch_size` named as DGF-1's. Reusing the temporal winner here would assume exactly
  what ADR-009 declines to assume.
- **The budget does not widen.** Two knobs, the same grid, no search beyond it (ADR-003, ADR-009).
- **The winner is recorded as a dated amendment to this ADR before the test mask is touched**, the
  same discipline ADR-012 clause 3 used.

### Clause 5 — Seeds: 3, with variance, and why not 1 and not 8

- **3 seeds per arm**, reported as mean ± SE.
- **Not 1:** a single-seed number placed beside published leaderboard figures invites exactly the
  over-reading ADR-005 exists to prevent. A positioning number without variance cannot say whether a
  gap is real.
- **Not 8:** ADR-012 derived 8 from a validation pilot *because a gate clause had to resolve*.
  Nothing here resolves, so there is no effect size to power for, and the marginal seeds buy
  precision no claim depends on.
- **ADR-005's ≥3 floor still applies as a floor**, not because this is a gate, but because 3 is the
  programme's minimum for stating variance at all.
- **If the floor's winning configuration is deterministic** (`subsample = 1.0`), its 3 rows are
  expected to be identical; that is evidenced rather than assumed, as ADR-012 clause 2 required.

### Clause 6 — The held-out mask is touched once

- **`official_test_mask` is a held-out eval set** (CLAUDE.md). It is scored **once**, after clause 4's
  winner and clause 5's seed count are fixed and recorded.
- **Every selection decision uses `official_val_mask` only.** No configuration, threshold, feature or
  seed count may be chosen by looking at test-mask performance.
- **The runner refuses the test mask unless this ADR is accepted**, parsed by the same guard the
  temporal runners use (`scripts/preregistration.py`), with the seed count read from this document's
  `**Stage-1 seeds:**` line rather than from a flag — so the batch cannot quietly run a different `n`
  than the one written here.
- **A dirty tree is refused**, as for every other batch.

### Clause 7 — Tests, written in the implementation session

1. **The two paths cannot be confused.** The official path never applies a temporal cutoff, and the
   temporal path never reads an official mask. Asserted directly, not by inspection.
2. **The transform is fitted on `official_train_mask` nodes only.** Mutation: fitting it on all
   labelled nodes must fail this test — the same shape of check that caught the frozen-transform hole
   in lab Session 25.
3. **Every official row records `split = "official"`**, so ADR-010's labelling rule is enforced in the
   row and not only in prose. A row is the durable artifact; a caption is not.
4. **The gate assemblers refuse official rows.** `scripts/assemble_gate_dgf1_1.py` currently filters
   on `arm` and `window` and has no notion of split, so an official row with a matching arm could be
   read into a gate table. This is the one place ADR-012's "may not share a table row with a gated
   number" can be made mechanical, and it is.
5. **Scored ids align to the official test mask**, the id-alignment property the gated path already
   tests.
6. **No official row carries a gate tag**, and no gate row carries `split = "official"`.

### Clause 8 — What this batch may never be used for

- To support or undermine **"structure beats features at scale"**, or any claim about the temporal
  gap. A random split cannot evidence a temporal claim (ADR-010).
- As a **comparator for Gate 3's ablations**, which run on the temporal split.
- To **revisit Gate 1**, which is signed and dated.
- As a **fallback** if a temporal number disappoints. The tracks are separate by design, and a
  positioning number is not a consolation gate.

## Alternatives rejected

- **Reuse ADR-012's temporal-split hyperparameters.** Cheaper by two grids, but it assumes HPO
  transfers across a change of split — the assumption ADR-009 exists to refuse. If the transfer
  question is ever worth measuring, it is its own experiment, not a silent premise here.
- **One seed.** Rejected: no variance, and the number sits beside published figures.
- **8 seeds, matching Gate 1.** Rejected: 8 was derived to power a resolvable gate clause; nothing
  here resolves.
- **Gate on the official split for comparability.** Rejected by ADR-010 clause 1 and not reopened.
- **Skip the official split entirely.** Rejected: doc-02 Phase 0 and Gate 1 require it as positioning,
  ADR-010 rejected dropping it, and `gates/GATE-DGF1-0.md` records it as owed.
- **Put the mask-based path in `gbe/`.** Rejected: official masks are a DGraph-specific artifact, and
  `gbe/` takes only what all four GBE models would implement identically.
- **Reuse `TemporalSplit` with sentinel values** to force the existing trainer through the official
  masks. Rejected: it would make a temporal-looking object that is not temporal, and clause 7 item 1
  exists precisely to keep the two paths distinguishable.

## Consequences

- **A second training path in `adapters/dgf1/`,** with its own tests, written in the same session
  (CLAUDE.md). It is additive: the gated path is not touched, so the ADR-013 clause 3 re-certification
  stands.
- **Cost, on this machine:** two 9-configuration retune grids plus 3 seeds × 3 arms. The GNN's
  temporal runs took ~9.6 min each, so the GNN grid is the bulk of it; the batch runs in the
  researcher's own terminal, not through the assistant.
- **The transductive departure is now on the record** in a versioned document, rather than being an
  unexamined consequence of running a random-split benchmark.
- **`gates/GATE-DGF1-0.md` §4's "not run" becomes historically true rather than currently true.** The
  gate file is not edited; the lab notebook and this ADR carry the answer.
- **The positioning claim will be weaker than the leaderboard's framing invites.** Published entries
  are tuned harder, on a protocol we have not verified (ADR-007's original reason for demoting the
  comparison). Being "competitive" is the claim; being ranked is not.

## Revisit when

- **The retune winner is measured.** Recorded here as a dated amendment before the test mask is
  touched.
- **The official leaderboard adopts a temporal split.** ADR-010's *Revisit when* already anticipates
  this; the two tracks would then converge and this ADR's separation would need rethinking.
- **Never, to attach a pass condition** to any number this batch produces, or to move an official-split
  figure into a gate table.
