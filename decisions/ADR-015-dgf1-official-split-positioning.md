# ADR-015 — DGF-1's official-split positioning batch

**Status:** proposed
**Date:** first proposed 2026-09-15 · **fifth draft 2026-10-06**, replacing `5f7f388`, `30fbf92`,
an uncommitted third, and `8296f4b`. See *Draft history*.
**Deciders:** coderback
**Positioning seeds:** 5
**Opens no gated window.** This document carries no `Stage-1 seeds:` line, so the shared guard's seed
regex (`scripts/preregistration.py:25`) cannot parse it, accepted or not (clause 7, test 6).
**Decides a protocol, not a result.** No metric from any official mask appears here.

**Sourcing.** Quotations are verbatim, with file and line. A figure taken from the registry names the
rows it came from, and every such figure here was read from validation-window (370–481) rows only. A
statement with no source is a decision of this ADR and is worded as one.

## Inherits, does not re-open

> **Reported, not gated:** official-split numbers, labelled explicitly as *random-split,
> leaderboard-comparable*, for positioning against published baselines. They acquire no pass
> condition and may not acquire one retroactively.
> — `decisions/ADR-010-dgraph-snapshot-reconciliation.md:77-79`

> The official-split track trains and scores on the official masks over the full graph as distributed.
> That is our own definition, not a claim about any third party's protocol. It is always labelled
> *random-split, leaderboard-comparable*, and by design carries none of this ADR's guarantees
> (ADR-010 clause 1).
> — `decisions/ADR-011-dgf1-temporal-split-derivation.md:402-405`

> **Official random-split numbers come from a separate positioning batch**, not from the gate
> batch: they require training on the official train mask, they are not part of clause 5's seed
> counts, and they are labelled *random-split, leaderboard-comparable* wherever they appear
> (ADR-010 clause 1). No gated number and no official-split number may share a table row.
> — `decisions/ADR-012-dgf1-gate1-preregistration.md:128-131`

> Every reported AUPRC carries its window's prevalence and positive count. AUPRC is **never**
> compared across datasets, across split variants with different prevalence, or against ELL-1.
> — `decisions/ADR-007-dgf1-gate-metric.md:134-136`

> **doc-00 §6.4's transfer clause becomes: retune `lr` + exactly *one* second knob, chosen per model
> as the one that binds on that graph, named in the model's build doc with a one-line reason.**
> — `decisions/ADR-009-hpo-transfer-second-knob.md:59-60`; and `:68`: "**DGF-1's second knob is
> `batch_size`**".

ADR-003's frozen architecture and ADR-012 clause 3's fixed training budget apply unchanged.

## Draft history

Four drafts were rejected. Drafts 1–3 (`5f7f388`, `30fbf92`, uncommitted) cited sources for
propositions they do not contain. Draft 4 (`8296f4b`) pasted its sources in place, which fixed line
accuracy, and two reviews still rejected it:

- **its once-only rule for `official_test_mask` was prose**, with four exits named in the draft
  itself: a re-run for a "bug" judged after the results; an interruption rule whose seed count was
  not frozen; the `--allow-dirty` flag every runner carries; and a *Revisit when* that pre-authorised
  a second, retuned scoring;
- **its transferred hyperparameters had partly seen its own test labels:** they were selected on
  validation AUPRC over steps 370–481, and the random mask puts official-test users in that window;
- **its seed argument** compared two ELL-1 arms that differ in code path and determinism settings,
  under two dispersion conventions, without the caveat ADR-005 requires of those bands;
- **it required an input transform for the XGBoost floors**, which take their inputs untransformed;
- **it presented as its own choice a protocol** ADR-011:402-405 had already fixed.

Quoting in place makes a citation checkable. It does not make the claim built on the citation true,
and it covers nothing unquoted; both reviews found errors of those two kinds.

`becb85c`, a draft-1 commit, also added draft 1 verbatim as a stray root file `xaa`, still carrying
`**Stage-1 seeds:** 3`. It decides nothing and is removed in its own commit before this ADR is
accepted (clause 8).

## Disclosure

- **No model has been trained or scored on any official mask, and no registry row mentions one**
  (`grep -ci official experiments/registry.csv` → 0, 2026-10-06).
- **Two scripts read the masks, for statistics only:** `scripts/measure_dgf1_temporal_split.py:108-110`
  and `scripts/verify_dgraph_snapshot.py:179-193`. The masks are attached to the loaded graph at
  `adapters/dgf1/datasource_dgraph.py:122-123`, from the mapping built at `:140-143`. No trainer or
  scorer reads them.
- **Per-mask fraud counts and prevalences are on record** in ADR-011 clause 2's table, produced for
  ADR-011's window derivation; ADR-010 Finding 1 records sizes and node-time quartiles. No figure is
  restated here, and none was recomputed.
- **Every official-test user's label has already been used by the temporal track.** The masks are
  random over time —

  > The three distributions are indistinguishable — medians differ by **2 steps out of 821**. This is a
  > random split. — `decisions/ADR-010-dgraph-snapshot-reconciliation.md:42-43`

  — so official-test users fall in all three temporal windows, and their labels served there as
  training labels (≤369), as validation labels that selected ADR-012's configurations (370–481), and
  as Gate-1 test labels (482–821). **This is why clause 2 retunes:** nothing fitted or selected on the
  temporal track's labels carries into this batch. It is not pre-registered against an unseen set,
  and clause 3 makes that a mandatory weakness.
- **What does carry over is design, fixed before any DGF-1 score existed:** ADR-003's architecture
  and budget (set on ELL-1), and ADR-012's inputs, fixed XGBoost parameters and both grids. ADR-012
  states the condition it was written under:

  > **No DGF-1 score of any kind exists.** The floor has never been run, on any window; no DGF-1 GNN
  > exists; no registry row is `model=dgf1`. Every threshold and count below is therefore fixed
  > blind in ADR-004's full sense.
  > — `decisions/ADR-012-dgf1-gate1-preregistration.md:68-70`

  Its record of what *had* been seen (`:71-79`: structural facts, and one look at test-window degree
  by label) applies here unchanged.

## Decision

### Clause 1 — The protocol is inherited; the leakage invariant is decided here

- **The protocol is ADR-011's**, quoted above: train and score on the official masks over the full
  graph as distributed. Every edge is visible in training. The number is not an inductive one and is
  never described as one. This ADR does not re-decide the protocol.
- **The leakage invariant:**

  > **No `official_test_mask` label enters anything this batch fits or selects. No
  > `official_val_mask` label enters anything it fits; validation labels only select among clause 2's
  > configurations, each already trained.** Training seeds are the nodes in both
  > `official_train_mask` and `labelled_mask` (`adapters/dgf1/datasource_dgraph.py:120`).

  It is stated over **labels**, not over a list of quantities, so it binds everything either model
  derives from its training labels: the GNN's loss and class weights (`gbe/gnn/train.py:148-150`),
  the floor's `scale_pos_weight` (`ADR-012:135-141`, "on the training rows"), and anything not named
  here. Clause 7 test 1 checks it by perturbing labels, not by enumerating quantities. Node *features*
  from every mask may enter message passing and the input transform; that is what "over the full
  graph" means. No input is derived from a label.
- **The input transform is the GNN's alone.** The floors take theirs untransformed — "the *same* 30
  columns, untransformed" (`ADR-012:111`); `STANDARDISED["xgboost"]` is `False`
  (`adapters/dgf1/baselines_tabular.py:91`). The GNN's transform is fitted over **all nodes**, because
  ADR-011's population is every user in the training view ("Fit on users existing by
  ``train_max``", `adapters/dgf1/features.py:49-52`) and the official training view is the whole
  graph. It reads features only.
- **Recency** is measured from the view's own cutoff (`ADR-011:241-242`). The official view is the
  whole graph, so its cutoff is the graph's last step, for training and scoring alike.
- **No early stopping, and no validation label during training,** in either mode. The official path
  takes no epoch hook. Validation is scored only after training completes, and only in the retune.
- **Scoring uses exact neighbourhoods**, as `ADR-012:190-191` requires "for every arm and every
  reported row".
- **All code lives in `adapters/dgf1/` and `scripts/`.** Nothing about official masks enters `gbe/`.

### Clause 2 — Arms, tuning and seeds

- **Three arms, ADR-012 clause 1's:** DGF-1 (GraphSAGE, ADR-003's architecture) on the 30 parity
  inputs; XGBoost on the same 30 columns; XGBoost on the raw 17. The parity inputs are the raw 17
  beside the view's 11 edge-type counts and two recency columns (`adapters/dgf1/features.py:44-46`,
  `adapters/dgf1/datasource_dgraph.py:45-48`).
- **What obliges each arm.** The floors:

  > Stand up baselines (§5) on **both** splits; log **ROC-AUC and AUPRC** for each, labelled by split
  > — `docs/02-dgraph-fin-embedding-model-BUILD.md:126`

  DGF-1, under Gate 1: "Leaderboard positioning on the official (random) split is **reported, not
  gated**." (`docs/02-dgraph-fin-embedding-model-BUILD.md:159-160`); and "**3. The official-split
  positioning numbers are owed.**" (`gates/GATE-DGF1-0.md:151`).
- **Both tuned arms are retuned on `official_val_mask`, at ADR-012's budget:** the same grids, one
  seed (seed 0) per configuration, selected on validation AUPRC.
  - GNN: "`lr ∈ {0.5×, 1×, 2×}` ADR-003's `6.636671097096978e-4`, `batch_size ∈ {512, 1024, 2048}`"
    (`ADR-012:180`) — ADR-009's two knobs for DGF-1, now retuned as ADR-009 says rather than carried.
  - Parity floor: "`max_depth ∈ {4, 6, 8}` × `subsample ∈ {1.0, 0.8, 0.6}`" (`ADR-012:146`), the
    other parameters fixed as `ADR-012:135-141` states.
  - Raw-17 floor: not tuned. It runs at the parity floor's winner, as on the temporal track, where the
    floor's nine retune rows are all `xgboost-parity` (validation rows at `e58ccb6`) and selection
    reads parity rows only (`scripts/run_dgf1_floor.py:122`).
  - **Selection is mechanical.** The positioning stage reads each arm's winner from its official
    retune rows: highest validation AUPRC, ties to the configuration listed first in the grid. Every
    positioning row records the winner's run id. No configuration is typed.
  - **An inherited weakness, stated:** "a 9-way selection on one seed can pick seed-luck rather than a
    better configuration" (`ADR-012:185-186`). ADR-012 checked its winners with a validation pilot;
    this batch has none, so the five positioning seeds are the only multi-seed measurement of each
    winner.
  - **Measured cost, temporal track:** GNN grid 3.03 h (9 retune rows at `2bd88e0`, `wall_clock_s`
    summing to 10,891 s; mean 20.2 min, max 33.6); floor grid 3.0 min (9 rows at `e58ccb6`, 177.7 s).
    The official grids train over the whole graph. Their cost has not been measured and is not
    predicted. **There is no runtime escape valve.** If a run proves infeasible, the batch is
    postponed, never shortened.
- **Positioning seeds: 5 per arm, `range(5)`,** reported as mean ± sample standard deviation
  (`ddof=1`), the convention `ADR-006:74-75` binds "Gate 3 onward" and both DGF-1 assemblers use
  (`scripts/assemble_gate_dgf1_0.py:82`, `scripts/assemble_gate_dgf1_1.py:105`).
  - **Five is a judgement, not a derivation.** It clears the floor of three (CLAUDE.md, "≥3 seeds
    with variance"). It matches the five seeds each validation pilot ran: GNN `…1acd5067` to
    `…ed807866` and floor `…c2af9760` to `…856b67ab`, seeds 0–4, validation rows at `2c8f484`. It is
    fixed before any official run. Nothing here resolves a difference, so no power argument applies
    and none is made.
  - ADR-007's "hard floor of 5 per arm" is not the authority. Its scope is the gates: "**Scope:**
    DGF-1's binary-classification gates — **Gate 1** (structure vs the tabular floor) and **Gate 3**
    (defending ablations)." (`ADR-007:104-105`).
  - The count is frozen once any positioning row exists (clause 5).

### Clause 3 — What may be claimed, and where

- **The claim available:** DGF-1's ROC-AUC and AUPRC on the official random split, beside its two
  tabular floors on the same split, each AUPRC with the official test mask's prevalence and positive
  count. It carries no gated claim and can never acquire one.
- **"Positioning" is read narrowly.** Published leaderboard figures may be shown beside ours, with
  source and access date, labelled as not reproduced here and as produced under protocols this
  programme has not verified (ADR-011:403, "not a claim about any third party's protocol"). They are
  never differenced against ours, and no ranking claim rests on them. That is narrower than doc-02:6's
  "ROC-AUC positioned against a public leaderboard" can be read; *Docs affected* records it.
- **Official and temporal numbers are never differenced, ranked, or explained against each other**,
  in either direction. The tracks differ in what training sees (the full graph here; edges dated ≤
  the cutoff there) and, after clause 2, in their tuning. doc-02:87's "random-vs-temporal is a
  controlled comparison" therefore does not hold, and *Docs affected* amends it. Cross-split AUPRC is
  forbidden regardless (ADR-007:135-136).
- **Never** to support or undermine "structure beats features at scale" or any claim about the
  temporal gap; never as a comparator for Gate 3; never to revisit Gate 1.
- **Five weaknesses, printed with the numbers wherever they appear:**
  1. transductive by ADR-011's definition, so not an inductive number;
  2. each configuration selected on one seed, with no multi-seed check before the test mask was
     scored;
  3. no fraud-specialised GNN: of doc-02 §5's comparators only the two XGBoost floors are run;
  4. no third party's protocol verified, so no ranking;
  5. not pre-registered against an unseen set: every official-test label was used by the temporal
     track (*Disclosure*), though nothing this batch fits or selects reads one.
- **Where:** a positioning report, `gates/POSITIONING-DGF1-OFFICIAL.md`, assembled from the registry
  by `scripts/assemble_dgf1_positioning.py`.
  - It is not a `GATE-*` file and carries no verdict, the treatment ADR-013 clause 2 gave
    `gates/ERRATUM-DGF1-1.md`.
  - The assembler, not a hand edit, prints: the label *random-split, leaderboard-comparable*; the
    five weaknesses; every AUPRC's prevalence and positive count; and the retune table, labelled
    *selection runs, one seed, not results*.
  - It refuses to overwrite the file once the researcher has dated it.
  - `gates/GATE-TEMPLATE.md:30-31` ("Papers are assembled from gate files; nothing is reported that is
    not in one.") gains this file type; see *Docs affected*.
- **Validation numbers are never reported as results.** The only ones are selection values, which
  carry a winner's curse (`ADR-012:160`), so doc-02's Val columns stay empty for official rows.

### Clause 4 — Row identity

- `arm ∈ {official-dgf1-parity, official-xgboost-parity, official-xgboost-raw17}`;
  `experiment ∈ {official-retune, official-positioning}`; `window` is `official-val` on retune rows
  and `official-test` on positioning rows; the hashed config's `split` value is
  `official-random-mask` in place of the temporal block; `phase` follows the temporal track, `P0` for
  the floors and `P1` for the GNN (`scripts/run_dgf1_gnn.py:286`).
- **Every row also carries the knobs it ran** (`lr` and `batch_size`, or `max_depth` and
  `subsample`). Positioning rows also carry `positioning_seeds`, the header's count at run time, and
  `retune_winner_run_id`.
- **The tags are logged at session entry, not at the end**, so an ERRORED row carries them. The
  temporal trainer logs its tags last (`adapters/dgf1/train_gnn.py:208-211`), and `RunSession` writes
  a row on error (`gbe/run/session.py:66-71`), so a temporal row that errors carries none.
- **The official path has its own config builders, for all three arms.** Both temporal entry points
  write `arm` and `window` after `**base_cfg` (`adapters/dgf1/train_gnn.py:166-168`,
  `adapters/dgf1/baselines_tabular.py:219-224`), so neither can be steered through config.
- **A tested predicate, `is_official(row)`**, lives in `adapters/dgf1/`: true when `experiment`
  starts with `official-`. Every DGF-1 assembler written from now on, Gate 3's included, excludes
  official rows through it.
- **Existing consumers, as checked.** This list records what was checked; it does not claim to be
  complete.
  - `scripts/assemble_gate_dgf1_1.py:83` keeps only rows with `window == "482-821"`, so official
    rows are invisible to it.
  - `scripts/assemble_gate_dgf1_0.py:206` matches `arm.startswith("xgboost")`, which the
    `official-` prefix avoids. But `:219`, `:221` and `:222` count every DGF-1 row. The script never
    runs again, though: it refuses to regenerate over GATE-DGF1-0's signed verdict (`:97-104`). It
    is not edited.
  - `scripts/check_extract_regression.py:237` keeps only its own `experiment` tag.
  - `scripts/freeze_extract_reference.py:100-104` keeps `phase == "P0"` rows whose `baseline` is
    `rf` or `lr` (`:55`). DGF-1's floor rows record `baseline: xgboost` (validation row
    `…c2af9760`), and the official runner offers no other model.
  - The repro-check's `read_row` and the tests that read the registry select by run id.

### Clause 5 — `official_test_mask` is scored once, and the code enforces it

- **A separate entry point, `scripts/run_dgf1_official.py`, with two modes, `retune` and
  `positioning`:**
  - **No pre-registration path argument.** It reads this file by a path fixed in its source, refuses
    unless `**Status:**` is `accepted`, and takes the seed count from this file's `**Positioning
    seeds:**` line, never from a flag. Both modes require acceptance.
  - **No `--allow-dirty` and no window, model, seed or knob options.** A dirty tree is refused in both
    modes. Smoke testing uses fixtures, never the snapshot.
  - **A mask check on every load, before anything trains.** The three official masks must be
    boolean, pairwise disjoint, and together exactly `labelled_mask`. It refuses otherwise, and
    prints only pass or fail: no class count and no prevalence.
  - **Deterministic rows only.** It refuses unless `torch.are_deterministic_algorithms_enabled()`
    holds after session entry.
  - **The gated paths are re-certified first.** It refuses unless the registry holds, for both
    temporal runners, a passing `repro_check` row at HEAD's code (clause 7, test 10). If either check
    fails, the batch waits while the mismatch is investigated to root cause. No other route exists.
- **Each stage runs once: the retune on `official_val_mask`, positioning on `official_test_mask`.**
  - A stage refuses if any of its rows exist, other than ERRORED ones.
  - `--resume` runs only the missing (arm, configuration) or (arm, seed) pairs. It runs only if every
    file under `adapters/`, `gbe/`, `scripts/` and `configs/` is identical at HEAD and at the commit
    the stage's existing rows record (one commit for all of them). An ERRORED pair counts as missing;
    the ERRORED row stays and is cited.
  - Positioning also refuses unless the retune is complete — exactly one clean, deterministic,
    non-ERRORED row per (arm, configuration) — and unless HEAD's code, in the same sense, is the
    retune's. The configuration is selected by the code that scores it.
  - **The seed count is frozen.** Positioning refuses if any existing positioning row records a
    `positioning_seeds` different from this file's header.
  - **No score is printed or read before its row is written.** A score file left by an ERRORED run
    is never read or reported.
- **No re-run for a defect.** A defect found once any row of a stage exists is first recorded in a
  new ADR that locates it in code, in ADR-013 clause 1's form (where, what, since when, why
  undetected; `ADR-013:225-236`). Only then may anything re-run. The original rows stay and are cited
  first, and only that ADR's implementation can change the refusal. This follows ADR-008's rule: "**A
  mismatch is investigated to root cause — never re-run until it matches** (that is the
  optional-stopping error in another costume)." (`ADR-008:108-109`).
- **Other models later.** A future ADR may score `official_test_mask` with a model this batch does
  not run; doc-02 §5's other baselines are owed. It never re-scores these three arms.

### Clause 6 — The shared guard binds document identity, in the same implementation session

- `require_accepted_preregistration` (`scripts/preregistration.py:28-57`) checks `**Status:**` and
  `**Stage-1 seeds:**` on whatever path it is handed.
  - The tests unlock the test window with arbitrary temporary files
    (`tests/test_dgf1_runner.py:68-83`, `:542-554`).
  - ADR-012, accepted at `**Stage-1 seeds:** 8`, unlocks 482–821 today
    (`tests/test_dgf1_runner.py:92`). So would any accepted file with a seed line.
- **The fix lands with this ADR's implementation, as its own commit.**
  - The guard takes the document its caller's mode names, and refuses any other, compared by resolved
    path.
  - Both temporal runners' gate modes name ADR-012; Gate 3's runner will name Gate 3's ADR.
  - The temporary-file tests are rewritten so that an accepted temporary file with a seed line is
    **refused**.
  - If Gate 3's pre-registration would otherwise be accepted first, this fix lands first.
- Draft 4 left this as a procedural commitment. It is now a deliverable with tests.

### Clause 7 — Tests, written in the implementation session, accepted by mutation

1. **Label perturbation, all three arms.** On a fixture:
   - **(a) positioning, configuration fixed:** flipping every `official_val_mask` and
     `official_test_mask` label leaves every arm's scored probabilities bit-identical;
   - **(b) retune:** flipping every `official_val_mask` label leaves each configuration's scored
     probabilities bit-identical (its metrics may change); flipping every `official_test_mask` label
     leaves every retune row and the selected winner identical.

   Mutations that must each fail it: class weights, `scale_pos_weight` or the loss computed over all
   labelled nodes; validation scored inside the epoch loop.
2. **Inputs.** The floors receive `node_inputs` untransformed. The GNN's transform is fitted over all
   nodes and reads no label. Mutation: standardising a floor's inputs.
3. **The two tracks cannot be confused.** The temporal runners expose no mask option, and the
   official runner exposes no window option. The official training edge set equals the full edge
   set. Mutation: a cutoff applied to the official view.
4. **The official runner's document check.** It accepts no path argument. It refuses this file unless
   it is accepted, refuses a missing or malformed `Positioning seeds` line, and takes its count from
   the file. Mutations: accepting a path argument; dropping the status check.
5. **Once-only.**
   - A stage whose rows exist is refused.
   - `--resume` runs only missing pairs, refuses at different code, and counts ERRORED pairs as
     missing.
   - Positioning refuses an incomplete or duplicated retune, a retune at different code, and a changed
     `positioning_seeds`.

   Mutation: removing the existing-row check.
6. **The shared guard.** It refuses an accepted temporary file with a seed line, still unlocks ADR-012
   for gate mode, and refuses this ADR, accepted or not. Mutation: dropping the identity comparison.
7. **Row identity and refusals.**
   - The tags are present on an ERRORED row (a fixture that raises mid-training).
   - No gate arm name appears.
   - `is_official` is true for every official name and false for every temporal arm and experiment.
   - A dirty tree is refused, and no `--allow-dirty` option exists.
8. **The mask check** refuses fixture masks that overlap, miss a labelled node, or include an
   unlabelled one. Its output carries no class count.
9. **Selection.** It picks the highest validation AUPRC under the stated tie-break, and positioning
   rows carry the winner's run id and knobs. Scored ids align to `official_val_mask` in the retune
   and to `official_test_mask` in positioning.
10. **The gated paths are unchanged, and the official runner checks it.**
    - `scripts/run_dgf1_gnn.py --mode repro-check` must still reproduce
      `dgf1-20260913T232506Z-1acd5067` bit-for-bit.
    - The floor runner gains the same check against its pilot's seed 0,
      `dgf1-20260913T232205Z-c2af9760`: validation, `max_depth=8`, `subsample=0.8`, commit `2c8f484`.
      Its row carries `scores_sha256`.
    - Unit tests cover the official runner's refusal when either passing row is missing or at
      different code. Both checks are run on the snapshot after implementation.

### Clause 8 — Execution order

0. Remove `xaa` in its own commit (`git rm xaa`), before this ADR is accepted.
1. Accept this ADR.
2. Implement clauses 1–7 in one session, with the shared-guard fix as its own commit. Every test must
   pass and the suite must be green.
3. Run both repro-checks at the implementation's code. Both must pass.
4. Run the official retune on `official_val_mask`.
5. Run positioning on `official_test_mask`: 5 seeds × 3 arms.
6. Assemble the positioning report. The researcher reviews and dates it.
7. Record the winners and run ids here as a dated amendment, and in `notebooks/lab/`.

The runner enforces steps 3–5 (clause 5). Steps 0, 6 and 7 are procedural.

## Docs affected — to apply on acceptance

- **CLAUDE.md** (gitignored, so this ADR and `notebooks/lab/` are the versioned account; `.gitignore:2-3`):
  - `:121`, "Temporal splits only: train ≤ cutoff, test after.", is amended to record its one
    exception: DGF-1's official random-split track, reported only (ADR-010 clause 1,
    ADR-011:402-405, this ADR).
  - The DataSource leakage-test sentence that follows (`:121-123`) gains the official track's
    equivalent: clause 7 test 1's label perturbation, since that track has no cutoff.
  - The DGF-1 bullet (`:127-129`) is marked as governing the temporal track.
  - The held-out rule (`:102`) gains: `official_test_mask` is scored once per arm (clause 5).
- **doc-00 §7** (`docs/00-shared-core-graph-embedding-GUIDE.md:154`), "**Time-split, not
  random-split.**", gains a note: it governs gated claims, and DGF-1's official track is the
  labelled, reported-only exception.
- **doc-02:**
  - `:87`, "random-vs-temporal is a controlled comparison", is amended per clause 3: the tracks
    differ in what training sees and in their tuning.
  - `:6` gains a pointer to clause 3's narrow reading of "positioned against a public leaderboard".
  - **§5's template** (`:203-213`) has one row per model and no split dimension, while `:239` forbids
    a gated and an official number sharing a row. Official numbers therefore get a second table. Their
    Val cells stay empty (clause 3).
  - `:220`'s story sentence, "is competitive with strong fraud GNNs on the official random split
    (reported, not gated)", is not deleted. It is marked owed: this batch runs no fraud-specialised
    GNN (`:199` lists CARE-GNN / PC-GNN / GTAN; the template's only such row is GTAN). The wording is
    the researcher's.
  - `:126`: this batch fills the official-split cells for DGF-1 and the two XGBoost floors. The rest
    of §5's comparators (`:195-199`: MLP and Random Forest on the raw 17, GCN, GraphSAGE, GAT,
    CARE-GNN / PC-GNN / GTAN) remain owed on the official split.
- **`gates/GATE-TEMPLATE.md:30-31`** gains positioning reports (clause 3) as files a paper may draw
  on.
- **`gates/GATE-DGF1-0.md` is not edited.** It is signed and dated.

## Alternatives rejected

- **Transfer ADR-012's winners** (draft 4). They were selected partly on official-test labels
  (*Disclosure*), and ADR-009 makes `lr` and `batch_size` the knobs that are retuned, not carried.
- **Type the winner on the command line**, ADR-012's pattern. A typed configuration can be any
  configuration, and when a held-out scoring follows, the selection rule belongs in code.
- **3 seeds.** Rejected as a judgement: five matches both validation pilots and costs two more runs
  per arm.
- **8 seeds.** Eight was derived to make a gate clause resolvable, and nothing here resolves.
- **Reuse ADR-012's `Stage-1 seeds:` line** (draft 1). It would open the temporal test window.
- **An inductive official protocol.** Not reopened: ADR-011:402-405 fixed the protocol.
- **Gate on the official split.** ADR-010 clause 1 forbids it.
- **Skip the batch.** doc-02:126, doc-02:159-160 and `gates/GATE-DGF1-0.md:151` owe it.
- **Leave the shared guard for Gate 3's ADR** (draft 4). That holds a known hole open with a promise.
- **Put the mask path in `gbe/`.** It is DGraph-specific, and `gbe/` takes only what all four models
  implement identically.

## Consequences

- **New code, all in `adapters/dgf1/` and `scripts/`:** the official runner; config builders for all
  three arms; `is_official`; the positioning-report assembler; a floor repro-check; and the
  shared-guard fix. Each comes with clause 7's tests.
- **Cost:** two 9-configuration grids and 15 positioning runs. The temporal track's measured cost is
  in clause 2; the official cost is not predicted. The batch runs in the researcher's own terminal.
- **The constitution records the non-temporal exception**, rather than leaving it implicit in a
  script.
- **The positioning claim is weak by construction**, for clause 3's five reasons, and every citation
  of it says so.

## Revisit when

- **The official leaderboard adopts a temporal split** (`ADR-010:166-167`).
- **A model this batch does not run is to be scored on the official split.** That needs its own ADR,
  and it never re-scores these arms (clause 5).
- **Never** to re-score these three arms except through clause 5's defect route; never to attach a
  pass condition to any number this batch produces; never to move an official-split figure into a
  gate table.
