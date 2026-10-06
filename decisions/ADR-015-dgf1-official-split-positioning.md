# ADR-015 — DGF-1's official-split positioning batch

**Status:** proposed
**Date:** first proposed 2026-09-15 · **seventh draft 2026-10-06**, replacing `5f7f388`, `30fbf92`,
an uncommitted third, `8296f4b`, `f769880` and `a029413`. See *Draft history*.
**Deciders:** coderback
**Positioning seeds:** 5
**Opens no gated window.** This document carries no `Stage-1 seeds:` line, so the shared guard's seed
regex (`scripts/preregistration.py:25`) cannot parse it, accepted or not (clause 7, test 13).
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

**The purpose this batch serves**, which clause 3 keeps rather than narrows:

> it would destroy the leaderboard comparison that is DGF-1's stated reason for existing
> — `decisions/ADR-007-dgf1-gate-metric.md:161-162`

> Discarding it would throw away the comparability the model exists to obtain.
> — `decisions/ADR-010-dgraph-snapshot-reconciliation.md:129`

The programme-wide rigor list asks the same of every model: "Positioning vs prior work with a named
delta" (`docs/research-plan-UNIFIED-GBE-GDE.md:110`). **What limits that comparison** is stated
twice: leaderboard figures are "a third party's run under a protocol we cannot verify"
(`ADR-007:173-174`; `docs/02-dgraph-fin-embedding-model-BUILD.md:166-167`). Clause 3 therefore
checks a publication's **stated** protocol, and calls a pass "protocol matches as published", never
"verified".

ADR-003's architecture and ADR-012 clause 3's 40 epochs apply, **without** clause 3's runtime
fallback to 20 epochs (`ADR-012:200-206`); see clause 2.

## Draft history

Six drafts were rejected.
- **Drafts 1–3** (`5f7f388`, `30fbf92`, uncommitted) cited sources for propositions they do not
  contain.
- **Draft 4** (`8296f4b`) left its once-only rule as prose and carried hyperparameters that had seen
  official-test labels.
- **Draft 5** (`f769880`) let the defect route re-score one arm and tied code identity to commit ids.
- **Draft 6** (`a029413`) was judged acceptable with edits; both reviews found four blocking gaps:
  - its code identity left out the configuration the runner reads at runtime
    (`adapters/dgf1/config.yaml`, read by path at `adapters/dgf1/eval.py:18`), and the environment
    pin;
  - a failed repro-check could be escaped by a cosmetic edit that made a new code identity, and a
    repro-check that errored could not be found by its tags;
  - the four-point check on published figures was judged after our numbers existed, on figures
    chosen then;
  - nothing bounded what counts as a "defect".

`becb85c`, committed after draft 2, re-added draft 1's text as a stray root file `xaa`. It was removed
at `54b8367`.

## Disclosure

- **No model has been trained or scored on any official mask, and no registry row mentions one**
  (`grep -ci official experiments/registry.csv` → 0, 2026-10-06).
- **Two scripts read the masks, for statistics only:** `scripts/measure_dgf1_temporal_split.py:108-110`
  and `scripts/verify_dgraph_snapshot.py:179-193`. The masks are attached to the loaded graph at
  `adapters/dgf1/datasource_dgraph.py:122-123`, from the mapping built at `:140-143`. No trainer or
  scorer reads them.
- **On record about the masks:**
  - sizes and node-time quartiles (`ADR-010:36-40`);
  - per-mask fraud counts and prevalences (ADR-011 clause 2's table);
  - zero background nodes in the official masks (that table's official row).

  The three mask sizes sum to 1,225,601. That is the labelled count of ADR-011's three windows
  together (`ADR-011:194-196`). Whether the masks are disjoint is not on record; clause 5 checks it.
  No figure is recomputed here.
- **Every official-test user's label has already been used by the temporal track.** The masks are
  random over time —

  > The three distributions are indistinguishable — medians differ by **2 steps out of 821**. This is a
  > random split. — `decisions/ADR-010-dgraph-snapshot-reconciliation.md:42-43`

  — so official-test users fall in all three temporal windows. Their labels served as training labels
  (≤369), as validation labels that selected ADR-012's configurations (370–481), and as Gate-1 test
  labels (482–821).
- **So no fitted parameter and no tuned configuration carries over:** clause 2 retunes. **The design
  does carry over.** That covers ADR-003's architecture (set on ELL-1), and ADR-012's 30 inputs,
  fixed XGBoost parameters and both grids. ADR-012 fixed these before any DGF-1 score existed:

  > **No DGF-1 score of any kind exists.** The floor has never been run, on any window; no DGF-1 GNN
  > exists; no registry row is `model=dgf1`. Every threshold and count below is therefore fixed
  > blind in ADR-004's full sense.
  > — `decisions/ADR-012-dgf1-gate1-preregistration.md:68-70`

  ADR-012 was accepted at `c0f2053`, before the first DGF-1 row. Its text has since changed only by
  dated amendments and the removal of its seed-count placeholder. But "blind" means blind to scores,
  not to labels:
  - ADR-012 records window prevalences among the facts already seen (`:71-74`);
  - it records one look at test-window degree by label, which "argued **for** floor parity"
    (`:75-79`) and so shaped the inputs.

  **The input design was therefore informed by labels from a window that contains official-test
  users.**
- **This ADR's own choices were made after Gate 1's test results were known.** Gate 1 passed on
  2026-09-14, and its test window shares users with `official_test_mask`. The decisions to retune
  here, to allow verified comparisons (clause 3) and to set the claim rules were made 2026-10-06.
  Published DGraph-Fin figures are public, and this ADR does not claim they were unseen.

  Clause 3 makes these a mandatory weakness.

## Decision

### Clause 1 — The protocol is inherited; the leakage invariant is decided here

- **The protocol is ADR-011's**, quoted above: train and score on the official masks over the full
  graph as distributed. Every edge is visible in training, so features of `official_val_mask` and
  `official_test_mask` nodes reach the fitted weights through message passing. That is what
  "transductive" means. The number is never described as inductive.
- **The leakage invariant:**

  > **No `official_test_mask` label enters anything this batch fits or selects. No
  > `official_val_mask` label enters anything it fits; validation labels only select among clause 2's
  > configurations, each already trained.** Training seeds are the nodes in both
  > `official_train_mask` and `labelled_mask` (`adapters/dgf1/datasource_dgraph.py:120`).

  It is stated over **labels**, not over a list of quantities. So it binds everything either model
  derives from its training labels: the GNN's loss and class weights (`gbe/gnn/train.py:148-150`),
  the floor's `scale_pos_weight` ("on the training rows", `ADR-012:135-141`), and anything not named
  here. Clause 7 test 1 checks it by flipping labels.
- **The GNN's input transform is fitted on every node outside `official_val_mask` and
  `official_test_mask`**: the official training nodes plus the unlabelled background. It reads
  features only. This carries over ADR-011's fit-on-train rule and its reason:

  > The feature tensor covers all 3.7M nodes, so a statistic computed over it would include
  > test-window users. — `decisions/ADR-011-dgf1-temporal-split-derivation.md:247-248`

  The floors take their inputs untransformed: "the *same* 30 columns, untransformed"
  (`ADR-012:111`), and `STANDARDISED["xgboost"]` is `False` (`adapters/dgf1/baselines_tabular.py:91`).
- **Recency** is measured from the view's own cutoff (`ADR-011:241-242`). The official view is the
  whole graph, so its cutoff is the graph's last step, for training and scoring alike.
- **No early stopping and no best-epoch selection.** The official path passes no epoch hook. It
  scores validation only after training completes, and only in the retune.
- **Scoring uses exact neighbourhoods**, as `ADR-012:190-191` requires "for every arm and every
  reported row".
- **Only deterministic rows count.** Every row records `deterministic`
  (`gbe/run/seeding.py:59`). Selection, completeness, resume and the assembler count only rows
  recording `true`.
- **All code lives in `adapters/dgf1/` and `scripts/`.** Nothing about official masks enters `gbe/`.

### Clause 2 — Arms, tuning and seeds

- **An arm is a model class and an input set,** whatever it is named or however it is tuned. Each row
  records both, as `model_class` and `input_set`. There are three arms, ADR-012 clause 1's:
  - GraphSAGE on the 30 parity inputs;
  - XGBoost on the same 30 columns;
  - XGBoost on the raw 17.

  The parity inputs are the raw 17 beside the view's 11 edge-type counts and two recency columns
  (`adapters/dgf1/features.py:44-46`, `adapters/dgf1/datasource_dgraph.py:45-48`).
- **What obliges each arm.** The floors:

  > Stand up baselines (§5) on **both** splits; log **ROC-AUC and AUPRC** for each, labelled by split
  > — `docs/02-dgraph-fin-embedding-model-BUILD.md:126`

  DGF-1, under Gate 1: "Leaderboard positioning on the official (random) split is **reported, not
  gated**." (`docs/02-dgraph-fin-embedding-model-BUILD.md:159-160`). And GATE-DGF1-0 records:
  "**3. The official-split positioning numbers are owed.**" (`gates/GATE-DGF1-0.md:151`).
- **The two tuned arms are retuned on `official_val_mask`, at ADR-012's budget**: the same grids, one
  seed (seed 0) per configuration, selected on validation AUPRC. The *Disclosure* requires the
  retune. ADR-009 fixes which knobs it covers, and does not itself require a retune per split.
  - GNN: "`lr ∈ {0.5×, 1×, 2×}` ADR-003's `6.636671097096978e-4`, `batch_size ∈ {512, 1024, 2048}`"
    (`ADR-012:180`). These are ADR-009's two knobs for DGF-1 (`ADR-009:59-60, 68`).
  - Parity floor: "`max_depth ∈ {4, 6, 8}` × `subsample ∈ {1.0, 0.8, 0.6}`" (`ADR-012:146`). The
    other parameters stay fixed as `ADR-012:135-141` states.
  - Raw-17 floor: not tuned. It runs at the parity floor's winner, as on the temporal track. There
    the floor's nine retune rows are all `xgboost-parity` (validation rows at `e58ccb6`), and
    selection reads parity rows only (`scripts/run_dgf1_floor.py:122`).
  - **Selection is mechanical.**
    - Each arm's winner is the configuration with the highest validation AUPRC.
    - A non-finite AUPRC ranks below every finite one. An arm with no finite value is refused, and
      that refusal is a defect (clause 5).
    - Ties go to the earlier configuration in the grid's product order: the first-listed knob outer
      (`lr`, `max_depth`), the second inner (`batch_size`, `subsample`), each in ADR-012's listed
      order.
    - The positioning stage reads the winner from the retune rows, and every positioning row records
      the winner's run id. No configuration is typed.
  - **An inherited weakness, stated:** "a 9-way selection on one seed can pick seed-luck rather than
    a better configuration" (`ADR-012:185-186`). This batch has no validation pilot. A pilot could
    not change the configuration anyway: re-selecting after one "would be tuning on the same data
    twice" (`ADR-012:188-189`). So the positioning seeds are the only multi-seed measurement of
    each winner.
  - **Measured cost, temporal track:**
    - GNN grid: 3.03 h (9 retune rows at `2bd88e0`, `wall_clock_s` summing to 10,891 s; mean
      20.2 min, max 33.6).
    - Floor grid: 3.0 min (9 rows at `e58ccb6`, 177.7 s).

    The official grids train over the whole graph; their cost has not been measured and is not
    predicted. **There is no runtime escape valve and no epoch fallback.** If a run proves
    infeasible on the pinned machine, the batch is postponed with no end date, never shortened.
- **Positioning seeds: 5 per arm, seeds 1 to 5 (`range(1, 6)`).** Seed 0 is excluded because, at the
  winning configuration, it would retrain exactly the model the retune selected and carry that
  selection into the band.
  - **Reporting:** mean ± sample standard deviation (`ddof=1`). That is the convention
    `ADR-006:74-75` binds "Gate 3 onward", and both DGF-1 assemblers use it
    (`scripts/assemble_gate_dgf1_0.py:82`, `scripts/assemble_gate_dgf1_1.py:105`).
  - **The header line** is matched as `^\*\*Positioning seeds:\*\*\s*(\d+)`: one line, one integer,
    the shape `ADR-012:345-347` fixed for its own seed line. Seeds run from 1 up to that count.
  - **Five is a judgement, not a derivation.** It clears the floor of three (CLAUDE.md, "≥3 seeds
    with variance"). It equals the five seeds each validation pilot ran (GNN `…1acd5067` to
    `…ed807866`, floor `…c2af9760` to `…856b67ab`, validation rows at `2c8f484`). It is fixed before
    any official run, and it supports no inference between arms (clause 3).
  - **ADR-007's "hard floor of 5 per arm" is not the authority.** Its scope is the gates: "**Scope:**
    DGF-1's binary-classification gates — **Gate 1** (structure vs the tabular floor) and **Gate 3**
    (defending ablations)." (`ADR-007:104-105`).
  - The count is frozen by clause 5.

### Clause 3 — What may be claimed, and where

- **Our arms are reported as separate bands.** Each arm's ROC-AUC and AUPRC, mean ± sample std over
  five seeds, every AUPRC with the official test mask's prevalence and positive count.
  - There is no difference, standard error, p-value or superiority claim between our arms.
  - Five seeds was chosen by judgement, not to power a comparison.
  - No gated claim attaches, now or later.
- **Published figures are fixed before the retune, in a file.** The file is
  `configs/dgf1-official-published-figures.yaml`, committed before the retune's first row (clause 8).
  - **Inclusion rule:** every ROC-AUC or AUPRC on DGraph-Fin's official test mask, for a model doc-02
    §5 lists as a comparator (`docs/02-dgraph-fin-embedding-model-BUILD.md:194-199`), reported in
    the sources ADR-007 names — "published DGraph/GADBench numbers" (`ADR-007:120-121`) — as of the
    file's commit date. Every figure meeting the rule goes in, whatever it shows.
  - **Each entry records:**
    - the source, citation and access date;
    - the model and its class (graph or tabular);
    - the metric and value;
    - the run count, spread and model-selection rule, where the source states them;
    - a verdict on each of the four points below: `matches`, `differs`, `not stated`, or
      `not applicable`. Each verdict quotes the publication's own words.
  - **The file is in the code identity** (clause 5), so it is frozen from the retune's first row.
    The assembler reads only this file. A figure found later is printed as *unverified*, never
    ranked.
- **The four points, judged against the publication's stated protocol:**
  1. **masks:** the DGraph-Fin official train/val/test masks, as distributed;
  2. **metric:** the same metric on `official_test_mask`;
  3. **graph:** training over the full graph, every edge visible, as ADR-011:402-405 defines our
     track. For a model that uses no graph, this point is `not applicable`;
  4. **labels and data:** fraud and normal labels of `official_train_mask` nodes only, no use of the
     background classes' labels, and no data beyond the distributed dataset.
- **Ranking:**
  - **Only a figure whose applicable points all read `matches` is ranked** ("protocol matches as
    published"). Every other figure is shown beside ours, labelled *unverified*, with no ranking and
    no delta.
  - **Each figure is ranked against one of our arms only.**
    - DGF-1 is ranked only against published graph models.
    - Each floor is ranked only against published figures of its own model class.

    Ranking DGF-1 against a published tabular model would be a structure-versus-features claim,
    which this clause forbids.
  - **A ranking is three-valued:** above, within, or below our arm's five-seed range.
  - **The named delta** is our five-seed mean minus the figure, printed beside our band.
  - **There is never a significance claim** against a published figure.
- **"Leaderboard-comparable" means:** the same masks and metric as the public leaderboard. It implies
  nothing about a particular published figure beyond what clause 3's check records.
- **Official and temporal numbers are never differenced, ranked, or explained against each other**,
  in either direction.
  - The tracks differ in what training sees: the full graph here, edges dated ≤ the cutoff there.
    After clause 2 they also differ in their tuning.
  - So ADR-011 clause 2's second reason for its windows, "**Random-vs-temporal becomes a controlled
    comparison.**" (`ADR-011:209`), does not hold for these numbers.
  - Its other two label-free reasons still carry clause 2: pilot/gate identity and the view match
    (`ADR-011:111-112`).
  - *Docs affected* amends ADR-011 and doc-02:87. Cross-split AUPRC is forbidden regardless
    (ADR-007:135-136).
- **Never** to support or undermine "structure beats features at scale" or any claim about the
  temporal gap. Never as a comparator for Gate 3. Never to revisit Gate 1.
- **Five weaknesses, printed with the numbers wherever they appear:**
  1. transductive by ADR-011's definition, so not an inductive number;
  2. each configuration selected on one seed, with no multi-seed check before the test mask was
     scored;
  3. no fraud-specialised GNN: of doc-02 §5's comparators, only the two XGBoost floors are run;
  4. a comparison with a published figure holds only on the four points, and only as the publication
     states its protocol;
  5. not pre-registered against an unseen set. Every official-test label was used by the temporal
     track. The input design was informed by a label look at a window containing official-test users.
     This ADR's choices were made after Gate 1's test results were known (*Disclosure*). Nothing this
     batch fits or selects reads an official-test label.
- **Where: a positioning report**, assembled from the registry and the figures file by
  `scripts/assemble_dgf1_positioning.py`.
  - The first report is `gates/POSITIONING-DGF1-OFFICIAL.md`. A report after a clause-5 re-run is
    `gates/POSITIONING-DGF1-OFFICIAL-<YYYY-MM-DD>.md`.
  - It is not a `GATE-*` file and carries no verdict, the treatment ADR-013 clause 2 gave
    `gates/ERRATUM-DGF1-1.md`.
  - **The assembler counts, for each arm, exactly `positioning_seeds` deterministic, non-ERRORED
    positioning rows, all at one code identity.** An identical concurrent duplicate is counted once.
    Rows at a superseded identity (clause 5) are printed beside the counted ones, never counted.
  - It prints, rather than relying on a hand edit:
    - the label *random-split, leaderboard-comparable*, and its meaning above;
    - the five weaknesses;
    - every AUPRC's prevalence and positive count;
    - the retune table, labelled *selection runs, one seed, not results*;
    - the figures file, with each entry's four verdicts and quotes.
  - It refuses to write a ranking or delta that this clause does not allow.
  - It refuses to overwrite a dated report.
  - `gates/GATE-TEMPLATE.md:30-31` ("Papers are assembled from gate files; nothing is reported that is
    not in one.") gains this file type; see *Docs affected*.
- **Validation numbers are never reported as results.** The only ones are selection values, which
  carry a winner's curse (`ADR-012:160`). doc-02's Val columns stay empty for official rows.

### Clause 4 — Row identity

- **Tags:**
  - `arm`: `official-dgf1-parity`, `official-xgboost-parity` or `official-xgboost-raw17`;
  - `model_class` and `input_set` (clause 2);
  - `experiment`: `official-retune` or `official-positioning`;
  - `window`: `official-val` on retune rows, `official-test` on positioning rows;
  - the hashed config's `split` value: `official-random-mask`, in place of the temporal block;
  - `phase`, as on the temporal track: `P0` for the floors, `P1` for the GNN
    (`scripts/run_dgf1_gnn.py:286`).
- **Every row also carries:**
  - the knobs it ran (`lr` and `batch_size`, or `max_depth` and `subsample`);
  - `code_identity` (clause 5);
  - the resolved device (`cpu` for the floors);
  - `positioning_seeds`, the header's count at run time, on retune and positioning rows alike.

  Positioning rows also carry `retune_winner_run_id`.
- **A stage's rows** are every registry row whose `window` is `official-val` (the retune) or
  `official-test` (positioning), and whose (`model_class`, `input_set`) is one of the three arms. That
  holds at any commit, code identity or arm name, and includes ERRORED rows. A later model's official
  rows are therefore never this stage's. The window strings are pinned literally by a test, and
  changing them is a defect under clause 5.
- **Logging order.**
  - **The tags are logged at session entry, not at the end**, so an ERRORED row carries them. The
    temporal trainers log theirs last (`adapters/dgf1/train_gnn.py:208-211`;
    `adapters/dgf1/baselines_tabular.py:232-234`), and `RunSession` writes a row on error
    (`gbe/run/session.py:66-71`). So a temporal row that errors carries no tags.
  - **`scores_sha256` is logged the moment the score file is saved.** Metrics are logged last.
- **The official path has its own config builders, for all three arms.** Both temporal entry points
  write `arm` and `window` after `**base_cfg` (`adapters/dgf1/train_gnn.py:166-168`,
  `adapters/dgf1/baselines_tabular.py:219-224`), so neither can be steered through config.
- **A tested predicate, `is_official(row)`**, lives in `adapters/dgf1/`. It is true when `window` is
  `official-val` or `official-test`. Every DGF-1 assembler written from now on, Gate 3's included,
  excludes official rows through it.
- **Existing consumers, as checked.** This list records what was checked; it does not claim to be
  complete.
  - `scripts/assemble_gate_dgf1_1.py:83` keeps only rows with `window == "482-821"`, so official
    rows are invisible to it.
  - `scripts/assemble_gate_dgf1_0.py:206` matches `arm.startswith("xgboost")`, which the `official-`
    prefix avoids. But `:219`, `:221` and `:222` count every DGF-1 row. The script never runs again:
    it refuses to regenerate over GATE-DGF1-0's signed verdict (`:97-104`). It is not edited.
  - `scripts/check_extract_regression.py:237` keeps only its own `experiment` tag.
  - `scripts/freeze_extract_reference.py:100-104` keeps `phase == "P0"` rows whose `baseline` is
    `rf` or `lr` (`:55`). `scripts/run_ell1_gnn.py:64-68` keeps every clean row whose `baseline` is
    `rf`, of any model, and requires exactly three. DGF-1's floor rows record `baseline: xgboost`
    (validation row `…c2af9760`), and the official runner offers no other model. **A future ADR that
    scores an RF or LR arm on the official split must reckon with both.**
  - The repro-check's `read_row` and the tests that read the registry select by run id.

### Clause 5 — `official_test_mask` is scored once, and the code enforces it

- **A separate entry point, `scripts/run_dgf1_official.py`, with modes `retune`, `positioning` and
  `adopt`:**
  - **No pre-registration path argument.** It reads this file by a path fixed in its source, refuses
    unless `**Status:**` is `accepted`, and takes the seed count from this file's `**Positioning
    seeds:**` line, never from a flag. Every mode requires acceptance.
  - **No `--allow-dirty` and no window, model, seed, knob or device options.** A dirty tree is refused
    in every mode. Smoke testing uses fixtures, never the snapshot.
  - **It refuses to start if the figures file (clause 3) is missing or malformed.**
  - **The GNN never runs on CPU, and runs only on the GPU the environment pin names.** It applies
    `device_problems` (`scripts/run_dgf1_gnn.py:205`), which refuses a CPU device. It also compares
    `torch.cuda.get_device_name()` with the pin's `gpu` field, a comparison `device_problems` does not
    make (`:219-228`). The floors run on CPU, as their temporal rows did.
  - **A mask check on every load, before anything trains.** The three official masks must be
    boolean, pairwise disjoint, and together exactly `labelled_mask`. It refuses otherwise, and
    prints only pass or fail: no class count and no prevalence. A failure is a fact about the
    snapshot, recorded in a new ADR. The check is never relaxed.
- **Code identity.** The runner lists, as a constant, **every repository file it imports or opens**.
  - The list includes `adapters/dgf1/config.yaml`, which holds the epochs, architecture and grid and
    is read by path (`adapters/dgf1/eval.py:18`), and the environment pin
    `experiments/extract_reference_env.txt` (`scripts/run_dgf1_gnn.py:49`). It also includes the
    figures file.
  - **Excluded:** the registry, this ADR, `experiments/scores/` and `data/`.
  - **The runner imports neither `scripts/run_dgf1_gnn.py` nor `scripts/preregistration.py`.** The
    helpers it needs from the former (`device_problems`, `repro_check_verdict` and what they call)
    move to `adapters/dgf1/`. Step 4's repro-check then confirms that the move changed nothing in
    the gated runner.
  - `code_identity` is a SHA-256 over the listed files' git blob ids at HEAD. Because a dirty tree is
    refused, HEAD equals the working tree. It is recorded on every official row.
  - At an older commit, a listed path that does not exist, or a `git_commit` of `"unknown"`, means
    "not matching".
  - **The listed files are frozen from the retune's first row to positioning's last.** Every
    comparison below is by `code_identity`, never by commit id. When it refuses, the runner prints
    which listed paths differ. Work that touches a listed file, such as Gate-3 changes to `gbe/`,
    waits or happens on another checkout.
- **The gated paths are re-certified first.**
  - **Finding repro rows.** The runner finds each temporal runner's `repro_check` rows by its rebuilt
    reference config hash, not by tag, so a repro-check that errored is found too.
  - **It refuses if any repro row of either runner, at any code identity, fails its recomputed
    verdict.** The verdict is the stored comparison plus the score-file re-hash (`repro_check_verdict`;
    the floor gets the same). An ERRORED row fails. So does a row whose score file is gone.
  - **The only exception** is a row whose run id appears in a constant, `RESOLVED_REPRO_FAILURES`,
    beside the ADR or dated lab entry that records its root cause. The runner checks that the named
    file exists.
  - **It also requires at least one passing repro row at HEAD's code identity**, for each runner.

  The registry records no verdict; the runner only prints it (`scripts/run_dgf1_gnn.py:342-353`).
  Retrying, or making a cosmetic edit to get a fresh identity, therefore cannot satisfy this rule.
  For official runs it supersedes ADR-013's "No runner checks for it" (`ADR-013:282-283`). If a check
  fails, the batch waits while the mismatch is investigated to root cause (ADR-008:108-109). A root
  cause in the environment bears on every DGF-1 row from that code path, not only on this batch.
- **Each stage runs once: the retune on `official_val_mask`, positioning on `official_test_mask`.**
  - **A stage's first row pins its code identity.** ERRORED rows count. Thereafter the stage runs
    only at that identity.
  - **No fresh start once any stage row exists**, ERRORED or not. `--resume` runs only the missing
    pairs, (arm, configuration) or (arm, seed): those with no deterministic, non-ERRORED row. It runs
    at the pinned identity. At identical code, a deterministic re-run reproduces what an ERRORED run
    would have produced, so nothing can be chosen by resuming.
  - **A crash that scored nothing.** A stage in which no row carries `scores_sha256`, and for which
    `experiments/scores/official/` holds no file, may restart at new code. Its identity must be named
    in `SUPERSEDED_IDENTITIES` beside a dated amendment to this ADR recording the crash. The runner
    checks that no row at that identity carries a score hash.
  - **Positioning requires a complete retune.** That is exactly one clean, deterministic,
    non-ERRORED row per (arm, configuration). A duplicate from a concurrent launch is accepted only
    if its `scores_sha256` equals the first's; otherwise the runner refuses. Positioning's code
    identity must equal the retune's.
  - **The seed count is frozen.** Every official row records `positioning_seeds`. The runner refuses
    if any existing official row records a value different from this file's header.
  - **Score files go to `experiments/scores/official/`, named by run id,** for both stages. The runner
    refuses if that directory holds a file whose run id has no registry row.
  - **`adopt <run_id>`.** A hard kill between saving a score file and writing its row (a closed
    console, the out-of-memory killer) leaves such a file. `adopt` appends an ERRORED row for it
    under the stage's pinned identity and keeps the file. The pair then counts as missing, and
    resuming reproduces it at identical code.
  - **What the code cannot close.**
    - Deleting rows together with their score files.
    - Merging rows written by another checkout.

    Only deliberate acts do either, and the registry is append-only (CLAUDE.md). The researcher
    commits the registry after each stage (clause 8). A revert of rows that scored nothing leaves no
    trace, and since nothing was seen, it gives nothing to choose.
- **The defect route.** A defect found after a stage has rows is not re-run until an accepted ADR
  allows it.
  - **What counts as a defect.** It is behaviour that an accepted clause, of this ADR or one it
    inherits, or the code's committed contract (its docstrings and tests) rules out. It must be shown
    by a test that uses fixtures only and fails at the pinned identity. ADR-013's defect met this bar:
    it broke accepted ADR-011 clause 3 (`ADR-013:229-230`), and it was "found by reading code" with
    "No number … computed to find it or to size it" (`ADR-013:119-120`). **An improvement the
    accepted text does not require is not a defect.** It is a new arm, and these three arms are
    never re-scored.
  - **What the defect ADR must contain.**
    - It locates the defect in ADR-013 clause 1's form: where, what, since when, why undetected
      (`ADR-013:225-236`).
    - It discloses how the defect was found and which official numbers its author had seen.
    - **After any positioning row exists**, it states the defect's expected effect on each arm and
      its direction, in ADR-013 clause 4's form ("state what, where and why, then the impact on the
      test's severity", `ADR-013:305-306`).
    - It is **accepted** before anything re-runs.
  - **What re-runs.**
    - Before any positioning row exists: the retune re-runs in full at the new identity.
    - After: both stages re-run **in full, for all three arms**. There is never a partial re-run.
  - **How the refusal changes.** The implementation records the superseded identity in
    `SUPERSEDED_IDENTITIES`, beside the ADR's path, and that is the only way the refusal changes. The
    runner checks that the named ADR exists, carries `**Status:** accepted`, and names that identity.
  - **What stays.** The original rows stay, printed beside the new ones in a new dated report. The
    original report is not edited.

  This follows ADR-008's rule: "**A mismatch is investigated to root cause — never re-run until it
  matches** (that is the optional-stopping error in another costume)." (`ADR-008:108-109`).
- **Other models later.** A future ADR may score `official_test_mask` with a model class or input
  set this batch does not run; doc-02 §5's other baselines are owed. These three arms are never
  re-scored, whatever a later arm is named or however it is tuned.

### Clause 6 — The shared pre-registration guard is a separate defect

- `require_accepted_preregistration` (`scripts/preregistration.py:28-57`) checks `**Status:**` and
  `**Stage-1 seeds:**` on whatever path it is handed.
  - The tests unlock the test window with temporary files (`tests/test_dgf1_floor.py:167-170`
    unlocks an "ADR-999" at 12 seeds; also `tests/test_dgf1_runner.py:68-83`).
  - ADR-012, accepted at `**Stage-1 seeds:** 8`, unlocks 482–821 today
    (`tests/test_dgf1_runner.py:92`).
- **This ADR's runner neither calls nor imports that guard** (clause 5), so nothing here depends on
  it.
- **It is a defect in a guard, fixed as its own code change**, as ADR-014 treated the CPU fallback.
  - **The fix:** the guard takes the document its caller's mode names and refuses any other, by
    resolved path.
  - **The tests:** they are rewritten so an accepted temporary file with a seed line is refused,
    including the tests that match on refusal messages.
- **It is owed before Gate 3's pre-registration is accepted, not before this ADR.**
- Whether Gate 1's gate mode should still unlock 482–821 now that Gate 1 is signed is a separate
  question, not decided here.

### Clause 7 — Tests, written in the implementation sessions, accepted by mutation

1. **Label flip, all three arms, driving the official runner's own stage functions.** The fixture's
   validation and test class counts are asymmetric, so a class-weight mutation changes the weights.
   - **Positioning, configuration fixed:** flipping every `official_val_mask` and
     `official_test_mask` label leaves every arm's scored probabilities bit-identical.
   - **Retune:** flipping every `official_val_mask` label leaves each configuration's probabilities
     bit-identical; its metrics may change. Flipping every `official_test_mask` label leaves every
     configuration's probabilities, metric fields and the selected winner identical.

   Mutations that must each fail it:
   - class weights, `scale_pos_weight` or the loss computed over all labelled nodes;
   - early stopping or best-epoch selection on validation.

   A structural test pins that the official path passes no epoch hook.
2. **Inputs.**
   - The floors receive `node_inputs` untransformed.
   - Perturbing the raw features `x` of `official_val_mask` and `official_test_mask` nodes leaves the
     GNN's fitted transform identical. Only `x` is perturbed: edges are not, since they feed training
     nodes' counts.

   Mutations: standardising a floor's inputs; fitting the transform over all nodes.
3. **The two tracks cannot be confused.**
   - The temporal runners expose no mask option, and the official runner exposes no window option.
   - The official training edge set equals the full edge set.

   Mutation: a cutoff applied to the official view.
4. **The document check.**
   - The runner accepts no path argument and refuses this file unless it is accepted.
   - It parses `Positioning seeds` with the stated regex, refuses a missing or malformed line, and
     runs seeds 1 to that count.

   Mutations: accepting a path argument; dropping the status check; starting seeds at 0.
5. **Code identity.**
   - Static import analysis (AST or `modulefinder`, since the runners import inside functions) finds
     every repo module the runner imports.
   - A `sys.addaudithook` "open" hook, during a fixture run, records every file it opens.
   - Both sets must lie within the listed files, apart from the four exclusions.
   - Changing a listed file changes `code_identity`.
   - The runner imports neither `scripts/run_dgf1_gnn.py` nor `scripts/preregistration.py`.

   Mutation: dropping `adapters/dgf1/config.yaml` from the list.
6. **Once-only.**
   - A fresh start is refused when any stage row exists, including an all-ERRORED stage.
   - `--resume` runs only missing pairs, and only at the pinned identity.
   - Positioning refuses each of: an incomplete retune; differing duplicates; a different identity; a
     changed `positioning_seeds`.
   - An orphan score file is refused, and `adopt` clears it.
   - A nothing-scored restart is honoured only through `SUPERSEDED_IDENTITIES`, and only when no row
     there carries a hash.
   - A later model's official rows are not counted as this stage's.

   Mutations: removing the existing-row check; treating an all-ERRORED stage as absent.
7. **The repro precondition.**
   - An ERRORED repro row is found by its config hash.
   - A failing row at **any** identity refuses until it is listed in `RESOLVED_REPRO_FAILURES` with an
     existing record.
   - A fresh identity does not clear a failure.
   - A passing row at HEAD's identity is required.

   Mutation: scoping the failure check to HEAD's identity.
8. **Row identity and refusals.**
   - Tags are present on an ERRORED row (a fixture that raises mid-training).
   - `scores_sha256` is logged at save.
   - The window strings are pinned literally.
   - `is_official` is correct on official and temporal rows.
   - Rows record the device (`cpu` for floors).
   - Only rows recording `deterministic: true` are counted.
   - A dirty tree is refused, and no `--allow-dirty` or `--device` option exists.
9. **The mask check** refuses fixture masks that overlap, miss a labelled node, or include an
   unlabelled one. Its output carries no class count.
10. **Selection.**
    - It picks the highest validation AUPRC under the stated tie-break.
    - Non-finite values rank last, and an arm with none finite is refused.
    - Positioning rows carry the winner's run id and knobs.
    - Scored ids align to `official_val_mask` in the retune and to `official_test_mask` in positioning.
11. **The assembler.**
    - It refuses an arm short of `positioning_seeds` rows at one identity.
    - It refuses each of: a ranking or delta for a figure not in the file, or with an applicable
      point not `matches`; DGF-1 ranked against a tabular figure; a floor ranked against another
      class; an overwrite of a dated report.
    - It prints three-valued rankings, superseded rows beside counted ones, the label, the five
      weaknesses, the prevalences, the retune table and the figures file.

    Mutation: dropping a weakness line.
12. **The defect route.** The runner refuses a `SUPERSEDED_IDENTITIES` entry whose named document is
    missing, is not accepted, or does not name that identity.
13. **The shared guard refuses this ADR**, accepted or not: the header claim.
14. **The gated paths are unchanged.**
    - The floor runner gains a repro-check against its pilot's seed 0,
      `dgf1-20260913T232205Z-c2af9760`: validation, `max_depth=8`, `subsample=0.8`, commit `2c8f484`.
      Its row carries `scores_sha256`.
    - The floor runner's inline config values are factored out so the reference hash can be rebuilt,
      and the floor gets its own compared-key list.
    - After implementation, both repro-checks are run on the snapshot, and test 7's precondition
      enforces the result.

### Clause 8 — Execution order

1. Accept this ADR.
2. Implement clauses 1–5 and 7, one component per session, each with its tests. The batch does not
   start until every clause-7 test passes and the suite is green.
3. Compile and commit the figures file (clause 3).
4. Run both repro-checks at the implementation's code identity. Both must pass.
5. Run the official retune on `official_val_mask`, then commit the registry.
6. Run positioning on `official_test_mask` (5 seeds × 3 arms), then commit the registry.
7. Assemble the positioning report. The researcher reviews and dates it.
8. Record the winners and run ids here as a dated amendment, and in `notebooks/lab/`.

The runner enforces steps 3–6: it refuses without the figures file, and its identity includes it.
Steps 7 and 8, and the registry commits, are procedural.

## Docs affected — to apply on acceptance

CLAUDE.md and `docs/` are gitignored (`.gitignore:2-3`), so this ADR and `notebooks/lab/` are the
versioned account. The constitution is amended only on the researcher's explicit instruction, as
ADR-011 was (`ADR-011:28-29`).

- **CLAUDE.md:**
  - `:121`, "Temporal splits only: train ≤ cutoff, test after.", records its one exception: DGF-1's
    official random-split track, reported only (ADR-010 clause 1, ADR-011:402-405, this ADR).
  - The DataSource leakage-test sentence (`:121-123`) gains the official track's equivalent, clause
    7 test 1's label flip, since that track has no cutoff.
  - The DGF-1 bullet (`:127-129`) is marked as governing the temporal track.
  - The held-out rule (`:102`) gains two sentences:
    - `official_test_mask` is scored once per arm (clause 5);
    - under the official protocol its nodes' features reach the model through message passing,
      while their labels enter nothing and their features enter no fitted statistic.
- **doc-00 §7** (`docs/00-shared-core-graph-embedding-GUIDE.md:154`), "**Time-split, not
  random-split.**", gains a note: it governs gated claims, and DGF-1's official track is the
  labelled, reported-only exception.
- **doc-02:**
  - `:83` ("the one training set, for every run") and `:90` (fit-on-train) are marked as governing
    the temporal track.
  - `:87`, "random-vs-temporal is a controlled comparison", is amended per clause 3.
  - `:6`, and `:166-167` ("a protocol we cannot verify"), gain a pointer to clause 3's check against
    a publication's stated protocol.
  - **§5's template** (`:203-213`) has one row per model and no split dimension, while `:239` forbids
    a gated and an official number sharing a row. Official numbers get a second table, and their Val
    cells stay empty.
  - `:220`'s story sentence ("is competitive with strong fraud GNNs on the official random split
    (reported, not gated)") stays, marked owed. This batch runs no fraud-specialised GNN (`:199`). A
    "competitive" claim needs clause 3's check against those GNNs' published figures. The wording is
    the researcher's.
  - `:126`: this batch fills the official-split cells for DGF-1 and the two XGBoost floors. The rest
    of §5's comparators remain owed on the official split (`:195-199`): MLP and Random Forest on the
    raw 17, GCN, GraphSAGE, GAT, and CARE-GNN / PC-GNN / GTAN.
- **ADR-011** gains a dated pointer at `:209` and `:446`: the random-vs-temporal comparison is not
  made (clause 3). Clause 2 stands on its other two label-free reasons (`:111-112`).
- **`docs/research-plan-UNIFIED-GBE-GDE.md:110`** ("Positioning vs prior work with a named delta")
  gains a pointer to clause 3, where a named delta is allowed.
- **`docs/timeline.md:24` and `:184-185`** say "both arms". This batch runs three. The wording is the
  researcher's.
- **`gates/GATE-TEMPLATE.md:30-31`** gains positioning reports (clause 3) as files a paper may draw
  on.
- **`gates/GATE-DGF1-0.md` is not edited.** It is signed and dated.

## Alternatives rejected

- **Transfer ADR-012's winners** (draft 4). They were selected partly on official-test labels
  (*Disclosure*).
- **Show published figures beside ours, never ranked** (draft 5). It contradicts DGF-1's stated
  purpose (ADR-007:161-162, ADR-010:129), and those sources were not amended.
- **Choose and check published figures after our numbers exist** (draft 6). Which figures appear, and
  which pass, could follow from whether they flatter DGF-1.
- **Rank any arm against any figure.** Ranking DGF-1 against a tabular model is a
  structure-versus-features claim, and ranking all our arms against one figure orders our arms.
- **Code identity by commit id** (draft 5), or **by import closure alone** (draft 6). The first
  refuses for ever after an unrelated commit. The second misses the configuration the runner reads by
  path.
- **Scope repro failures to the current identity** (draft 6). A cosmetic edit escapes it.
- **Seeds 0–4.** Seed 0 retrains the model the retune selected.
- **A lock file against concurrent launches.** A stale lock refuses for ever. Identical duplicates are
  harmless, and differing ones are refused.
- **Type the winner on the command line**, ADR-012's pattern. A typed configuration can be any
  configuration.
- **Fit the transform over all nodes** (draft 5). It puts scored users into a fitted statistic, which
  ADR-011:245-248 exists to prevent.
- **3 seeds.** Rejected as a judgement: five matches both validation pilots and costs two more runs
  per arm.
- **8 seeds.** Eight was derived to make a gate clause resolvable, and nothing here resolves.
- **Reuse ADR-012's `Stage-1 seeds:` line** (draft 1). It would open the temporal test window.
- **An inductive official protocol.** Not reopened: ADR-011:402-405 fixed the protocol.
- **Gate on the official split.** ADR-010 clause 1 forbids it.
- **Skip the batch.** doc-02:126, doc-02:159-160 and `gates/GATE-DGF1-0.md:151` owe it.
- **Fix the shared guard inside this ADR** (draft 5). This runner never calls it, and it is a code
  defect, not a decision.
- **Put the mask path in `gbe/`.** It is DGraph-specific.

## Consequences

- **New code, all in `adapters/dgf1/` and `scripts/`:**
  - the official runner, with its code identity, once-only, orphan and adopt logic;
  - mask-based equivalents of the `TemporalSplit`-based functions it needs (`train_dgf1`,
    `score_view`, `floor_design`, `run_floor`, `run_dgf1`, `train_seed_mask`, `window_target_mask`);
  - config builders for all three arms;
  - `is_official`;
  - `device_problems` and the repro verdict moved into `adapters/dgf1/`;
  - a floor repro-check;
  - the figures file;
  - the positioning-report assembler.

  Each comes with clause 7's tests.
- **Cost:** two 9-configuration grids and 15 positioning runs. The temporal track's measured cost is
  in clause 2; the official cost is not predicted. The batch runs in the researcher's own terminal.
- **The constitution records the non-temporal exception**, rather than leaving it implicit in a
  script.
- **The numbers are weak by construction**, for clause 3's five reasons, and every citation of them
  says so. A comparison with published work is as strong as the four-point check behind it, as the
  publication states its protocol, and no stronger.

## Revisit when

- **The official leaderboard adopts a temporal split** (`ADR-010:166-167`).
- **A model this batch does not run is to be scored on the official split.** That needs its own ADR,
  and it never re-scores these arms (clause 5).
- **Never** to re-score these three arms except through clause 5's defect route; never to attach a
  pass condition to any number this batch produces; never to move an official-split figure into a
  gate table.
