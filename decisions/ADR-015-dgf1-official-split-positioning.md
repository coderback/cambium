# ADR-015 — DGF-1's official-split positioning batch

**Status:** accepted · erratum accepted 2026-10-08 (15:30) by coderback, at the end of this file
**Date:** first proposed 2026-09-15 · **accepted 2026-10-07 by coderback**, as the fifteenth draft.
It folds the fourteenth (`7118b1a`) review's non-blocking items in, under the review's stopping
rule. It replaces `5f7f388`, `30fbf92`, an uncommitted third, `8296f4b`, `f769880`, `a029413`,
`3b58edc`, `a01d865`, `e9c3ea7`, `b98fac3`, `d86b55e`, `6f7ccef`, `55e537d` and `7118b1a`. See
*Draft history*.
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

Thirteen drafts were rejected.

- **Drafts 1–3** (`5f7f388`, `30fbf92`, uncommitted) cited sources for propositions they do not
  contain.
- **Draft 4** (`8296f4b`) left its once-only rule as prose and carried hyperparameters that had seen
  official-test labels.
- **Drafts 5–7** (`f769880`, `a029413`, `3b58edc`) enforced once-only scoring with a growing state
  machine. Each review closed holes in it and found new ones in the parts added.
- **Draft 8** (`a01d865`) replaced that machine with two invariants, and its review judged the
  argument sound. It found three completeness gaps:
  - **the identity omitted factors that change results:** the data, `CUBLAS_WORKSPACE_CONFIG`, thread
    counts and the CPU. A re-run that varied one of them would disagree, refuse, and leave only the
    defect route, which is a full redraw after the numbers were seen;
  - **the pin was read only from gitignored score files**, so deleting them re-pinned;
  - **a route let an "unrestorable" environment change take the defect route.** It could never
    certify, and it was a second draw the author would adjudicate.

  The review also found rules that would have caught a later ADR's rows. Draft 9 completed the
  identity, compared before saving, read the pin from rows as well as files, deleted that route,
  and scoped every rule to this batch.
- **Draft 9** (`e9c3ea7`) was checked narrowly, on clause 5 alone. Probes run on this machine, and
  reproduced, showed three unset environment variables that change results while lying outside the
  identity:
  - `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`;
  - `DISABLE_ADDMM_CUDA_LT`;
  - `ATEN_CPU_CAPABILITY`.

  The check also found three further gaps: a disagreement row whose keys were unspecified; an
  identity whose components were not recorded, so "what differs" could not be printed; and
  automatic updates that would make a stop likely. Draft 10 fixes these:
  - it hashes environment variables by family;
  - it hashes the loaded data rather than its files;
  - it names the disagreement keys;
  - it logs the identity's components;
  - it pauses updates for the batch.

  One claim in that check did not reproduce. `torch.save` is byte-stable for a fixed filename, and
  the probe had compared two filenames.
- **Draft 10** (`b98fac3`) was reviewed on its diff alone. It said to read the environment variables
  "after `gbe.run.seeding` is imported". But importing `gbe.eval` loads scikit-learn, which then sets
  `KMP_DUPLICATE_LIB_OK` and `KMP_INIT_AT_FORK` (`sklearn/__init__.py:56,60`). That put both inside a
  hashed family, so the identity depended on when it was read. This was reproduced in a fresh process.
  The review also found three further problems:
  - the update step claimed to detect updates it cannot see;
  - the disagreement block could be cleared by a lab entry, or by any accepted ADR;
  - logging variable values could commit a credential.

  Draft 11 fixed these:
  - the variables were read once, at a fixed point after every import;
  - the stated detection was narrowed;
  - a resolution had to name the row;
  - secret-like values were hashed.
- **Draft 11** (`d86b55e`) was reviewed on its diff. **Both blocking findings were in text draft 11
  itself added:**
  - **An escape hatch.** "If the root cause shows that the pinned result itself ran outside its
    identity, that is a defect" let the author blame an already-seen pinned result and redraw.
  - **A deadlock.** "The key is never re-run" could not be honoured, because every invocation
    re-runs its whole stage.

  The review also found three smaller problems:
  - test 5's equality check could not catch its own mutation;
  - a sentence contradicted the run-id rule for mismatch ADRs;
  - resolved disagreements would vanish from the report.

  Draft 12 deleted both sentences and fixed the rest. Its fresh-process probes confirmed that,
  after `gbe.eval`, no import on the planned runner's path, no CUDA initialisation and no GPU work
  sets a family variable.
- **Draft 12** (`6f7ccef`) was reviewed on its diff. Its blocking finding was again in its own new
  text. "Whatever the root cause, the pinned result stands", together with "a later re-run can add
  a new disagreement row", left no exit when the pinned result is the one that will not reproduce:
  - every re-run would disagree again, each time needing a new ADR;
  - a stage that was not yet complete could finish only by re-running until a run agreed;
  - the sentence also contradicted the defect route, which does change the identity.

  The review also found two smaller problems:
  - the session-exit re-read came after the result was saved and pinned;
  - the new print rule had no test.

  Draft 13 stopped the batch in that case, scoped the "stands" sentence to resolutions, moved the
  re-read before the comparison, and added mutations to tests 5 and 12. It added none to test 6,
  which is supplied in draft 14.
- **Draft 13** (`55e537d`) was reviewed on its diff. **Its blocking finding was a principle, not a
  mechanism.** Its new stop rules could withhold results already seen:
  - each positioning seed's metrics are visible as it runs, and a stopped batch produced no report;
  - the author controls two of the triggers, by resuming updates or by setting a variable outside
    the hashed families.

  That offers no second draw, but it does let a choice follow from seeing a result, which is what
  clause 5 exists to prevent. The review also found:
  - a stop that was adjudicated ("found") rather than mechanical;
  - a lapse that ended the batch even before anything was scored;
  - session reads compared only with themselves, never with the pinned identity;
  - a post-scoring re-read that certify cannot perform;
  - a test fixture that could not catch the ordering error.

  Draft 14 fixes these. A stop withholds nothing, the stop condition is mechanical, and a lapse
  before the first official score pins nothing.

- **Draft 14** (`7118b1a`) was reviewed on its diff and **nothing blocking was found**. Its review
  judged the mechanical stop, the entry read compared with the pinned identity, the re-read before
  comparison, and the test-6 mutations to hold. Under the stopping rule below, its should-fix items
  and nits are folded into this fifteenth draft without another round:
  - **A stop is recorded as a dated amendment here, and write-ups cite it.** Without that, a stopped
    batch's numbers sat on record but could not be cited, and listing them in a lab entry would
    break CLAUDE.md's ten-line cap.
  - **The ways a batch stops are listed exactly.** A batch has stopped, from row timestamps alone,
    five weeks after its first official score with no dated report. That closes the halts that
    were not stops.
  - **The registry is committed after every scoring invocation**, and no metric is printed before
    its row is written.
  - **Certify's post-scoring re-read is dropped.** Its bit-for-bit verdict already catches any
    variable that changes its result, and the re-read's result could not have been stored.
  - **Test 6** checks Invariant 1's boundary rather than an untestable lapse.
  - **"The batch's first official score" is defined.**
  - **Smaller items:** the restart after a lapse is marked procedural, and ignored entries are
    printed by the assembler too.

**How this review ends.** The researcher set this rule on 2026-10-07:
- **When a diff pass finds nothing blocking,** its should-fix items and nits are folded in without
  another round, and the ADR goes to the researcher's accept-or-reject decision.
- **When it finds something blocking,** that finding is fixed and the diff pass is repeated.
- **What is left after that is implementation detail:** read points, fixtures, certify's
  mechanics. Clause 7's mutation tests and the implementation's code review check those
  mechanically, which prose review cannot.

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
  here, to allow comparisons that pass clause 3's check, and to set the claim rules were made
  2026-10-06. Published DGraph-Fin figures are public, and this ADR does not claim they were unseen.

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
- **Only deterministic rows count.** Every row records `deterministic` (`gbe/run/seeding.py:59`).
  Only rows recording `true` count, and clause 5 checks that results actually reproduce.
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
    - A non-finite AUPRC ranks below every finite one. An arm with no finite value is refused.
    - Ties go to the earlier configuration in the grid's product order: the first-listed knob outer
      (`lr`, `max_depth`), the second inner (`batch_size`, `subsample`), each in ADR-012's listed
      order.
    - Positioning reads the winner from the retune rows, and every positioning row records the
      winner's run id. No configuration is typed.
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
    infeasible on the pinned machine before the batch's first official score, the batch is
    postponed with no end date, never shortened. After the first official score, infeasibility is
    covered by clause 5's five-week stop, or by an earlier dated declaration, and a stop withholds
    nothing.
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

### Clause 3 — What may be claimed, and where

- **Our arms are reported as separate bands.** Each arm's ROC-AUC and AUPRC, mean ± sample std over
  five seeds, every AUPRC with the official test mask's prevalence and positive count.
  - There is no difference, standard error, p-value or superiority claim between our arms.
  - Five seeds was chosen by judgement, not to power a comparison.
  - No gated claim attaches, now or later.
- **Published figures are fixed before the retune, in a file:**
  `configs/dgf1-official-published-figures.yaml`, committed before the batch's first official score
  (clause 8). An empty file is valid.
  - **Sources.** Exactly the three ADR-007 names as "published DGraph/GADBench numbers"
    (`ADR-007:120-121`):
    - DGraph-Fin's own publication;
    - its official leaderboard;
    - the GADBench publication.

    Each is taken at the version current on the file's commit date, and the file's header records
    each one's citation, version and access date. For a leaderboard entry, only the leaderboard's own
    text counts as its stated protocol.
  - **Inclusion.** Every ROC-AUC or AUPRC that those sources report for DGraph-Fin goes in, with the
    model's name verbatim, whatever it shows. Whether it used the official masks is judged by point
    1 below, not used to filter.
  - **Each entry records:**
    - the metric and value;
    - the run count, spread and model-selection rule, where the source states them;
    - its row in the table below, with the quote that places it there;
    - a verdict on each of the four points: `matches`, `differs`, `not stated` or
      `not applicable`. Each verdict quotes the publication's own words.
  - **Procedural, not enforced.** Three things are procedural: that the file is complete, that its
    verdicts are right, and which table row each model falls in. The runner checks only that the file
    exists and parses against its schema.
  - **The file is frozen by clause 5's identity.** Rows record its git blob id, and the assembler
    reads that blob, never the working copy. **No change of identity may change the figures blob**
    (clause 5).
  - **Later figures and corrections** go in a second file, `experiments/dgf1-official-figures-errata.yaml`,
    outside the identity. They are printed beside the frozen entries as *unverified* or *erratum*.
    They are never ranked, and never change a frozen entry's ranking.
- **The four points, judged against the publication's stated protocol:**
  1. **masks:** the DGraph-Fin official train/val/test masks, as distributed;
  2. **metric:** the same metric on `official_test_mask`;
  3. **graph:** training over the full graph, every edge visible, as ADR-011:402-405 defines our
     track. For a model that uses no graph, this point is `not applicable`;
  4. **labels and data:** fraud and normal labels of `official_train_mask` nodes only, no use of the
     background classes' labels, and no data beyond the distributed dataset.
- **Which of our arms a figure may be ranked against** is fixed here, by model class and input set:

  | published model | ranked against |
  |---|---|
  | a GNN using the distributed graph and node features | DGF-1 |
  | XGBoost on the distributed node features alone | the raw-17 floor |
  | anything else, or a model whose class or inputs the source leaves unclear | nothing: shown beside, unranked |

  Our parity floor is ranked against no published figure. Ranking DGF-1 against a published
  tabular model would be a structure-versus-features claim, which this clause forbids.
- **Ranking:**
  - **Only a figure whose applicable points all read `matches` is ranked** ("protocol matches as
    published"). Every other figure is shown beside ours, labelled *unverified*, with no ranking and
    no delta.
  - **A ranking is three-valued.** It says whether the published figure lies above, within, or below
    our arm's five-seed range. "Within" means the closed interval from our lowest seed to our
    highest. The publication's own spread is printed but not used.
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
- **Where: a positioning report**, assembled from the registry and the figures by
  `scripts/assemble_dgf1_positioning.py`.
  - The first report is `gates/POSITIONING-DGF1-OFFICIAL.md`. A report after a change of identity
    (clause 5) is `gates/POSITIONING-DGF1-OFFICIAL-<YYYY-MM-DD>.md`. The assembler refuses to
    overwrite any existing report.
  - It is not a `GATE-*` file and carries no verdict, the treatment ADR-013 clause 2 gave
    `gates/ERRATUM-DGF1-1.md`.
  - **What it counts:**
    - only rows at the pinned identity (clause 5), refusing if this batch has more than one
      unsuperseded identity;
    - the seed count from the rows' `positioning_seeds`, refusing if that differs from this file's
      header;
    - one result per (arm, seed), requiring every arm to have every seed;
    - refusing any key whose results disagree, and refusing while any unresolved disagreement row
      exists at the pinned identity. Disagreement rows carry no `scores_sha256`, so the assembler
      looks for them explicitly.

    Rows at a superseded identity are printed beside the counted ones, never counted.
  - It prints, rather than relying on a hand edit:
    - the label *random-split, leaderboard-comparable*, and its meaning above;
    - the five weaknesses;
    - every AUPRC's prevalence and positive count;
    - the retune table, labelled *selection runs, one seed, not results*;
    - every resolved disagreement row, beside the key it belongs to, with the ADR that resolved it;
    - the frozen figures with their verdicts and quotes, and the errata.
  - It prints no temporal-track number and no difference between our arms.
  - It refuses to write a ranking or delta that this clause does not allow.
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
- **This batch's rows** are the rows with `model == "dgf1"`, one of the two `experiment` values
  above, and one of the three `arm` values above. Every rule in clause 5 applies to these rows only,
  so a later ADR's official rows never touch this batch, and this batch never touches them.
- **Every row also carries:**
  - the knobs it ran (`lr` and `batch_size`, or `max_depth` and `subsample`);
  - `code_identity`, `env_identity` and `data_identity` (clause 5), and the **components** each is
    computed from: code paths with their blob ids, every environment field's value, and the hash of
    each data tensor. These are logged at session entry through the unhashed metrics channel
    (clause 5), so a refusal can name what differs and a drifted environment can be restored.
    **A variable whose name contains `KEY`, `TOKEN`, `SECRET` or `PASSWORD` enters the identity and
    the log by name only.** Its value is never logged or hashed. So no credential, and no unsalted
    hash of one, reaches the committed registry, and rotating a credential does not stop the batch.
    The components are logged under a single key, so on certify rows they join
    `DELIBERATELY_UNCOMPARED` as one entry. Certify rows log the same components;
  - the figures file's git blob id;
  - the resolved device (`cpu` for the floors);
  - `positioning_seeds`, the header's count at run time.

  Positioning rows also carry `retune_winner_run_id`.
- **Logging order:**
  1. **The tags are logged at session entry, not at the end**, so an ERRORED row carries them. The
     temporal trainers log theirs last (`adapters/dgf1/train_gnn.py:208-211`;
     `adapters/dgf1/baselines_tabular.py:232-234`), and `RunSession` writes a row on error
     (`gbe/run/session.py:66-71`). So a temporal row that errors carries no tags.
  2. **The new result's hash is computed in memory and compared first.**
     - The hash is `content_hash` over the result's node ids, labels and probabilities. Those are the
       same dtype casts `save_scores` applies, so the hash equals the saved file's.
     - It is checked against every recorded hash for the same key at the pinned identity, before a
       score file is written and before any metric is computed or printed.
     - A disagreeing result is discarded unsaved. Its row is marked ERRORED and records the two hashes
       as `scores_sha256_expected` and `scores_sha256_observed`, **never** as `scores_sha256`. The
       runner refuses (clause 5, Invariant 2).

     The temporal trainers compute metrics before saving (`adapters/dgf1/train_gnn.py:197` before
     `:201`; `adapters/dgf1/baselines_tabular.py:232` before `:244`), so the official stages do not
     reuse `run_dgf1` or `run_floor` as they stand.

     **Before that comparison, the environment variables are re-read** and must equal the session's
     entry read (clause 5). A mismatch is an identity refusal, not a disagreement: the row is
     ERRORED, nothing is saved or pinned, and no metric is computed.
  3. Only after that comparison is the score file saved, atomically.
     - It is written through an open file handle to a temporary name and then renamed. Writing
       through a handle avoids `np.savez` appending `.npz` to a path.
     - Pin reading ignores temporary names.
     - Its `scores_sha256` is logged next, and the metrics last.
- **Score files** go to `experiments/scores/official/adr-015/<batch_identity>/`, one file per result,
  named `<stage>-<arm>-<configuration or seed>-<run_id>.npz`.
- **The official path has its own config builders, for all three arms.** Both temporal entry points
  write `arm` and `window` after `**base_cfg` (`adapters/dgf1/train_gnn.py:166-168`,
  `adapters/dgf1/baselines_tabular.py:219-224`), so neither can be steered through config.
- **A tested predicate, `is_official(row)`**, lives in `adapters/dgf1/`. It is true when `window` is
  `official-val` or `official-test`. Every DGF-1 assembler written from now on, Gate 3's included,
  excludes official rows through it. The tag strings are pinned literally by a test.
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

### Clause 5 — One pinned identity, and every result at it reproduces

**The principle.** The held-out rule exists so that no choice can follow from seeing a test result.
A re-computation that reproduces the same number offers no such choice. So this clause does not
count runs. It fixes every condition that determines a result, refuses any change to them once a
result exists, and requires every re-computation to reproduce. "Scored once" is enforced as **one
result per arm and seed, which every re-computation must reproduce**.

- **The runner, `scripts/run_dgf1_official.py`, has three modes: `certify`, `retune` and
  `positioning`.**
  - **Its command line exposes `--mode` and nothing else.** There is no pre-registration path, no
    `--allow-dirty`, and no window, model, seed, knob or device option.
  - **It reads this file by a path fixed in its source.** It refuses unless `**Status:**` is
    `accepted`, and it takes the seed count from the `**Positioning seeds:**` line. Every mode
    requires acceptance.
  - **A dirty tree is refused.** Smoke testing uses fixtures, never the snapshot.
  - **It refuses to start if the figures file is missing or fails its schema.**
  - **The GNN never runs on CPU, and runs only on the GPU the environment pin names.** It applies
    `device_problems` (`scripts/run_dgf1_gnn.py:205`), which refuses a CPU device. It also compares
    `torch.cuda.get_device_name(0)` with the pin's `gpu` field, which `device_problems` does not do
    (`:219-228`). The floors run on CPU, as their temporal rows did.
  - **A mask check on every load, before anything trains.** The three official masks must be
    boolean, pairwise disjoint, and together exactly `labelled_mask`. It refuses otherwise, and
    prints only pass or fail: no class count and no prevalence. A failure is a fact about the
    snapshot, recorded in a new ADR. The check is never relaxed.
  - **It imports neither `scripts/run_dgf1_gnn.py` nor `scripts/preregistration.py`.** The helpers it
    needs from the former (`device_problems`, `repro_check_verdict` and what they call) move to
    `adapters/dgf1/`.
- **The batch identity** is computed afresh at every invocation and recorded on every row. It has
  three parts:
  - **`code_identity`** is a SHA-256 over the git blob ids of every repository file the runner imports
    or opens, **its own file included**. That covers `adapters/dgf1/config.yaml`, which holds the
    epochs, architecture and grid and is read by path (`adapters/dgf1/eval.py:18`); the environment
    pin `experiments/extract_reference_env.txt` (`scripts/run_dgf1_gnn.py:49`); and the figures
    file.

    **Excluded:** the registry; this ADR; `experiments/scores/`; `data/` (covered by
    `data_identity`); the records file below and the documents its entries name; and the errata file.
    Because a dirty tree is refused, HEAD equals the working tree.
  - **`env_identity`** is a SHA-256 over:
    - every installed Python distribution and version (`importlib.metadata`) and the Python version;
    - the CUDA and cuDNN versions, the GPU name, and the NVIDIA driver version as `nvidia-smi`
      reports it;
    - the CPU model, and the operating system as `platform.platform()` reports it;
    - `torch.get_num_threads()` and `torch.get_num_interop_threads()`;
    - **environment variables, by family.** These are sorted `name=value` pairs: every variable
      whose name starts with `TORCH_`, `PYTORCH_`, `ATEN_`, `CUDA_`, `CUBLAS`, `CUDNN`, `NVIDIA_`,
      `OMP_`, `KMP_`, `MKL_` or `OPENBLAS_`, plus `DISABLE_ADDMM_CUDA_LT`.
      - **Matching is strict prefix matching.** `TORCHINDUCTOR_CACHE_DIR` does not match `TORCH_`.
      - **A name containing `KEY`, `TOKEN`, `SECRET` or `PASSWORD` enters by name only** (clause 4).

    **Why families, not a list.** Three unset variables change results on this machine, measured by
    hashing a fixed synthetic computation in separate processes; the baseline reproduced exactly:
    - `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` changes the GPU matmul and `Linear` outputs;
    - `DISABLE_ADDMM_CUDA_LT=1` changes the GPU `Linear` output;
    - `ATEN_CPU_CAPABILITY=default` changes CPU arithmetic, which moves initialisation and the
      fitted transform.

    Families are still a list. A variable outside them that changes a result is not caught by the
    identity. It is caught by Invariant 2, as a disagreement.

    **Read at each session's entry, after the runner's module-level imports.** The official runner
    will open one session per configuration or seed, as the temporal runner does
    (`scripts/run_dgf1_gnn.py:320-322`), so every session reads.
    - **Every session's entry read must yield the pinned `env_identity`.** A session is compared with
      the pinned identity, not only with itself, so a variable set during an earlier session's save
      or metrics is caught at the next entry.
    - **`retune` and `positioning` re-read right after scoring**, before the in-memory comparison
      (clause 4). The two reads must be equal. A mismatch is an identity refusal: nothing is saved,
      nothing is pinned, and no metric is computed. That catches any lazy import that sets a
      variable. xgboost, for one, is imported inside the fit
      (`adapters/dgf1/baselines_tabular.py:176`).
    - **`certify` needs no post-scoring re-read.** Its verdict is a bit-for-bit comparison with a
      stored reference. A variable that changed its result fails certification by itself, and one
      that changed nothing does not matter. A re-read after the gated trainer returns would also come
      after that trainer's row is written (`gbe/run/session.py:66-71`), so its result could not be
      stored.
    - **Imports set variables inside the hashed families:**
      - `gbe.run.seeding` sets `CUBLAS_WORKSPACE_CONFIG` with `setdefault`, keeping any value set
        outside the process (`gbe/run/seeding.py:32`);
      - importing `gbe.eval` loads scikit-learn, which sets `KMP_DUPLICATE_LIB_OK` and
        `KMP_INIT_AT_FORK` (`sklearn/__init__.py:56,60`).

    So the read point decides the identity. Reproduced in a fresh process, the families read as
    follows:
    - empty at start;
    - `CUBLAS_WORKSPACE_CONFIG` only, after seeding;
    - those two variables added, after `gbe.eval`.

    After `gbe.eval`, fresh-process probes found nothing further that sets a family variable:
    - no import on the planned runner's path: `gbe.run`, `gbe.gnn`, `gbe.features`, the adapters'
      dependencies, `torch_geometric.datasets` or xgboost;
    - no CUDA initialisation;
    - no GPU matmul or `SAGEConv`;
    - no xgboost fit.

    Every mode reads at the same point, and clause 7 test 5 checks it in fresh processes.

    `PYTHONHASHSEED` belongs to no family. `seed_everything` rewrites it to each run's seed
    (`gbe/run/seeding.py:79`).

    This replaces a hand comparison with the pin: `environment_drift`
    (`scripts/check_extract_regression.py:155`) compares only what the pin lists, and the pin lists
    neither xgboost nor the driver.
  - **`data_identity`** hashes the **loaded tensors**, not the files: `x`, `edge_index`,
    `edge_time`, `edge_type`, `y`, `node_time` and the three official masks. Each tensor is hashed in
    `content_hash`'s style, together with its dtype and shape.
    - `node_time` is derived from the edges, so hashing it is redundant but harmless.
    - `reverse_edges` is not data. `edge_index` stays forward-only as shipped
      (`adapters/dgf1/datasource_dgraph.py:83-84`), and reverse edges are added per view
      (`:246-248`). The flag that governs this is `load_dgraph`'s default (`:128`), because the
      runners pass no value (`scripts/run_dgf1_gnn.py:311`). It is code, which `code_identity`
      covers.

    The repository already holds this principle for score files:

    > **The hash is over the *content*, not the file.** `.npz` is a zip, whose bytes carry
    > timestamps, so a file hash would differ on every rewrite of identical data.
    > — `gbe/eval/scores.py:13-14`

    A file hash would also move with stray files in `data/dgraph/`, or with PyG's processed cache
    (`data/dgraph/processed/`) if a different PyG version re-pickled it. The loader itself checks
    only counts (`adapters/dgf1/datasource_dgraph.py:149-159`).
  - **Updates are paused for the batch, and nothing is installed for it** (clause 8). Windows Update
    and driver updates are paused before `certify` and stay paused until positioning is committed.
    Installing a driver just before certifying would risk certification against an old driver whose
    version is recorded nowhere: the pin does not hold it (`ADR-014:71-72`).
    - **The pause has a ceiling.** Windows 11 Pro can pause updates for at most five weeks.
      - **A lapse before the batch's first official score pins nothing.** Install the pending
        updates, pause again, and restart at clause 8's step 5 (`certify`) at the new identity.
        Nothing had been scored, so nothing can be chosen.
        - If an installed driver then makes certification impossible, the batch is postponed, as
          in clause 2.
        - The restart at `certify` is procedural. If the identity did not change, the old certify
          rows still satisfy the precondition.
      - **A lapse after the first official score stops the batch**, as for an unrestorable
        environment, and the lapse is recorded in `notebooks/lab/`. The batch does not resume
        after updates are installed: Windows installs pending updates before it allows another
        pause, and a cumulative update may then have changed what the identity cannot see. **The
        stop withholds nothing.** Anything further is a new ADR's question.
    - **What the identity detects:** a feature update, which changes `platform.platform()`; a
      driver change; any package change; and any change to a variable in a hashed family.
    - **What it does not detect:** a monthly cumulative update. On this machine `platform.platform()`
      reads `Windows-11-10.0.26100-SP0`, with no build revision, so the pause alone guards against
      it. The build revision is deliberately not added; it would make every patch a stop.

  On a refusal, the runner prints which code paths, environment fields or data tensors differ. It
  compares the current components with the components the pinned rows logged (clause 4).
- **Invariant 1 — one pinned identity.**
  - **The batch's pinned identity is its one unsuperseded identity.** It is read from two places:
    - this batch's score directories under `experiments/scores/official/adr-015/`;
    - this batch's registry rows that carry a `scores_sha256`.
  - **Any second unsuperseded identity, from either place, refuses everything**, in the runner and in
    the assembler.
  - **"The batch's first official score"**, as used throughout, is the first result that pins: the
    retune's first validation score. A restart at a new identity after it is refused here. It would
    be a second validation draw.
  - **A score directory alone cannot be lost silently.** The score files are gitignored as
    "Regenerable" (`.gitignore:15-17`), so deleting them breaks no rule. The committed rows still
    carry the pin.
  - **A run that crashes before its score file is written pins nothing**, because nothing was
    scored. A score directory pins only if it holds a readable score file. Files are written
    atomically (clause 4), so a crash cannot leave a truncated file that pins.
  - **Every `retune` and `positioning` invocation must run at the pinned identity**, or the runner
    refuses.
  - **The only change of identity** is clause 5's defect route below.
- **Invariant 2 — every result at the pinned identity reproduces.**
  - **Re-runs at the pinned identity are allowed**, after a crash, an interrupt or a registry revert.
  - For each (stage, arm, configuration or seed), every score file and every row's `scores_sha256` at
    the pinned identity must agree.
  - The comparison happens before the new result is saved or any metric computed (clause 4), so a
    disagreeing result is never seen.
  - A disagreement refuses, and it is investigated to root cause as a determinism failure (ADR-005).
  - **The disagreement row** carries `scores_sha256_expected` and `scores_sha256_observed`, never
    `scores_sha256`, so it neither pins nor counts as a result.
  - **While any unresolved disagreement row exists at the pinned identity, every invocation refuses.**
    That includes the runner in every mode and the assembler.
  - **Only an accepted ADR resolves it.**
    - The records entry names the row's run id. The runner checks that the named ADR exists, is
      accepted, and contains that run id, as supersession must name its identity.
    - **A lab entry never resolves a disagreement.**
  - **A resolution leaves the pinned result standing.** The disagreeing result was discarded unseen,
    so continuing with the pinned one offers nothing to choose.
    - A later re-run of the key is compared like any other. It can add a new disagreement row, but
      it can never displace the pinned result.
    - A resolution is never a reason to re-run until a run agrees (ADR-008:108-109).
    - **A resolution never displaces the pinned result, whatever the root cause.** Only the defect
      route, on its own bar, changes the identity. Treating the pinned run itself as the defect
      would be a second draw, because its numbers have been seen.
  - **The resolving ADR states whether its root cause bears on the pinned result.** That statement is
    procedural: the runner checks only that the ADR exists, is accepted and contains the run id. The
    report prints that ADR beside the key (clause 3), and every citation of that number cites it.
  - **When a key will not reproduce, the batch stops.** The condition is mechanical, worked out from
    existing rows: **a key that has a disagreement row recorded after a resolved one has stopped the
    batch.**
    - The runner refuses in every mode, and the assembler refuses.
    - No records entry clears the second disagreement. An entry that names its run id resolves
      nothing.
    - That covers the case where the pinned result is the one that is wrong, and it applies whether
      or not the stage is complete.
    - No per-key count is kept beyond what the rows already show, and no seed is dropped: either
      would be a selection.
    - Anything further is a new ADR's question.
  - With every factor that determines a result inside the identity, a re-run that reproduces adds
    nothing to choose from.
  - Rows at any other identity are never counted.
- **A batch stops in exactly four ways, and every stop is final:**
  - a key that will not reproduce (above);
  - a lapsed update pause after the first official score;
  - an environment that cannot be restored;
  - **five weeks after the batch's first official score, if no positioning report has been
    dated.** This follows from the row timestamps alone. It covers a stage never resumed, a first
    disagreement never resolved, and a run that proves infeasible. The researcher may declare
    infeasibility earlier, in a dated amendment recording the wall-clock times observed. Stopping
    earlier changes nothing below.

  A stop is judged on rows at the pinned identity. Only the defect route, on its own bar, leads past
  it.
- **A stop withholds nothing.** Every result scored at the pinned identity is reported, whether it
  is found in the rows or in the score files, the two places the pin is read.
  - **The stop is recorded here as a dated amendment**, as clause 8's step 9 records the winners. It
    lists:
    - every scored result, with clause 3's five weaknesses and every AUPRC's prevalence and positive
      count;
    - every resolved disagreement, with its ADR;
    - the stop's cause.

    A lab entry, within CLAUDE.md's ten lines, points to the amendment.
  - **Any write-up on DGF-1's official positioning cites that amendment**, as any write-up drawing on
    GATE-DGF1-1 reads its erratum (*Docs affected*).
  - Any later report on these arms or this split prints these results beside its own.

  The assembler is not changed for this. Stopping is never a way to leave a seen number unreported,
  which is this clause's principle.
- **Nothing scored waits to be committed.** The registry is committed after every invocation that
  scored anything, not only after each stage (clause 8). That day's lab entry names the invocation's
  run ids. A batch that halts without stopping still leaves every seen result on record.
- **No metric is printed before its row is written.** The runner prints a session's metrics only
  after the session's row is written. A hard kill therefore cannot leave a seen number whose only
  trace is a gitignored score file.
- **Certification first.** `certify` re-runs the two temporal references through the gated trainers:
  - the GNN pilot's seed 0, `dgf1-20260913T232506Z-1acd5067`;
  - the floor pilot's seed 0, `dgf1-20260913T232205Z-c2af9760` (validation, `max_depth=8`,
    `subsample=0.8`, commit `2c8f484`, carrying `scores_sha256`).

  **What a certify row records.** Each certify run writes an ordinary `repro_check` row on the
  validation window.
  - **The three identities are logged at session entry, through a metrics channel that is not
    hashed.** The gated trainers gain an optional keyword, default empty, passed to `log_metrics` and
    never to `run_config_values`. ADR-014 verified that channel leaves the config hash unchanged:

    > **Environment provenance can reach a row without touching the config hash**, via a non-hashed
    > keyword on `run_dgf1` passed to `log_metrics` and never to `run_config_values`.
    > — `decisions/ADR-014-dgf1-repro-check-environment.md:60-61`
  - With the keyword empty, gated rows are unchanged.
  - The identity keys join `DELIBERATELY_UNCOMPARED` (`scripts/run_dgf1_gnn.py:77`), as the coverage
    test requires (`tests/test_dgf1_runner.py:266-284`).

  **How a certify row is judged.** A fresh certify row's verdict is the full `repro_check_verdict`:
  the stored comparison plus the score-file re-hash, with a floor key list for the floor.
  - **Before running, `retune` and `positioning` refuse** unless both references have a passing
    certify row at the current identity.
  - **They also refuse if any `repro_check` row of either reference fails.**
    - Rows other than the fresh certify rows are judged on the stored comparison alone, which already
      compares `scores_sha256` (`REPRO_METRIC_KEYS`). Their gitignored score files are not re-hashed,
      so a clone or a cleanup cannot turn a past pass into a failure.
    - ERRORED repro rows are found by their config hash, rebuilt with `experiment` set to
      `repro_check`. The gated helper rebuilds the pilot's hash instead (`scripts/run_dgf1_gnn.py:241`),
      so it cannot be reused for this lookup.
    - The hash finds an ERRORED row only if it ran at the current reference config. One at an older
      config is not found, and this rule does not claim to catch it.
  - **A failure stays blocking until the records file resolves it.**
    - A dated lab entry may resolve only a `repro_check` row that errored before its comparison ran.
      It never resolves a mismatch, and never a disagreement row (Invariant 2). A mismatch needs an
      accepted ADR that names the row's run id.
    - The runner checks only that the named document exists, and for an ADR that it is accepted and
      contains the row's run id. The quality of the root cause is procedural.
  - **This replaces ADR-013's "No runner checks for it"** (`ADR-013:282-283`) for official runs. A root
    cause in the environment bears on every DGF-1 row from that code path, not only on this batch.
- **The records file**, `experiments/dgf1-official-records.yaml`, holds three kinds of entry:
  resolved repro failures, resolved disagreements, and superseded identities. It is outside the
  identity, so a record never moves the identity.
  - That it is append-only is procedural. Deleting an entry only makes the runner refuse more.
  - **No entry may cite this ADR.** ADR-015 is accepted, and after clause 8's step 9 it contains run
    ids, so citing it would satisfy every check.
    - An entry whose ADR path resolves to this file **resolves nothing**. It is ignored, not refused,
      so a bad append-only entry never blocks for ever. It is **not ignored silently**: the runner and
      the assembler print every ignored entry, with its reason, on every invocation.
    - Paths are compared by resolved path, as in clause 6, because Windows paths are
      case-insensitive.
    - This applies to all three kinds of entry.
- **Positioning requires a complete retune**: an agreed result for every (arm, configuration) at the
  pinned identity. "Agreed" means the key has a pinned result and no unresolved disagreement row.
  **The seed count is frozen**: the runner refuses if any of this batch's rows records a
  `positioning_seeds` different from this file's header.
- **The defect route — the only change of identity.**
  - **What counts as a defect.** It is behaviour ruled out by a clause **accepted**, or a docstring or
    test **committed**, before the batch's first official score. It must be shown by a test that uses
    fixtures only and fails at the pinned identity. ADR-013's defect met this bar: it broke accepted
    ADR-011 clause 3 (`ADR-013:229-230`), and it was "found by reading code" with "No number …
    computed to find it or to size it" (`ADR-013:119-120`). **An improvement the accepted text does
    not require is not a defect.** It is a new arm, and these three arms are never re-scored.
  - **An environment that cannot be restored is not a defect.** The batch cannot run at the pinned
    identity, so it stops. Anything further is a question for a new ADR, decided then and not
    pre-authorised here. The logged components (clause 4) say exactly what to restore.
  - **What the ADR must contain.**
    - It locates the defect in ADR-013 clause 1's form: where, what, since when, why undetected
      (`ADR-013:225-236`).
    - It discloses how the defect was found and which official numbers its author had seen.
    - Once any positioning score exists, it states the defect's expected effect on each arm and its
      direction, in ADR-013 clause 4's form ("state what, where and why, then the impact on the
      test's severity", `ADR-013:305-306`).
    - It is **accepted** before anything runs at the new identity.
  - **How the runner applies it.**
    - The records file names the superseded identity beside the ADR's path.
    - The runner checks only that the ADR exists, carries `**Status:** accepted`, and names that
      identity.
    - **The new identity must record the same figures blob as the one it supersedes**; corrections to
      figures go to the errata file only.
    - Whether the defect meets the bar is procedural, and judged when the ADR is accepted.
  - **What re-runs.** Every stage with a result at the superseded identity re-runs **in full, for all
    three arms**, at the new identity. There is never a partial re-run.
  - **What stays.** The original rows and files stay, printed beside the new ones in a new dated
    report.

  This follows ADR-008's rule: "**A mismatch is investigated to root cause — never re-run until it
  matches** (that is the optional-stopping error in another costume)." (`ADR-008:108-109`).
- **What the code cannot close.**
  - Only one act defeats Invariant 1: removing this batch's rows from the registry together with
    its score files.
  - A registry revert can also erase an **uncommitted** disagreement row.

  The registry is append-only (CLAUDE.md), and the researcher commits it after each stage (clause
  8). Once committed, removing rows means rewriting committed history, a deliberate act.
- **Other models later.** A future ADR may score `official_test_mask` with a model class or input
  set this batch does not run; doc-02 §5's other baselines are owed.
  - Its rows and score directories are outside this batch's scope (clause 4).
  - These three arms are never re-scored, whatever a later arm is named or however it is tuned.

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
   - The command line exposes `--mode` only.
   - The runner refuses this file unless it is accepted.
   - It parses `Positioning seeds` with the stated regex, refuses a missing or malformed line, and
     runs seeds 1 to that count.
   - It refuses a missing figures file, and one that fails its schema.

   Mutations: adding a path argument; dropping the status check; starting seeds at 0.
5. **Identity.**
   - Static, transitive import analysis (AST or `modulefinder`, since the runners import inside
     functions) finds every repo module the runner imports.
   - A `sys.addaudithook` "open" hook records every file opened while `main()` is driven on fixtures
     in every mode.
   - After stdlib, site-packages and `__pycache__` paths are filtered out, every repo path found must
     be in `code_identity`, apart from the stated exclusions. The runner's own file must be in it.
   - The runner imports neither `scripts/run_dgf1_gnn.py` nor `scripts/preregistration.py`.
   - Changing a listed file, any environment field, or a variable in any listed family changes the
     identity. So does changing any loaded data tensor.
   - Re-saving the same tensors, or adding a stray file to the data directory, does not.
   - Neither does editing the records file or the errata file.
   - `PYTHONHASHSEED` is not in the identity.
   - The identity is the same whether `CUBLAS_WORKSPACE_CONFIG` was unset or set to the seeding
     default before import.
   - Every component is logged at session entry, certify rows included.
   - A family variable whose name contains `KEY`, `TOKEN`, `SECRET` or `PASSWORD` enters by name
     only. Changing its value changes neither the identity nor the log.
   - **The read point is fixed.**
     - `main()` runs in each of the three modes in a **fresh subprocess**, whose wrapper imports only
       the standard library before the runner. pytest already has scikit-learn loaded and would hide
       the problem.
     - All three modes must yield the same `env_identity`.
     - It must be the same whether `KMP_DUPLICATE_LIB_OK` and `KMP_INIT_AT_FORK` were unset or preset
       to scikit-learn's defaults.
     - **The post-scoring re-read:**
       - the fixture runs at a key that already has a pinned result;
       - it sets a family variable mid-run, and the variable changes the result;
       - it must be refused before anything is saved or pinned and before any metric is computed;
       - its row is ERRORED, with no disagreement keys. A re-read placed after the comparison would
         instead record a disagreement, so this fixture tells the two orders apart.
     - **Each session's entry read yields the pinned `env_identity`:** a variable set during one
       session's save is refused at the next session's entry.
     - **`certify` does no post-scoring re-read.** A fixture variable that changes a certify result
       fails the bit-for-bit verdict.

   Mutations:
   - dropping `adapters/dgf1/config.yaml`;
   - omitting the runner's own file;
   - hashing the four named variables instead of the families, which setting
     `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` must catch;
   - hashing data files rather than tensors;
   - reading the variables before `gbe.eval` is imported, which the preset-`KMP_` check must catch;
   - deleting the post-scoring re-read, or moving it after the comparison or the save;
   - comparing a session's entry read only with its own re-read, not with the pinned identity;
   - logging or hashing a `*_KEY` variable's value.
6. **The two invariants.**
   - **Pinning:**
     - the pin is read from score directories and from rows that carry a hash;
     - deleting the score directory does not re-pin, because the rows still pin;
     - a crash before the score file is written pins nothing.
   - **A second unsuperseded identity** is refused by the runner and by the assembler.
   - **A run at a different identity** is refused, naming what differs.
   - **A re-run at the pinned identity:**
     - one that disagrees is discarded before saving: no file is written, and no function of the
       probabilities other than `content_hash` is computed or printed;
     - its row is ERRORED, and carries `scores_sha256_expected` and `scores_sha256_observed` but no
       `scores_sha256`;
     - every later invocation refuses until a records entry resolves the row. The entry must name
       the row's run id, and the ADR it cites must be accepted and contain that run id. A lab entry
       or an ADR that does not name the run id is refused;
     - after resolution, the pinned result stands. A later re-run of the key is compared like any
       other: it can add a new disagreement row, never displace the pinned result;
     - an entry of any of the three kinds whose ADR path resolves to ADR-015, in any letter case,
       resolves nothing;
     - a key with a disagreement row recorded after a resolved one stops the batch. Every mode and
       the assembler refuse, and a records entry naming the second row resolves nothing;
     - ignored records entries are printed with their reason, by the runner and by the assembler;
     - **Invariant 1's boundary:**
       - before any pinned result, a new identity is accepted;
       - after one, a new identity is refused.

       The lapse rule itself is procedural, because a cumulative update can leave the identity
       unchanged;
     - the mechanical stop counts rows at the pinned identity. Only a supersession through the
       defect route leads past it;
     - five weeks after the first official score with no dated report, the runner refuses as
       stopped. The test uses fixture timestamps;
     - no metric is printed before the session's row is written;
     - one that agrees is accepted.
   - **Atomic writes:** a crash mid-write leaves no readable file and pins nothing.
   - **Rows outside this batch** do not count. A later model's rows and directories change nothing.
   - **Refused:** a changed `positioning_seeds`, and positioning on an incomplete retune.

   Mutations:
   - accepting a second unsuperseded identity;
   - saving before comparing;
   - reading the pin from score directories only;
   - recording the disagreeing hash as `scores_sha256`;
   - letting an agreeing re-run clear a disagreement;
   - accepting any accepted ADR, or a lab entry, as a disagreement's resolution;
   - a temporary score file pinning;
   - accepting a resolution of a key's second disagreement;
   - honouring an entry whose ADR path resolves to ADR-015;
   - ignoring an entry silently.
7. **Certification.**
   - **The certify row:**
     - its config hash equals the reference's rebuilt with `experiment=repro_check`, so the identity
       keys are not hashed;
     - those keys are in `DELIBERATELY_UNCOMPARED`;
     - an ERRORED certify row carries them;
     - with the new keyword empty, a gated row is unchanged.
   - **Refusals:**
     - `retune` and `positioning` are refused without a passing certify row for each reference at the
       current identity;
     - a failing `repro_check` row of either reference blocks until the records file resolves it.
   - **Judging rows:**
     - an older row is judged on its stored comparison, so a missing score file for a past pass does
       not block;
     - a fresh certify row's score file is re-hashed;
     - an ERRORED row at the current config is found by the hash rebuilt with `repro_check`;
     - a lab entry resolves only a `repro_check` row that errored before its comparison ran. It
       never resolves a mismatch or a disagreement row.

   Mutations: rebuilding the lookup hash with the pilot's tag; re-hashing old rows' score files;
   logging the identities through `base_cfg`; letting a lab entry resolve a mismatch; accepting a
   mismatch ADR that does not contain the row's run id.
8. **The defect route.**
   - A records entry that supersedes an identity is honoured only if its named ADR exists, is
     accepted, and names that identity.
   - A new identity with a different figures blob is refused.
   - After a supersession, the assembler counts the new identity only once every superseded stage has
     re-run in full.
9. **Row identity and refusals.**
   - Tags are present on an ERRORED row (a fixture that raises mid-training).
   - Score files land under `adr-015/<identity>/`.
   - The tag strings are pinned literally, and `is_official` is correct on official and temporal
     rows.
   - Rows record the device (`cpu` for floors). Only rows recording `deterministic: true` are counted.
   - A dirty tree, a CPU device for the GNN, and a GPU whose name differs from the pin's are each
     refused.
10. **The mask check** refuses fixture masks that overlap, miss a labelled node, or include an
    unlabelled one. Its output carries no class count.
11. **Selection.**
    - It picks the highest validation AUPRC under the stated tie-break.
    - Non-finite values rank last, and an arm with none finite is refused.
    - Positioning rows carry the winner's run id and knobs.
    - Scored ids align to `official_val_mask` in the retune and to `official_test_mask` in positioning.
12. **The assembler.**
    - It reads the figures from the blob id the rows record, never the working copy.
    - It counts as clause 3 states:
      - the pinned identity only;
      - `n` from the rows, checked against the header;
      - one result per (arm, seed), with every seed present;
      - refusing a disagreeing key;
      - refusing while an unresolved disagreement row exists.
    - It ranks only per clause 3's table and four points, three-valued over the closed range.
    - It prints errata and superseded rows beside the counted ones, never ranked or counted.
    - It prints every resolved disagreement row beside its key, with the resolving ADR.
    - It prints no temporal-track number and no difference between our arms.
    - It refuses to overwrite any existing report.
    - It prints the label, the five weaknesses, the prevalences and the retune table.

    Mutations: reading the working-tree figures file; dropping a weakness line; taking `n` from the
    header; reading only rows that carry `scores_sha256`, which would miss disagreement rows;
    omitting resolved disagreement rows, or their resolving ADR, from the report.
13. **The shared guard refuses this ADR**, accepted or not: the header claim.

### Clause 8 — Execution order

1. Accept this ADR.
2. Implement clauses 1–5 and 7, one component per session, each with its tests. The batch does not
   start until every clause-7 test passes and the suite is green.
3. Compile and commit the figures file (clause 3).
4. Pause Windows Update and NVIDIA driver updates, installing nothing, until step 7 is committed
   (clause 5).
5. Run `certify` at the identity after steps 3 and 4. Both references must pass.
6. Run `retune` on `official_val_mask`. Commit the registry after every invocation that scored
   anything.
7. Run `positioning` on `official_test_mask` (5 seeds × 3 arms). Commit the registry after every
   invocation that scored anything.
8. Assemble the positioning report. The researcher reviews and dates it.
9. Record the winners and run ids here as a dated amendment, and in `notebooks/lab/`.

The runner enforces steps 3 and 5–7: the figures file, certification, the pinned identity and
reproduction. Step 4 is procedural:
- the runner detects a feature update, a driver change or a package change slipping through, as a
  change of `env_identity`;
- it does not detect a monthly cumulative update (clause 5).

Also procedural:
- steps 8 and 9;
- the registry commits;
- the figures file's completeness;
- the restart at `certify` after a lapse before the first official score;
- an early declaration of infeasibility;
- the dated amendment that records a stop (clause 5).

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
    - `official_test_mask` yields one result per arm and seed, at one pinned identity, which every
      re-computation must reproduce (clause 5);
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
- **ADR-013** gains a dated pointer at `:282-283`: for official runs, the official runner does check
  the re-certification (clause 5).
- **`docs/research-plan-UNIFIED-GBE-GDE.md:110`** ("Positioning vs prior work with a named delta")
  gains a pointer to clause 3, where a named delta is allowed.
- **`docs/timeline.md:24` and `:184-185`** say "both arms". This batch runs three. The wording is the
  researcher's.
- **`gates/GATE-TEMPLATE.md:30-31`** gains positioning reports (clause 3) as files a paper may draw
  on. If the batch stopped, a paper draws instead on the dated stop amendment in this ADR (clause 5).
  Any write-up on DGF-1's official positioning cites whichever exists, as any write-up drawing on
  GATE-DGF1-1 reads `gates/ERRATUM-DGF1-1.md`.
- **`gates/GATE-DGF1-0.md` is not edited.** It is signed and dated.

## Alternatives rejected

- **Count runs with a state machine** (drafts 5–7): resume, `adopt`, restarts, override constants.
  Each piece added to close a hole opened another. Invariant 2 makes a reproducing re-run harmless,
  so runs need not be counted.
- **Forbid every re-run.** A crash would then need an ADR, which pushes towards reverting the
  registry. Re-computation at the pinned identity offers nothing to choose.
- **Let an unrestorable environment take the defect route** (draft 8). It can never certify, and it
  would be a second draw the author adjudicates.
- **Require results a defect ADR declares unaffected to reproduce.** It would add machinery back for
  a case the full re-run already covers.
- **Name the environment variables individually** (draft 9). Three unnamed ones were measured to
  change results. Families cover more of them, and Invariant 2 catches any that fall outside.
- **A "stopped" mode for the assembler** (draft 13's gap). It would be machinery. A stop's record is
  the committed rows and the lab entry, and nothing seen goes unreported.
- **Restart after the first official score** (draft 13's lapse rule, inverted). A restart at a new
  identity after results exist is a second draw. Before the first score, a restart costs nothing,
  and it is allowed.
- **Re-read the variables at session exit** (draft 12). By then the result is saved and pinned, so a
  mismatch would either change nothing or invalidate a result already seen. The re-read now comes
  before the comparison.
- **Count a key's disagreements, or drop a seed that will not reproduce** (draft 12). A count is
  machinery, and dropping a seed is a selection. A key that will not reproduce stops the batch.
- **Read the variables after one named import** (draft 10). Later imports set variables inside the
  families, so the read point would decide the identity.
- **Treat the pinned result's own run as the defect** (draft 11). Its numbers have been seen, so
  re-running everything on that ground is a second draw. No fixture test can show what a past run's
  environment was.
- **Never re-run a resolved key** (draft 11). Every invocation re-runs its stage, so honouring it
  needs a skip list, which is machinery. A re-run is compared like any other and cannot displace
  the pinned result.
- **Hash secret-like variable values** (draft 11). Rotating a credential would stop the batch, and
  an unsalted hash of a credential would sit in an open release. Names alone are enough.
- **Install updates before certifying** (draft 10). It risks certifying against a driver whose
  previous version is recorded nowhere. Pausing them is enough.
- **Add the Windows build revision to the identity.** Every monthly patch would become a stop, so
  the pause guards against those instead.
- **Hash the data directory's files** (draft 9). The repository's own principle is to hash content,
  not bytes (`gbe/eval/scores.py:13-14`). File hashes also move with stray files or a re-pickled
  cache.
- **Include `PYTHONHASHSEED`.** `seed_everything` rewrites it to each run's seed, so it would vary
  with the seed and not with the environment.
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
- **Seeds 0–4.** Seed 0 retrains the model the retune selected.
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
  - the official runner, with its three-part identity and logged components, two invariants and
    certification;
  - official stage functions that hash before saving and save before computing metrics, rather than
    reusing `run_dgf1` and `run_floor` as they stand;
  - mask-based equivalents of the `TemporalSplit`-based functions it needs (`train_dgf1`,
    `score_view`, `floor_design`, `run_floor`, `run_dgf1`, `train_seed_mask`, `window_target_mask`);
  - config builders for all three arms;
  - an optional unhashed-metrics keyword on the gated trainers, empty by default;
  - `is_official`;
  - `device_problems` and the repro verdict moved into `adapters/dgf1/`, with a floor key list;
  - the figures, errata and records files;
  - the positioning-report assembler.

  Each comes with clause 7's tests.
- **Cost:** two references re-certified, two 9-configuration grids and 15 positioning runs. The
  temporal track's measured cost is in clause 2; the official cost is not predicted. The batch runs
  in the researcher's own terminal.
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

## Erratum, 2026-10-08 (accepted 2026-10-08 (15:30) by coderback)

**Source:** research audit 2026-10-07, §2
(`notebooks/audit/2026-10-07/README.md`; finding A S5).
The text above is not edited, so citations of its lines stay valid. **No clause changes.**

**`:791-797` rests on a verification that exists in no commit.** It quotes ADR-014:60-61: a keyword
on `run_dgf1`, passed to `log_metrics` and never to `run_config_values`, leaves the config hash
unchanged. That check ran on code no commit contains.
- **`run_dgf1`'s signature never had such a keyword.** It is the same, with no such keyword, in
  every commit of `adapters/dgf1/train_gnn.py` (`6b02164`, `2bd88e0`, `ccb986f`, `cd4912d`).
- **The name appears in no code.** `git log --all -S extra_metrics -- adapters gbe scripts tests`
  finds it only as a test fixture of the evidence checker (`508ab04`).

So the channel is a design, not a verified property. CLAUDE.md's *Next* list already requires
ADR-015's implementation to establish it by test (audit A S5).
