# ADR-023 — Matched-time sensitivity pre-registration (reported, not gated)

**Status:** proposed
**Date:** proposed 2026-11-04 · draft 2 2026-11-04
**Tier:** A. It exercises access to a held-out set, as a reported-only batch, and it narrowly amends
ADR-021 (CLAUDE.md *Process*).
**Review rounds:** 1/4
**Deciders:** coderback
**Implemented by:** plan rows 6b, 6c and 7, after the one-night wait. Row 5a's backfill lands before
acceptance (ADR-021 clause 9).
**Docs affected:**
- `docs/timeline.md`'s rows 6 to 7, and CLAUDE.md's *Next* item 6;
- CLAUDE.md's post-appearance qualification, once the report is signed;
- clause 9's amendments: `decisions/ADR-021-heldout-ledger-and-look-tiers.md:140`, CLAUDE.md's "Re-reading
  a signed gate file is not a look", `docs/subagent-preamble.md`'s signed-file line, and
  `gates/GATE-TEMPLATE.md:30-31`.

No gate file changes.

**Held-out look:** dgf1-temporal-test-482-821 · scores · scripts/run_dgf1_matched.py · matched-time
**Held-out look:** dgf1-temporal-test-482-821 · scores · scripts/assemble_dgf1_matched.py · matched-time

`scripts/run_dgf1_matched.py` runs both arms and the exploratory rows.

## Disclosure — what had been seen when this was written

**Generated, not typed:** `python scripts/heldout_ledger.py disclose --set dgf1-temporal-test-482-821`,
pasted here at acceptance once row 5a's backfill exists, with the validation runs this ADR uses
(ADR-021 clauses 4 and 6, and clause 9). The post-look rule applies: every choice here uses no held-out
label or score, or only removes claims, or, since this batch is reported only, is disclosed as informed.

**Disclosed as informed:** the question exists because of Gate 1's result and its withdrawn view
figures (ADR-013 clauses 1 and 2). No number from them enters this document's design or rules.

## Context

**The question.** Does DGF-1's gain over the parity floor still appear when every graph-derived input
stops at each user's own node time? Gate 1 scored users on the graph at the window's end, so its gain
may rest on activity that happened after appearance. CLAUDE.md therefore reports that claim as "not
established", and ADR-013 clause 5 routes it here.

**The researcher's decision of 2026-10-07** (`notebooks/lab/2026-10-07.md:13`): this pre-registration
with a go/no-go rule on informativeness, never direction, a GPU view builder with leakage tests, and a
validation pilot. The test batch, called stage 2 there, runs only on a go.

**What a label-free run found (ADR-017 item 3),** on steps up to 369 only
(`notebooks/measurements/2026-11-04-matched-time/output.txt`, from `measure_matched_epoch.py` beside it).
It replaces the 2026-10-07 timings in that lab line, which timed view statistics alone and whose script
was lost.
- **Cost:** one full epoch under clause 2's rule took 41.4 s over all 2,065,417 users up to step 369,
  with 0.35 GB of GPU memory allocated at its peak. Three runs took 38.2 s to 43.6 s (commit `d7b00a6`).
  Forty such epochs are about 25 to 30 minutes per seed by arithmetic, before scoring, and a
  labelled-seed epoch is smaller.
- **How:** a node's statistics at its seed's time come from a binary search over its dated edges, so
  batches shuffled across times need no per-time view. The search equalled `view_node_features` on every
  user in the view at three cutoffs.
- **The graph is thin at appearance:** 87.42% of those users have exactly one edge at their own node
  time. In a 50,000-user sample, 51.05% have at most one other user within three hops, the model's depth,
  and the same share within two.
- **A transform fitted at each user's own time would fail:** it zeroes both recency columns, because
  they are constant there. Fitted on the view as of 369, only `edge_type_8` is zeroed, as at Gate 1.

**The rule was prototyped first (ADR-017 item 3):**
`notebooks/measurements/2026-11-04-matched-time/prototype/` has 42 tests and 30 of 30 mutants killed, in
its `mutation-output.txt`. ADR-017 item 3 puts a prototype outside the repo. The researcher chose on
2026-11-04 to keep this one here, as ADR-018 and ADR-021 kept theirs, because scratch can be lost.

## Decision

### Clause 1 — What this answers, and what it never can

- **It answers one question:** on this window, is DGF-1 resolvably above the parity floor, or is its gain
  below a pre-registered margin, when all *graph-derived* inputs stop at each user's own node time?
- **It cannot clear the raw features.** The 17 raw features are an undated snapshot (ADR-011
  *Limitations*). A user's own snapshot reaches both arms. A neighbour's reaches only the GNN
  (`decisions/ADR-011-dgf1-temporal-split-derivation.md:416-417`), so later information can still reach
  DGF-1 through a neighbour joined by an early edge.
- **Labels carry no date,** so absolute numbers are never read as detection at sign-up (ADR-013
  clause 5.2).
- **Edges dated at `t_u` count as earlier than `u`:** 12.58% of users have more than one edge at their
  own node time, by arithmetic from *Context*, so same-step edges are inside the view. The report says so.
- **The graph has little in it at this point** (*Context*), so message passing has little to pass for
  most users. A small or absent gain may reflect that, and the report says so beside its reading.
- **Reported only.** It has no pass condition and makes no deployment claim (ADR-013 clause 5.6), and it
  does not revise Gate 1's view or verdict.

### Clause 2 — The matched-time view (ADR-013 clause 5.2)

For a seed or scored user `u` with node time `t_u` (ADR-011 clause 1):
- **Edges:** every hop of `u`'s subgraph uses only edges dated `≤ t_u`, reverse edges keeping their
  forward edge's date. `adapters/dgf1/sampling.py` already applies the seed's time at every hop.
- **Inputs:** every node in `u`'s subgraph, `u` included, carries view statistics computed from those
  edges, with recency counted from `t_u` (ADR-011 clause 3). Static inputs would repeat the defect
  ADR-013 clause 1 records.
- **Forward edges only** feed the statistics (ADR-013 clause 5.7).
- **Recency at `u` itself is constant:** both of `u`'s recency features are 0 at `t_u`. That is expected,
  it is disclosed in the batch's report, and a run-time check holds it (clause 8). Neighbours' recency
  still varies.
- **The statistics table stops at the window's end:** 369 in training, 481 for the pilot's scoring, and
  821 for the test batch's. A table over all edges would make the pilot an unrecorded structure look
  (ADR-021:162).
- **The transform** is ADR-011 clause 3's, unchanged: fitted on every user with `t_node ≤ 369`, from the
  training view as of 369, and frozen for scoring, for the GNN only. A fit at each user's own time would
  zero neighbours' recency along with `u`'s (*Context*). So the shared scaling constants summarise
  activity up to 369, which is training data but later than some training seeds. The batch's report
  discloses it.
- **The floor** gets `u`'s own row of the same inputs, untransformed (ADR-011 clause 4's parity), for
  training and scoring alike. Reusing `floor_design`, which builds window-end rows, is what clause 8's
  floor test refuses.
- **Training is otherwise Gate 1's:** the same seeds' labels, the same loss with its class weights, the
  same optimiser and epochs, and mini-batches shuffled across times. Grouping batches by time would
  change what is trained, so it is not allowed without an amendment. The loop lives in
  `adapters/dgf1/`, because `train_model` takes one input tensor; `gbe/` does not change.

### Clause 3 — Arms and configuration

- **DGF-1 matched** and **parity floor matched**, each under its Gate-1 configuration (ADR-012 clauses 2
  and 3, as amended). **There is no retune:** only the view changes, and neither arm gets a selection
  advantage. The cost is disclosed: those settings were chosen on the gated view, so both arms may be
  below their best.
- **Metrics:** ROC-AUC and AUPRC (ADR-007), with AUPRC's chance level and the positive count printed.
- **Rows have a fixed identity.** The arms are `dgf1-parity-matched` and `xgboost-parity-matched`, and
  the experiment tags are `matched_pilot`, `matched_repro`, `matched_time` and `matched_exploratory`. No
  row reuses a withdrawn key (ADR-013:240-242). `pilot_arm` gains an `experiment` filter in row 6b, with a
  test and a mutant, because a reproduction row shares its arm and window with a pilot row.

### Clause 4 — Exploratory rows (ADR-013 clause 5.3)

ADR-011's two reported views run in the test batch with correctly built inputs, from models trained on
the gated view, and are labelled **exploratory**. They are never compared inferentially and never decide
anything.
- **Both arms, and the same seed count as the batch,** extended if it takes a second look.
- **Their models produce no gated test-window score.** They run through a path that scores only these two
  views (ADR-016:28-29), never `run_dgf1`.
- **Before the test window:** each view's builder runs on validation, and its inputs pass clause 8's
  comparison. Window-only cannot match the temporal sampler, so its edges are compared with
  `window_only_edges` instead.
- **Reproduction:** seed 0 of each arm's gated model reproduces ADR-012's validation pilot hash where one
  exists. Other seeds run from the same code, and a mismatch fails the batch before any test-window score.

### Clause 5 — The pilot and the go/no-go rule

**Before the pilot,** DGF-1's repro-check has passed on this trainer, and its run id is cited in row 6c's
commit (ADR-013:280-281, ADR-021:248-251).

**The pilot** runs on validation only (370–481), with **exactly seeds 0 to 4 per arm**, deterministic, on
one clean commit, through `gbe.run` (ADR-018 clause 4). `scripts/gate_stats.py`'s `pilot_arm` reads its
rows.

**Before the rule is computed, reproduction runs (ADR-013 clause 5.7).** Seed 0 of each arm runs again on
the same commit, tagged `matched_repro`, and its metrics and score-file SHA-256 are compared with the
pilot's. On a mismatch: root cause, never a retry, no row from those weights, and no rule.

**The rule is computed by a function that receives no mean of either arm and no gap,** only the higher
arm's mean validation ROC-AUC, without saying whose it is, and each arm's standard deviation. Go requires
both:

1. **Sanity.** The higher mean validation ROC-AUC is at least **0.55**. This checks the pipeline, not the
   graph: raw features sit in both arms, so the floor can clear it alone.
2. **Precision, on at least one metric.** `welch_power(s_dgf1, s_floor, 20, m, α/2) ≥ 0.80`, with:
   - `m = ½ × Δ_gated`, the margin clause 7 tests. The ½ is ADR-018 clause 4's planning factor;
   - `Δ_gated`, that metric's validation gap between ADR-012's own pilot rows, selected by their `pilot`
     tag. A script fixes it before this ADR is accepted, and it prints the value beside the standard
     deviation at which the rule starts to fail, so the rule is known not to be vacuous
     (as ADR-012:270-272). It never reads a Gate-1 test-window figure.

**Every other case has one outcome:**

| Case | Outcome |
|---|---|
| A missing or extra seed, or `pilot_arm` raises | A broken pilot: root cause, then every seed again on one commit. The rule is not computed |
| Neither arm varies on a metric | That metric fails precision, and the batch reports it as refused. `n1` is planned on the other metric |
| `Δ_gated ≤ 0` on a metric | It has no margin clause and fails precision, which is known before the pilot |
| The sanity bar fails, or neither metric is precise enough | A no-go |

**Once all the pilot's rows exist and reproduction has passed,** the rule is computed and its result
recorded that day, before any decline. It is recorded in this ADR's body as a line
`**Go/no-go:** <go or no-go> · <date> · <run ids>`. On a go, `**Stage-1 seeds:**` joins it. Both are
mechanical output, not amendments (ADR-018:255-256), and neither goes on the Status or Date line, so
neither restarts the wait.

**A no-go** ends matched-time. The dated reason is recorded here and in Gate 3's ADR. That is the decline
ADR-013:327's guard reads, and it satisfies ADR-013 clause 5.1 and resolves it for EDR-1
(`decisions/ADR-020-edr1-start-condition.md:71-74`). The post-appearance claim stays "not established",
and ADR-011's two reported rows stay unproduced.

**A decline** may come at any point before the test batch runs (ADR-020:72-73), with a dated reason in
both documents (ADR-013:324-325). If the pilot's rows exist, it prints the pilot's gaps and the rule's
result, so a reader can judge it. After Gate 3's pre-registration is accepted, it also needs a Tier-A
amendment to that ADR (ADR-022:177-179). If the budget runs out before the rows exist, it is declined
(ADR-022:68-70).

**A go** leads to the test batch. Pilot gaps may appear in the report, and they size `n1` as ADR-018
clause 4 allows, but they never enter the rule.

### Clause 6 — What is dropped

**ADR-013 clause 5.4's optional GNN-only-channel arm is dropped.** It sits near a withdrawn figure, so it
could only be disclosed as informed, and it adds a third training per seed.

### Clause 7 — The test batch

- **Two clauses per metric,** each with one floor arm shared across metrics, so they are never described
  as independent confirmations (ADR-018:85-86):
  - **Gain:** ADR-018 clause 2, one-sided, DGF-1 above floor.
  - **Margin:** ADR-018's one-sided absence claim (ADR-018:91-92): the gain is below `m`, at the look's
    `ℓ`. As ADR-018:97-99 requires this ADR to fix: `m` is clause 5's; the clause does not size the
    batch; "powered" means clause 5's power of at least 0.80; and its non-resolution triggers no second
    look.
- **Design:** ADR-018's default `n1 → 20`, with `n1` from ADR-018 clause 4's rule on the pilot's gain
  clauses. Only a gain clause can trigger the second look. The table prints before the first test-window
  run. Only ledger and header commits may separate a second look's rows from the first's (ADR-021:264).
- **Access:** a mode `matched-time` in `scripts/preregistration.py`'s map (ADR-016 clause 5), and
  `open_test_window` at the scores tier under this ADR's "Held-out look" lines, batch `matched-time`
  (ADR-021). The runner refuses unless this ADR carries a `**Go/no-go:** go` line, because the map's guard
  checks only acceptance and the seed line, and the look lines stay valid after a no-go.
- **Order:** it runs after Gate 3's pre-registration is accepted (plan row 7, ADR-022) and before Gate 3's
  batch (ADR-013 clause 5.1).
- **The file:** `gates/REPORT-DGF1-matched-time.md`, assembled with `scripts/assemble_dgf1_matched.py`. Its
  disclosure is generated from the ledger and the validation runs (ADR-021 clause 6).

**The report's reading, fixed now.** Each sentence names its metric. No sentence claims that at least one
metric holds, which would need α/2 (ADR-018:81-84). A metric that failed clause 5's precision prints
"under-resolved" and its power beside its row.

| Result on a metric | What may be written |
|---|---|
| Gain resolvable, margin not | DGF-1 is resolvably above the parity floor on this window when graph-derived inputs stop at each user's node time. Its size is printed beside Gate 1's and not tested against it. Not attributable to message passing: model class differs, and neighbours' raw snapshot features are a GNN-only route to later information |
| Margin resolvable, gain not | Under matched time the gain is below the pre-registered margin, half the gated view's validation gap. Not "absent". The graph is thin at appearance and the settings were not retuned, so this does not show that later activity produced the gated gain |
| Both | Both sentences |
| Neither | Not resolvable at planned power `p`, which supports no claim either way (ADR-018:192) |
| Refused | "refused: <reason>" (ADR-018:120) |

CLAUDE.md's qualification changes only after the researcher signs this report, and it carries this
table's wording.

### Clause 8 — Tests, with the implementation (row 6b)

The prototype's rule tests and mutants move onto the real module. **Every test below reads the property,
not a symptom: ADR-013:232-233 records what checking that scores differ cost.**

- **The rule never sees the gap:** its input holds no mean of either arm. Swapping the arms, or moving a
  mean while the higher one stays above the bar, leaves the decision unchanged.
- **Inputs, per arm and per path:** for DGF-1 and the floor, in training and in scoring, the pre-transform
  inputs captured at the model's door equal `view_node_features` on independently filtered edges `≤ t_u`.
  A fixture has later edges and later neighbour statistics, and a reverse edge whose date would be wrong.
- **Equivalence on real data (ADR-013 clause 5.5):** before any test-window run, on validation, the same
  comparison runs on real training and validation seeds. Edges are compared with an independent filter,
  because the builder's edges come from the temporal sampler, which comparing with itself would not test.
- **A run-time check, on every batch:** seed rows' two recency values are 0 before the transform, and
  `assert_temporal_batch` passes.
- **Training is Gate 1's:** a test pins the loss with its class weights, the optimiser, the epochs and the
  seeds.
- **ADR-013 clause 5.7's traps:** no row from weights that fail reproduction, statistics count forward edges
  only, and reproduction is checked on score-file hashes.
- **Rows and access:** `pilot_arm`'s `experiment` filter; the runner's refusal without a go line; and the
  window-end stop on each statistics table.
- **Each test has a mutant,** with its output cited in the implementing commit. They include: static inputs;
  window-end inputs at scoring only; window-end inputs at training only; floor inputs static on each path;
  a misdated reverse edge; and a loss with no class weights.

### Clause 9 — The signed report's status

**Once signed, `gates/REPORT-DGF1-matched-time.md` counts as a signed gate file,** so re-reading it is not a
look and a paper may draw on it. This narrowly amends three lines:
- `decisions/ADR-021-heldout-ledger-and-look-tiers.md:140` reads "re-reading a signed gate file or
  `gates/REPORT-DGF1-matched-time.md`, or a figure either prints";
- CLAUDE.md's sentence and `docs/subagent-preamble.md`'s signed-file line say the same;
- `gates/GATE-TEMPLATE.md:30-31` gains this file, as ADR-015:461-462 did for its reports.

**This restarts ADR-021's wait, and that costs nothing.** ADR-019:271's narrower reading applies to its own
amendment only, so it is not relied on here. Row 5a, which implements ADR-021, is finished before this ADR
can be accepted (ADR-022 clause 1's row 6a), and ADR-021 itself authorises no look.

## Alternatives rejected

- **Decline now.** Little has been tried, and the question stays open forever on Gate 1.
- **Skip the pilot.** It would spend a test-window look on a batch that may not be able to answer.
- **A go rule using the validation gap.** That is choosing by direction, which the researcher ruled out.
- **Keep ADR-018's "powered failure" as the only reading of a failed clause.** Its effect comes from the
  matched pilot's own gap, so when that gap is near zero it could only say "not answered", which is when a
  shrunken gain most needs saying. ADR-019:288-289 rejected the same construction.
- **Set the margin from Gate 1's test-window gain.** That would use a held-out score to set a rule.
- **Static inputs with the temporal sampler.** That repeats ADR-013's defect.
- **Fitting the transform at each user's own time.** It zeroes recency (*Context*).
- **Caching every view.** The size in `notebooks/lab/2026-10-07.md:13` rules it out on this machine, and the
  binary search makes it unnecessary.
- **Folding this into Gate 3.** ADR-013's alternatives reject tangling the two questions.

## Consequences

- **Cost:** a GPU builder, 5 pilot seeds per arm, reproduction, then up to 20 seeds per arm plus the
  exploratory rows' training. *Context* puts one seed's training at about 25 to 30 minutes by arithmetic,
  before scoring.
- **A second look at the test window,** at scores, on a window already opened: disclosed and recorded. It
  also counts as a look at `dgraph-official-test` (ADR-021:118-119).
- **Gate 3 waits on this,** by ADR-013 clause 5.1, and this batch waits for Gate 3's pre-registration.
- **Its budget is ADR-022 clause 2's:** 13 working days, from 2026-11-04.

## Revisit when

- The pilot returns a no-go, or matched-time is declined: this ADR closes, with the dated reason.

## Draft history

- **Draft 1** (2026-11-04, `45bb43b`). The researcher set these on 2026-11-04, before any review: the signal
  bar of 0.55 (clause 5), no retune (clause 3), keeping the exploratory rows at their cost (clause 4), and
  dropping ADR-013 clause 5.4's optional arm (clause 6). The label-free run then changed clause 2's transform
  from a fit at each user's own time to ADR-011 clause 3's, before any review.
- **Round 1, full review** (2026-11-04). 5 blocking, 16 should-fix, nits:
  - the batch could not report a shrunken gain;
  - clause 8 tested symptoms, and left the floor and the scoring path unpinned;
  - the reading's first row claimed more than the batch can show;
  - reproduction was named, but no run was defined;
  - the rule's trigger and refusals were undefined.
- **Draft 2** (2026-11-04). All of them are folded in. The researcher set four choices on 2026-11-04:
  - a margin clause per metric, at half the gated validation gap;
  - exactly 5 pilot seeds per arm;
  - a decline after the pilot's rows exist prints their gaps and the rule's result;
  - a signed report counts as a gate file (clause 9).

  The rule was prototyped with mutants before this draft, and the neighbour counts were re-run at three hops.
