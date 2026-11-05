# ADR-023 — Matched-time sensitivity pre-registration (reported, not gated)

**Status:** proposed
**Date:** proposed 2026-11-04 · draft 2 2026-11-04 · draft 3 2026-11-04 · draft 4 2026-11-05
**Tier:** A. It exercises access to a held-out set, as a reported-only batch, and it narrowly amends
ADR-021 (CLAUDE.md *Process*).
**Review rounds:** 3/4
**Deciders:** coderback
**Implemented by:** plan rows 6b, 6c and 7, after the one-night wait. Row 5a's backfill landed first
(`a462ef2`, ADR-021 clause 9).
**Docs affected:**
- `docs/timeline.md`'s rows 6 to 7, and CLAUDE.md's *Next* item 6;
- CLAUDE.md's post-appearance qualification, once the report is signed;
- `decisions/ADR-013-dgf1-sensitivity-view-inputs.md`: a dated note on three departures, as ADR-013:442-443
  requires: clause 4's from its clause 5.5; clause 5's from its "No row from weights that fail reproduction"
  (ADR-013:355), since a pilot row exists before its reproduction can run; and clause 7's broken batch, which
  its guard (ADR-013:327) reads as a dated decline;
- `decisions/ADR-022-order-of-work-and-working-day-budgets.md:69-70` and the wording ADR-022:109 gives
  `docs/timeline.md`: a dated note on clause 5's reading of "the pilot's rows";
- clause 9's amendments: `decisions/ADR-021-heldout-ledger-and-look-tiers.md:140`, CLAUDE.md's "Re-reading
  a signed gate file is not a look", `docs/subagent-preamble.md:20-21` and `:36`, `gates/GATE-TEMPLATE.md:30-31`,
  and `docs/research-plan-UNIFIED-GBE-GDE.md:137`.

No gate file changes.

**Held-out look:** dgf1-temporal-test-482-821 · scores · scripts/run_dgf1_matched.py · matched-time
**Held-out look:** dgf1-temporal-test-482-821 · scores · scripts/assemble_dgf1_matched.py · matched-time

`scripts/run_dgf1_matched.py` runs both arms and the exploratory rows. One batch label covers both looks
of the design, deliberately. ADR-021:242-243 lets a label name a stage, but this batch is reported only, a
second look extends the first's arms, and each look's ledger row carries its own time.

## Disclosure — what had been seen when this was written

**Generated, not typed:** `python scripts/heldout_ledger.py disclose --set dgf1-temporal-test-482-821`,
pasted here at acceptance, with the validation runs this ADR uses (ADR-021 clauses 4 and 6, and clause 9).
The post-look rule applies: every choice here uses no held-out label or score, or only removes claims, or,
since this batch is reported only, is disclosed as informed.

**Disclosed as informed:**
- The question exists because of Gate 1's result and its withdrawn view figures (ADR-013 clauses 1 and 2).
  No number from them enters this document's rules.
- The margin rule was chosen with Gate 1's test-window gains known. Its inputs are validation rows and it
  never reads a test-window figure (clause 5), but ADR-021:188 defines use by whether a value could change
  the choice, so this is disclosed as informed (ADR-021:194-195).
- One comparison was made on purpose, on 2026-11-04, at the researcher's request. Clause 5's margins were set
  beside the gains GATE-DGF1-1 prints: `m` is 73.6% of ROC-AUC's +0.0251 and 57.4% of AUPRC's +0.0070,
  not half. It was used only to remove claims (ADR-021:191): clause 7's second reading, and the test rejected
  below. It set no number here.
- On 2026-11-05, while folding in round 3, a whole-file search of ADR-011 for the views' definitions printed
  test-window figures not then on the skip list. They are in the ledger (`incidental-2026-11-05` and
  `incidental-2026-11-05-structure`) and in the preamble. Clause 4's definition of window-only's inputs was
  written straight after. It follows from ADR-011:286's "recomputed from that view", and no figure could
  change it.

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
  with 0.35 GB of GPU memory allocated at its peak. Other runs took 38.2 s (`9a55249`) and 43.6 s
  (`d7b00a6`'s message). Forty such epochs are about 25 to 30 minutes per seed by arithmetic, before
  scoring, and a labelled-seed epoch is smaller.
- **How:** a node's statistics at its seed's time come from a binary search over its dated edges, so
  batches shuffled across times need no per-time view. The search equalled `view_node_features` on every
  user in the view at three cutoffs.
- **The graph is thin at appearance:** 87.42% of those users have exactly one edge at their own node
  time. In a 50,000-user sample, 51.05% have at most one other user within three hops, the model's depth,
  and the same share within two.
- **A transform fitted at each user's own time would fail:** it zeroes both recency columns, because
  they are constant there. Fitted on the view as of 369, only `edge_type_8` is zeroed, as at Gate 1.

**The rule and its guards were prototyped first (ADR-017 item 3):**
`notebooks/measurements/2026-11-04-matched-time/prototype/` has 232 tests on synthetic data, and 204 of 204
mutants killed, each by a failing test. `mutation-output.txt` holds the results, and its attribution section
says which tests catch the mutants clause 8 relies on. ADR-017 item 3 puts a prototype outside the
repo. The researcher chose on 2026-11-04 to keep this one here, as ADR-018 and ADR-021 kept theirs, because
scratch can be lost.

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
  `adapters/dgf1/matched.py`, because `train_model` takes one input tensor; `gbe/` does not change.

### Clause 3 — Arms and configuration

- **DGF-1 matched** and **parity floor matched**, each under its Gate-1 configuration (ADR-012 clauses 2
  and 3, as amended). **There is no retune:** only the view changes, and neither arm gets a selection
  advantage. The cost is disclosed: those settings were chosen on the gated view, so both arms may be
  below their best.
- **Metrics:** ROC-AUC and AUPRC (ADR-007), with AUPRC's chance level and the positive count printed.
- **Rows have a fixed identity.** The arms are `dgf1-parity-matched` and `xgboost-parity-matched`. The
  experiment tags are `matched_pilot_<k>` and `matched_repro_<k>` for attempt `k`, counting from 1, then
  `matched_batch_repro`, `matched_time`, and clause 4's two. No row reuses a withdrawn key (ADR-013:240-242).
- **`pilot_arm` gains an `experiment` filter in row 6b,** with a test and a mutant. A caller names one
  attempt, because a reproduction row shares its arm and window with a pilot row, and the registry keeps a
  failed attempt's rows beside the next one's.

### Clause 4 — Exploratory rows (ADR-013 clause 5.3)

ADR-011's two reported views run in the test batch with correctly built inputs, from models trained on
the gated view, and are labelled **exploratory**. They are never compared inferentially and never decide
anything.
- **Both arms, and the same seed count as the batch,** extended if it takes a second look.
- **Rows have a fixed identity.** The arms are `dgf1-parity-exploratory` and `xgboost-parity-exploratory`,
  so no row shares an arm with Gate 1's. Each row has a `view` key, `window_only` or `first_appearance`, and
  the tag `matched_exploratory`.
- **Their models produce no gated test-window score** (ADR-016:28-29). They never run through `run_dgf1`,
  and the path refuses a gated view on the test window (clause 8).
- **Their inputs.** First appearance's are clause 2's. Window-only's come from that view's own forward edges,
  because ADR-011:286 has them "recomputed from that view": recency runs from the window's end, age from a
  user's first edge in the view, and a user the view isolates gets zeros. The GNN's go through the frozen
  transform and the floor's do not. Each floor is trained on the gated view's rows.
- **Before the test window:** each view's builder runs on validation, and its inputs pass clause 8's
  comparison. Window-only cannot match the temporal sampler, so its edges are compared with
  `window_only_edges` instead. This departs from ADR-013 clause 5.5, whose same-edge-sets check only a
  first-appearance builder can meet, and ADR-013 gains a dated note saying so at acceptance.
- **Reproduction:** seed 0 of each arm's gated model scores validation first, under the gated view, in a row
  tagged `matched_exploratory_repro`. Its hash must match ADR-012's validation pilot row for that arm and
  seed. A mismatch drops the exploratory rows, with a dated reason in the report, and the matched batch goes
  on: these rows never decide anything, so they cannot stop it. ADR-013 clause 5.7 allows no row from those
  weights, so there is no retry. Other seeds run from the same code.

### Clause 5 — The pilot and the go/no-go rule

**Before the pilot,** DGF-1's repro-check has passed on code that includes row 5a step 3 (`c53f200`), and
its run id is cited in row 6c's commit (ADR-013:280-281, ADR-021:248-251). It covers `run_dgf1`, and not the
new loop, whose own reproduction is below. Clause 8's real-data check has also passed on the pilot's code, so
a builder that fails it cannot leave a clean attempt with no outcome.

**The pilot** runs on validation only (370–481), with **exactly seeds 0 to 4 per arm**, deterministic, on
one clean commit shared by both arms and the reproduction, through `gbe.run` (ADR-018 clause 4). One command
runs the ten rows and both reproductions, with no stop between them.

**Before the rule is computed, reproduction runs (ADR-013 clause 5.7).** Seed 0 of each arm runs again on
the same commit, tagged `matched_repro_<k>`, and its metrics and score-file SHA-256 are compared with the
pilot's, on a row that is itself clean. On a mismatch the attempt is void: root cause, no further row from
those weights, and no rule. A pilot row exists before its reproduction can run, so this is as near as the
design comes to ADR-013:355's "No row".

**A reproduction is owed once all ten rows are clean on one commit.** If the command stops after them, the
reproduction is the only thing that may run next, on that commit and no other. No decline of any kind comes
first, even past the budget's last day, which ADR-022:70-71 already allows. So a reproduction cannot be skipped
to avoid the rule, and no line can say one failed when it never ran.

**An attempt's status is read once,** from its first ten pilot rows and first two reproduction rows in
registry order, and the runner refuses any further row under its tags. A row added later changes nothing.

**A void attempt.** Its rows stay, because the registry is append-only. The rule never reads them, and the
report lists them with their root cause. The next attempt runs every seed again, on one commit, under its
own tags. Its root cause is recorded first. It is refused on the void attempt's commit unless that cause is
environmental, such as a run stopped for low memory: a mismatch is investigated, never retried (ADR-013:280,
ADR-008 clause 2).

**The first clean attempt is final.** The rule takes no attempt number and reads that attempt alone, and no
pilot starts after it. So a no-go cannot be followed by another attempt.

**A void attempt's rows are not "the pilot's rows" in ADR-022:69-70.** Only ten clean rows are, whether their
reproduction has passed or is still owed, so the budget can still run out while no attempt has reproduced.
Otherwise, after a failed reproduction, the budget could never force an end: only a voluntary decline could,
and attempts could go on past the last day for as long as the researcher chose, against ADR-022:65. No attempt
starts after the last day. One already running may finish in that run. If it stops past the last day short of
its ten rows it is void, and with all ten its reproduction is owed.

**The rule is computed by a function that receives neither arm's mean and no gap from this pilot,** only the
higher arm's mean validation ROC-AUC, without saying whose it is, each arm's standard deviation, and the fixed
`Δ_gated`. Go requires both:

1. **Sanity.** The higher mean validation ROC-AUC is at least **0.55**. This checks the shared data path and
   that at least one arm learns. It does not check DGF-1's arm: a DGF-1 arm stuck near 0.5 gets a go if the
   floor clears the bar. Clause 8's tests check that arm's code, and *Alternatives* says why the bar is
   not on each arm.
2. **Precision, on at least one metric.** `welch_power(s_dgf1, s_floor, 20, m, α/2) ≥ 0.80`, with
   `m = ½ × Δ_gated`, the margin clause 7 tests. The ½ is ADR-018 clause 4's planning factor.

**`Δ_gated` is fixed here,** as ADR-018:98 requires. It is that metric's validation gap between ADR-012's
own pilot rows, selected by their `pilot` tag: exactly seeds 0 to 4 per arm, on `2c8f484`.
`scripts/matched_margin.py` computes it and never reads a Gate-1 test-window figure. Its output is
`decisions/ADR-023-margin.txt`, beside this ADR, because a metric gap cannot go in `notebooks/measurements/`
(CLAUDE.md:268-269).

| Metric | `Δ_gated` | `m` | Gated pilot's spreads, DGF-1 and floor | Rule fails at DGF-1's spread, floor's as piloted | Or both spreads at |
|---|---|---|---|---|---|
| ROC-AUC | +0.036935 | 0.018467 | 0.007872, 0.002422 | 0.024556 | 0.018078 |
| AUPRC | +0.008036 | 0.004018 | 0.001755, 0.000318 | 0.005354 | 0.003933 |

The code uses these printed values. The rule is not vacuous, as ADR-012:270-272 asked of its own: with the
floor's spread as piloted, it fails once DGF-1's spread about trebles on both metrics, since one precise metric
is enough. This is arithmetic, and it predicts nothing about the matched arms' spreads.

**Every other case has one outcome:**

| Case | Outcome |
|---|---|
| A missing or extra seed, a missing metric, a non-finite value, more than one commit, or `pilot_arm` raises | The attempt is void, as above. The rule is not computed |
| Ten clean rows, and the reproduction has not run | It is owed, as above. Nothing else may happen first |
| The reproduction ran and failed, errored, or was not clean | The attempt is void, as above. The rule is not computed |
| The budget runs out with no clean attempt and no reproduction owed | It is declined. The decline prints each void attempt's gaps, labelled void, its root cause, and `rule not computed: reproduction failed` or `rule not computed: broken pilot` |
| Neither arm varies on a metric | That metric fails precision, and `n1` is planned on the other. At assembly `judge` decides that metric: refused if neither arm varies on the test rows, and otherwise it runs, labelled under-resolved |
| The sanity bar fails, or neither metric is precise enough | A no-go |

**Once the first clean attempt exists,** the rule is computed and its result recorded that day, before any
decline. It is recorded in this ADR's header as a line
`**Go/no-go:** <go or no-go> · <date> · <run ids>`, naming that attempt's twelve rows. On a go, `**Stage-1 seeds:**` joins it, and the table
ADR-018 clause 6 prints is committed beside this ADR (ADR-018:253). All are mechanical output, not
amendments (ADR-018:255-256), and none goes on the Status or Date line, so none restarts the wait.

**A no-go** ends matched-time. The dated reason is recorded here and in Gate 3's ADR. That is the decline
ADR-013:327's guard reads, and it satisfies ADR-013 clause 5.1 and resolves it for EDR-1
(`decisions/ADR-020-edr1-start-condition.md:71-74`). The post-appearance claim stays "not established",
and ADR-011's two reported rows stay unproduced.

**A decline** may come at any point before the test batch runs (ADR-020:72-73), once any reproduction owed
has run, with a dated reason in both documents (ADR-013:324-325). If a clean attempt's rows exist, it prints
their gaps and the rule's result, so a reader can judge it, and any void attempt prints as in the table. After
Gate 3's pre-registration is accepted, it also needs a Tier-A amendment to that ADR (ADR-022:177-179). If the
budget runs out with no clean attempt and no reproduction owed, it is declined (ADR-022:68-70, read as above).

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
    batch; and its non-resolution triggers no second look.
- **For the margin clause, "powered" means power of at least 0.80 at a true gain of 0,** which is clause 5's
  check. ADR-018's own label stays on the gain clauses. At a true gain `g`, the margin clause's planned power
  is `welch_power(s_dgf1, s_floor, 20, m − g, α/2)` on the clean attempt's spreads. The report prints it at
  `g = m/2`, beside the same power with the realised spreads at the `n` and `ℓ` of the look that decided the
  clause (as ADR-018:196-199). So a gain that merely shrank is harder to call:
  - at the precision bar, with equal spreads, a true gain of `m/2` resolves each clause 24% of the time
    (`prototype/margin-table-output.txt`, arithmetic);
  - at the gated pilot's spreads, the margin clause's power at `m/2` is 0.99 on each metric
    (`decisions/ADR-023-margin.txt`, arithmetic). Neither predicts the matched arms' spreads.
- **Design:** ADR-018's default `n1 → 20`, with `n1` from ADR-018 clause 4's rule on the pilot's gain
  clauses. Only a gain clause can trigger the second look. The table prints before the first test-window
  run. The assembler needs each row in HEAD's ledger (ADR-021:264), so a ledger commit may separate a second
  look's rows from the first's. Between the looks no commit may touch `adapters/`, `gbe/`, `scripts/` or
  `configs/`, and a test reads the paths. The ledger, this ADR's header and the lab may change.
- **The batch's own reproduction, before the first look.** Commits will separate the pilot from the batch,
  because row 7 waits for Gate 3's pre-registration. So seed 0 of each arm runs on validation at the batch's
  commit, tagged `matched_batch_repro`, and its metrics and score-file SHA-256 must match the clean attempt's,
  on a clean row. The runner refuses the test window without it. On a mismatch no test-window row exists: root
  cause, restore what moved, and check again. If it cannot be restored, the batch ends as broken, below.
- **Stage 2's result is final for both clauses on both metrics** (ADR-018:148), so a margin resolved at
  stage 1 can un-resolve.
- **Access:** a mode `matched-time` in `scripts/preregistration.py`'s map (ADR-016 clause 5), and
  `open_test_window` at the scores tier under this ADR's "Held-out look" lines, batch `matched-time`
  (ADR-021).
- **The go line.** The map's guard checks only acceptance and the seed line, and the look lines stay valid
  after a no-go. So the runner reads this ADR from HEAD and refuses unless exactly one line opens with the
  go/no-go field, reading go, a date and run ids. It uses an anchored pattern, never a substring (as
  ADR-016:150-152), because this prose mentions the field. The runner then recomputes the rule, and refuses
  unless the line names the first clean attempt's twelve rows and the rule on them says go. The check comes
  before `open_test_window`, so a refused run writes no ledger row. The exploratory rows open through the same
  check.
- **Order:** it runs after Gate 3's pre-registration is accepted (plan row 7, ADR-022) and before Gate 3's
  batch (ADR-013 clause 5.1).
- **The file:** `gates/REPORT-DGF1-matched-time.md`, assembled with `scripts/assemble_dgf1_matched.py`. Its
  disclosure is generated from the ledger and the validation runs (ADR-021 clause 6). For each metric it
  prints `m`, the matched gain and Gate 1's gain on one line, tested against nothing. It lists every attempt,
  void ones with their root cause, as a decline does.

**A broken test batch has one outcome:**

| Case | Outcome |
|---|---|
| A run errors or stops with no metric written | Root cause, then that seed again on the same commit. The report lists the errored row and its re-run, and the assembler reads only the completed row |
| A completed row | It never runs again |
| Anything else: a fix that needs a code change, an errored row holding a metric, a seed twice, a row that is not clean, or a seed or metric still missing | The batch ends. The report prints `refused: broken batch: <why>` in place of any reading. It prints no gap, and the rows stay in the registry, unread. This counts as reported for ADR-020:75. Its dated reason is recorded in both documents like a decline after a go (ADR-022:177-179), so ADR-013:327's guard reads it. The claim stays "not established". Any later batch needs its own Tier-A amendment, disclosed as informed |

**The report's reading, fixed now.** Each sentence names its metric. No sentence claims that at least one
metric holds, which would need α/2 (ADR-018:81-84). A metric that failed clause 5's precision prints
"under-resolved" and its power beside its row.

| Result on a metric | What may be written |
|---|---|
| Gain resolvable, margin not | DGF-1 is resolvably above the parity floor on this window when graph-derived inputs stop at each user's node time. Its size is printed beside Gate 1's and not tested against it. Not attributable to message passing: model class differs, and neighbours' raw snapshot features are a GNN-only route to later information |
| Margin resolvable, gain not | Under matched time the gain is below the pre-registered margin, half the gated view's validation gap. Not "absent". The margin comes from validation, which ADR-018 treats as optimistic, so this does not say the gain shrank against Gate 1's test-window gain. Its size is printed beside Gate 1's and not tested against it. The graph is thin at appearance and the settings were not retuned, so this does not show that later activity produced the gated gain |
| Both | Both sentences |
| Neither | Not resolvable at planned power `p`, which supports no claim either way (ADR-018:192). `p` is clause 5's power at `m` |
| Refused | "refused: <reason>" (ADR-018:120) |

CLAUDE.md's qualification changes only after the researcher signs this report, and it carries this
table's wording.

### Clause 8 — Tests, with the implementation (row 6b)

The prototype's rule, guard and door tests and mutants move onto the real modules:
- `adapters/dgf1/matched.py`, tested in `tests/test_dgf1_matched.py`: the view, the loop and the floor's rows;
- `scripts/matched_rule.py`, tested in `tests/test_matched_rule.py`: the rule, the attempts and the budget;
- `scripts/run_dgf1_matched.py`, tested in `tests/test_dgf1_matched_runner.py`: the go line and the paths;
- `scripts/assemble_dgf1_matched.py`, tested in `tests/test_dgf1_matched_assembly.py`: the reading;
- `scripts/gate_stats.py`'s `pilot_arm` filter, tested in `tests/test_gate_stats.py`.

**Every test below reads the property, not a symptom: ADR-013:232-233 records what checking that scores
differ cost.**

- **The rule never sees the gap:** its input holds neither arm's mean and no gap from this pilot. Swapping the
  arms, or moving a mean while the higher one stays above the bar, leaves the decision unchanged.
- **The GNN's door, in training and in scoring:** the tensor passed to `model(...)` equals the frozen 369
  transform applied to raw `x` plus `view_node_features` on independently filtered edges `≤ t_u`, row by row
  through `batch.n_id` and `batch.batch`.
- **The GNN's edges:** under `[-1]` exact neighbourhoods, `batch.edge_index` mapped through `n_id` equals the
  same filter in both directions, for every seed. So a reverse edge let in by a wrong date fails here, where
  `assert_temporal_batch` cannot see it, because it reads the sampler's own dates. With the statistics' own
  raise removed too, a reverse edge dated 3 steps early fails both edge tests (`mutation-output.txt`,
  attribution). At a fan-out, a sample cannot equal the filter, so it is held to a subset of it.
- **The floor's door, in training and in scoring:** the `X_train` and `X_score` handed to `fit_predict` equal
  raw `x` plus the same statistics, untransformed.
- **Equivalence on real data (ADR-013 clause 5.5):** before the pilot, on validation, those three comparisons
  run on a fixed generator's 4,000 training seeds and 4,000 validation seeds. Edges are compared with an
  independent filter, because the builder's edges come from the temporal sampler, which comparing with itself
  would not test. A builder wrong on 0.1% of seeds is missed about 0.03% of the time if the fault is on both
  paths (0.999^8000), and about 1.8% if it is on one (0.999^4000), by arithmetic. The check's run time is not
  measured yet; the builder's plan row measures it.
- **A run-time check, on every batch:** the rows about to go through the transform hold 0 in both of each
  seed's recency columns, read from that tensor and not recomputed, and `assert_temporal_batch` passes.
- **Training is Gate 1's:** a test pins the loss with its class weights, the optimiser, the epochs, the seeds,
  the fan-out and the batch size. It also pins that batches are shuffled across times, and that each batch's
  labels are its own seeds', because the temporal loader's graph carries none (`adapters/dgf1/sampling.py:53`).
- **ADR-013 clause 5.7's traps:** no further row from weights that fail reproduction, statistics count
  forward edges only, and reproduction is checked on score-file hashes.
- **Attempts:** `pilot_arm`'s `experiment` filter reads one attempt, past another attempt's errored, dirty or
  other-commit rows, and never a reproduction row. Both arms and the reproduction must share one commit,
  because `pilot_arm` checks only one arm at a time, and a reproduction row must itself be clean. A row added
  after an attempt's first twelve changes nothing. A void attempt never reaches the rule. A clean attempt or an
  owed reproduction locks the budget, and a void attempt does not. With a reproduction owed, nothing prints, and
  the runner refuses a reproduction on another commit. The rule takes no attempt number. No pilot starts after a
  clean attempt, before a root cause is recorded, or on a void attempt's commit unless its cause is environmental.
  A decline and the report print attempts through one function, as the table says.
- **The test batch:** a run that errored with no metric may run again, on its commit only. A completed row never
  may. Everything else in clause 7's table prints `refused: broken batch`. The batch's reproduction passes before
  `open_test_window`. A commit between the looks that touches code is refused.
- **The second look and the plan:** only a gain clause triggers a second look, and each result says which
  clause it is; `n1` is planned on the other metric where neither arm varies on one, since `plan` raises on
  such a pair; stage 2 is final for both clauses on both metrics; the commits between looks are limited; and
  "under-resolved" prints with its power.
- **Rows and access:** the go line is refused unless it is exactly one anchored line. On this ADR's own text,
  only the header may open with the field, which stays true once the result is recorded. The line must name the
  first clean attempt's rows, and the rule on them must say go. The check comes before `open_test_window`, on
  both the batch's path and the exploratory path, each driven by a test with no go line. The window-end stop
  holds on each statistics table.
- **The exploratory path:** each view and each arm has its own door test. Window-only's reads its statistics
  against an independent filter and its edges against `window_only_edges`. First appearance's reads clause 2's
  oracle, and each floor trains on the gated view's rows. The path refuses a gated view on the test window. A
  reproduction mismatch drops only the exploratory rows, and raises nothing.
- **Each test has a mutant,** with its output cited in the implementing commit. They include: static inputs;
  window-end inputs at scoring only; window-end inputs at training only; each neighbour read at its own time;
  a transform refit on the scoring view; floor inputs static on each path; a misdated reverse edge; a read of
  a void attempt; a reproduction skipped, or run on another commit; a row added after a clean attempt; a retry
  on the same commit; a go line on a later attempt; labels out of step with a batch's seeds; a window-only
  statistic counting reverse edges; and a loss with no class weights. The prototype's no-margin path is not
  moved, since `Δ_gated` is fixed and positive, and its four mutants go with it.

### Clause 9 — The signed report's status

**Once signed, `gates/REPORT-DGF1-matched-time.md` counts as a signed gate file,** so re-reading it is not a
look and a paper may draw on it. It has no verdict to date, so signed means the researcher's own committed line
`**Signed:** coderback · <date>` in its header. An assembled but unsigned report is not exempt, and it goes on
the preamble's list of test-window locations until then. Five documents change, at acceptance, taking effect
on signature, each with a dated "*Amended (ADR-023), was …*" note, as ADR-019:268-269 did:
- `decisions/ADR-021-heldout-ledger-and-look-tiers.md:140` reads "re-reading a signed gate file or a signed
  `gates/REPORT-DGF1-matched-time.md`, which counts as one throughout this ADR, or a figure either prints".
  That covers its definition of a look (`:136`) and its post-look rule (`:189-190`), and no line moves;
- CLAUDE.md's sentence says the same;
- `docs/subagent-preamble.md:20-21` and `:36` say the same;
- `gates/GATE-TEMPLATE.md:30-31` gains this file, as ADR-015:461-462 did for its reports;
- `docs/research-plan-UNIFIED-GBE-GDE.md:137` gains it too, since it holds the same sentence.

**This restarts ADR-021's wait and, for clause 5's reading, ADR-022's, and that costs nothing.** ADR-019:271's
narrower reading applies to its own amendment only, so it is not relied on here. Row 5a, which implements
ADR-021, finished before this ADR can be accepted (ADR-022 clause 1's row 6a), and ADR-021 itself authorises no
look. Clause 5's reading acts only on row 6c, which runs after this ADR's own wait.

## Alternatives rejected

- **Decline now.** Little has been tried, and the question stays open forever on Gate 1.
- **Skip the pilot.** It would spend a test-window look on a batch that may not be able to answer.
- **A go rule using the validation gap.** That is choosing by direction, which the researcher ruled out.
- **The sanity bar on each arm.** The rule could do it without seeing the gap, by receiving only the lower
  mean. But if the floor scores well and DGF-1 falls below 0.55, a no-go would end the batch, when a collapse
  on own-time inputs could be the answer to the question. That is choosing by direction. The researcher chose
  on 2026-11-04 to keep the bar on the higher arm. The cost: a bug that passes clause 8's tests would spend a
  test-window look. The reported-only wording limits the damage, and a decline that prints the gaps stays open.
- **On a reproduction failure at the budget's end, holding the lock with no limit.** Only a voluntary decline
  could then end the item, so nothing would force Gate 3 and EDR-1 to stop waiting, against ADR-022:65. A
  reproduction failure cannot be produced by disliking a gap: the runner gives no way to skip the run or move it
  to another commit, and the decline prints the void gaps. The cost: a bug that would take a day to fix could end
  matched-time. The researcher chose this on 2026-11-04 for a reason since found wrong, and kept it on
  2026-11-05 for this one.
- **A second batch after a broken one, under this ADR.** A fix made once test-window rows exist could be chosen
  with some of them in view, and nothing in the rows could show it was not. A Tier-A amendment is the honest
  route, because it discloses that. The cost: a bug found mid-batch ends the batch unless that amendment is written.
- **Testing the matched gain against Gate 1's.** The matched arms' spreads are unknown, but if they matched
  Gate 1's at 20 seeds, a drop of about 13% of the gain on ROC-AUC or 6% on AUPRC would resolve (twice the
  standard error, by arithmetic from GATE-DGF1-1's spreads). Seeds measure seed noise on one window, and this batch also
  changes the inputs with untuned settings, so a resolved drop would claim more than the design can show.
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

- **Cost:** a GPU builder, 5 pilot seeds per arm, reproduction, the batch's own reproduction, then up to 20
  seeds per arm plus the exploratory rows' training. *Context* puts one seed's training at about 25 to 30 minutes by arithmetic,
  before scoring.
- **A second look at the test window,** at scores, on a window already opened: disclosed and recorded. It
  also counts as a look at `dgraph-official-test` (ADR-021:118-119).
- **Gate 3 waits on this,** by ADR-013 clause 5.1, and this batch waits for Gate 3's pre-registration.
- **Its budget is stage 1's, from ADR-022 clause 2:** 13 working days, from 2026-11-04. The pilot and
  reproduction take about 2.5 to 3 hours of training by arithmetic, so the plan is to finish them by about
  working day 10, leaving time for a root cause. That is a plan, not a rule.

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
- **Draft 2** (2026-11-04, `839ac41`). All of them are folded in. The researcher set four choices on 2026-11-04:
  - a margin clause per metric, at half the gated validation gap;
  - exactly 5 pilot seeds per arm;
  - a decline after the pilot's rows exist prints their gaps and the rule's result;
  - a signed report counts as a gate file (clause 9).

  The rule was prototyped with mutants before this draft, and the neighbour counts were re-run at three hops.
- **Round 2, diff-only** (2026-11-04). 3 blocking, 16 should-fix, nits:
  - after a failed attempt, the pilot could never be read again;
  - a reproduction failure once all rows exist had no outcome;
  - the input test contradicted itself, and left the frozen transform and reverse edges' dates unchecked.
- **Draft 3** (2026-11-04, committed 2026-11-05 as `7b0d5ae`). All of them are folded in. The researcher set
  four choices on 2026-11-04:
  - a void attempt's rows are not the pilot's rows, so the budget can decline;
  - no test against Gate 1's gain: the sizes are printed side by side, and the comparison made is disclosed;
  - the door capture, the go line and the attempts' filter are prototyped before acceptance (done);
  - the sanity bar stays on the higher arm, and says what it does not check.

  `Δ_gated` is now fixed and positive on both metrics, so the table's row for a metric with no margin is gone.
  The prototype's headline mutant, which round 2 found was killed by an import error, now dies on a failing
  test, and the harness reports any such kill separately. Writing the prototype found that the fixture held
  no user in the later windows, which its own check caught. The real-data check's 2,000 seeds per window
  were proposed while drafting, and the researcher kept them on 2026-11-05, before this draft was committed.
- **Round 3, diff-only** (2026-11-05). 2 blocking, 17 should-fix, nits:
  - a reproduction could simply not be run, so a disliked gap could reach a decline that said it had failed, and
    a row added after a clean attempt could void it;
  - a broken test batch had no outcome, and an exploratory mismatch could stop the comparison it never decides.
- **Draft 4** (2026-11-05). All of them are folded in. The guards were prototyped first (*Context*). Also
  changed: a clean attempt is final and the go line is recomputed from its rows; no retry without a root cause;
  a reproduction at the batch's commit; the real-data check moved before the pilot; window-only's inputs defined
  and tested per arm; and the run-time recency check reads the tensor. The stated reason for draft 3's budget
  reading was wrong: a voluntary decline could always end the item, so the real difference is whether the budget
  can force an end. The researcher set three choices on 2026-11-05:
  - the budget reading stands, on the corrected reason;
  - a broken batch follows clause 7's table;
  - the real-data check rises to 4,000 seeds per window, because the "about 2%" they kept 2,000 on holds only for
    a fault on both paths, and a fault on one was missed about 13.5% of the time.

  Writing this draft, a whole-file search of ADR-011 printed test-window figures (*Disclosure*).
