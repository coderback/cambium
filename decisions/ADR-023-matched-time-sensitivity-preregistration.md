# ADR-023 — Matched-time sensitivity pre-registration (reported, not gated)

**Status:** proposed
**Date:** proposed 2026-11-04 · draft 1 2026-11-04
**Tier:** A. It exercises access to a held-out set, as a reported-only batch (CLAUDE.md *Process*).
**Review rounds:** 0/4
**Deciders:** coderback
**Implemented by:** plan rows 6b, 6c and 7, after the one-night wait. Row 5a's backfill lands before
acceptance (ADR-021 clause 9).
**Docs affected:** none beyond `docs/timeline.md`'s rows 6 to 7, and CLAUDE.md's post-appearance
qualification when the rows exist. No gate file changes.

**Held-out look:** dgf1-temporal-test-482-821 · scores · scripts/run_dgf1_matched.py · matched-time
**Held-out look:** dgf1-temporal-test-482-821 · scores · scripts/assemble_dgf1_matched.py · matched-time

## Disclosure — what had been seen when this was written

**Generated, not typed:** `python scripts/heldout_ledger.py disclose --set dgf1-temporal-test-482-821`,
pasted here at acceptance once row 5a's backfill exists (ADR-021 clauses 4 and 9). The post-look rule
applies: every choice here uses no held-out label or score, or only removes claims, or, since this
batch is reported only, is disclosed as informed.

**Disclosed as informed:** the question exists because of Gate 1's result and its withdrawn view
figures (ADR-013 clauses 1 and 2). No number from them enters this document's design or rules.

## Context

**The question.** Does DGF-1's gain over the parity floor still appear when neither arm can use activity
from after a user first appeared? Gate 1 scored users on the graph at the window's end, so its gain may
rest on activity that happened after appearance. CLAUDE.md therefore reports that claim as "not
established", and ADR-013 clause 5 routes it here.

**The researcher's decision of 2026-10-07** (`notebooks/lab/2026-10-07.md:13`): two stages. Stage 1 is this
pre-registration with a go/no-go rule on informativeness, never direction, a GPU view builder with
leakage tests, and a validation pilot. Stage 2, the test batch, runs only on a go.

**What a label-free run found (ADR-017 item 3),** on steps up to 369 only
(`notebooks/measurements/2026-11-04-matched-time/output.txt`, from `measure_matched_epoch.py` beside it).
It replaces the 2026-10-07 timings in that lab line, which timed view statistics alone and whose script
was lost.
- **Cost:** one full epoch under clause 2's rule took 38.2 s over all 2,065,417 users up to step 369,
  with 0.35 GB of GPU memory allocated at its peak. Forty such epochs are about 25 minutes per seed by
  arithmetic, before scoring, and a labelled-seed epoch is smaller.
- **How:** a node's statistics at its seed's time come from a binary search over its dated edges, so
  batches shuffled across times need no per-time view. The search equalled `view_node_features` on every
  user in the view at three cutoffs.
- **The graph is thin at appearance:** 87.42% of those users have exactly one edge at their own node
  time, and 51.05% of a 50,000-user sample have at most one other user within two hops.
- **A transform fitted at each user's own time would fail:** it zeroes both recency columns, because
  they are constant there. Fitted on the view as of 369, only `edge_type_8` is zeroed, as at Gate 1.

## Decision

### Clause 1 — What this answers, and what it never can

- **It answers one question:** is the gain resolvable when all *graph-derived* inputs stop at each user's
  own node time?
- **It cannot clear the raw features.** The 17 raw features are an undated snapshot (ADR-011
  *Limitations*), so post-appearance information may remain in both arms.
- **Labels carry no date,** so absolute numbers are never read as detection at sign-up (ADR-013
  clause 5.2).
- **The graph has little in it at this point** (*Context*), so message passing has little to pass for
  most users. A small or absent gain may reflect that. The report says so beside its reading.
- **Reported only.** It has no pass condition and makes no deployment claim (clause 5.6), and it does
  not revise Gate 1's view or verdict.

### Clause 2 — The matched-time view (ADR-013 clause 5.2)

For a seed or scored user `u` with node time `t_u` (ADR-011 clause 1):
- **Edges:** every hop of `u`'s subgraph uses only edges dated `≤ t_u`, reverse edges keeping their
  forward edge's date. `adapters/dgf1/sampling.py` already applies the seed's time at every hop.
- **Inputs:** every node in `u`'s subgraph, `u` included, carries view statistics computed from those
  edges, with recency counted from `t_u` (ADR-011 clause 3). Static inputs would repeat the defect
  ADR-013 clause 1 records.
- **Forward edges only** feed the statistics (ADR-013 clause 5.7).
- **Recency at `u` itself is constant:** both of `u`'s recency features are 0 at `t_u`. That is expected
  and is disclosed in the batch's report. Neighbours' recency still varies.
- **The transform** is ADR-011 clause 3's, unchanged: fitted on every user with `t_node ≤ 369`, from the
  training view as of 369, and frozen for scoring. A fit at each user's own time would zero neighbours'
  recency along with `u`'s (*Context*). So the shared scaling constants summarise activity up to 369,
  which is training data, and the batch's report discloses it.
- **The floor** gets `u`'s own row of the same inputs (ADR-011 clause 4's parity), for training and
  scoring alike.
- **Training is otherwise Gate 1's:** the same seeds' labels, and mini-batches shuffled across times.
  Grouping batches by time would change what is trained, so it is not allowed without an amendment.

### Clause 3 — Arms and configuration

- **DGF-1 matched** and **parity floor matched**, each under its Gate-1 configuration (ADR-012 clauses 2
  and 3, as amended). **There is no retune:** only the view changes, and neither arm gets a selection
  advantage. The cost is disclosed: those settings were chosen on the gated view, so both arms may be
  below their best.
- **Metrics:** ROC-AUC and AUPRC (ADR-007), with AUPRC's chance level printed.

### Clause 4 — Exploratory rows (ADR-013 clause 5.3)

ADR-011's two reported views run in stage 2 with correctly built inputs, from models trained on the gated
view, and are labelled **exploratory**. They are never compared inferentially and never decide anything.

### Clause 5 — Stage 1: the pilot and the go/no-go rule

**The pilot** runs on validation only (370–481), with at least 5 seeds per arm, the same count in each, deterministic, on committed code,
through `gbe.run` (ADR-018 clause 4). `scripts/gate_stats.py`'s `pilot_arm` reads its rows.

**The rule is computed by a function that never receives the gap between the arms.** Go requires both:

1. **Signal.** At least one arm's mean validation ROC-AUC is at least **0.55**. If neither arm is above
   chance, the batch would compare noise.
2. **Precision, on at least one metric.** `t_crit × SE_diff(20) ≤ ½ × Δ_gated`, with:
   - `SE_diff(20)` from the pilot's standard deviations at 20 seeds, and `t_crit` at level α/2, the
     level of a two-look design's final look (ADR-018 clause 3);
   - `Δ_gated`, that metric's validation gap between ADR-012's own pilot rows. A script fixes it from
     the registry's validation rows before this ADR is accepted, and it is printed beside the rule.

   So the batch could resolve a gain half the size of the gated one. The ½ is ADR-018 clause 4's
   planning factor, and a metric that fails this is reported as under-resolved.

**A no-go** ends matched-time. The dated reason is recorded here and in Gate 3's ADR, which satisfies
ADR-013 clause 5.1 and resolves it for EDR-1 (`decisions/ADR-020-edr1-start-condition.md:71-74`).
The post-appearance claim stays "not established".

**Once all the pilot's rows exist,** the rule is computed and its result recorded that day, before any
decline.

**A go** leads to stage 2. Pilot gaps may appear in the report, but they never enter the rule.

### Clause 6 — What is dropped

**ADR-013 clause 5.4's optional GNN-only-channel arm is dropped.** It sits near a withdrawn figure, so it
could only be disclosed as informed, and it adds a third training per seed.

### Clause 7 — Stage 2: the test batch

- **Statistics:** ADR-018 applies (clause 10), unchanged: one clause per metric, DGF-1 above floor, the
  default design `n1 → 20`, and `n1` from ADR-018 clause 4's rule on the pilot. The table prints before the first
  test-window run, and `**Stage-1 seeds:**` joins this header mechanically (ADR-018 clause 6).
- **Access:** a mode `matched-time` in `scripts/preregistration.py`'s map (ADR-016 clause 5), and
  `open_test_window` at the scores tier under this ADR's "Held-out look" lines, batch `matched-time`
  (ADR-021).
- **Order:** it runs after Gate 3's pre-registration is accepted (plan row 7, ADR-022) and before Gate 3's
  batch (ADR-013 clause 5.1).
- **The file:** `gates/REPORT-DGF1-matched-time.md`, assembled with `scripts/assemble_dgf1_matched.py`. Its
  disclosure is generated from the ledger (ADR-021 clause 6).

**The report's reading, fixed now:**

| Result on a metric | What may be written |
|---|---|
| Resolvable | The gain does not need graph activity from after appearance. Raw features may still carry it |
| Not resolvable, and a powered failure (ADR-018 clause 4) | A gain as large as the planning effect is not supported under matched time. Not "absent" |
| Not resolvable, not powered | Not answered, with its planned power |

CLAUDE.md's qualification changes only after the researcher signs this report, and it carries this
table's wording.

### Clause 8 — Tests, with the implementation (row 6b)

- **The rule never sees the gap:** swapping the arms, or moving a mean while the arm stays above the bar,
  leaves the decision unchanged.
- **Time at every hop, for edges and inputs:** a fixture with later edges and later neighbour statistics
  moves the scores if either leaks.
- **Equivalence on real data (ADR-013 clause 5.5):** before any test-window run, on validation, the
  builder's edge sets match the temporal sampler's.
- **ADR-013 clause 5.7's traps:** no row from weights that fail reproduction, `assert_temporal_batch` runs
  on each batch, statistics count forward edges only, and reproduction is checked on score-file hashes.
- **Each test has a mutant,** with its output cited in the implementing commit.

## Alternatives rejected

- **Decline now.** No attempt has been made, and the question stays open forever on Gate 1.
- **Skip the pilot.** It would spend a test-window look on a batch that may not be able to answer.
- **A go rule using the validation gap.** That is choosing by direction, which the researcher ruled out.
- **Static inputs with the temporal sampler.** That repeats ADR-013's defect.
- **Caching every view.** The size in `notebooks/lab/2026-10-07.md:13` rules it out on this machine.
- **Folding this into Gate 3.** ADR-013's alternatives reject tangling the two questions.

## Consequences

- **Cost:** a GPU builder and the pilot, then up to 20 seeds per arm plus the exploratory rows' training.
  *Context* puts one seed's training at about 25 minutes by arithmetic, before scoring.
- **A second look at the test window,** at scores, on a window already opened: disclosed and recorded.
- **Gate 3 waits on this,** by ADR-013 clause 5.1.

## Revisit when

- Stage 1 returns a no-go: this ADR closes, with the dated reason.

## Draft history

- **Draft 1** (2026-11-04). The researcher set these on 2026-11-04, before any review: the signal bar of
  0.55 (clause 5), no retune (clause 3), keeping the exploratory rows at their cost (clause 4), and
  dropping ADR-013 clause 5.4's optional arm (clause 6). The label-free run then changed clause 2's
  transform from a fit at each user's own time to ADR-011 clause 3's, before any review.
