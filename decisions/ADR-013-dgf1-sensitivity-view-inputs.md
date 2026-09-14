# ADR-013 — DGF-1's reported sensitivity views: the defect recorded, the numbers withdrawn, the question given its own pre-registration

**Status:** proposed
**Date:** 2026-09-14
**Deciders:** coderback
**Replaces the first draft of this ADR** (commits `70f61a4`, `4667dfe`), withdrawn before acceptance.
See *Draft history*.
**Inherits, does not re-open:** the split, the gated view, floor parity and clause 3's feature rule
(**ADR-011**); the arms, the stage-1 seed count and the gated/reported boundary (**ADR-012**);
determinism (**ADR-005**); the no-retry rule for a reproduction check (**ADR-008** clause 2).
**Changes no gated quantity, and schedules no test-window run.** Both signed gates stand exactly as
recorded: `gates/GATE-DGF1-0.md` and `gates/GATE-DGF1-1.md`, both PASSED 2026-09-14.
**Docs affected — to apply on acceptance.** Sites were found by grepping `docs/`, `CLAUDE.md` and
`decisions/` before drafting. `docs/` is gitignored, so ripgrep skips it. Re-run the grep after
applying.
- `docs/02-dgraph-fin-embedding-model-BUILD.md` §5: the two sensitivity rows of the reporting
  template (lines 212–213) and the ADR-011 note beneath them (lines 228–229) are marked *not produced
  in ADR-011's form — ADR-013; see its clause 5*.
- `docs/timeline.md`: the Phase-1 row "Floor under the window-only and first-appearance views"
  becomes "Withdrawn; routed to the matched-time pre-registration (ADR-013)". Header Next item 1
  changes to match.
- `CLAUDE.md`: phase-block Next item 1, and the Gate-1 qualification at lines 50–51 (clause 2 below
  gives the wording).
- `decisions/ADR-011-dgf1-temporal-split-derivation.md` gets a dated pointer to this ADR in three
  places: clause 4 (lines 285–307), *Limitations* (lines 403–408) and *Consequences* (lines 485–487).
- `decisions/ADR-012-dgf1-gate1-preregistration.md` gets the same pointer in clause 1 (lines
  113–116), clause 7 (lines 317–318) and clause 9 (lines 351–352).

## Draft history — why the first draft was withdrawn

The first draft proposed recomputing each reported view's node statistics from that view's own
edges, for both arms, and re-running both arms on the test window for 8 seeds. An adversarial pass
revised it (`4667dfe`). A second review then used three independent read-only agents, each forbidden
from test-window labels, the dataset, score files and package installs: one arguing for acceptance,
one against, and a literature review. The draft was withdrawn for five reasons.

1. **Its own clause 7 said the corrected rows could not answer their question.**
   - Under first appearance, every scored user has age 0. Only 0.148% of training seeds do.
   - Under window-only, 36.66% of scored users get all-zero statistics. No training seed does.

   So a shrinking GNN-vs-floor gap fits post-appearance dependence and worse transfer to users with
   no history equally well (*Context* below).
2. **The literature makes the ambiguity worse, not better.** The frameworks that define a valid
   design featurise every training row and every test row as of its own prediction time. The
   DGraph-Fin paper names post-appearance edge timing as a fraud signal, and GNNs are reported weak
   on new, low-degree nodes. The gap is therefore expected to shrink **under either hypothesis**,
   so the likely outcome carries no information (*Context*).
3. **The one harm the draft repaired needs no new run.** An inference in a signed verdict rests on
   invalid numbers. Withdrawing those numbers repairs that without another look at the test window.
4. **The draft had implementation defects, confirmed against the code:**
   - `RunSession` writes a row even when a run raises (`gbe/run/session.py:40,66–71`), so "no row
     if reproduction fails" was false;
   - `assert_temporal_batch` needs `batch.batch`, which the installed PyG builds only for temporal or
     disjoint sampling (`torch_geometric/sampler/neighbor_sampler.py:390–391`), so the guard would
     have stopped running under grouped scoring;
   - building window-only statistics from `window_only_edges` would double-count, because that
     filter keeps reverse edges (`adapters/dgf1/train_gnn.py:86–90`,
     `adapters/dgf1/datasource_dgraph.py:246–248`);
   - the reproduction check compared metrics rather than score hashes.
5. **A gap no draft had caught.** `run_dgf1` still writes the flawed reported-view keys on **every**
   run. Gate 3's real-graph arm would add fresh flawed numbers to the registry (clause 3 below).

**The case for accepting was not wrong. It bought less than it cost.** One thing would reverse this
decision: a reason to think the gap surviving the corrected views is plausible enough for the rows'
single informative outcome to be worth a test-window run.

## Disclosure — what had been seen when this was written

- **Both DGF-1 gates are signed.** Every gated number is known, on validation and test.
- **The GNN's reported-view numbers on the test window have been seen**, because GATE-DGF1-1 prints
  them at lines 137–139. Against gated 0.7533 ± 0.0037 ROC-AUC / 0.0388 ± 0.0005 AUPRC:
  - window-only: **0.7437 ± 0.0045 / 0.0350 ± 0.0007**;
  - first appearance: **0.7479 ± 0.0038 / 0.0382 ± 0.0005**.

  The verdict's qualification 3 cites the first-appearance figure (lines 209–213).
- **The floor has never been scored under either reported view.** No floor number exists for them.
- **The defect was found by reading code** (`adapters/dgf1/train_gnn.py:110–116`), while starting on
  the floor rows. No number was computed to find it or to size it.
- **Every measurement below is label-free.** They use node times, edges and label **presence**
  (`y ∈ {0, 1}`); no fraud label was read.
- **The literature was read by a review agent, not by the researcher.**
  - The RelBench and Relational Deep Learning definitions and code, and the DGraph-Fin paper, were
    read in full.
  - The low-degree-GNN results (Tail-GNN, KDD 2021; Zhu et al., arXiv:2310.09787) were seen only as
    abstracts.
  - Spot-check every quotation before citing it in a write-up.
- **This ADR computes no new test-window number and schedules no test-window run.**

## Context

### What ADR-011 required

Clause 3, accepted before any DGF-1 score existed:

> Every feature derived from edges (doc-02 §2.2.4b's 11-wide edge-type histogram and recency
> features) is computed from **the same edge set as the graph or view it feeds**.

Clause 4 pre-registered two reported rows, window-only and first-appearance:

> Floor and GNN are both scored under each view, with the floor's view-derived features recomputed
> from that view, so each row stays a GNN-vs-floor comparison.

*Limitations* says that after parity, what remains GNN-only "is *who* a user's later neighbours are
and what their features say. The first-appearance sensitivity row measures how much the result
depends on it."

### What was built

`score_view` builds the GNN's inputs once, from the graph as of the window's last step, for all three
views (`adapters/dgf1/train_gnn.py:110–111`):

```python
view = graph_view(data, split.test_max)
x = transform.transform(node_inputs(data, view))
```

Only the message-passing edges change with the view (`:114–120`). `run_dgf1` scores both reported
views on every run and logs them under `window_only_*` and `first_appearance_*` (`:184–186`). **All
24 DGF-1 GNN rows carry these keys: 11 retune, 5 pilot, 8 gate.** In each of them:
- under **window-only**, each user's histogram and recency still count its edges to pre-window users;
- under **first appearance**, every scored user and every neighbour reached carries its histogram,
  degree and recency **as of 821**.

Clause 3 is broken for both reported views.

**Why no test caught it.** The trainer test for reported views checks only that their *scores* differ
from the gated view's (`tests/test_dgf1_trainer.py:129–136`). They do, because the message-passing
edges differ. No test asserted that a view's *inputs* come from that view's edges.

### What it affects

- **The gated rows of both gates are unaffected.** On the gated view, both arms' statistics come from
  the graph as of 821, which is the view that feeds them. Training uses the ≤ 369 view for both arms.
- **GATE-DGF1-1's view table (lines 133–141) prints numbers from inputs that break clause 3.** Its
  label, "edges up to each user's own node time", is accurate about *edges* only.
- **Qualification 3's inference (lines 209–213) is unsupported.** First-appearance AUPRC 0.0382
  against 0.0388 gated shows that dropping later edges from message passing barely matters *while
  post-appearance counts stay in the inputs*. It does not show that post-appearance activity barely
  matters.
- **The verdict's claims table (line 224) already records the claim as "not established".**

### Why correcting the rows in place does not rescue them

**1. The corrected inputs sit outside the training distribution.** Measured label-free:

| | training seeds as of 369 (858,702) | test targets under the corrected view (183,469) |
|---|---|---|
| age 0 and last-edge recency 0 | 1,270 (0.148%) | **all**, under first appearance |
| all 13 statistics zero | **0** | 67,266 (36.66%, ADR-011), under window-only |
| mean degree / share at degree 1 | 1.874 / 47.61% | 1.382 / 74.09%, under first appearance |

**2. Valid designs featurise training and test symmetrically.**
- **RelBench** (arXiv:2407.20060, §2–§3 and `relbench/base/task_base.py`) builds every training row
  and every test row as of its own seed time.
- **Relational Deep Learning** (Fey et al., arXiv:2312.04615, §2.2 and Algorithm 1) applies the same
  time filter to both.
- **Google's *Rules of ML*, rule #29** treats training features that differ from serving features as
  a failure mode.

The corrected rows score at-arrival inputs with models trained on a snapshot, so they measure
**transfer under shift**.

**3. The likely outcome is expected under either hypothesis.** This is a prior, not evidence:
- the DGraph-Fin paper (arXiv:2207.03579, §4.2) finds that fraudsters' post-appearance edge timing
  differs from normal users', so removing it should cost both arms;
- GNNs are reported to degrade on new, low-degree nodes (Tail-GNN; Zhu et al., both seen as
  abstracts), so a history-less test population should cost the GNN more;
- the review found **no** direct evidence on how GNNs and gradient-boosted trees compare under this
  shift, so neither direction is claimed.

**4. The corrected view asks a broader question than qualification 3.** It strips post-appearance
information from **both** arms at once. A change in the gap would then mix three things: the shared
node-level channel, the GNN-only channel (*who* the later neighbours are), and the shift.

### Why the flawed scoring cannot simply be left in the code

Gate 3's pre-registration will re-run DGF-1's real-graph arm through `run_dgf1`, which would write new
flawed `window_only_*` and `first_appearance_*` numbers into the append-only registry. Numbers that
exist get read.

## Decision

### Clause 1 — The defect is recorded

- **Where:** `adapters/dgf1/train_gnn.py:110–116`, where the inputs are built, and `:184–186`, where
  they are logged.
- **What:** the GNN's reported-view inputs were built from the window-end graph, breaking ADR-011
  clause 3.
- **Since when:** the trainer's first commit (lab Session 25). Every DGF-1 GNN row carries it.
- **Why undetected:** the only reported-view test checked a symptom, that the scores differ, not the
  property, that the inputs follow the view. This is the fifth test in this programme that shared
  its source of truth with what it tested, or read a symptom the failure does not produce.

### Clause 2 — The reported-view numbers are withdrawn from all inference

- **What is withdrawn:** every `window_only_*` and `first_appearance_*` key in the 24 DGF-1 GNN rows.
  They stay in the registry, which is append-only. **No document, write-up, figure or assembler may
  cite, aggregate or compare them, except to report this withdrawal.**
- **GATE-DGF1-1 is not edited.** Its view table and the figures in qualification 3 are read as
  withdrawn under this ADR. Its claims table already records the claim as "not established".
- **CLAUDE.md's Gate-1 qualification** (lines 50–51) becomes:
  > **Post-appearance activity: not established.** The first-appearance figure cited in the verdict
  > is withdrawn (ADR-013); no valid row bears on this claim.

### Clause 3 — The flawed scoring leaves the code path, and the trainer is re-certified on validation

- **What changes:** `run_dgf1` scores and logs the **gated view only**. `score_view` accepts only
  `"gated"` and raises for any other kind. `VIEWS` and `REPORTED_VIEWS` are removed, and the module
  docstring is corrected.
- **What is kept, as tested building blocks wired into no row:**
  - `adapters/dgf1/sampling.py` (`first_appearance_loader`, `assert_temporal_batch`) with its tests
    and `scripts/verify_dgf1_temporal_sampler.py`;
  - `window_only_edges` with its test.

  Clause 5's pre-registration decides whether and how either is used.
- **What is not changed:** `scripts/assemble_gate_dgf1_1.py`, which reads the withdrawn keys to print
  its view table. It belongs to a signed gate, and it refuses on the signed verdict before reading
  any row.
- **Real-data re-certification, validation window only.**
  - **The run:** after the change, re-run the validation pilot's seed 0
    (`dgf1-20260913T232506Z-1acd5067`) tagged `experiment = "repro_check"`.
  - **The bar:** its gated metrics must equal the pilot row's bit-for-bit, **and** its score file's
    SHA-256 must equal the pilot row's `scores_sha256`.
  - **On a mismatch:** investigate to root cause, never retry (ADR-008 clause 2). No later DGF-1 run,
    Gate 3's included, may rely on the trainer until the check passes.
  - **The row is written either way,** because `RunSession` writes on error. The check is decided by
    the comparison, never by whether the row exists.

### Clause 4 — The deviation from pre-registration is recorded

This follows the form recommended by Lakens (2024, *Collabra: Psychology* 10(1):117094) and Nosek et
al. (2018, *PNAS* 115(11)): state what, where and why, then the impact on the test's severity.

| pre-registered | what happens | why | impact on severity |
|---|---|---|---|
| ADR-011 clause 4, **window-only** row, GNN vs floor | not produced in ADR-011's form; the GNN's numbers withdrawn (clause 2); the floor never scored | defective implementation (clause 1); the corrected form shifts inputs off the training distribution (36.66% all-zero vs 0) | none on Gate 1; no claim rests on this row |
| ADR-011 clause 4, **first-appearance** row, GNN vs floor | as above | as above (age 0 for all scored users vs 0.148% of training seeds); the likely outcome is expected under either hypothesis | none on Gate 1; qualification 3's claim stays **not established** |
| ADR-012 clauses 1 and 7: these rows reported beside the gated numbers | GATE-DGF1-1 printed the GNN side only; now withdrawn | as above | none on the gated test |

**Net direction:** the only number withdrawn favoured the GNN, and no claim is strengthened by this
deviation. Qualification 3's claim moves from "partly supported" to "not established", which is
harder for the GNN.

### Clause 5 — The question goes to its own pre-registration, under binding conditions

A new ADR, the **matched-time sensitivity pre-registration**, numbered when drafted, answers the
question the withdrawn rows were meant to answer. It must meet all of the following:

1. **Ordering.** It is accepted, or explicitly declined by the researcher with a dated reason
   recorded in both it and Gate 3's ADR, **before Gate 3's pre-registration is accepted**. Gate 3's
   ADR must cite this clause and state which view each of its arms is scored under.
2. **A matched-time design.**
   - Both arms are **trained and scored** on inputs dated to each user's own time, the
     RelBench-style seed time. Statistics come from the edge set that feeds them (ADR-011 clause 3).
   - It has its own seed count, derived from a validation pilot (ADR-007).
3. **ADR-011's literal rows run in the same batch.** Both reported views, with inputs correctly built
   under ADR-011 clause 3, are labelled **exploratory**. The pre-registered comparison then appears,
   beside one that can be read.
4. **Optionally, a GNN-only-channel arm.** Each scored user keeps its own gated-view statistics, which
   the floor shares. Message passing and neighbour statistics are limited to its own time. This arm
   sits near the withdrawn figure, so it is **disclosed as informed**, not blind.
5. **A real-data equivalence check before any test-window run.** On the validation window, every new
   view builder must give the same edge sets as the temporal sampler, which was verified on real
   data (lab Session 18).
6. **Reported only.** No pass condition, and no deployment claim ("detect at sign-up"). ADR-011's
   *Revisit when* reserves that for an ADR decided before any score exists.

**Until that ADR's rows exist**, every report of DGF-1 Gate 1 states the post-appearance claim as
**not established** (clause 2's CLAUDE.md wording).

### Clause 6 — Tests, written in the implementation session

1. **No reported-view key in a row.** A `run_dgf1` row carries no key beginning `window_only_` or
   `first_appearance_`, with the prefixes written literally. Mutation: restoring the reported-view
   loop must make this test fail.
2. **`score_view` refuses every view kind except `"gated"`.**
3. **The gated path is unchanged.** Every existing gated-view trainer test passes unmodified,
   including the frozen-transform and id-alignment tests added after lab Session 25's mutation
   check.
4. **The retained building blocks keep their tests:** the temporal sampler's tests and the
   window-only edge filter's test.
5. **The existing reported-view tests are rewritten, not deleted silently.**
   - The two tests that loop over every view (`tests/test_dgf1_trainer.py:119–136`) are restricted
     to the gated view, and test 2 is added beside them.
   - In the row test, the reported-key assertions (`:184–191`) are replaced by test 1's absence
     assertions. The gated-key and configuration assertions around them stay.

The real-data re-certification is clause 3's.

## Alternatives rejected

- **Accept the first draft (with the nine edits its review listed).** Its rows cannot separate the
  hypotheses, and their likely outcome is expected under both. They would cost new code, a second
  statistics path and a new look at the test window. The one harm it repaired is repaired here by
  withdrawal.
- **Produce the corrected rows now, labelled exploratory only.** This keeps the same cost and
  test-window look for no discriminating evidence. The same rows appear under clause 5, built once
  by a tested builder and beside a readable comparison.
- **Score the floor under the views as literally worded, and keep the GNN rows.** The floor would lose
  post-appearance counts while the GNN kept them, recreating the asymmetry parity exists to remove,
  in rows that look like a fair comparison.
- **Give the floor the GNN's inputs as implemented.** The floor's rows would equal its gated rows by
  construction and measure nothing, and clause 3 would stay broken.
- **Put the matched-time question into Gate 3's criteria.** Gate 3 asks whether the gain is
  structural. Folding a temporal-dependence question into its criteria would tangle two
  pre-registrations. They share a builder and an ordering instead (clause 5.1).
- **Withdraw the numbers without routing the question anywhere.** A pre-registered comparison would
  vanish with no path to exist, which is the selective-reporting pattern pre-registration exists to
  prevent. Clause 5's binding is the mitigation.
- **Leave `run_dgf1` unchanged and "ignore" the keys.** Gate 3's re-runs would write new flawed
  numbers, and numbers that exist get read.
- **Delete the keys from the registry.** The registry is append-only.
- **Edit GATE-DGF1-1's view table or verdict.** Dated gate files are never edited. A correction is a
  new record (ADR-008 clause 2's precedent).

## Consequences

- **Compute:** one validation re-run, about 10 minutes at batch 2048 (lab Session 30). No test-window
  run.
- **Code:**
  - `adapters/dgf1/train_gnn.py`: `score_view`, `run_dgf1`, the view constants and the docstring;
  - `tests/test_dgf1_trainer.py`, per clause 6;
  - the sampler module, its verify script and `window_only_edges` stay.
- **Reporting:**
  - GATE-DGF1-1's view table and qualification 3's figures are read as withdrawn;
  - the post-appearance claim has no valid evidence either way, which is where ADR-011 placed it
    until a valid row exists;
  - CLAUDE.md carries the wording.
- **Gate 3 gains an ordering constraint:** clause 5.1.
- **GNN rows get cheaper.** Each gate-length GNN run drops two scoring passes, measured at about
  11 s together on validation in lab Session 27.
- **The selective-reporting risk moves rather than disappears.** It is held by clause 5's binding
  and by the researcher's decision on the matched-time ADR. If that ADR is declined, the dated reason
  is the record.

## Revisit when

- **The matched-time pre-registration is drafted.** It meets clause 5's conditions or records, in both
  ADRs, why one does not apply.
- **Never, to re-admit the withdrawn numbers.** They broke a clause accepted blind. A valid row is a
  new row.
