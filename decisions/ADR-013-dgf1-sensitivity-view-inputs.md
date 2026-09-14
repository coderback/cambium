# ADR-013 — DGF-1's sensitivity views: node statistics come from each view's own edges, for both arms

**Status:** proposed
**Date:** 2026-09-14
**Deciders:** coderback
**Inherits, does not re-open:** the split, the gated view and floor parity (**ADR-011** clauses 2–4),
the arms, the stage-1 seed count and the reported/gated boundary (**ADR-012** clauses 1, 5 and 7),
determinism (**ADR-005**), and the no-retry rule for a reproduction check (**ADR-008** clause 2).
**Changes no gated quantity.** Both signed gates stand exactly as recorded: `gates/GATE-DGF1-0.md`
and `gates/GATE-DGF1-1.md` (both PASSED 2026-09-14).
**Docs affected — to apply on acceptance. Sites found by grepping `docs/` and `CLAUDE.md` before
drafting (ADR-009 practice; `docs/` is gitignored, so ripgrep skips it). Re-run the grep after
applying:**
- `docs/02-dgraph-fin-embedding-model-BUILD.md` §5: the two sensitivity rows of the reporting
  template (lines 212–213), and the ADR-011 note beneath it (line 228). "The window-only and
  first-appearance rows score the **same trained weights** (floor and GNN) under a different view"
  gains: *with every node statistic recomputed from that view's edges, for both arms (ADR-013)*.
- `docs/timeline.md`: the Phase-1 "floor under the window-only and first-appearance views" row
  becomes "both arms re-scored under ADR-013", and item 1 of the header's Next list changes to match.
- `CLAUDE.md`: item 1 of the phase block's Next list, and the Gate-1 qualification "only partly clear of
  post-appearance activity", which gains a pointer to this ADR.
- `decisions/ADR-011-dgf1-temporal-split-derivation.md` clause 4 and *Consequences* ("logged in the
  same registry row"), and `decisions/ADR-012-dgf1-gate1-preregistration.md` clause 1's reported
  list: a dated pointer to this ADR in each, because corrected reported scores now live in separate
  rows (clause 4).

> **Review note (2026-09-14, pre-acceptance): adversarial pass.** Two findings were serious and
> changed what the ADR claims. Seven were defects in the mechanics. All nine are fixed below.
> - **The corrected rows trade a leak for a distribution shift, and the draft did not say so.**
>   Measured label-free:
>   - under first appearance, every scored user has age 0 and last-edge recency 0; of 858,702
>     training seeds as of 369, only 1,270 (0.148%) do;
>   - under window-only, 36.66% of labelled test users get all-zero statistics, and no training seed
>     has them (0 of 858,702);
>   - at first appearance, test users have mean degree 1.382 with 74.09% at degree 1, against 1.874
>     and 47.61% for training seeds.
>
>   Both arms were trained on the ≤ 369 view, so a gap that shrinks under these views cannot be put
>   down to post-appearance activity alone. The draft's "that is the answer to qualification 3's
>   question" was an overclaim. New clause 7 states what the rows can show, and the addendum must
>   print these figures.
> - **Clause 4 edited the assemblers of two signed gates for no protection.** Both call their
>   signed-verdict refusal before reading a single row. If regenerated for audit, their existing
>   checks refuse on a sensitivity row, so they fail closed. The edit would only have changed the
>   script a signed file names as its provenance. They are now left untouched.
> - **Per-view score vectors were not persisted**, so clause 5's paired bootstrap could not be
>   recomputed from stored files, which is ADR-012 clause 10's standard. They are now persisted.
> - **Moving corrected reported scores into separate rows departs from ADR-011's *Consequences***
>   ("logged in the same registry row"). Now acknowledged: the pairing ADR-011 wanted survives
>   because each new row carries the reproduced gated metrics. Pointer notes are added to *Docs
>   affected*.
> - **No outcome was stated for a failed reproduction.** Now: no rows, the failure is recorded, and
>   there is no fallback.
> - **The runner guard named "this ADR accepted" as if it carried a seed line**, which it does not.
>   The four checks are now spelled out.
> - **The addendum's name matched `gates/GATE-*.md`**, the files CLAUDE.md says gates are decided
>   from, while carrying no verdict. Renamed.
> - **The addendum printed `2 × SE_diff` without ruling out a pass/fail reading of it.** Its wording
>   is now fixed.
> - **The cost omitted that view building repeats for every seed** (projected 46–57 minutes across 8
>   GNN passes, from single timings), and that view statistics do not depend on weights, so each
>   group's view may be built once and shared. Both are now stated, and the timings are marked as
>   single measurements.

## Disclosure — what had been seen when this was written

- **Both DGF-1 gates are signed.** Every gated number is known, on validation and test.
- **The GNN's reported-view numbers on the test window have been seen**, because GATE-DGF1-1
  prints them: window-only **0.7437 ± 0.0045 ROC-AUC / 0.0350 ± 0.0007 AUPRC**, first appearance
  **0.7479 ± 0.0038 / 0.0382 ± 0.0005**, against gated 0.7533 ± 0.0037 / 0.0388 ± 0.0005.
  GATE-DGF1-1's verdict, qualification 3, draws on the first-appearance number.
- **The floor has never been scored under either reported view.** No floor number exists for them.
- **The defect was found by reading code, not results.** It turned up while starting on the floor
  rows, at `adapters/dgf1/train_gnn.py:110–111`. No number was computed to find it or to size it.
- **No new label look.** The only measurements taken for this ADR are label-free: the number of
  distinct node times among scored users, where "scored" means label *presence* (`y ∈ {0, 1}`) and
  no fraud label is read, and the time to build one graph view. The adversarial pass added the
  training-support figures in clause 7, measured the same way.
- **What these rows will show is not predicted here.** The fix could move the GNN's reported
  numbers either way, and the floor's too.

## Context

### What ADR-011 requires

Clause 3, accepted before any DGF-1 score existed:

> Every feature derived from edges (doc-02 §2.2.4b's 11-wide edge-type histogram and recency
> features) is computed from **the same edge set as the graph or view it feeds**. **Recency is
> measured from the view's own cutoff** […], never from step 821.

Clause 4, on the two reported rows:

> Floor and GNN are both scored under each view, with the floor's view-derived features recomputed
> from that view, so each row stays a GNN-vs-floor comparison.

And *Limitations*: after appearance, what remains GNN-only is *who* a user's later neighbours are,
and "the first-appearance sensitivity row measures how much the result depends on it."

### What was built

`score_view` builds the GNN's node inputs once, from the graph as of the window's last step, for
**all three** views:

```python
view = graph_view(data, split.test_max)
x = transform.transform(node_inputs(data, view))
```

Only the **message-passing edges** change with the view. So in every GNN row logged so far (retune,
pilot and gate):

- **window-only:** each user's histogram and recency still count its edges to pre-window users,
  which that view removes from message passing;
- **first appearance:** each scored user carries its histogram, degree and recency **as of 821**.
  So does every neighbour reached. The only thing restricted to the user's own time is which edges
  carry messages.

This breaks clause 3 for the reported views: the features do not come from the edge set of the view
they feed.

**Why no test caught it.** ADR-011's clause-5 tests pin the training view and the gated scoring view.
The trainer test for reported views checks only that they produce *different scores* from the gated
view (`tests/test_dgf1_trainer.py:135–136`), and they do, because the message-passing edges differ.
No test asserted that a reported view's *inputs* derive from that view's edges.

### What it does and does not affect

- **The gated rows of both gates are unaffected.** On the gated view, both arms' statistics come
  from the graph as of 821, the view that feeds them, so clause 3 and parity both hold. Training uses
  the ≤ 369 view for both arms.
- **GATE-DGF1-1's view-sensitivity table is exactly what the code computed**, and its label
  "first appearance (edges up to each user's own node time)" is accurate about *edges*. **The
  inference qualification 3 draws from it is not supported.** A first-appearance AUPRC of 0.0382 against
  0.0388 gated shows that removing later edges from message passing barely matters *while each
  user's post-appearance counts are still in its inputs*. It does not show that post-appearance
  activity barely matters.
- **The gate file is not edited** (it is dated). This ADR and the rows it specifies are the
  correction.

### Why an ADR rather than a fix

- **ADR-011's *Revisit when*:** "Never for DGF-1 once any DGF-1 score exists … changing the split,
  the view or the floor is re-thresholding after seeing a result."
  - This ADR changes no gated view, no floor definition and no threshold. It brings two reported
    rows into line with clause 3, which was accepted blind.
  - **But two choices below are new**, and they are made after the GNN's reported numbers were seen:
    how the window-only view encodes a user with no edge in it, and what statistics a neighbour
    carries under first appearance. Both follow from one principle, stated in clause 1 and taken
    from clause 3's own wording. Every alternative below is rejected on that principle, never on the
    direction it would move a number.
  - **The researcher decides whether that argument holds.**
- **ADR-011's post-look rule:** a later change must make Gate 1 harder for the GNN or rest entirely
  on label-free evidence. This ADR rests on reading code and clause 3, touches nothing gated, and
  cannot move either verdict.
- **Doc-first:** the fix needs specifications ADR-011 does not give. They must not be invented
  inside an implementation session.

## Decision

### Clause 1 — A view's statistics are a function of that view's edge set alone, for both arms

For every scoring view, the 11-wide edge-type histogram and both recency features of **every node the
scoring touches** are computed from **that view's edge set only**, by the same function for the GNN
and the floor. Recency is measured from the view's own cutoff.

| view | edge set | cutoff for recency |
|---|---|---|
| **gated** (unchanged) | edges dated ≤ the window's last step | the window's last step |
| **window-only** | edges dated ≤ the window's last step with **both** endpoints' `t_node` inside the window | the window's last step |
| **first appearance** | for a scored user `v`: edges dated ≤ `t_node(v)` | `t_node(v)` |

The consequences are stated so they cannot be reinterpreted later:

- **Window-only: a node with no edge in the set gets the existing "not in view" encoding, all 13
  statistics zero**, exactly as `graph_view` encodes a user who does not yet exist.
  `steps_since_first_edge` is measured from the node's earliest edge **in the set**, not from its
  global `t_node`: that date can come from an edge to a pre-window user, which this view excludes.
  ADR-011 measured **67,266 of 183,469 labelled test users (36.66%)** left isolated by this view, so
  those users score with all-zero statistics under it.
- **First appearance: the scored user and every neighbour in its scoring subgraph take statistics
  from edges dated ≤ `t_node(v)`**, with recency from `t_node(v)`. So the scored user's own two
  recency features are 0, and its histogram counts only the edges dated exactly at its node time.
  A neighbour's statistics are also as of the **seed's** time, not its own and not 821.
- **The frozen training transform applies unchanged** to every view's inputs (ADR-011 clause 3).
- **The raw 17 features do not vary by view.** They are a release snapshot (ADR-011 *Limitations*),
  so the raw-17 floor's predictions are identical under all three views, and it is not re-run.

### Clause 2 — How first appearance is scored

Scored users are **grouped by node time**. For each distinct time `t`, `graph_view(data, t)` is
built once, and that group is scored on it with exact neighbourhoods (`num_neighbors = [-1] * layers`).

- **This is ADR-011's first-appearance view by construction.** Every edge in `graph_view(t)` is dated
  ≤ `t`, so every hop is bounded by the seed's time, just as the temporal sampler bounds it.
- **The per-batch guard stays.** Every message-passing edge is asserted dated ≤ its seed's time,
  where the numbers are produced.
- **Measured, label-free:** scored users span **340** distinct node times on test (183,469 users)
  and **112** on validation (183,430). Building one view took **1.01 s** as of step 482 and
  **1.26 s** as of step 821 on this machine, one measurement each.
  - **Projection only, from those two timings:** 340 view builds per test scoring pass takes about
    6–7 minutes. Built separately for each of 8 GNN seeds, that becomes about 46–57 minutes.
  - **View statistics do not depend on weights**, so building each time group's view once and
    scoring every seed's model on it is permitted, provided each seed still writes its own row. The
    floor needs only the scored users' own statistics, so it builds them once for all seeds.
  - The first real run measures the actual cost, and the result is recorded.
- **Holding every view in memory at once is ruled out:** 340 × 3.7M × 13 statistics do not fit
  in 15 GB. Views are built and released one group at a time.

### Clause 3 — Both arms re-run on the test window, and must first reproduce their gate rows

- **DGF-1:** seeds 0–7 at ADR-012's winning configuration (`lr = 3.318335548548489e-4`,
  `batch_size = 2048`, 40 epochs), trained on the ≤ 369 view.
  - **No trained weights were saved**, so the model is retrained. Under ADR-005 that must reproduce
    the gate's weights.
  - **Checked, not assumed:** each new row's gated metrics must equal its GATE-DGF1-1 row
    bit-for-bit. A mismatch **stops the batch**, and no reported number is logged from weights that
    are not the gate's. It is investigated to root cause, never retried (ADR-008 clause 2).
- **Parity floor:** seeds 0–7 at `max_depth = 8`, `subsample = 0.8`. It is trained once per seed on the
  ≤ 369 view, because training does not depend on the scoring view, then scored under all three views.
  Its gated metrics must equal its GATE-DGF1-1 rows bit-for-bit, under the same stop rule.
- **Eight seeds**, ADR-012's stage-1 count. No new count is derived, because these rows have no
  pass condition (ADR-011 clause 4, ADR-012 clause 1).
- **If either reproduction check fails, no sensitivity row is logged.** The failure and its root
  cause are recorded in the addendum, which then reports that the rows could not be produced. There
  is no fallback to unreproduced weights, and none to the superseded scoring path.
- **Every row persists its scored vectors for all three views**, with node ids and SHA-256 hashes
  (ADR-012 clause 10's standard), so the addendum's bootstrap can be recomputed from stored files.
- **Separate rows, not the gated row.** ADR-011's *Consequences* put reported scores "in the same
  registry row" as the gated number, so the two could never be reported apart. The gate rows are
  written and cannot be amended. Each sensitivity row instead carries its reproduced gated metrics
  beside its reported ones, which keeps that pairing inside the new row.
- **Validation is not re-run.** The retune and pilot GNN rows carry the same defect in their
  reported keys. They were selection and sizing runs, their reported views fed no decision, and
  they are reported nowhere.

### Clause 4 — The rows can never be read as gate rows

- **Tag:** `experiment = "sensitivity"`.
- **The reproduced gated metrics are logged under a `gated_` prefix, never bare.** A sensitivity row
  therefore has **no** `fraud_auc` or `fraud_auprc` key, and a reader looking for a gated number finds
  nothing to mistake. The reported views keep their `window_only_` and `first_appearance_` prefixes.
- **The two signed gates' assemblers are not changed.**
  - Each calls its signed-verdict refusal before reading any row, so neither reaches its loader
    while a verdict exists.
  - If one is ever regenerated for audit, its existing checks refuse on a sensitivity row rather
    than misread it: `assemble_gate_dgf1_1.py` refuses any non-gate row on the test window, and
    `assemble_gate_dgf1_0.py` refuses test-window rows from more than one commit. That is the safe
    failure.
  - An audit regenerates against the registry prefix that ends with the gate batch, which the
    append-only registry makes well-defined.
  - Editing either script would change only the provenance a signed file names.
- **Every later reader of DGF-1 test-window rows** selects on `experiment` explicitly. That covers
  the addendum assembler and Gate 3's.
- **The runner mode refuses to start** unless all four hold:
  1. ADR-013's status line reads `accepted`. This is a status check only, since this ADR carries no
     seed line.
  2. ADR-012 is accepted and carries its `**Stage-1 seeds:**` count, read by the existing
     `require_accepted_preregistration`.
  3. GATE-DGF1-1's Verdict section is signed.
  4. The tree is clean.
- **The reported-view keys already in the 24 GNN rows (retune, pilot, gate) are superseded, not
  deleted** (the registry is append-only). Any reader of DGF-1's reported views reads the sensitivity
  rows.

### Clause 5 — Where the result is reported

- **A dated addendum, `gates/ADDENDUM-DGF1-1-sensitivity.md`**, assembled by script from the
  sensitivity rows, with every cell computed. It is deliberately not named `GATE-*`: CLAUDE.md
  names those as the files gates are decided from, and this file decides nothing.
- **Per view, for both arms:** mean ± std (`ddof = 1`), the difference, and `2 × SE_diff`, stated
  explicitly as having **no pass condition**. The words *pass*, *fail* and *resolvable* do not
  appear in the file. It also gives prevalence and positive count (ADR-007), and a paired bootstrap
  per seed pair (ADR-012 clause 10's settings: 1,000 replicates, seed 0).
- **Clause 7's training-support figures are printed beside the view rows**, so no reader meets a
  gap without the distribution shift that qualifies it.
- **The superseded GNN numbers are printed beside the new ones**, with this ADR as the reason, so
  the correction is visible where the old numbers were.
- **The addendum carries no verdict.** It ends with a *Reading* section left blank for the
  researcher: an interpretation of reported rows, not a gate.
- **GATE-DGF1-1 is not edited.** Its qualification 3 is read together with the addendum, and
  CLAUDE.md's Gate-1 qualification says so.

### Clause 6 — Tests, written in the implementation session

1. **Inputs follow the view.** For each reported view, on a fixture where a user's as-of-window-end
   statistics differ from its view statistics, both the GNN's scoring inputs and the floor's design
   matrix equal the statistics computed from that view's edge set. **This test must fail against
   today's `score_view`** (the mutation check is the existing code).
2. **First appearance, per seed.** Every node in a seed's scoring subgraph carries statistics as of
   the seed's time, and the seed's recency features are 0.
3. **Window-only isolation.** A target whose only edges reach pre-window users gets all-zero
   statistics.
4. **Parity per view.** The GNN's and the floor's inputs for the scored users are identical under
   each view.
5. **Gated reproduction.** On a fixture, the new scoring path's gated scores equal the old path's
   bit-for-bit.
6. **Determinism.** Two runs of the grouped first-appearance scoring are identical.
7. **The registry guard.**
   - A sensitivity row has no bare `fraud_*` key.
   - GATE-DGF1-1's loader, handed a sensitivity-tagged test-window row, refuses rather than
     returning it.
   - The runner refuses each missing precondition, one test per refusal.
8. **Persistence.** Each view's stored vector reloads with its ids aligned to the scored users, and
   its hash matches the row.

### Clause 7 — What these rows can and cannot show

Correcting the inputs removes a leak, and it introduces a distribution shift. Both arms were
trained on the ≤ 369 view, where users have a history. Measured label-free (node times, edges and
label presence only):

| | training seeds as of 369 (858,702) | test targets under the view (183,469) |
|---|---|---|
| age 0 and last-edge recency 0 | 1,270 (0.148%) | **all of them**, under first appearance |
| all 13 statistics zero | **0** | 67,266 (36.66%, ADR-011), under window-only |
| mean degree / share at degree 1 | 1.874 / 47.61% | 1.382 / 74.09%, under first appearance |

So:
- **A gap under these views compares the arms on inputs they rarely or never saw in training.** If
  the GNN's advantage shrinks or vanishes, that fits the advantage depending on post-appearance
  activity. It **equally** fits the GNN transferring worse than the tree to users with no history.
  These rows cannot separate the two.
- **If the advantage survives**, it survives both the removal and the shift, which is the stronger
  statement.
- **Neither reading licenses a deployment claim** ("detect at sign-up"). ADR-011 reserves that for its
  own ADR, with first-appearance views decided before any score exists.
- **Separating removal from shift needs arms trained on first-appearance inputs.** That is a
  different experiment, not pre-registered here (ADR-011 clause 4).

## Alternatives rejected

- **A. Score the floor under the views as literally worded, and keep the GNN rows as logged.** The
  floor would lose post-appearance counts while the GNN kept them, recreating the asymmetry parity
  exists to remove. The rows would look like a fair comparison and not be one. It also leaves clause 3
  broken for the GNN.
- **A′. Give the floor the GNN's inputs as implemented (as-of-821 statistics under every view).**
  Parity would hold, but the floor's reported rows would be its gated rows by construction and
  measure nothing. It still breaks clause 3.
- **C. Produce no rows, and record the defect.** ADR-011 and ADR-012 both list these rows as
  reported, GATE-DGF1-1's qualification 3 would stay unsupported permanently, and the cost is modest
  (clause 3).
- **First appearance with the seed's statistics at its own time but neighbours' at 821.**
  Neighbours' counts would still include edges dated after the seed's time, which is the
  post-appearance channel again, arriving through the neighbours.
- **Window-only `steps_since_first_edge` from the global `t_node`.** It brings in a date that can
  come from an edge the view excludes, which breaks clause 1's principle.
- **Recompute only the scored users' statistics, not the neighbours'.** Same objection as the
  neighbours-at-821 alternative, for both views.
- **Re-score from saved weights instead of retraining.** No weights were saved. Retraining under
  ADR-005, checked bit-for-bit against the gate rows, recovers the same model with evidence that it
  is the same model.
- **Also re-run the validation pilots.** Their reported keys informed no decision and are reported
  nowhere. Re-running them adds compute and no information any document uses.
- **Retrain both arms on first-appearance inputs, to remove clause 7's shift.** It would separate
  information removal from distribution shift, but it is a new experiment with its own training sets
  and seeds. ADR-011 clause 4 leaves it out of the pre-registration, and deciding it now, after the
  gated results, is what ADR-011's *Revisit when* forbids. It belongs in its own ADR, if a write-up
  needs it.
- **Add `experiment == "gate"` filters to the two signed gates' assemblers.** Neither can reach its
  loader while a verdict exists, and both already fail closed on a sensitivity row. The edit would
  protect nothing and would change the provenance script of two signed files.
- **Edit GATE-DGF1-1's view table or verdict.** Dated gate files are never edited. A corrected result
  is a new result (ADR-008 clause 2's precedent).
- **Report the rows in Gate 3's file.** Gate 3 has its own question and needs its own ADR. ADR-011
  requires reported views beside the gated numbers they qualify, not beside ablations.

## Consequences

- **Compute, run in the researcher's own terminal:**
  - 8 GNN retrains at about 10 minutes each, the measured cost of a batch-2048 run (Session 30; gate
    row seed 7 took 590 s), plus the grouped first-appearance scoring of clause 2. Its view building
    is projected at about 6–7 minutes if views are shared across seeds, or 46–57 minutes if not, and
    is measured on the first run;
  - 8 floor runs taking seconds each, plus their view builds.
- **Code:**
  - a statistics function for an arbitrary edge set. `view_node_features` asserts that `node_time`
    was derived from the same edges, which the window-only set breaks by design, so window-only
    needs its own path, under its own tests;
  - `score_view` builds inputs per view, and first appearance is scored in time groups;
  - the floor gains view scoring from one fit;
  - a `sensitivity` runner mode, persisting three score vectors per row;
  - the addendum assembler. The two signed gates' assemblers are untouched (clause 4).
- **The reported-view keys in all 24 existing GNN rows are superseded.** GATE-DGF1-1's view table
  must be read with the addendum.
- **The trainer's test for reported views (`test_dgf1_trainer.py:135–136`) was a symptom check.** It
  passed because the edges differed, which is the pattern this programme has hit four times before.
  Clause 6 test 1 is written to fail against the code it replaces.
- **Whatever the rows show is reported** (ADR-011 clause 4: "That is a result, not an embarrassment"),
  always with clause 7's reading beside it. The rows sharpen qualification 3 but cannot close it: a
  shrinking gap is ambiguous between post-appearance dependence and worse transfer to users with no
  history.

## Revisit when

- **Never for DGF-1 once any `experiment = "sensitivity"` row exists.** From that point, changing a
  view's edge set, its isolation encoding or its neighbour statistics is choosing a definition after
  seeing its result.
- **Gate 3's pre-registration ADR is drafted.** Its arms reuse these scoring paths, so it must state
  which view each arm is scored under and cite clause 1 for how inputs are built.
