# ADR-007 — DGF-1 gate metric: ROC-AUC and AUPRC jointly, neither alone

**Status:** accepted
**Date:** 2026-07-26 · **accepted** 2026-07-26
**Deciders:** coderback
**Docs affected — all amendments applied 2026-07-26 on acceptance:**
`docs/02-dgraph-fin-embedding-model-BUILD.md` header (Role), §1, §2.3, §4 (Phases **0**, 1, 3), §5,
**§6**, §7, **§8**; `docs/00-shared-core-graph-embedding-GUIDE.md` §4 (`gbe.eval` metric list), §5
(DGF-1 `EvalConfig` cell), §7 (metrics bullet), §9 (build-order row);
`docs/timeline.md` DGF-1 Gate 0 / 1 / 3 rows (derived mirror — the docs stay upstream).

> **Enumeration corrected on application.** The proposed draft listed doc-02 §2.3 / §4 / §5 / §7.
> Applying it surfaced three further places that specify the metric and would have been left
> contradicting this decision: **§6**'s risk row read *"Report AUC **only**; match leaderboard
> metric"* — a direct contradiction — plus §1's passing reference to the *"AUC-based `EvalConfig`"*
> and the header Role line's *"judged by **AUC** against a public leaderboard"*. All three are
> amended. §2.1's *"Official leaderboard at dgraph.xinye.com (metric: **AUC**)"* is a fact about
> the leaderboard rather than about our gate, and is deliberately left unchanged.

## Context

Doc-02 fixes **AUC** as DGF-1's metric in six places: §2.3 ("`EvalConfig`, **metric = AUC**"),
§4 Gate 1 ("beats the tabular floor on AUC"), §4 Gate 3 ("all on AUC"), §5 ("judged by **AUC**"),
§6 ("Report AUC only; match leaderboard metric"), §7. Doc-00 §7 backs this: *"Metrics, matched to
the dataset's convention (never invent your own to look good): AUC where there is extreme
imbalance (DGraph, EDGAR)."* The reason is sound — DGraph's official leaderboard uses ROC-AUC,
and DGF-1's stated role is validating scale *"judged by AUC against a public leaderboard"*.

**ELL-1 produced a measured counterexample to the assumption underneath that choice** — that
ROC-AUC tracks usable performance under imbalance. From the local-94 diagnostic batch
(`experiments/registry.csv`, 8 seeds/arm, deterministic, reported-not-gated per ADR-006 clause 5;
`git_dirty=true`, which is why these are cited as a diagnostic and not as a gate number):

| arm | illicit precision | illicit recall | ROC-AUC |
|---|---|---|---|
| GNN-on-local-94 | **0.5337 ± 0.1402** | 0.7244 ± 0.0667 | **0.9230 ± 0.0053** |
| RF-on-all-165 | **0.9125 ± 0.0044** | 0.7231 ± 0.0027 | **0.9394 ± 0.0004** |

At **matched recall** (0.7244 vs 0.7231, a difference of 0.0013), precision differs by **0.379**
— and ROC-AUC compresses that into **0.016**. One model flags roughly one licit transaction per
illicit one; the other flags one per ten. ROC-AUC calls them near-equivalent.

**The arithmetic gets worse at DGraph's prevalence, not better.** ELL-1's test window is 6.5%
illicit. DGraph's labelled prevalence is 15,509 / (15,509 + 1,210,092) = **1.27%** (doc-02 §2.1)
— five times lower. ROC-AUC's x-axis is FPR = FP/N_neg; with 1.21M negatives, 12,000 false
positives moves FPR by 0.01 while destroying precision, because there are only 15.5k true
positives to compete with. This is Davis & Goadrich (2006): ROC and PR space are not
order-equivalent under class skew.

The following is a **synthetic illustration of metric behaviour, not a DGraph result** — no
DGraph data exists (see Disclosure). Equal-variance Gaussian scores at DGraph's exact labelled
counts, separation `mu`:

| `mu` | ROC-AUC | AUPRC | precision @ recall 0.70 |
|---|---|---|---|
| 2.0 | 0.9221 | 0.3047 | 0.1135 |
| 2.4 | 0.9564 | 0.4808 | 0.2366 |
| 2.8 | 0.9763 | 0.6309 | 0.4286 |
| 3.2 | 0.9884 | 0.7729 | 0.7117 |

Across that range ROC-AUC moves **+0.066** while AUPRC moves **+0.468** and precision at matched
recall moves **+0.598**. AUPRC is **7.1× more responsive** to the same change in model quality.
A ROC-AUC gap of 0.03 — smaller than several seed-variance bands ELL-1 measured — spans a
**doubling** of precision.

**Three facts make this decision urgent rather than tidy-up:**

1. **doc-02 §0 already predicts a small structural delta.** *"At average degree ~1.16, most nodes
   have ≤1 neighbour and message passing has almost nothing to aggregate… This is the model most
   likely to show a *small* structural delta."* A small true effect measured with an insensitive
   instrument is the regime where a gate most easily passes a null result. Gate 1's whole job is
   to catch "structure isn't earning its keep at scale".
2. **The metric cannot be added retroactively.** The registry stores *summary metrics*, not
   per-node scores. AUPRC is not recoverable from any existing row — adding it later means
   **re-running every arm**. A gate metric chosen after the batch exists is therefore either
   forking-paths or expensive, and in practice the pressure would be to gate on whatever was
   already logged.
3. **ADR-006 explicitly deferred this question to its own ADR.** Reconsidering the program's gate
   metric was ruled *"legitimate, but it must happen in its own ADR, argued on its merits, not
   mid-gate after seeing which metric wins."* This is that ADR.

## Disclosure — informed by ELL-1, blind to DGraph

This repo's practice is to state what the decider had seen. ADR-006 carried a disclosure because
its clause-1 bar was set with that comparison's own provisional numbers in view. **This ADR is
not in that position, and the difference is the point:**

- **No DGF-1 number of any kind exists.** Verified 2026-07-26: `data/` contains only `elliptic/`;
  all 100 registry rows are `model=ell1`; `adapters/dgf1/` is a one-line scaffold with no logic.
  Nothing about DGraph has been observed, so no threshold here can be tuned to a result. This is
  blind in ADR-004's full sense.
- **The decision is informed by a *different* dataset's result** (ELL-1's local-94 diagnostic) and
  by the arithmetic of the two metrics under skew. That is a program learning from its own cheap
  first gate — which is what ELL-1 was built to deliver (doc-01 §0, doc-02 §Role).
- **The change can only make passing harder.** A forking-paths concern bites when a change can
  help a result pass. This one *adds* a clause to a conjunction: every model that would have
  cleared the new Gate 1 would also have cleared the old one, and some that would have cleared
  the old one will now fail. That asymmetry is the defense, and it is also why the change must
  land now rather than after the first batch — afterwards, the same edit would be indefensible
  regardless of its merits.

## Decision

**Scope:** DGF-1's binary-classification gates — **Gate 1** (structure vs the tabular floor) and
**Gate 3** (defending ablations). Gate 2 is a retrieval/clustering task with its own metrics and
is **out of scope** here.

**Every DGF-1 classification run reports both metrics on every row:**

| metric | role | why it cannot be dropped |
|---|---|---|
| **ROC-AUC** | comparability | the leaderboard's metric; doc-00 §7's "matched to the dataset's convention"; the only number comparable to published baselines |
| **AUPRC** (average precision) | diagnosticity | prevalence-sensitive; the instrument ELL-1 proved is needed at low prevalence |

**Gate 1 passes iff both gated clauses hold**, each against a **tabular floor re-run in the same
batch** (ADR-006 clause 4: never against rows from a different code path):

1. **ROC-AUC.** `mean(DGF-1) − mean(floor)` positive and **resolvable**.
2. **AUPRC.** `mean(DGF-1) − mean(floor)` positive and **resolvable**.
3. **Leaderboard positioning — reported, not gated.** ROC-AUC on the official split alongside
   published DGraph/GADBench numbers, for positioning only. It acquires no pass condition, and
   may not acquire one retroactively.

**Resolvability is inherited unchanged from ADR-006:** a difference is resolvable iff
`mean(A) − mean(B) > 2 × SE_diff`, `SE_diff = sqrt(s_A²/n_A + s_B²/n_B)`, with **`s` the sample
standard deviation (`ddof=1`)** and the Welch p-value reported alongside for context.

**Two implementation pins.** Both are the same class of pin as ADR-006's `ddof=1` — a criterion
called *mechanical* must not depend on which script evaluates it:

- **AUPRC is `sklearn.metrics.average_precision_score`, never `auc(recall, precision)`.** Linear
  interpolation between PR points is optimistically biased (Davis & Goadrich 2006); the two
  disagree on the same scores, and the difference is largest exactly at low prevalence.
- **AUPRC's chance level is the prevalence of the scored window, not 0.5.** Every reported AUPRC
  carries its window's prevalence and positive count. AUPRC is **never** compared across datasets,
  across split variants with different prevalence, or against ELL-1.

**Cost: zero.** Both metrics are computed from the same score vector in the same pass — one extra
sklearn call per run, no extra training.

**Seed count is deliberately not fixed here.** ADR-005 requires each gate's count to be derived
**in advance from measured variance on a deterministic batch**, never assumed — and DGF-1's
variance cannot be known before DGF-1 exists. The *rule* is fixed now: the count is derived from a
pilot measured on the **validation split only** (never the test split), fixed before any
test-split run, with a **hard floor of 5 per arm** and ADR-006's fixed two-stage design inherited
(**no open-ended top-up** — that is optional stopping). The number itself goes in DGF-1's Gate-1
pre-registration ADR. ELL-1 is the cautionary precedent: 3 seeds gave ±0.031 where 8 gave ±0.070
on the same arm.

**Gate 3's criteria and seed count still require their own ADR** before those runs, per ADR-006's
*Revisit when*. This ADR fixes only the metric pair they will use.

## Alternatives rejected

- **ROC-AUC alone (doc-02 as written).** Rejected: measured on ELL-1 to compress a 0.379 precision
  gap at matched recall into 0.016 ROC-AUC, and the compression is worse at DGraph's 1.27%
  prevalence. Combined with doc-02 §0's own prediction of a small structural delta, this is the
  configuration most likely to pass a null result. Note this rejection is *not* "the doc was
  wrong" — ROC-AUC is retained in full as clause 1; the doc under-specified by making it the
  *only* clause.
- **AUPRC alone.** Rejected: it would destroy the leaderboard comparison that is DGF-1's stated
  reason for existing, and violate doc-00 §7's "matched to the dataset's convention / never invent
  your own to look good". Published DGraph baselines report ROC-AUC; AUPRC has no comparator.
- **F1, for continuity with ELL-1's gates.** Rejected: threshold-dependent, and ELL-1 measured the
  threshold to be the noisy part — the calibration diagnostic's two honest transfer folds came
  back **−0.018 and +0.032**, opposite signs. At 1.27% prevalence an argmax operating point is
  worse still. Both metrics chosen here are threshold-free, which also removes that noise from the
  gate.
- **Precision@k or recall at fixed FPR.** Rejected as the *gated* metric: each needs an arbitrary
  `k` or FPR with no operational anchor available to us (we do not have the platform's review
  budget), and AUPRC summarises the same region of the curve without a free parameter. Either may
  be **reported** for interpretability.
- **Gating on the leaderboard comparison** (doc-02 §4/§7's "competitive with the published
  baseline"). Rejected: it would gate on a third party's run under a protocol we cannot verify.
  ADR-006 already established that even *our own* rows from a different code path may not serve as
  a gate arm; someone else's number is strictly weaker. Demoted to reported. **This is the one
  clause this ADR makes softer, and it is a real trade** — offset by clause 2, which is both new
  and enforceable, so Gate 1 is net strictly harder.
- **Deciding after the first DGF-1 batch.** Rejected twice over: it is the forking-paths move, and
  it is mechanically dishonest — AUPRC cannot be recovered from logged summary metrics, so the
  choice would be made under pressure to gate on what was already written.
- **Retro-fitting AUPRC to ELL-1's gates.** Rejected: dated gate files are sacred, and AUPRC is
  not computable from the registry without re-running ELL-1. ELL-1's verdicts stand exactly as
  recorded.

## Consequences

- **`gbe/eval/metrics.py::classification_metrics` gains an `auprc` key.** It stays domain-agnostic;
  the module exists precisely so a model and its baselines score identically, so both the GNN and
  tabular-baseline paths pick it up together and the adapters rename it as they do today.
- **Gate 0 acquires a dependency on Gate 1 — found while applying the amendments.** doc-02 Phase 0
  said *"log AUC"* for the tabular floor, and the floor is exactly what clause 2 is measured
  **against**. A floor logged on ROC-AUC alone would leave clause 2 with no comparator, and because
  AUPRC is not recoverable from a logged row (consequence below), the only remedy would be
  re-running the entire floor. **Both metrics are therefore required from Phase 0 onward**, not
  from Phase 1 — doc-02 §4 Phase 0, §7 Gate-0 row, and §8 step 3 amended accordingly. This is the
  first concrete instance of the "cannot be added retroactively" argument in the Context biting
  something other than Gate 1 itself.
- **The registry's `metrics_json` key set stops being uniform across the file.** ELL-1's 100 rows
  have no `auprc`; DGF-1's will. Any reader — including future gate assemblers — must treat the
  key set as per-row, not per-file. This is the append-only registry working as designed, but it
  is a new sharp edge.
- **New guard tests required in the same session as the implementation** (constitution: untested
  guards don't exist): that `auprc` is `average_precision_score` and not the trapezoidal `auc`
  form, and that it returns ≈ prevalence on random scores.
- **Nothing is implemented in this session.** Doc-first: an idea is not implemented in the session
  it was conceived. On acceptance, the doc amendments land first, then the code.
- **Doc amendments on acceptance:** doc-02 §2.3 (`metric = AUC` → the metric pair), §4 Phase 1 and
  Phase 3, §5 (reporting template gains an AUPRC column; the template's "fill from YOUR runs"
  discipline is unchanged), §7 (Gate 1 and Gate 3 rows); doc-00 §5 (DGF-1 `EvalConfig` cell) and
  §7 (the metrics bullet gains the skew caveat). Each amendment carries an inline note of the
  previous wording and why it changed, as ADR-006 did.
- **EDR-1 inherits a live question, not an answer** — see below.

## Revisit when

- **Before EDR-1's gate ADR is written.** EDGAR is the program's other extreme-imbalance node task
  (doc-00 §7 names DGraph and EDGAR in the same breath). The same argument plausibly applies, but
  it must be **re-argued in that ADR** against EDR-1's own prevalence and its own published
  comparators — inherited, not assumed.
- **If DGraph's official split proves non-temporal** (doc-02 §2.3 flags this as a verification
  task: *"do not assume"*). Official-split and own-temporal-split windows would then carry
  different prevalences, making the "AUPRC carries its window's prevalence" rule load-bearing
  rather than hygienic, and the two AUPRC numbers strictly non-comparable.
- **Not for DGF-1 Gate 1 once that gate is assembled and dated.**
