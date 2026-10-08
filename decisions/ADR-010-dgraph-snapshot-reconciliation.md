# ADR-010 — DGraph snapshot reconciliation: the official split is random, and two shape corrections

**Status:** accepted · erratum proposed 2026-10-08, at the end of this file
**Date:** 2026-07-27 · **accepted** 2026-07-27
**Deciders:** coderback
**Data snapshot:** `DGraphFin.zip` (150,476,320 bytes) from dgraph.xinye.com, loaded via
`torch_geometric.datasets.DGraphFin` 2.8.0.post1 → `data/dgraph/` (gitignored). Verified by
`scripts/verify_dgraph_snapshot.py` on 2026-07-27.
**Docs affected (amendments applied on acceptance, not before):**
`docs/02-dgraph-fin-embedding-model-BUILD.md` §0 (both paragraphs), §2.1, §2.2 (items 2, 4, 4b),
§2.3, §4 (Phase 0, Phase 1 Gate 1), §5 (comparators, template, story), §7 (Gate 1 row), §8 (steps
1, 3), §9 (PyG loader reference); `docs/timeline.md` DGF-1 rows;
`scripts/verify_dgraph_snapshot.py` (`EXPECTED_EDGE_TYPES` 12 → 11).

## Context

The registration-gated snapshot landed and the verifier ran before any DGF-1 code exists — which is
the point of having written it first (ELL-1 precedent: the loader was written before the data and
escalated on arrival). Most of doc-02 §2.1 is confirmed **exactly**:

| pinned | measured | |
|---|---|---|
| 3,700,550 nodes / 4,300,999 edges / 17 features | identical | ✓ |
| 15,509 fraud / 1,210,092 normal / 2,474,949 background | identical | ✓ |
| edge timestamps 1–821 | 1–821, 821 distinct | ✓ |
| ~1.2% anomaly ratio | **1.2654%** labelled prevalence | ✓ — and this is the figure **ADR-007** assumed when fixing the ROC-AUC + AUPRC pair; now measured, not assumed |

Three things did not match, and per ADR-001 the verifier **escalated rather than absorbing them**.
No constant has been edited.

### Finding 1 — the official split is a **random mask**, not temporal

doc-02 §2.3 made this an explicit verification task (*"do not assume"*). Answered, using a node-time
proxy (earliest incident edge time; DGraph dates **edges**, not nodes):

| split | n | p25 | median | p75 | mean |
|---|---|---|---|---|---|
| train | 857,899 | 99 | **216** | 400 | 266.5 |
| val | 183,862 | 99 | **217** | 400 | 266.6 |
| test | 183,840 | 99 | **215** | 399 | 265.5 |

The three distributions are indistinguishable — medians differ by **2 steps out of 821**. This is a
random split.

It follows that **the published leaderboard numbers are not a temporal holdout.** doc-00 §7 makes
temporal holdout *"the only honest way to claim prediction rather than memorisation"*, so the
official split cannot carry DGF-1's structural claim, however comparable it is.

### Finding 2 — edge subtypes are **1–11 (11 types)**, not 0–11 (12)

Measured: contiguous values `[1..11]`, **no type 0**, no gaps. doc-02 §0/§2.1/§2.2 say "0–11" and
"12 subtypes". PyG's own class docstring says "ranging from 0 to 11" and is likewise wrong — recorded
here so nobody later "fixes" our number back to match it.

### Finding 3 — "average degree ~1.16" is the *directed* convention, and the doc reads as if it were the sampled one

| quantity | value |
|---|---|
| `E/N` — mean **out**-degree, directed | **1.162** ← doc-02's "~1.16" |
| `2E/N` — mean **total** degree | **2.325** ← what reverse-edges-ON gives |
| median total degree | **2** (p90 = 4, max 882) |
| isolated nodes | **0** |
| degree ≤ 1 | 1,439,084 (**38.9%**) |
| degree ≥ 10 (smallest fan-out) | 9,501 (**0.26%**) |

doc-02 §2.2.4 already makes reverse edges the **default** ("doubles usable connectivity"), so the
graph the GNN actually samples has mean degree **2.325 — essentially Elliptic's 2.30**, not a
dramatically sparser object. §0's *"most nodes have ≤1 neighbour"* holds only in the directed
convention; with reverse edges it is 38.9%, which is a lot but is not "most".

## Decision

### Clause 1 — DGF-1's gates run on **our own temporal split**; the official split is reported, never gated

- **Gated:** ROC-AUC and AUPRC (ADR-007) on a temporal holdout DGF-1 constructs from `edge_time`,
  against a tabular floor **re-run on that same temporal split** in the same batch.
- **Reported, not gated:** official-split numbers, labelled explicitly as *random-split,
  leaderboard-comparable*, for positioning against published baselines. They acquire no pass
  condition and may not acquire one retroactively.
- Every reported number states **which split produced it**. A table mixing the two without labels
  is the specific failure doc-02 §2.3 warned about.

This inherits ADR-007's structure, and that structure now has a **second, independent justification**:
ADR-007 demoted the leaderboard comparison because gating it means gating on a third party's run
under an unverified protocol. We now know the protocol is not merely unverified but *different in
kind* — a random split cannot evidence a temporal claim.

**The split derivation is deliberately NOT decided here.** DGraph dates edges while
`gbe.eval.temporal.split_masks` takes a **per-node** time tensor, so DGF-1 needs either a node-time
derivation or a harness extension that splits on edge time directly. That choice is pre-registered
in DGF-1's Phase-0/Gate-1 ADR **before any run**, and it must satisfy one constraint fixed here:

> **A node's assigned time may not depend on any edge after the cutoff.** A max- or mean-based
> proxy reads post-cutoff edges into a node's "birth" time and leaks the future into the training
> split — the exact failure `gbe.eval.assert_no_temporal_leakage` exists to catch.

### Clause 2 — edge subtypes are 1–11 (11 types)

doc-02 §0, §2.1, §2.2 item 4b and §9 amended. Concretely: the per-node **edge-type histogram is 11
wide, not 12** (§2.2.4b), and `EXPECTED_EDGE_TYPES` becomes 11.

### Clause 3 — state degree in both conventions; keep §0's expectation, correct its basis

doc-02 §2.1 states `E/N` **and** `2E/N` with the measured distribution. §0's expectation —
*"the model most likely to show a small structural delta"* — **stands**, but rests on the
independent GADBench finding that tree baselines match GNNs on DGraph and on the 38.9% of nodes with
≤1 neighbour, **not** on a mean-degree figure that describes a graph we do not sample.

**ADR-009 is confirmed on measurement.** Its premise — fan-out cannot bind at this sparsity — was
argued from the assumed 1.16. Measured: only **0.26%** of nodes reach the smallest fan-out of 10, so
fan-out is non-binding for 99.74% of them. Batch size was the right second knob, and the conclusion
holds under the *larger* of the two degree conventions.

**Also recorded (no amendment needed):** background nodes carry **two** label values, `y=2`
(1,620,851) and `y=3` (854,098), summing to the documented 2,474,949. The adapter's labelled mask is
therefore `(y == 0) | (y == 1)` — not `y != <one background value>`. doc-02 §2.1's "**binary** fraud
label" is true of the *task* and false of the *tensor*; §2.1 gains a parenthetical.

## Alternatives rejected

- **Edit the constants so the verifier passes.** Rejected — the script forbids exactly this in its
  own error message, and ADR-001 set the precedent when Elliptic's 166 features turned out to be 165.
  A doc-vs-data mismatch is reconciled in the doc.
- **Gate on the official split, for leaderboard comparability.** Rejected: it is a random split, so
  a gate on it would evidence memorisation-resistance nowhere. It would also make DGF-1's headline
  claim weaker than ELL-1's, which was temporal throughout.
- **Drop the official split entirely and report only our temporal numbers.** Rejected: positioning
  against published baselines is DGF-1's stated role (doc-02 §Role), and doc-02 §2.3 requires both,
  labelled separately. Discarding it would throw away the comparability the model exists to obtain.
- **Treat "0–11" as right and assume type 0 exists but is unused in this release.** Rejected: zero
  edges carry it, a 12-wide histogram would ship a permanently-zero column into every node feature
  vector, and the doc would assert something false about the data.
- **Revise §0's expectation now that the sampled degree ≈ Elliptic's.** Rejected as
  over-correction: the expectation has independent support (GADBench; 38.9% of nodes at ≤1
  neighbour). Only its *stated basis* was wrong, and that is what this ADR fixes. Whether the
  structural delta is in fact small is Gate 1's question, not this ADR's.
- **Decide the node-time derivation here.** Rejected: it is a modelling choice with leakage
  consequences and several defensible options. It belongs in a pre-registration ADR before any run,
  not folded into a data-reconciliation ADR.
- **Adopt DGraphFin-2.** Out of scope and not assessed — see *Revisit when*.

## Consequences

- **DGF-1 Phase 0 gains a real, previously-invisible task:** construct and validate a temporal split
  from `edge_time`, with a leakage test in the same session (constitution: untested guards don't
  exist). It was hidden behind an assumption that the official split might already be temporal.
- **A likely `gbe.eval` extension:** the harness splits on node time; DGraph dates edges. Whether
  that becomes an edge-time split path in the core or a node-time derivation in the adapter is an
  **EXTRACT design input** — it should be decided while the `DataSource` seam is being cut, not after.
- **The tabular floor is run twice** — once on the temporal split (the gate's comparator, "the
  numbers to beat") and once on the official split (positioning). doc-02 §8 step 3 moves accordingly.
  Both carry ROC-AUC and AUPRC per ADR-007, and neither can be back-filled.
- `scripts/verify_dgraph_snapshot.py` sets `EXPECTED_EDGE_TYPES = 11` and then passes end-to-end;
  its temporal-split probe becomes a recorded finding rather than an open question.
- **Nothing here touches ELL-1, ADR-003's frozen config, or ADR-008's reference set.** No registry
  row is written by verification.

## Revisit when

- **DGraphFin-2.** A second dataset is published on the same site and has **not** been assessed. If
  it is an *extension*, the Elliptic++ precedent applies (doc-01 §2.1: defer, it complicates the
  clean baseline comparison). If it *replaces* the original — original withdrawn, leaderboard
  migrated — then doc-02 §2.1's pinned shape and the entire comparability argument move, and that
  needs its own ADR **before** DGF-1 Phase 0. These require opposite actions; determine which it is
  before relying on either.
- **The official leaderboard adopts a temporal split.** Clause 1's two-track reporting would
  collapse into one, and the gated/reported division could be revisited.

## Erratum, 2026-10-08 (proposed)

**Source:** research audit 2026-10-07, §2
(`notebooks/audit/2026-10-07/README.md`; the literature re-check of GADBench).
The text above is not edited, so citations of its lines stay valid. **No clause changes.**

**GADBench does not find that "tree baselines match GNNs on DGraph" (`:106`, relied on at `:134`).**
On DGraph-Fin with tuned hyperparameters (Tables 4 and 13;
`notebooks/audit/2026-10-07/literature-recheck.md` §3 and §5):
- **Plain RF and XGBoost sit below the GNNs:** AUPRC 2.57 and 2.75 against 3.77–3.97, and AUROC
  70.37 and 72.43 against 75.51–76.30.
- **XGB-Graph matches them:** a tree with one-hop neighbour aggregation, at AUPRC 3.79 and AUROC
  75.83. RF-Graph does not.

**What it changes.** doc-02 §0's expectation of a small structural delta still has independent
support, but that support is the tree-plus-aggregation result and the 38.9% figure, not plain trees
matching GNNs. DGF-1's Gate 1 later found its GNN above a tree given the same node-level
information (GATE-DGF1-1). Whether that gain survives one-hop neighbour aggregation is the question
of Gate 3's reported tree arm (CLAUDE.md, *Next* item 7).
