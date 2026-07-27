# ADR-009 — HPO transfer: lr + one *graph-appropriate* second knob

**Status:** accepted
**Deciders:** coderback
**Date:** 2026-07-26 · **accepted** 2026-07-27
**Supersedes:** the **transfer clause** of ADR-003 (its frozen config is untouched)
**Docs affected — all amendments applied 2026-07-27 on acceptance:**
`docs/00-shared-core-graph-embedding-GUIDE.md` §6.4;
`docs/01-elliptic-embedding-model-BUILD.md` §4 (Phase 1);
`docs/02-dgraph-fin-embedding-model-BUILD.md` §4 (Phase 1) — citation only, wording stands;
`docs/research-plan-UNIFIED-GBE-GDE.md` (HPO protocol bullet);
`docs/timeline.md` — ELL-1 Phase-1 row;
`decisions/ADR-003-ell1-hpo-frozen-config.md` — superseded-clause note, body **not** rewritten;
`adapters/ell1/hpo.py` — module docstring (documentation only, no behaviour).

> **Enumeration corrected *before* applying — the first time that has happened.** The proposed draft
> listed four sites. Grepping for the phrase first (ADR-007's enumeration missed three sites,
> ADR-008's missed one, both found only after the fact) turned up **six more**: doc-01 §4, the
> unified research plan, ADR-003's *Alternatives rejected* entry, the timeline's ELL-1 Phase-1 row,
> and a module docstring in `adapters/ell1/hpo.py`. The contradiction was roughly twice as
> widespread as the draft claimed.
>
> **Deliberately not amended:** `docs/document-amendments-v0.2.md` (lines 25, 95). That file is the
> dated record of the v0.2 amendment round — a historical document, and amended no more freely than
> a dated gate file. It correctly records what was decided *then*.

## Context

The program runs **one** HPO sweep, on ELL-1, and transfers the winning region upward (doc-00 §6.4,
no NAS). What "transfer" permits downstream is currently specified **four times, inconsistently**:

| where | says |
|---|---|
| doc-00 §6.4 | "Transfer the winning region to `DGF-1`/`EDR-1`, tuning only **lr + fan-out** there" |
| ADR-003, *Transfer rule* | "DGF-1 and EDR-1 inherit this winning region and tune only **lr + fan-out**" |
| ADR-003, *Revisit when* | "if the transfer to DGF-1 (**lr + fan-out** retune only) clearly underperforms…" |
| doc-02 §4 Phase 1 | "tune only **lr + batch size** here — fan-out tuning is mostly irrelevant at this sparsity (full neighbourhoods are tiny); don't burn budget on it" |

An accepted ADR and a build doc give different instructions for the same runs, so one of them will
be violated by construction. Under doc-first that must be resolved before Phase-1 budget is spent —
not discovered afterwards, when whichever knob was actually tuned becomes the retrospective
justification.

**doc-02 is substantively right on the facts.** DGraph's average degree is ~1.16 (doc-02 §2.1):
most nodes have ≤1 neighbour, so full neighbourhoods are already smaller than the smallest fan-out
in the frozen space. Sweeping fan-out {[10,10], [15,10], [25,10]} there would mostly re-run the
same computation — ELL-1 already showed the milder version of this, where at mean degree 2.30
fan-out 25 truncated only **0.26%** of train nodes (lab 2026-07-24). Batch size, by contrast, is
the knob that actually binds at 3.7M nodes on 4 GB of VRAM.

**But the general rule is what needs fixing, not DGF-1's instance of it.** "lr + fan-out" was
written when ELL-1 was the only graph in view, and it silently encodes an assumption — that
neighbourhood sampling is the binding constraint — which is a property of the *graph*, not of the
transfer protocol. EDR-1's typed corporate graph will have its own answer, and if the rule stays
phrased as a specific knob the same contradiction simply recurs one model later.

## Decision

**doc-00 §6.4's transfer clause becomes: retune `lr` + exactly *one* second knob, chosen per model
as the one that binds on that graph, named in the model's build doc with a one-line reason.**

- The **budget is unchanged**: two knobs, no full-space search, no NAS. This ADR does not widen
  what may be tuned; it only stops the shared guide from hard-coding *which* second knob every
  future graph must use.
- The second knob must be named **in the model's build doc before its Phase-1 runs begin**, with
  its justification. A knob chosen after seeing Phase-1 results is a searched hyperparameter
  wearing a transfer rule's clothes.
- **DGF-1's second knob is `batch_size`** (average degree ~1.16; fan-out cannot bind). doc-02 §4
  Phase 1 already says this and stands as written — it becomes the first instance of the rule
  rather than a violation of it.
- **EDR-1's second knob is explicitly not decided here.** It is named in EDR-1's own build doc or
  gate ADR, argued from its own graph's density. Deciding it now, with no EDGAR graph in
  existence, would repeat exactly the error this ADR fixes.
- **ADR-003's frozen config is untouched.** Backbone, layers, hidden, aggregator, dropout, lr,
  fan-out, and the fixed budget all stand exactly as frozen. Only the *transfer clause* changes.

**On ELL-1:** nothing. ELL-1 *is* the sweep; it has no transfer step, and its gates are dated and
closed.

## Alternatives rejected

- **doc-00 / ADR-003 win — keep "lr + fan-out" everywhere, amend doc-02.** Rejected: it spends
  DGF-1's Phase-1 budget on a knob doc-02 argues cannot bind at degree 1.16, and it discards
  correct dataset-specific reasoning in favour of a rule written before that dataset was examined.
  Preserving an accepted ADR's exact words is not worth running a sweep known to be uninformative.
- **doc-02 wins narrowly — the rule becomes "lr + batch size".** Rejected: it fixes DGF-1 and
  breaks EDR-1 in the same motion. The rule would still name a specific knob, so the next model
  whose graph disagrees reopens this ADR. Same defect, one model later.
- **Allow both knobs (lr + fan-out + batch size).** Rejected: three knobs is a wider search than
  doc-00 §6.4 budgets, and "transfer the region, retune a little" stops being a meaningful
  constraint once the list grows. Two is the budget; this ADR keeps it.
- **Leave the contradiction and let each doc govern its own model.** Rejected: it is exactly the
  state that makes a violation invisible. Whichever knob got tuned would be defensible after the
  fact by citing the other document — the ambiguity is the problem, not either wording.
- **Fold this into DGF-1's Gate-1 pre-registration ADR.** Rejected: that ADR needs measured
  variance from a validation pilot, which needs the data, which is registration-gated and not yet
  in hand. The contradiction is resolvable now and blocks nothing else; bundling it would hold a
  two-paragraph fix behind a download.

## Consequences

- **doc-00 §6.4** carries the generalised clause plus an inline note of the previous wording and why
  it changed (ADR-005/006/007 house style).
- **ADR-003** gains a superseded-clause note pointing here. Its body is **not** rewritten — dated
  ADRs, like gate files, are not silently edited. Its *Revisit when* still reads "lr + fan-out
  retune only"; the note records that this now means "lr + the model's named second knob", which
  for DGF-1 is batch size.
- **doc-02 §4 Phase 1** gains an ADR-009 citation. No wording change: it was already correct.
- **Every future model doc owes one line** naming its second knob and why, before its Phase-1 runs.
  For EDR-1 that is a real open item, not a formality.
- No code changes. No registry rows. `adapters/ell1/config.yaml`'s frozen `gnn:` block is untouched,
  so ELL-1's config hashes — and therefore ADR-008's reference set — are unaffected.

## Revisit when

A model's graph makes *both* candidate knobs non-binding, or makes a third knob obviously the
binding one — in which case the question is whether "two knobs" is still the right budget, which is
a doc-00 §6.4 decision and needs its own ADR. Not revisited for DGF-1 once its Phase-1 runs begin.
