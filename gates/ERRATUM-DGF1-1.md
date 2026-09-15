# ERRATUM-DGF1-1 — withdrawn passages in GATE-DGF1-1

**Dated:** 2026-09-15
**Decision:** `decisions/ADR-013-dgf1-sensitivity-view-inputs.md` (accepted 2026-09-15), clause 2
**Concerns:** `gates/GATE-DGF1-1.md`, signed PASSED 2026-09-14. That file is **not edited**; dated
gate files never are.

**This is not a gate file and carries no verdict.** Gate 1's gated result is unaffected: every gated
number, the clause-6 check, the paired bootstrap and the verdict's PASS stand exactly as recorded.

## What is withdrawn, and why

GATE-DGF1-1 reports DGF-1 under two sensitivity views, window-only and first appearance (ADR-011
clause 4). The code built DGF-1's node inputs under **both** views from the graph as of step 821:
the edge-type histogram, degree and recency. That breaks ADR-011 clause 3, which requires every
edge-derived feature to come from the edge set of the view it feeds. The numbers are exactly what
the code computed, but they do not measure what their labels say.

| GATE-DGF1-1 passage | lines | status |
|---|---|---|
| *Scoring-view sensitivity*: the window-only and first-appearance rows, and the implementation-gap note beneath them | 133–141 | **withdrawn**: cite them for no inference |
| *Verdict*, qualification 3: the first-appearance and gated AUPRC figures it compares | 209–213 | **the figures are withdrawn**; the qualification's conclusion, that the question is unsettled, stands; its closing "that pre-registered comparison … is owed" is **superseded** — the comparison moves to ADR-013 clause 5, and no run is pending under ADR-011 |
| *Verdict*, claims table: "the gain does not depend on activity after users appear — **not established**" | 224 | **the status stands**; its stated reason ("floor unscored under the first-appearance view") is superseded by: no valid row bears on the claim |

The registry keys behind these passages are withdrawn too: `window_only_*` and `first_appearance_*`
in all 24 DGF-1 GNN rows (retune, pilot and gate). The rows stay, because the registry is
append-only.

## How to state the claim now

> **Post-appearance activity: not established.** The first-appearance figure cited in the verdict is
> withdrawn (ADR-013); no valid row bears on this claim.

That holds until the matched-time pre-registration in ADR-013 clause 5 produces rows. Any write-up
drawing on GATE-DGF1-1 reads this notice with it.
