# GATE-<MODEL>-<N> — <short title>

**Date assembled:** <YYYY-MM-DD>
**Phase / doc:** <e.g. ELL-1 Phase 0 — docs/01-elliptic-embedding-model-BUILD.md §X>
**Assembled from registry rows:** <run_ids or config_hashes>

## Question
<The single go/no-go question this gate answers. One sentence.>

## Pass condition (from the doc — quote it, do not invent)
<Exact threshold and metric as stated in the phase doc. Thresholds live in the
adapter config / doc, never in gbe/.>

## Results (≥3 seeds — a gate does not pass on a single seed)

| seed | <metric-1> | <metric-2> | … | notes |
|------|-----------|-----------|---|-------|
| <s0> |           |           |   |       |
| <s1> |           |           |   |       |
| <s2> |           |           |   |       |
| **mean ± std** |  ± |  ± |   |       |

_All numbers copied from `experiments/registry.csv`. No cell is filled by
estimate, extrapolation, or smoothing. Empty cells mean the run does not exist yet._

## Verdict
<!-- Left blank. Decided by the researcher, not by Claude. -->

---
_Verdict, seeds, and table are sacred once dated. Papers are assembled from gate
files; nothing is reported that is not in one._
