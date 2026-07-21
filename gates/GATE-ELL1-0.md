# GATE-ELL1-0 — ELL-1 tabular floor (RF + LR) logged

**Date assembled:** 2026-07-21
**Phase / doc:** ELL-1 Phase 0 — docs/01-elliptic-embedding-model-BUILD.md §4, §5, §8
**Assembled from registry rows:** ell1-20260721T221616Z-7e957080, ell1-20260721T221620Z-353b97ac, ell1-20260721T221624Z-8277e1a4, ell1-20260721T221628Z-b794fad3, ell1-20260721T221629Z-518d41da, ell1-20260721T221630Z-53fc88ad
**Git commit:** ea7868462f77a8f53f16d4d9f672e2074aa45749  ·  **Data snapshot:** elliptic-kaggle-v1

## Question
Is the ELL-1 pipeline standing (clean temporal graph) and are reproducible tabular-floor numbers logged on the steps 35–49 holdout?

## Pass condition (from the doc — quote it, do not invent)
> **Gate 0:** clean PyG graph at scale + reproducible baseline numbers logged. (§4, Phase 0)
>
> Gate 0 — *"Pipeline works, numbers reproducible."* (§7)

## Results (≥3 seeds — a gate does not pass on a single seed)

### Random Forest on raw 165 features (illicit = positive class, steps 35–49)

| seed | illicit F1 | illicit recall | illicit AUC | notes |
|------|-----------|----------------|-------------|-------|
| 0 | 0.8072 | 0.7211 | 0.9396 | ell1-20260721T221616Z-7e957080 |
| 1 | 0.8025 | 0.7202 | 0.9387 | ell1-20260721T221620Z-353b97ac |
| 2 | 0.8078 | 0.7258 | 0.9398 | ell1-20260721T221624Z-8277e1a4 |
| **mean ± std** | **0.8058 ± 0.0024** | **0.7224 ± 0.0024** | **0.9394 ± 0.0005** | |

### Logistic Regression on standardised 165 features (illicit = positive class, steps 35–49)

| seed | illicit F1 | illicit recall | illicit AUC | notes |
|------|-----------|----------------|-------------|-------|
| 0 | 0.3023 | 0.8753 | 0.8813 | ell1-20260721T221628Z-b794fad3 |
| 1 | 0.3023 | 0.8753 | 0.8813 | ell1-20260721T221629Z-518d41da |
| 2 | 0.3023 | 0.8753 | 0.8813 | ell1-20260721T221630Z-53fc88ad |
| **mean ± std** | **0.3023 ± 0.0000** | **0.8753 ± 0.0000** | **0.8813 ± 0.0000** | |

**Support:** train = 29894 labelled nodes (steps 1–34); test = 16670 labelled nodes (steps 35–49), of which 1083 illicit.

_All numbers copied from `experiments/registry.csv`. No cell is filled by estimate, extrapolation, or smoothing._

_Notes: **165** features per ADR-001 (`time_step` excluded); illicit is the positive class; accuracy deliberately not reported (doc §5); RF on raw features, LR on a train-fit StandardScaler; edges unused — this is the no-graph floor._

## Verdict
<!-- Left blank. Decided by the researcher, not by Claude. -->

---
_Verdict, seeds, and table are sacred once dated. Papers are assembled from gate files; nothing is reported that is not in one._
