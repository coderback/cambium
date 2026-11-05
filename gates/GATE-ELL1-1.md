# GATE-ELL1-1 — structure vs the tabular floor (strict inductive)

**Date assembled:** 2026-07-24
**Phase / doc:** ELL-1 Phase 1 — docs/01-elliptic-embedding-model-BUILD.md §4, §5, §7
**Assembled from registry rows:** ell1-20260724T183214Z-a0027115, ell1-20260724T183302Z-3f5b51e8, ell1-20260724T183347Z-518d36c9, ell1-20260724T183431Z-2b490f17, ell1-20260724T183521Z-2bb54922, ell1-20260724T183610Z-97c3148e
**Git commit:** 2306fa4e2d57a92884db1398c3689c07ba08e85b  ·  **Data snapshot:** elliptic-kaggle-v1  ·  **Config:** ADR-003

## Question
Under the strict inductive protocol, does the structure-aware GNN beat the tabular floor (RF on raw 165 features) convincingly, and at least match GCN, on steps 35–49?

## Pass condition (from the doc — quoted; operationalised in ADR-004)
> **Gate 1 (the load-bearing gate):** the GNN model **beats the tabular floor (Random Forest / LR on raw 165 features) convincingly**, and at least **matches/beats plain GCN**, on the held-out steps 35–49. (§4, Phase 1)
>
> Gate 1 — *"GNN > RF on raw features and ≥ GCN on steps 35–49."* (§7)

ADR-004 pre-registers "convincingly" as **non-overlapping mean±std F1 bands** vs the RF floor, **and** GraphSAGE mean ≥ GCN mean; ≥3 seeds; strict inductive.

## Results (≥3 seeds — a gate does not pass on a single seed)

### RF on raw 165 features — the floor (from GATE-ELL1-0, no graph)

| **mean ± std** | **0.8058 ± 0.0024** | **0.7224 ± 0.0024** | **0.9394 ± 0.0005** |

(F1 | recall | AUC — carried from the Gate-0 clean batch, not recomputed here.)

### GCN (capacity-matched comparator) (illicit = positive class, steps 35–49, strict inductive)

| seed | illicit F1 | illicit recall | illicit AUC | run_id |
|------|-----------|----------------|-------------|--------|
| 0 | 0.6752 | 0.6824 | 0.9154 | ell1-20260724T183431Z-2b490f17 |
| 1 | 0.5466 | 0.4303 | 0.8915 | ell1-20260724T183521Z-2bb54922 |
| 2 | 0.6781 | 0.6223 | 0.9057 | ell1-20260724T183610Z-97c3148e |
| **mean ± std** | **0.6333 ± 0.0613** | **0.5783 ± 0.1075** | **0.9042 ± 0.0098** | |

### GraphSAGE (ELL-1, frozen ADR-003 config) (illicit = positive class, steps 35–49, strict inductive)

| seed | illicit F1 | illicit recall | illicit AUC | run_id |
|------|-----------|----------------|-------------|--------|
| 0 | 0.6390 | 0.5614 | 0.9210 | ell1-20260724T183214Z-a0027115 |
| 1 | 0.6848 | 0.6270 | 0.9059 | ell1-20260724T183302Z-3f5b51e8 |
| 2 | 0.7131 | 0.7036 | 0.9279 | ell1-20260724T183347Z-518d36c9 |
| **mean ± std** | **0.6790 ± 0.0306** | **0.6307 ± 0.0581** | **0.9182 ± 0.0092** | |

**Support:** train = 29894 labelled nodes (steps 1–34); test = 16670 labelled nodes (steps 35–49), of which 1083 illicit. Message passing: train over the ≤34 induced subgraph; eval over the 35–49 induced subgraph only (no full-graph forward).

## Pre-registered criterion check (ADR-004) — mechanical, not the verdict

- **RF floor band** (Gate-0 clean batch): [0.8034, 0.8082]  (mean ± std = 0.8058 ± 0.0024)
- **GraphSAGE band**: [0.6484, 0.7095]  (mean ± std = 0.6790 ± 0.0306)
- **GCN mean**: 0.6333

1. Beats RF, non-overlapping bands — `(GS_mean − GS_std) > (RF_mean + RF_std)`: `0.6484 > 0.8082` → **FAIL**
2. At least matches GCN — `GS_mean ≥ GCN_mean`: `0.6790 ≥ 0.6333` → **PASS**

**Both clauses (ADR-004): FAIL.** This is the pre-registered mechanical result; the verdict below is the researcher's.

_All GNN numbers copied from `experiments/registry.csv`. No cell is filled by estimate, extrapolation, or smoothing. RF floor carried from GATE-ELL1-0._

## Verdict
**FAILED — Gate 1 not passed.** 2026-07-24, coderbackf.

Per the pre-registered criterion (ADR-004): GraphSAGE illicit-F1 **0.679 ± 0.031** (band
[0.648, 0.710]) sits entirely below the RF floor **0.806 ± 0.002** (band [0.803, 0.808]) —
clause 1 (beats RF, non-overlapping bands) **fails**; clause 2 (≥ GCN, 0.679 ≥ 0.633) passes;
overall **FAIL**. Reproduced across two independent batches; the clean batch (commit
`2306fa4`, `dirty=false`) is cited above.

This is the **pre-registered informative outcome, not a thesis refutation** (doc-01 §0): under
the strict inductive protocol the GNN does not beat the tabular floor, because Elliptic's
features 94–164 already encode one hop and RF on them is a hard 0.806 bar. The regimes the
thesis rests on (node text + typed edges) are untested here by design — EDR-1 proceeds
regardless, and this lesson is written into P0. Per doc §4 ("stop and diagnose"), the next
step is characterisation: edge-scramble (does the GNN use the graph at all?) and
GNN-on-local-94 vs RF-on-165.

---
_Verdict, seeds, and table are sacred once dated. Papers are assembled from gate files; nothing is reported that is not in one._
