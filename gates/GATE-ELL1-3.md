# GATE-ELL1-3 — is the gain structural? (defending ablations)

**Date assembled:** 2026-07-25
**Phase / doc:** ELL-1 Phase 3 — docs/01-elliptic-embedding-model-BUILD.md §4, §7; doc-00 §8
**Git commit:** 1a852a356ea5081e2f5d95ea3726bff537cd08fe  ·  **Data snapshot:** elliptic-kaggle-v1  ·  **Config:** ADR-003  ·  **Criteria:** ADR-006
**Determinism:** all rows `deterministic=true`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` (ADR-005) — these numbers are bit-for-bit reproducible on the recorded environment.
**Batch:** stage 1 of ADR-006 clause 4 — 8 seeds × 5 arms, all `git_dirty=false`. No stage-2 top-up was triggered.

> **Disclosure — this is a partial pre-registration, and one clause is not blind.**
> Reproduced verbatim from ADR-006, which requires this gate file to carry it.
>
> ADR-004 worked because it fixed the bar while **no** test-window number existed. That is not
> the situation here. A **provisional edge-scramble batch had already been run and seen** before
> the bar was set: 6 seeds per arm on CUDA, `git_dirty=true`, run ids
> `ell1-20260724T202110Z-70e9fab9` … `ell1-20260724T202947Z-f7dc7b7b`, observed real illicit-F1
> 0.6763 ± 0.0790 vs scrambled 0.5597 ± 0.0288.
>
> - **Clause 1 (edge-scramble) is _informed_, not blind.** Its bar is derived from a principle —
>   directional sign plus statistical resolvability — and **not** from the observed 0.117
>   magnitude. No threshold in ADR-006 is a number the provisional result happens to clear.
> - **Clauses 2 and 3 (random-graph control, GNN-removed) are genuinely blind.** Neither had been
>   run when the criteria were fixed. They carry ADR-004's full force.
> - The provisional rows remain in the append-only registry, unused for this gate.
>
> A reader is entitled to discount clause 1 accordingly.

## Question
Gate 1 established that GraphSAGE does not beat the RF tabular floor. It did **not** establish whether the GNN's score comes from the graph or from the 165 node features alone — features 94–164 are hand-built one-hop aggregates, so a GNN could score well while ignoring message passing. Does destroying the structure cost performance?

## Pass condition (doc-01 §4/§7 as amended by ADR-006; operationalised in ADR-006)
> **Gate 3:** edge-scramble drops **materially and resolvably below the real-graph run**; real graph ≫ random graph; removing the GNN removes the gain.

A difference is **resolvable** iff `mean(real) − mean(ablated) > 2 × SE_diff`, where `SE_diff = sqrt(s_r²/n_r + s_a²/n_a)` and `s` is the **sample** standard deviation (ddof=1, pinned by ADR-006). Primary metric is illicit-F1; recall and AUC are reported but are **not** pass/fail inputs. Gate 3 passes iff clauses 1–3 all hold.

## Results (8 seeds per arm — a gate does not pass on a single seed)

### real

_The true graph. Every clause below is measured against this arm, re-run in the same batch — never against the Gate-1 rows, whose code path and determinism settings differ._

| seed | illicit F1 | illicit recall | illicit AUC | run_id |
|------|-----------|----------------|-------------|--------|
| 0 | 0.5655 | 0.5577 | 0.9096 | ell1-20260725T095727Z-a14a4337 |
| 1 | 0.6858 | 0.6842 | 0.9179 | ell1-20260725T095828Z-d0905a25 |
| 2 | 0.6325 | 0.7018 | 0.9194 | ell1-20260725T095926Z-e4e28e51 |
| 3 | 0.6618 | 0.5863 | 0.9226 | ell1-20260725T100023Z-10dcacbb |
| 4 | 0.6924 | 0.5716 | 0.9314 | ell1-20260725T100122Z-ec761169 |
| 5 | 0.5720 | 0.5226 | 0.9023 | ell1-20260725T100219Z-632725eb |
| 6 | 0.7624 | 0.6593 | 0.9270 | ell1-20260725T100316Z-73a257f5 |
| 7 | 0.7304 | 0.6353 | 0.9271 | ell1-20260725T100416Z-b10907e4 |
| **mean ± std** | **0.6629 ± 0.0702** | **0.6148 ± 0.0646** | **0.9197 ± 0.0097** | |

### scrambled

_Node identities permuted within each time step. Topology, degree sequence and the temporal split are preserved exactly; only the correspondence between graph position and node content is destroyed. **Clause 1, gated.**_

| seed | illicit F1 | illicit recall | illicit AUC | run_id |
|------|-----------|----------------|-------------|--------|
| 0 | 0.5628 | 0.5688 | 0.8813 | ell1-20260725T100514Z-aeaed570 |
| 1 | 0.5857 | 0.6057 | 0.8884 | ell1-20260725T100611Z-a91278df |
| 2 | 0.5219 | 0.7147 | 0.9044 | ell1-20260725T100708Z-71e14e3e |
| 3 | 0.5148 | 0.4894 | 0.8727 | ell1-20260725T100805Z-6c07b33f |
| 4 | 0.5750 | 0.6177 | 0.8951 | ell1-20260725T100901Z-5e163a8e |
| 5 | 0.5402 | 0.6602 | 0.8933 | ell1-20260725T100958Z-75f56ba2 |
| 6 | 0.6002 | 0.6611 | 0.8958 | ell1-20260725T101054Z-a9c2daae |
| 7 | 0.5882 | 0.6048 | 0.8856 | ell1-20260725T101151Z-0bb71b0e |
| **mean ± std** | **0.5611 ± 0.0320** | **0.6153 ± 0.0678** | **0.8896 ± 0.0098** | |

### random

_Erdős–Rényi within each time step: same per-step edge count, degree sequence destroyed. **Clause 2, gated.**_

| seed | illicit F1 | illicit recall | illicit AUC | run_id |
|------|-----------|----------------|-------------|--------|
| 0 | 0.5852 | 0.6187 | 0.8782 | ell1-20260725T101249Z-eef0ec62 |
| 1 | 0.6305 | 0.6168 | 0.8899 | ell1-20260725T101351Z-e01e213c |
| 2 | 0.6305 | 0.5845 | 0.8980 | ell1-20260725T101456Z-ae5bb2de |
| 3 | 0.4829 | 0.4875 | 0.8715 | ell1-20260725T101602Z-5813cebb |
| 4 | 0.5390 | 0.6251 | 0.8905 | ell1-20260725T101712Z-bde6a105 |
| 5 | 0.5976 | 0.4903 | 0.8684 | ell1-20260725T101822Z-f93416c8 |
| 6 | 0.6046 | 0.5337 | 0.8893 | ell1-20260725T101933Z-c2e025f7 |
| 7 | 0.6378 | 0.5845 | 0.8861 | ell1-20260725T102044Z-ab4c7db5 |
| **mean ± std** | **0.5885 ± 0.0533** | **0.5676 ± 0.0566** | **0.8840 ± 0.0103** | |

### no_edges

_Empty edge set — message passing disabled at identical parameter count and training budget. **Clause 3, gated.**_

| seed | illicit F1 | illicit recall | illicit AUC | run_id |
|------|-----------|----------------|-------------|--------|
| 0 | 0.5802 | 0.5596 | 0.8844 | ell1-20260725T102150Z-b579551f |
| 1 | 0.6198 | 0.5743 | 0.8661 | ell1-20260725T102218Z-b9bfd000 |
| 2 | 0.6115 | 0.7128 | 0.9125 | ell1-20260725T102247Z-624ba875 |
| 3 | 0.5299 | 0.4340 | 0.8898 | ell1-20260725T102316Z-d415740e |
| 4 | 0.5731 | 0.6639 | 0.8895 | ell1-20260725T102345Z-316e939c |
| 5 | 0.6120 | 0.5753 | 0.8814 | ell1-20260725T102414Z-9b7f81c7 |
| 6 | 0.6612 | 0.5965 | 0.9005 | ell1-20260725T102442Z-38a10704 |
| 7 | 0.5551 | 0.5863 | 0.8750 | ell1-20260725T102509Z-a2e0786d |
| **mean ± std** | **0.5928 ± 0.0415** | **0.5878 ± 0.0813** | **0.8874 ± 0.0145** | |

### config

_Degree-preserving rewire by double-edge swaps: every node keeps its own degree and its own features, and only *who it connects to* is randomised. **Reported, not gated** (ADR-006 clause 5)._

| seed | illicit F1 | illicit recall | illicit AUC | run_id |
|------|-----------|----------------|-------------|--------|
| 0 | 0.5716 | 0.5937 | 0.8921 | ell1-20260725T102544Z-821c4c09 |
| 1 | 0.5230 | 0.7091 | 0.8982 | ell1-20260725T102651Z-a9018ab5 |
| 2 | 0.6635 | 0.5826 | 0.8942 | ell1-20260725T102754Z-8e67d93f |
| 3 | 0.5362 | 0.5125 | 0.8794 | ell1-20260725T102911Z-cd05ac05 |
| 4 | 0.5762 | 0.5780 | 0.8911 | ell1-20260725T103018Z-80f7c5a4 |
| 5 | 0.5832 | 0.5762 | 0.8878 | ell1-20260725T103128Z-b4496beb |
| 6 | 0.6342 | 0.6427 | 0.9076 | ell1-20260725T103236Z-afd6655b |
| 7 | 0.4619 | 0.6630 | 0.8759 | ell1-20260725T103342Z-e356d176 |
| **mean ± std** | **0.5687 ± 0.0634** | **0.6072 ± 0.0613** | **0.8908 ± 0.0101** | |

**Support:** train = 29894 labelled nodes (steps 1–34); test = 16670 labelled nodes (steps 35–49), of which 1083 illicit. Every arm rewires **within a time step**, so no ablation forges an edge across the 34/35 cutoff and the strict inductive protocol is identical across arms.

## ADR-006 criterion check — mechanical, not the verdict

Real-graph arm: **0.6629 ± 0.0702** illicit-F1 (n=8).

| arm | role | drop vs real | 2×SE_diff | resolvable | Welch p |
|---|---|---|---|---|---|
| scrambled | **clause 1, gated** | +0.1017 | 0.0546 | **yes** | 0.0041 |
| random | **clause 2, gated** | +0.0744 | 0.0624 | **yes** | 0.0329 |
| no_edges | **clause 3, gated** | +0.0700 | 0.0577 | **yes** | 0.0329 |
| config | reported | +0.0941 | 0.0669 | **yes** | 0.0139 |

**All gated clauses (1–3): PASS** (3/3 resolvable). This is the pre-registered mechanical result; the verdict below is the researcher's.

Two properties of the conjunction, so the result is not over-read: the three tests share the same `real` arm, so they are **correlated, not independent**; and requiring all three makes this gate stricter than any single clause.

_All numbers read from `experiments/registry.csv`. No cell is filled by estimate, extrapolation, or smoothing. Arms were matched to rows by rebuilding each (arm, seed) config hash, not by position._

## Verdict

**PASSED** — 2026-07-25, coderback.

The gain is structural. All three pre-registered clauses (ADR-006) are positive and resolvable on
illicit-F1 against a real-graph arm re-run in the same batch: edge-scramble **+0.1017**
(p=0.0041), random-graph **+0.0744** (p=0.0329), GNN-removed **+0.0700** (p=0.0329). 8 seeds per
arm, deterministic (ADR-005), `git_dirty=false`. The result survives the stricter exact Welch
critical values (2.16–2.23×SE rather than 2.00×SE), so it does not rest on the one loosening
ADR-006 admitted to. No stage-2 top-up was triggered.

**Two qualifications are part of this verdict, not footnotes to it.**

**1. Clause 1 is informed, not blind** — see the disclosure above. Clauses 2 and 3 carry ADR-004's
full force; clause 1 does not, and a reader is entitled to discount it. The gate does not rest on
clause 1 alone.

**2. This does not rescue Gate 1, and must never be cited as though it does.** Both of the
following are true at once:

| claim | status |
|---|---|
| the GNN's performance depends on graph structure | **supported here** — +0.070 over no-graph, resolvable |
| structure beats the tabular floor | **failed** (GATE-ELL1-1) — 0.663 vs RF 0.806 |

*Structure demonstrably contributes, and demonstrably is not enough.* What this gate licenses is
the claim that ELL-1 is not merely exploiting features — it retires the "critical finding" branch
of doc-01 §4. It licenses nothing about the model being good on Elliptic.

**What the ablations establish beyond the pass.** Gate 1's failure now decomposes, which is the
useful output for P0. The strict inductive protocol is exonerated — zero of 468,710 edges cross
the 34/35 cutoff, so induction drops nothing. Absence of structural signal is excluded by the
three clauses above. What remains is that the *neural* feature path sits ~0.21 F1 below a tree on
identical features (`no_edges` 0.5928 vs RF 0.806), and the +0.070 structural gain does not cover
that deficit. The honest P0 sentence is therefore **"trees beat neural nets on this tabular data;
the graph helps, but not by enough"** — not "structure did not help".

**The reported arm sharpens it.** `config` — every node's degree preserved exactly, only wiring
randomised — falls to 0.5687, level with full scramble and *below* both the random graph and no
graph at all. Preserving degree rescues nothing, so message passing reads something the hand-built
one-hop aggregates (features 94–164) do not contain. That is the claim which survives the standing
objection to this dataset. It was **not** a gate input (ADR-006 clause 5) and is recorded as a
diagnostic only.

**An ordering nobody pre-registered:** `no_edges` (0.5928) and `random` (0.5885) both score above
`scrambled` (0.5611) and `config` (0.5687) — a plausible-but-wrong graph hurts more than no graph
at all, because message passing confidently aggregates misleading neighbours. The arms are
therefore not monotonic in how much structure was destroyed and must not be read as a severity
ladder.

---
_Verdict, seeds, and table are sacred once dated. Papers are assembled from gate files; nothing is reported that is not in one._
