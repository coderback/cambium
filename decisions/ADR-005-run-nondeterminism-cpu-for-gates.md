# ADR-005 — Gate-grade runs are computed on CPU (CUDA run nondeterminism exceeds our effect sizes)

**Status:** proposed
**Date:** 2026-07-24
**Deciders:** coderback
**Docs affected:** `CLAUDE.md` (integrity rule "≥3 seeds with variance on every gate number");
`docs/00-shared-core-graph-embedding-GUIDE.md` §7 (the temporal-holdout harness). Amend both
as part of accepting this ADR.

## Context

The Gate-1 implementation review proposed three candidate handicaps in the ELL-1 GNN path
(no model selection; an uncalibrated 0.5 threshold under a train→test prevalence shift; heavy-
tailed features into the MLP). Measuring their headroom on the leak-free inner window
(train 1–29, validate 30–34; `scripts/diagnose_ell1_inner.py`) produced effects of
**+0.021 to +0.031 illicit-F1**, with a rank-transform arm that looked worth **+0.030** in one
run and **+0.004** in a repeat of the same run.

That inconsistency prompted a direct check. Training the **same** frozen ADR-003 config at the
**same** seed, three times **in one process** on CUDA:

| repeat | inner-window val illicit-F1 @0.5 |
|---|---|
| 0 | 0.915055 |
| 1 | 0.928322 |
| 2 | 0.893891 |
| **spread** | **0.034431** |

The same config on **CPU** (5 epochs, seed 0, two repeats) returned `0.74519846` both times —
spread exactly **0.0**, bit-for-bit.

So the measurement floor on CUDA is ≈0.034 F1, which is **larger than every effect the
diagnostic set out to resolve**, and comparable to the ±0.0306 3-seed band that
`gates/GATE-ELL1-1.md` reports for GraphSAGE. Two consequences follow. First, none of the
three handicaps is refuted — they are *unmeasurable* at current precision. Second, and more
seriously, the "seed variance" column in the Gate-0/Gate-1 tables substantially reflects run
nondeterminism rather than sensitivity to the seed, so those bands claim more precision than
the instrument delivers.

**Cause.** `gbe/run/seeding.py` sets `cudnn.deterministic = True` and `cudnn.benchmark = False`,
but never calls `torch.use_deterministic_algorithms(...)` and never sets
`CUBLAS_WORKSPACE_CONFIG`. More fundamentally, PyG's scatter-based neighbour aggregation — the
dominant operation in a GNN — reduces via CUDA atomics whose accumulation order varies run to
run, and the cuDNN flags do not govern it. The existing flags were never going to make a GNN
reproducible; they only looked like they would.

**This does not disturb the Gate-1 verdict.** The deficit there is
`0.806 − 0.679 = 0.127`, roughly 4× the 0.034 noise floor. FAILED stands on any reading.

## Decision

1. **Load-bearing gate numbers are computed on CPU**, where determinism is verified. Numbers
   that a gate verdict or a paper table rests on must be exactly reproducible from
   `experiments/registry.csv` — for a repo intended as an open release, reproducibility of six
   decisive runs outranks their wall-clock cost. HPO sweeps, smokes, and exploration stay on
   GPU, where run noise is tolerable and is not reported as a result.
2. **`gbe/run/seeding.py` is hardened and, more importantly, honest**: attempt
   `torch.use_deterministic_algorithms(..., warn_only=True)` and set `CUBLAS_WORKSPACE_CONFIG`,
   and state in the docstring that **CUDA scatter aggregation remains nondeterministic
   regardless**, so no future reader mistakes the cuDNN flags for a guarantee. The documented
   caveat is the deliverable; the flags are secondary.
3. **"≥3 seeds" is qualified, not weakened.** Three seeds measure seed sensitivity only on a
   deterministic backend. Any GNN number reported from CUDA requires **≥20 repeats** (σ≈0.034
   ⇒ standard error ≈0.008) before an effect below ~0.03 may be claimed.

**Prerequisite before adopting (1):** measure wall-clock for one full 40-epoch CPU run on steps
1–34. It has not been measured, and the 5-epoch inner-window probe does not extrapolate to it.
If CPU proves impractical, fall back to (3) — ≥20 GPU repeats — which achieves comparable
precision without exact reproducibility.

## Alternatives rejected

- **Force full CUDA determinism via `torch.use_deterministic_algorithms(True)`.** Rejected as a
  primary fix: PyG's scatter reductions have no deterministic CUDA implementation, so this
  either raises at runtime or (with `warn_only=True`) warns and changes nothing. Kept only as
  the honest-documentation half of decision (2).
- **Accept the noise and report it as seed variance.** Rejected: it is not seed variance, and
  labelling it so in a gate table misrepresents the uncertainty to a reviewer — the precise
  failure the gate files exist to prevent.
- **Re-run Gate 1 on CPU now and supersede the numbers.** Rejected: the verdict does not change
  (0.127 ≫ 0.034), and re-running a dated gate invites exactly the after-the-fact revision
  ADR-004 was written to block. A future gate may cite CPU numbers; this one stands as recorded.
- **Raise seed count on GPU and keep everything there.** Not rejected — retained as the
  documented fallback if CPU wall-clock proves prohibitive.

## Consequences

- `gates/GATE-ELL1-1.md` and `gates/GATE-ELL1-0.md` are **not edited** (sacred once dated). This
  ADR is the record that their 3-seed bands conflate seed variance with run nondeterminism; any
  future write-up citing them must carry that caveat.
- The three Gate-1 handicaps are closed as **unmeasurable at current precision**, not refuted.
  Re-opening any of them requires the fixed measurement protocol first.
- `scripts/diagnose_ell1_inner.py` and `experiments/ell1_inner_diag.csv` are retained as the
  provenance for this finding. Their numbers are **validation** numbers and are never results.
- DGF-1 inherits this: its Gate-1 AUC comparison against a leaderboard needs the same
  reproducibility guarantee, at a scale where CPU runs may not be viable — decide there.

## Revisit when

CPU wall-clock is measured and proves prohibitive (fall back to ≥20 GPU repeats); or a PyG
release ships deterministic scatter kernels; or DGF-1's scale makes CPU gate runs impossible,
at which point the repeat-count route becomes the program default.
