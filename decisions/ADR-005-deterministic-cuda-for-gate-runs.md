# ADR-005 — Runs are deterministic by default (CUDA scatter nondeterminism exceeded our effect sizes)

**Status:** accepted · erratum accepted 2026-10-08 (15:30) by coderback, at the end of this file
**Date:** 2026-07-24
**Deciders:** coderback
**Docs affected:** `CLAUDE.md` (integrity rule "≥3 seeds with variance on every gate number");
`docs/00-shared-core-graph-embedding-GUIDE.md` §7 (the temporal-holdout harness). **Both
amendments applied 2026-07-24 on acceptance**, §7 carrying a dated inline note of what changed
and the measured evidence for it.

> **Revision note.** The first draft of this ADR concluded that gate-grade runs must move to
> **CPU**, on the assumption that PyG's scatter aggregation had no deterministic CUDA kernel.
> That assumption was never tested, and it was wrong: `torch.use_deterministic_algorithms(True)`
> selects a deterministic scatter and makes CUDA bit-for-bit reproducible at ~5% wall-clock cost.
> The CPU proposal is retained below only as a rejected alternative. The lesson is recorded
> rather than quietly edited out — an untested assumption about the toolchain is the same class
> of error as an untested assumption about the data.

## Context

Pricing three candidate handicaps in the ELL-1 GNN path on the leak-free inner window
(`scripts/diagnose_ell1_inner.py`) produced effects of **+0.021 to +0.031 illicit-F1**, and a
rank-transform arm worth **+0.030** in one run and **+0.004** in a repeat of the same run. That
inconsistency prompted a direct check.

Same frozen ADR-003 config, same seed, three repeats **in one process**, CUDA:

| repeat | inner-window val illicit-F1 @0.5 |
|---|---|
| 0 | 0.915055 |
| 1 | 0.928322 |
| 2 | 0.893891 |
| **spread** | **0.034431** |

The measurement floor was therefore ≈0.034 F1 — **larger than every effect being measured**, and
comparable to the ±0.0306 3-seed band `gates/GATE-ELL1-1.md` reports for GraphSAGE. The
Phase-3 edge-scramble batch then showed the sampling problem is worse than nondeterminism alone:
6 seeds on the real-graph arm gave **±0.0790** (range 0.5345–0.7534) where 3 seeds had suggested
±0.0306.

**Cause.** GraphSAGE's neighbour aggregation is a scatter-reduce: each edge atomically adds into
its destination's accumulator, and CUDA atomics complete in scheduler-dependent order. Float
addition is not associative, so runs diverge at ~1e-7 and ~1200 optimizer steps amplify that into
a different model. `gbe/run/seeding.py` set `cudnn.deterministic` and `cudnn.benchmark`, but
those govern only cuDNN's convolution/RNN algorithm choice — a GNN's hot path contains neither,
so the flags were inert. They looked like a guarantee and never were.

**Measured remedies:**

| configuration | per 40-epoch run | reproducible |
|---|---|---|
| CUDA, previous defaults | 1.3 min | ✗ (σ ≈ 0.034) |
| **CUDA + `use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG`** | **1.3 min** | **✓ bit-for-bit** |
| CPU | 12.4 min | ✓ |

**This does not disturb the Gate-1 verdict.** The deficit there is `0.806 − 0.679 = 0.127`,
roughly 4× the noise floor. FAILED stands on any reading.

## Decision

1. **Determinism is the default for every run, not a gate-time special case.**
   `seed_everything(seed, deterministic=True)` calls
   `torch.use_deterministic_algorithms(True)`, and `gbe.run.seeding` sets
   `CUBLAS_WORKSPACE_CONFIG` **at module import** — it must precede the first cuBLAS handle, so
   setting it inside `seed_everything` (which runs later, at `RunSession` entry) would be a
   silent no-op. The module warns if it is imported after CUDA is already initialised, because
   that failure mode is otherwise invisible.
2. **Determinism is strict, not `warn_only`.** If a future backbone uses an op with no
   deterministic kernel, the run **raises**. A loud failure beats silently irreproducible gate
   numbers — that is the whole content of this ADR. `deterministic=False` exists only for
   throughput-bound exploration whose numbers are never reported.
3. **The escape hatch is recorded in the run row.** `RunSession` writes the determinism state
   into every registry row, so a row states for itself whether it was reproducible. Without this
   the hatch would be a hole in exactly the provenance this ADR exists to establish: a
   `deterministic=False` exploration run would otherwise be indistinguishable, after the fact,
   from a gate-grade one. An unusable-but-honest default would be better than a silent hatch;
   a recorded hatch is better than both.
4. **Seed counts are set per gate, not program-wide.** Determinism removes run noise but not
   *seed* sensitivity, and the 6-seed batch shows that sensitivity is large on the real-graph
   arm. This ADR therefore sets **no** global number: each gate's ADR fixes its own count from
   the measured variance of a **deterministic** batch, in advance (ADR-006 clause 4 does this
   for Gate 3, with a fixed two-stage 8→20 rule). `CLAUDE.md`'s "≥3 seeds" remains the
   constitutional floor — a minimum below which nothing is a gate number, not a target.

## Alternatives rejected

- **Move gate-grade runs to CPU.** The first draft's proposal, and now unnecessary: deterministic
  CUDA gives the same guarantee 9.5× faster. Retained as the fallback if a future model hits an
  op with no deterministic CUDA kernel.
- **`warn_only=True`.** Rejected: it downgrades an unsupported op to a warning and returns
  silently irreproducible numbers — reintroducing precisely the failure being fixed.
- **Accept the noise and report it as seed variance.** Rejected: it is not seed variance, and
  labelling it so in a gate table misrepresents uncertainty to a reviewer.
- **Re-run Gate 1 deterministically and supersede its numbers.** Rejected: the verdict does not
  change (0.127 ≫ 0.034), and re-running a dated gate invites the after-the-fact revision ADR-004
  exists to block. Future gates cite deterministic numbers; Gate 1 stands as recorded.

## Consequences

- `gates/GATE-ELL1-1.md` and `gates/GATE-ELL1-0.md` are **not edited** (sacred once dated). This
  ADR is the record that their 3-seed bands conflate seed variance with run nondeterminism, and
  any future write-up citing them must carry that caveat.
- Runs before this ADR are not bit-comparable with runs after it. Registry rows remain valid and
  append-only; the boundary is this ADR's commit.
- **The guarantee is same-machine, not universal — do not oversell it.** Bit-for-bit
  reproducibility holds for this GPU, driver, and library versions. A reviewer on different
  hardware will *not* reproduce `0.56554307`. What this buys is real and worth the change — a
  gate can be re-run and audited, and a number can be traced to an exact config — but a paper
  must claim "reproducible on the recorded environment", never "reproducible anywhere". The
  registry already pins config hash, commit, and data snapshot; the environment is the piece it
  does not pin.
- The three Gate-1 handicaps are closed as **unmeasurable at the precision then available**, not
  refuted. Re-opening any requires re-measurement under determinism.
- `scripts/diagnose_ell1_inner.py` and `experiments/ell1_inner_diag.csv` are retained as this
  finding's provenance. Their numbers are **validation** numbers and are never results.
- A guard test must assert that `seed_everything` leaves deterministic algorithms enabled and
  that `CUBLAS_WORKSPACE_CONFIG` is set — untested guards don't exist.
- DGF-1 inherits this. At 3M nodes the CPU fallback is not viable, so an op without a
  deterministic CUDA kernel becomes a blocking issue there rather than an inconvenience.

## Revisit when

A model hits an op with no deterministic CUDA implementation (decide: CPU fallback, or a
different formulation); or determinism's wall-clock cost rises materially above the ~5% measured
here at ELL-1's scale.

## Erratum, 2026-10-08 (accepted 2026-10-08 (15:30) by coderback)

**Source:** research audit 2026-10-07, §2
(`notebooks/audit/2026-10-07/README.md`; findings A2 E-B1, E-B2, E-S2 and E-S3).
The text above is not edited, so citations of its lines stay valid. **No clause changes.**

- **Clause 3 (`:72-77`), "a row states for itself whether it was reproducible", did not hold until
  `ca8181d` (2026-10-07).**
  - **Before:** `RunSession` read the state just after forcing determinism on, so every row
    recorded `deterministic=true` by construction. For rows written before `ca8181d`, the field
    shows only that determinism was switched on at the start. What shows that a run reproduces
    is a re-run, and that evidence differs by gate:
    - **ELL-1's Gate-0 and Gate-3 rows** were reproduced bit-for-bit by ADR-008's batches (49/49
      at `d614671` and again at `8060cf2`);
    - **DGF-1:** one validation pilot row was reproduced bit-for-bit by the amended gated path
      (ADR-013's re-certification, `dgf1-20260915T151348Z-0d390b65`), on a stack matching
      ADR-008's pin, which says nothing of the pilot's own stack. Two pre-echo validation rows
      also match same-config runs on every float (GATE-DGF1-0:58-59). The 24 Gate-1 test rows
      were never re-run, and ADR-016 has closed their window;
    - **ELL-1's Gate-1 rows** predate this ADR and ran without determinism, so no re-run can
      reproduce them (ADR-008:93-95).
  - **Since:** the field is true only when strict determinism held both at entry and at exit
    (`gbe/run/session.py:62`, `:73-74`). `tests/test_determinism_guard.py` fails if a run that
    switches determinism off, or to `warn_only`, is recorded true.
- **Clause 2 (`:68-71`), "strict, not `warn_only`", was neither recorded nor tested until
  `ca8181d`.** The recorded field now requires `warn_only` to be off (`gbe/run/seeding.py:64-65`).
- **Clause 4's example (`:81-82`) contradicts its rule.**
  - **The rule:** a count derived from the variance of a deterministic batch.
  - **The example:** ADR-006's 8→20 design was fixed in advance, with no deterministic batch, and
    ADR-006 says the provisional σ "cannot size the batch" (ADR-006:109-110).
  - **The first count derived as the rule requires** is ADR-012's `Stage-1 seeds: 8`, from its
    validation pilots. Gates pre-registered after ADR-018's acceptance follow ADR-018.
- **The context labels a range as a standard deviation (`:33-36`, `:52`).**
  - **The figures:** the three repeats (0.915055, 0.928322, 0.893891) span 0.034431. Their sample
    standard deviation is 0.0174.
  - **`:35-36`** compares that range with GATE-ELL1-1's "±0.0306", a standard deviation computed
    with `ddof=0` (ADR-006:71-72).
  - **At `:56-57`,** Gate 1's deficit of 0.127 is about 7 standard deviations of the repeats, not 4.
  - **What stands:** the decision. The repeats were real nondeterminism, which bit-for-bit runs
    removed.
