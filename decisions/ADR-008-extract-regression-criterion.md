# ADR-008 — What "ELL-1 still passes" means after EXTRACT: bit-for-bit reproduction, not re-passing

**Status:** accepted
**Date:** 2026-07-26 · **accepted** 2026-07-26
**Deciders:** coderback
**Docs affected — all amendments applied 2026-07-26 on acceptance:**
`docs/02-dgraph-fin-embedding-model-BUILD.md` §4 (Phase 0 / Gate 0), §6 (EXTRACT risk row), §7
(Gate 0 row), **§8 step 4**; `docs/timeline.md` DGF-1 Gate-0 row.

> **Enumeration corrected 2026-07-26,** found by auditing the clauses programmatically rather than
> by eye: **§8 step 4** ("Refactor `ELL-1` onto `gbe/` and confirm it **still passes its gates**")
> was a *fourth* live statement of the unsatisfiable condition, missed by the original three-site
> enumeration. Amended. The phrase survives elsewhere in doc-02 only inside §4's quoted record of
> the previous wording, which is intentional.

**Artifacts created on acceptance:** `experiments/extract_reference_env.txt` (clause 4 environment
pin) and `scripts/run_extract_reference.py` (clause 3 GCN reference runner, which refuses to start
on a dirty tree).

**Artifacts created on implementation (2026-07-26):**
`experiments/extract_reference_manifest.json` — the 49 reference identities, **frozen pre-refactor**
(see the amendment note under *Consequences*); `scripts/freeze_extract_reference.py` (one-shot, wrote
it); `scripts/check_extract_regression.py` (the acceptance test);
`tests/test_extract_reference_manifest.py` + `tests/test_extract_regression_checker.py` (22 guards).

## Context

The EXTRACT milestone (doc-00 §9, doc-02 §4 Phase 0) refactors `ELL-1` onto a shared `gbe/` core
while generalising `DataSource`/`FeatureEncoder` and adding inductive sampling. doc-02 states the
acceptance condition three times, identically:

> **Gate 0:** sampled training runs without OOM; baseline AUCs logged; **`ELL-1` still passes its
> gates on the refactored core.** (§4)
> | EXTRACT introduces regressions in ELL-1 | **Re-run ELL-1 gates after refactor**; CI the core | (§6)
> | 0 | … | No OOM; AUCs logged; **ELL-1 passes** | (§7)

**That condition is unsatisfiable as written, and its failure mode is silent.** ELL-1's actual
recorded state is:

| gate | verdict | date |
|---|---|---|
| Gate 0 | **PASSED** | 2026-07-23 |
| Gate 1 | **FAILED** (ADR-004 clause 1) | 2026-07-24 |
| Gate 2 | deferred, never run (doc-01 §6) | — |
| Gate 3 | **PASSED** | 2026-07-25 |

"Still passes its gates" has no truth value for Gate 1, which did not pass. Read literally the
condition can never be met; read charitably it silently narrows to Gates 0 and 3 — and even there
it is far weaker than it sounds, because **Gate 3 could still pass while every underlying number
moved.** Its clauses are inequalities between arms; a refactor that shifted all five arms
together would satisfy them while having changed the model. A pass/fail re-check cannot detect
that, which is precisely the class of bug a refactor produces.

The question a refactor actually poses is not *"does it still pass?"* but **"does it still compute
the same thing?"** — and that has an exact answer available here.

**Two measurements taken 2026-07-26, before any refactoring exists:**

1. **Gate 3's rows reproduce bit-for-bit across independent batches.** The Gate-3 `real` arm and
   the local-94 `gnn_all165` arm are the same configuration run by two different scripts about an
   hour apart. All 8 seeds are identical to full float precision on `illicit_f1` and `illicit_auc`
   — e.g. `0.5655430711610487` vs `0.5655430711610487`, 8/8.
2. **Gate 0's rows still reproduce bit-for-bit today,** five days after they were recorded:
   re-running RF (seeds 0, 1) and LR (seed 0) through `fit_predict`/`evaluate` on current code
   returns `0.807235142118863`, `0.8024691358024691`, `0.3022959183673469` — exactly the recorded
   values. (Run without `RunSession` so the append-only registry was not written to.)

So exact equality is not an aspiration: it is the *observed* behaviour of this codebase under
ADR-005 determinism. A criterion looser than equality would knowingly tolerate drift it is capable
of detecting.

**Why this must be fixed now.** Deciding what counts as "no regression" *after* seeing the
post-refactor numbers is setting a threshold to fit a result — the exact move ADR-004 and ADR-006
exist to forbid, and the pressure would be acute, since the alternative to accepting a small
discrepancy is debugging a large refactor. This ADR is written while **no refactored-core number
exists and no refactoring has begun**: blind in ADR-004's full sense.

## Decision

**Replace the acceptance condition.** For EXTRACT, `ELL-1`'s Gate-0 condition is *not* that its
gates re-pass; it is that **the refactored core reproduces ELL-1's recorded numbers bit-for-bit
over a fixed reference set, in an unchanged environment.** ELL-1's gate verdicts are historical
facts on dated files and are neither re-earned nor re-litigated.

### 1. The reference set (fixed here, before the refactor)

| source | rows | metrics compared | bar |
|---|---|---|---|
| **Gate 0** clean batch (`git_dirty=false`) | 6 (RF×3, LR×3) | `illicit_f1`, `illicit_recall`, `illicit_precision`, `illicit_auc` | **exact equality** |
| **Gate 3** batch (`git_dirty=false`, `deterministic=true`) | 40 (5 arms × 8 seeds) | same four | **exact equality** |
| **GCN reference** — *to be created pre-refactor*, see §3 | 3 (GCN × seeds 0,1,2) | same four | **exact equality** |

**Gate 1's rows are deliberately excluded as a reference.** They predate ADR-005 and ran on
non-deterministic CUDA; their recorded ±0.031 is largely run noise, not seed variance (three
*identical* repeats spanned 0.034). Equality against them is impossible and a band around them
would be uninformative. The GraphSAGE path is instead covered — better — by Gate 3's `real` arm,
which is the same model, config and protocol run deterministically at 8 seeds. **This is a
coverage improvement, not a gap:** Gate 1's rows were never reproducible and cannot be made so
retroactively.

All 40 Gate-3 rows are included rather than a cheaper subset because the five arms exercise all
four ablation functions in `gbe/eval/ablations.py` — core code the refactor touches directly.

### 2. The bar is equality, not a tolerance

A differing row is a **regression**, full stop. There is no tolerance band, because no principled
width exists: any band wide enough to absorb genuine nondeterminism is wide enough to hide a
refactor bug, and equality is measured to be achievable. **A mismatch is investigated to root
cause — never re-run until it matches** (that is the optional-stopping error in another costume).

If a discrepancy turns out to be an *intended* change — extraction uncovers a real defect in
ELL-1's code and fixing it moves the numbers — then: the fix requires **its own ADR** stating what
was wrong and what moved; the new numbers are recorded as a new post-refactor batch; and
**`gates/GATE-ELL1-*.md` are not edited.** Dated gate files are sacred; a corrected result is a
new result, not a revision of an old one.

**Key-set note (ADR-007).** Post-refactor runs emit `auprc`, which the Gate-0 and Gate-3 reference
rows do not carry. Comparison is over the four metrics listed above — **the presence of a new key
is not a regression.** The GCN reference in §3, being created after ADR-007, will carry `auprc`
and is compared on all five.

### 3. Two actions required *before* the first extraction commit

- **Create the GCN deterministic reference (3 runs, ~4 min).** No reproducible GCN row exists
  anywhere in the registry — the only GCN rows are Gate 1's non-deterministic ones — so the `gcn`
  entry in `BACKBONES` would pass through the refactor completely unchecked. Run GraphSAGE's
  frozen ADR-003 config with `backbone=gcn`, seeds 0–2, deterministic, on committed code.
  Tag the rows `experiment: "extract_reference"` via `PROVENANCE_KEYS`. *This doubles as the first
  live exercise of that mechanism, which has been committed since 2026-07-25 but has never
  produced a row.*
- **Pin the environment.** Capture `pip freeze` and the GPU/driver identity, and store it
  alongside the reference set. See clause 4.

### 4. Environment is part of the criterion

ADR-005 established that reproducibility here is **same-environment** — this GPU, this driver,
these library versions. A torch/PyG upgrade during the refactor would break bit-for-bit equality
for reasons that have nothing to do with the extraction, and would be indistinguishable from a
regression.

**Therefore: do not upgrade torch, torch-geometric, pyg-lib, numpy, scikit-learn or the driver
during EXTRACT.** If any of them changes, equality is void; the check degrades to "every arm
within its recorded seed band", the result is explicitly labelled as such, and the environment
delta is recorded. Do the upgrade before or after the refactor, never during — and if before,
re-establish the whole reference set first.

### 5. Consequence for DGF-1 Gate 0

**DGF-1's Gate 0 cannot be signed until this check is green.** The comparison is performed
programmatically against `experiments/registry.csv` — not read off by eye across 49 rows — and its
result is recorded in `gates/GATE-DGF1-0.md`.

## Alternatives rejected

- **Keep "ELL-1 still passes its gates."** Rejected: unsatisfiable for Gate 1, and too weak
  where it is defined — Gate 3's clauses are between-arm inequalities that survive a uniform shift
  of every arm, so a re-pass does not demonstrate the core still computes the same thing.
- **A tolerance band (e.g. |Δ| < 0.01).** Rejected: no principled width, and equality is measured
  to be achievable (8/8 on Gate 3, 3/3 on Gate 0). A band would accept drift the test can see.
  Having *observed* the reproduction before writing this, choosing a band would also have meant
  choosing a number I already knew the code could beat.
- **Use Gate 1's rows as the reference.** Rejected: pre-ADR-005, non-deterministic; equality is
  impossible and a band around run noise tests nothing.
- **Re-run only Gate 3's `real` and `no_edges` arms (~21 min).** Rejected: the excluded arms are
  the only coverage of `scramble_edges`, `random_graph_edges` and `configuration_model_edges` —
  core `gbe.eval` code the refactor moves. The saving is ~35 minutes against the cost of shipping
  a silently broken ablation into DGF-1's Gate 3.
- **Rely on the existing 62 tests.** Rejected: they are contract and guard tests on synthetic
  fixtures; **not one asserts a real-data number.** They would not catch a subtly different
  standardisation, a changed sampling order, or an off-by-one in mask construction — all of which
  preserve every type signature and every invariant the suite checks.
- **Decide the criterion after the refactor.** Rejected: post-hoc thresholding under maximum
  pressure to accept.
- **Skip the GCN reference and accept the coverage hole.** Rejected: 4 minutes now versus an
  unexercised backbone entering DGF-1, where GCN is a named comparator (doc-02 §5).

## Consequences

- **~57 min of post-refactor compute**: 40 Gate-3 re-runs (~52 min at ~1.3 min each), 6 Gate-0
  re-runs (~25 s), 3 GCN re-runs (~4 min). Plus ~4 min pre-refactor for the GCN reference.
- **A checker script** (e.g. `scripts/check_extract_regression.py`) that rebuilds each reference
  row's identity, pairs it with its post-refactor counterpart, and diffs the four/five metrics
  exactly — reporting per-row rather than in aggregate, so a single moved row is visible.
  - *Amended 2026-07-26 on implementation, was "rebuilds each reference row's identity" **at check
    time**.* That mechanism is unsafe and would have failed in the one situation it exists for.
    Gate-3's 40 rows carry no `ablation` key, so their arm is recoverable only by rebuilding config
    hashes — via `arm_run_ids()`, which depends on `frozen_hparams()` and `GNNHParams` in the
    adapter. **A refactor changes config shape, which changes config hashes, and moves the very code
    the rebuild calls.** So identities are now resolved **once, pre-refactor**, and frozen in
    `experiments/extract_reference_manifest.json` (`scripts/freeze_extract_reference.py`); the
    checker pairs on **`(arm, seed)`** — the 8 arm names are globally unique across the three
    sources — and never touches a hash. `config_hash` is stored per row for audit only.
    **The criterion is unchanged**: same 49 rows, same exact-equality bar, same no-retry rule, same
    environment clause. Only row *identification* changed.
- The re-runs append ~49 new rows to the registry. That is correct: they are genuinely new runs on
  new code, and the append-only property is preserved.
- **`PROVENANCE_KEYS` gets its first live exercise** before DGF-1's batches depend on it.
- The extraction itself is unconstrained by this ADR — it fixes only the acceptance test. What
  moves into `gbe/`, and the shape of the new `DataSource` interface, is design work this ADR
  deliberately does not pre-empt.

## Revisit when

Never for EXTRACT once the check is run and recorded. **EXTEND (EDR-1) must derive its own
criterion in its own ADR** — the same principle (reproduce recorded numbers bit-for-bit in an
unchanged environment) but a different reference set, and EDR-1 will additionally have to decide
how a *frozen text encoder's* outputs are pinned, which has no analogue here.
