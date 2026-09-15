# ADR-014 — The DGF-1 re-certification's environment precondition, and the re-baseline that replaces a deadlock

**Status:** proposed
**Date:** 2026-09-15 · **second draft**, replacing the first (commit `d7c453e`), rejected before
acceptance. See *Draft history*.
**Deciders:** coderback
**Also replaces the withdrawn in-place amendment to ADR-013 clause 3** (commit `bd422c3`), struck
from ADR-013 on 2026-09-15 independently of this ADR.
**Inherits, does not re-open:** the bit-for-bit bar and the hard-wiring of `repro-check`
(**ADR-013** clause 3); the no-retry rule for a reproduction check (**ADR-008** clause 2); the
environment pin and its five packages (**ADR-008** clause 4); same-environment reproducibility
(**ADR-005**).
**Relaxes nothing.** ADR-013 clause 3's rule — "No later DGF-1 run, Gate 3's included, may rely on
the trainer until the check passes" — stands unaltered and unqualified. This ADR adds a
*precondition* and a second *route to satisfying* the bar; it creates no exception, no disclosure
and no way to proceed uncertified. The first draft did, which is why it was rejected.
**Changes no gated quantity, no metric and no claim, and schedules no test-window run.** Both signed
gates stand exactly as recorded. **Nothing rests on this ADR:** the registry holds **zero**
`repro_check` rows (verified 2026-09-15).

**Docs affected — to apply on acceptance:**
- `decisions/ADR-013-dgf1-sensitivity-view-inputs.md` clause 3: the withdrawal note added 2026-09-15
  gains a dated pointer here. Clause 3's own text stays unchanged.
- `CLAUDE.md` (line 39) and `docs/timeline.md` (lines 23, 204) gain the precondition. **Both are
  gitignored**, so this ADR and `notebooks/lab/` are the versioned account.
- Gate 3's pre-registration ADR, when drafted, resolves clause 7 in its own text.

## Draft history — why the first draft was rejected

The first draft (`d7c453e`) was reviewed on two angles. Both returned blocking findings.

1. **Its provenance clause could never run.** It routed environment fields into the row "through
   `base_cfg`", asserting this "needs no `gbe/` change — `base_cfg` already carries `experiment` this
   way". `run_config_values` returns `{**base_cfg, …}` and `resolve_config` hashes all of it, so the
   pre-run guard would have refused permanently: measured `d825a002…` (reference) against
   `854552d6…` (with the four keys added). Wrong a second way as well — `base_cfg` keys reach
   `metrics_json` only if they appear in `PROVENANCE_KEYS` (`adapters/dgf1/train_gnn.py:52,208`), so
   `experiment` arrives *despite* `base_cfg`, through an allow-list. The analogy was backwards.
2. **It repeated the defect it was written to fix.** Its clause 5 let Gate 3 proceed on a
   certification that was never obtained — a relaxation of ADR-013 clause 3, inside a document whose
   header claimed to inherit it, with no honest sentence saying so and no alternative recording the
   option it overrode. A laundering path followed: FAIL → permanently unattributable → upgrade a
   package → now REFUSED → declare unobtainable → proceed. It also required its own declaration to be
   recorded as an in-place amendment, the second thing its draft history condemned.
3. **Its field list needed a parser that does not exist.** `_pinned_versions` reads only the pin's
   `[pip freeze]` section (52 entries); `python`, `cudnn`, `compute_capability` and `gpu` are absent
   from it. Importing `environment_drift` buys item 1 plus two substring checks, not the list.
4. **Its scope argument rested on a false premise:** that gate batches "compare arms run against each
   other". Gate 1's arms were three separate invocations between 00:37 and 01:55 at commit
   `09c6e726`.

**What changed in this draft.** Clause 5's disclosure is gone entirely, replaced by clause 5's
re-baseline, which makes the deadlock disappear rather than licensing an exit from it. Clause 3 names
the code change it actually needs. Clause 1 names the parser as a deliverable and adds the two fields
the reference row actually records. Clause 7 drops the false premise.

## Disclosure — what had been seen when this was written

- **No test-window data.** Nothing under `data/`, no score file, and no label or score from steps
  482–821 was read by me or by any reviewing agent, in this session or those that produced ADR-013.
- **Test-window figures already seen**, from the signed `gates/GATE-DGF1-1.md` and carried into
  ADR-013: the gated test metrics, and the reported-view figures this programme has since withdrawn.
  This ADR touches none of them.
- **Measured 2026-09-15, read-only:** `environment_drift` against ADR-008's pin returns `[]`. Zero
  `repro_check` rows. The pilot row records `deterministic` and `cublas_workspace_config` and **no
  package version**. The gated path is bit-identical across `ccb986f` on the synthetic fixture.

## Context

ADR-013 clause 3 requires a real-data re-certification of the trainer: re-run the validation pilot's
seed 0 and require its gated metrics and score-file hash to equal the stored pilot row's
bit-for-bit. It fixes that bar and says nothing about the stack the bar is measured on.

**ADR-005 settles the principle** (*Consequences*): a claim "must claim 'reproducible on the recorded
environment', never 'reproducible anywhere'", and "the registry already pins config hash, commit, and
data snapshot; **the environment is the piece it does not pin**." ADR-008 clause 4 applies it to a
reproduction check. The DGF-1 check inherited neither.

**Two harms, in opposite directions.** On a pass, `REPRODUCED … bit-for-bit` with no statement of the
stack reads as unconditional reproduction — the claim ADR-005 forbids. On a failure, a stack move
changes the arithmetic for reasons unrelated to ADR-013's code change, and clause 3's rule then
blocks Gate 3 on a sound trainer.

**Two concrete routes to that second harm exist today, and neither needs a driver failure.**

1. `environment_drift` guards its GPU and CUDA checks behind `if torch.cuda.is_available()`
   (`scripts/check_extract_regression.py`), and `resolve_device("auto")` silently falls back to CPU
   (`gbe/run/device.py:11`). Lose CUDA and the check reports no drift, trains on CPU, and records a
   sound trainer as a regression.
2. `device: auto` lives in `adapters/dgf1/config.yaml` but is returned by `dgf1_hparams()`, **not**
   `dgf1_base_config()`, so it is in neither the hashed config nor the row. Editing that one line to
   `cpu` passes the pre-run hash guard untouched.

**What the pin can establish.** ADR-008's pin was captured 2026-07-26 at `3fb0632`. The pilot ran
2026-09-13 at `2c8f484`, and its row records no version. "Matches the pin" therefore establishes *the
live stack equals the July pin*, never *the live stack equals the pilot's stack*. That gap is
unclosable retrospectively, and clauses 4 and 5 are built around it rather than around a pretence
that it is closed.

## Decision

### Clause 1 — The environment is a precondition, checked before the run

- **Checked in `main`, after `resolve_batch` and `require_clean_tree`, immediately before ADR-013's
  configuration-hash refusal, and before `load_dgraph`.** The order is fixed here so an operator with
  two problems sees the environment one first: it is the cheaper to fix and the more likely.
- **The pin is ADR-008's**, `experiments/extract_reference_env.txt`. No new pin is captured: one taken
  now would record today's stack rather than the pilot's while looking like a measurement of the
  latter. `scripts/assemble_gate_dgf1_0.py` already reuses this pin for DGF-1.
- **The compared fields.** The authority is a single module constant, pinned literally in a test —
  the pattern `REPRO_METRIC_KEYS` already uses. This list is its content, and the check prints it:
  1. the five ADR-008 packages: `torch`, `torch-geometric`, `pyg-lib`, `numpy`, `scikit-learn`;
  2. the GPU name and the CUDA runtime, **compared unconditionally** — absence of CUDA where the pin
     names a GPU is drift, not a skipped check;
  3. **the resolved device**, obtained by calling `resolve_device` in the runner and **passed into
     `run_dgf1`** so the run uses the object the check inspected. The check must not resolve the
     device a second time and assume the run agrees: that is two sources of truth for one fact, the
     defect ADR-013 clause 1 records;
  4. **`deterministic` and `CUBLAS_WORKSPACE_CONFIG`** (ADR-005's determinism state). These are the
     **only** environment fields the reference row records, so they are the only ones comparable
     against the *pilot's own stack* rather than against the July pin. Checking them post-run, as the
     first draft did, spends ten minutes to learn what is knowable in advance;
  5. the Python version and the cuDNN version.
- **The criterion for inclusion, stated so the list is arguable:** a field is compared if a change in
  it can alter the arithmetic or the determinism guarantee. `platform` is **excluded** under that same
  criterion — a Windows build bump moves the string without touching either, and comparing it would
  refuse sound runs. Python and cuDNN are included as stack-move signals rather than as established
  arithmetic dependencies; **if either proves to cause spurious refusals in practice, drop it by dated
  amendment** rather than loosening the rule for everything.
- **Two known gaps, recorded rather than papered over.** The **driver** is named by ADR-008 clause 4
  but is absent from the pin and from any check; only its downstream symptom (CUDA disappearing) is
  detectable. The pin's other `pip freeze` entries (`scipy`, `pandas`, `networkx`, …) are not read.
  The check therefore reports **"matches the pin on the fields listed above"**, never "the
  environment".
- **New code is required and is a deliverable of this ADR:** `_pinned_versions` parses only the pin's
  `[pip freeze]` section, so items 2–5 need a `[runtime]`-block parser. Importing `environment_drift`
  does not deliver this clause.

### Clause 2 — Three outcomes, only two of which follow a run

| stage | condition | outcome | exit |
|---|---|---|---|
| before the run | any compared field drifts | **REFUSED** — no training, no row, nothing certified | non-zero |
| after the run | every compared field equal | **PASS** — reproduced *on a stack matching ADR-008's pin, on the fields clause 1 lists*; never unconditional | 0 |
| after the run | any compared field differs | **FAIL** — did not reproduce; clause 4 governs | non-zero |

- **There is no post-run "inconclusive".** Drift refuses before the run, so a drifted comparison
  cannot occur, and no clause of this ADR is unreachable.
- **A refusal is not a run**, so re-running after restoring the stack is not a retry. This is a
  statement about what ADR-008 clause 2 governs — the re-running of a *completed, failed comparison* —
  not a new permission. Clause 4 forbids the case clause 2 actually governs, without exception.
- **Deliberate drift buys nothing.** A researcher facing an impending FAIL cannot convert it to a
  REFUSED for advantage: a REFUSED leaves the trainer uncertified and Gate 3 blocked exactly as a
  FAIL does, and clause 5's re-baseline is available in both cases. There is no state in which being
  refused is better than being tested.
- **Exit codes follow the only precedent**, `0` on pass and non-zero otherwise
  (`scripts/check_extract_regression.py`). No third code is invented.
- **A run that raises** is neither PASS nor FAIL: `RunSession` writes its row marked `ERRORED` and the
  comparison is never reached. Such a row certifies nothing, must carry no environment provenance
  (clause 3), and **may be re-run** — an aborted attempt is not a completed comparison, which is the
  one case ADR-008 clause 2 does not reach. The first draft left this undecided.
- **The PASS wording is normative.** Any document quoting it carries the qualifier in the same
  sentence; bare "re-certified" is not an available phrasing.

### Clause 3 — The run records its own environment, through a channel that is not hashed

- **What is recorded** in the `repro_check` row's `metrics_json`: `env_pin` (repo-relative path),
  `env_pin_sha256` (the pin file's content hash, so a later edit to the pin is detectable from the
  row alone), `env_fields` (what clause 1 compared), and `env_state` (the observed values).
- **The mechanism, corrected.** These go through a **new non-hashed channel**: a keyword argument on
  `run_dgf1` (e.g. `extra_metrics: dict | None = None`) whose contents are passed to
  `run.log_metrics` and **never** to `run_config_values`. The first draft's `base_cfg` route is wrong
  twice: `base_cfg` is inside the hashed config, so the pre-run guard would refuse forever; and
  `base_cfg` keys do not reach `metrics_json` at all, only `PROVENANCE_KEYS` do. **Adding these names
  to `PROVENANCE_KEYS` is not a fix** — that allow-list reads the hashed `cfg_values`.
- **This is a code change to `adapters/dgf1/train_gnn.py`**, stated plainly because the first draft
  denied needing one. It is outside `gbe/` and outside ADR-008's certified paths
  (`scripts/assemble_gate_dgf1_0.py`), so it does **not** re-trigger the 49-row reproduction bar.
- **Why in the row.** `docs/timeline.md` and `CLAUDE.md` are gitignored, so a record kept only there
  is absent from the versioned repo — the reasoning ADR-013 clause 2 used to create
  `gates/ERRATUM-DGF1-1.md`. A row is append-only and cannot be quietly restated. ADR-005 named this
  gap: "the environment is the piece it does not pin." This closes it for one run kind, by the
  cheapest means that touches no shared code; closing it for every row is clause 7's follow-up.
- **Comparison handling.** The reference pilot row predates these keys, so they join
  `DELIBERATELY_UNCOMPARED` with that reason. **The coverage test must exercise the env-logging path**,
  or the classification guards nothing — the test currently calls `run_dgf1` without these keys and
  would never see them.

### Clause 4 — What a FAIL means

- **The no-retry rule is absolute and unchanged.** A completed comparison that failed is never re-run
  in the hope of a different answer (ADR-008 clause 2). This ADR adds no exception.
- **A FAIL has two possible causes:** ADR-013's code change altered the gated computation; or the
  stack moved, unrecorded, between the pin's capture and the pilot.
- **It will in practice never be attributed, and the ADR says so rather than implying patience.** The
  pilot's row records no version, so cause 2 can never be excluded, and "cause not yet established" is
  the terminal state, not an interim one. A FAIL is therefore recorded as **"did not reproduce;
  cause not attributable"** — never as "regression", never as "environment".
- **What it costs:** ADR-013 clause 3 continues to apply in full, and clause 5 is the only route
  onward. A FAIL does not license proceeding, and does not license a disclosure in place of evidence.

### Clause 5 — The re-baseline: a second route to the bar, not an exit from it

The first draft treated the stored pilot row as the only possible reference, concluded that
unrestorable drift deadlocks Gate 3 forever, and then invented a disclosure to escape. **The premise
was false.** The reference need not be the 2026-09-13 row: it can be regenerated from the code that
produced it.

- **The procedure.** On the current stack, whatever it is: run the pilot configuration under the
  **pre-ADR-013 trainer**, at the commit preceding `ccb986f`, in a detached worktree; then run it
  under the current trainer. Require the two to match bit-for-bit, score-file hash included.
- **Why this is valid.** Both sides share one environment, so the only difference between them is
  ADR-013's code change — which is exactly what the re-certification is testing, and exactly the
  design ADR-008 used by freezing its reference pre-refactor. It is a *stronger* test than the stored
  comparison, because it controls the environment instead of assuming it.
- **When it is used.** Whenever the stored-row comparison is unavailable or has failed: after a
  REFUSED that cannot be cleared, and after a FAIL. A FAIL is not re-run against the stored row
  (clause 4); the re-baseline is a different comparison, against a reference generated now, and is
  therefore not a retry of it.
- **What it settles.** If the two runs match, ADR-013's change is confirmed inert on this stack, the
  trainer is certified **on that basis**, and the recorded claim names the basis. If they differ, the
  code change is implicated directly, the stack having been held fixed — the attribution clause 4
  cannot make against the stored row.
- **Its cost, stated:** two training runs rather than one (~20 minutes), a detached worktree, and two
  rows tagged `repro_baseline_old` and `repro_baseline_new` so neither is mistaken for a
  stored-row comparison. It is not the default for that reason, not because it is weaker.
- **There is no disclosure route, and no unobtainable declaration.** They are removed. If neither the
  stored comparison nor the re-baseline can be run, the trainer stays uncertified and Gate 3 stays
  blocked, which is ADR-013 clause 3 operating as accepted.

### Clause 6 — Tests, written in the implementation session

1. **Each of the three outcomes is produced and asserted**, driving `main`. PASS and FAIL are already
   driven through `main` as of `cd4912d`; **REFUSED is new**, as are the env fields.
2. **Drift refuses before the dataset is loaded**, asserted by observing the loader was never called.
3. **A refusal writes no registry row**, which is what makes clause 2's "a refusal is not a run" true
   in code and not only in prose.
4. **The CPU-fallback case gets its own test:** pin naming a CUDA GPU, resolved device CPU → REFUSED,
   not FAIL. **The trap:** that test must monkeypatch `torch.cuda.is_available`, which also silences
   `environment_drift`'s own GPU branch — so unless the new device check is genuinely unconditional,
   the test passes vacuously. It must be written to fail against a conditional implementation, and
   that mutation must be demonstrated.
5. **The `config.yaml` route is covered:** `device: cpu` in the adapter config, with the pin naming a
   GPU, yields REFUSED. This route bypasses the hash guard entirely, so it needs its own case.
6. **The compared-field list is pinned literally**, and each field is shown to be compared — the
   failure mode that survived seven mutations in `ccb986f` and is now guarded for the metric
   comparison.
7. **The env-provenance keys do not change the config hash.** A test asserts the rebuilt reference
   hash still equals the stored pilot row's with `extra_metrics` populated. This is the first draft's
   blocking defect, converted into a standing guard.

### Clause 7 — Scope: why this check, and what is deferred

- **This check alone compares one run against one stored row at bit-for-bit equality.** That is the
  only place in DGF-1 where an environment move is indistinguishable from the effect being measured.
- **No claim is made about other run kinds.** The first draft asserted that gate batches "compare arms
  run against each other" so a stack move moves both; Gate 1's arms were in fact three separate
  invocations, and nothing in ADR-011 or ADR-012 requires a shared session or stack. The premise is
  withdrawn, and with it any implied exemption.
- **Gate 3's ADR must decide its own case and cite this clause.** If it compares new rows against
  Gate 1's **stored** test rows, it inherits this precondition and says so; if it re-runs all arms in
  one batch, it states that instead. This is a genuine deferral of a question that belongs to that
  document, not of this one's hard part.
- **Deferred to its own ADR:** that registry rows record no environment at all is a `gbe/run` change
  affecting all four GBE models. So is extracting the shared environment check out of
  `scripts/check_extract_regression.py`, which currently pulls ELL-1 adapters into any importer
  transitively — as `assemble_gate_dgf1_0.py` already does, accepted at Gate 0. Clause 3 fixes only
  the instance it can reach without touching shared code.

## Alternatives rejected

- **The first draft's "unobtainable certification" disclosure** (`d7c453e` clause 5). Rejected: it
  relaxed ADR-013 clause 3 while claiming to inherit it, was reachable after a FAIL, was adjudicated
  by its own author, and rested on console text no artifact preserved. Clause 5's re-baseline answers
  the same problem with evidence instead of a declaration.
- **Keep clause 3's block absolute and accept a possible deadlock.** This is the alternative the first
  draft failed to record. It is no longer *necessary* — clause 5 removes the deadlock — but had the
  re-baseline been impossible, this, and not a disclosure, would have been the right answer: an
  uncertified trainer blocks Gate 3, and the programme carries the cost.
- **Amend ADR-013 clause 3 in place** (`bd422c3`). Withdrawn and struck; see *Draft history*.
- **No environment check at all.** Rejected: the ADR-005 overclaim on a pass, and two live routes to
  condemning a sound trainer.
- **Capture a fresh DGF-1 pin now.** Rejected: it records the stack of 2026-09-15, not the pilot's,
  while looking like a measurement of the latter. ADR-008's pin is no better anchored to the pilot,
  but it does not pretend to be.
- **Let a drifted run proceed, labelled uninterpretable.** Rejected: ten minutes spent to write a row
  that certifies nothing, and a row in the registry is a standing invitation to be cited later.
- **Move `environment_drift` into `gbe/run/`.** Domain-agnostic and defensible under the core
  inclusion rule, but any `gbe/` change re-triggers ADR-008's 49-row bar. Deferred to clause 7's
  follow-up.

## Consequences

- **The check becomes refusable for reasons unrelated to the code**, which is the intent. A spurious
  refusal costs a second and a decision, never a run or a false verdict.
- **The pin comparison is textually brittle** and inherited as such: `environment_drift` matches the
  GPU by substring against the whole pin file and the CUDA runtime by exact spacing. Reformatting the
  pin will refuse an unmoved stack. Recorded as a known property.
- **The July-to-September gap is not closed and cannot be.** Clause 2's PASS wording, clause 4's
  refusal to attribute, and clause 5's re-baseline are how the programme lives with it honestly.
- **A pass is weaker than it sounds and is written that way:** "reproduced on a stack matching
  ADR-008's pin, on the fields ADR-014 clause 1 lists".
- **Implementation is a later session's** (`CLAUDE.md`, doc-first). Until this is accepted and
  implemented, no `repro_check` run is treated as certifying — and none has been run.

## Revisit when

- **The follow-up ADR is drafted** — rows recording their own environment, and the extraction of the
  shared environment check.
- **Gate 3's pre-registration is drafted.** It resolves clause 7 for its own arms.
- **A field in clause 1 causes a spurious refusal.** Drop that field by dated amendment, with the
  refusal quoted; never by loosening the outcome rules.
- **Never, to convert a REFUSED or a FAIL into a pass**, and never to reintroduce a route that lets an
  uncertified trainer carry a later gate.
