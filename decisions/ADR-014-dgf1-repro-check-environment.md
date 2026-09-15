# ADR-014 — The DGF-1 re-certification states the environment it ran under

**Status:** proposed
**Date:** 2026-09-15 · **third draft**, replacing the first (`d7c453e`) and second (`3a21d3b`), both
rejected before acceptance. See *Draft history*.
**Deciders:** coderback
**Also replaces the withdrawn in-place amendment to ADR-013 clause 3** (commit `bd422c3`), struck
from ADR-013 on 2026-09-15 independently of this ADR.
**Inherits, does not re-open:** the bit-for-bit bar, the reference run and the hard-wiring of
`repro-check` (**ADR-013** clause 3), *including its rule that no later DGF-1 run may rely on the
trainer until the check passes*; the no-retry rule (**ADR-008** clause 2); the environment pin and
its five packages (**ADR-008** clause 4); same-environment reproducibility (**ADR-005**).
**This ADR adds a precondition and nothing else.** It creates no second route to certification, no
disclosure, no exception, and no way for an uncertified trainer to carry a later gate. **ADR-013
clause 3's bar is unchanged: the only way to certify the trainer is the stored-row comparison it
specifies.** Both previous drafts invented a way around a blocked certification, and both were
rejected for it; this draft does not try.
**Changes no gated quantity, no metric and no claim, and schedules no test-window run.** Both signed
gates stand exactly as recorded. **Nothing rests on this ADR:** the registry holds **zero**
`repro_check` rows (verified 2026-09-15).

**Docs affected — to apply on acceptance:**
- `decisions/ADR-013-dgf1-sensitivity-view-inputs.md` clause 3: the withdrawal note added 2026-09-15
  gains a dated pointer here. Clause 3's own text stays unchanged, and this ADR contradicts none of
  it. **Correction to that note:** it says "the environment is not part of this clause's bar", which
  is not quite true — `deterministic` and `cublas_workspace_config` are already in
  `REPRO_METRIC_KEYS` and already compared post-run. The note should read *"the environment is not
  checked before the run, and only those two fields are compared after it."*
- `CLAUDE.md` (line 39) and `docs/timeline.md` (lines 23, 204) gain the precondition. **Both are
  gitignored**, so this ADR and `notebooks/lab/` are the versioned account.
- Gate 3's pre-registration ADR, when drafted, resolves clause 6 in its own text.

## Draft history — three drafts, and what each got wrong

Each draft was rejected by adversarial review before acceptance. The pattern is worth recording,
because it is the same mistake three times.

1. **`bd422c3`, an in-place amendment to ADR-013.** Created an exception to ADR-008 clause 2's
   no-retry rule inside a document whose header declares that rule inherited; its own items made each
   other unreachable; the escape it borrowed from ADR-008 does not exist here; and it attributed
   "certifies nothing" to ADR-008 clause 4, which actually prescribes a degraded-but-interpretable
   result. **Struck from ADR-013 on 2026-09-15**, not made to wait on a replacement.
2. **`d7c453e`, ADR-014 first draft.** Routed environment provenance "through `base_cfg`", which
   `resolve_config` hashes in full — the pre-run guard would have refused permanently. Its clause 5
   let Gate 3 proceed on a certification never obtained: a relaxation of ADR-013 clause 3 inside a
   header claiming to inherit it.
3. **`3a21d3b`, ADR-014 second draft.** Replaced that disclosure with a "re-baseline": run the old
   and new trainers on the current stack and certify on their agreement. **That test cannot fail for
   the reason the check exists.** The removed reported-view loop sits *after* the gated scoring and
   *before* `save_scores`, so old-vs-new on the gated path is bit-identical by construction — a fact
   already recorded in `notebooks/lab/2026-09-15.md` before the clause was written. It spent twenty
   GPU-minutes re-deriving a known fact and called the result certification, while the proposition
   ADR-013 clause 3 asks went untested. Its clause 1 also mandated a pre-run comparison of
   `deterministic`, which is `False` until `seed_everything` runs inside `RunSession.__enter__`, so
   the check would have refused on every stack forever. And its procedure was not executable: the
   old runner has no `repro-check` mode, `compare_repro` rejects the tags it mandated, and `data/`
   and the registry are unreachable from a fresh worktree.

**What this draft does differently: it stops trying to solve the deadlock.** If the stack has moved
and the stored comparison cannot pass, the trainer is not certified and ADR-013 clause 3 blocks the
work — which is that clause operating as accepted, not a problem this ADR must fix. The environment
matches the pin today, so the check is runnable now; the recovery question is hypothetical, and
three attempts to decide it in advance produced three unsound answers. Clause 7 records what to do if
it ever becomes real.

## Disclosure — what had been seen when this was written

- **No test-window data.** Nothing under `data/`, no score file, and no label or score from steps
  482–821 was read by me or by any reviewing agent, in this session or those that produced ADR-013.
- **Test-window figures already seen**, from the signed `gates/GATE-DGF1-1.md` and carried into
  ADR-013: the gated test metrics, and the reported-view figures since withdrawn. This ADR touches
  none of them.
- **Measured 2026-09-15, read-only:** `environment_drift` against ADR-008's pin returns `[]`, and
  every `[runtime]` field matches the pin. Zero `repro_check` rows. The pilot row records
  `deterministic: true` and `cublas_workspace_config: ':4096:8'` and **no package version**. In a
  fresh process, `torch.are_deterministic_algorithms_enabled()` is `False`.
- **Known before drafting:** the gated path is bit-identical across `ccb986f` on the synthetic
  fixture. This ADR does not treat that as a substitute for the real-data check.

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
changes the arithmetic for reasons unrelated to ADR-013's code change, and the failure is recorded as
a regression under a rule that forbids retrying.

**Two concrete routes to that second harm exist today, neither needing a driver failure.**

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
unclosable retrospectively; clauses 2 and 4 are written around it rather than around a pretence that
it is closed.

## Decision

### Clause 1 — The environment is a precondition, checked before the run

- **Scope: `repro-check` mode only.** The retune, pilot and gate modes are unaffected; clause 6 says
  why, and makes no claim of exemption for them beyond this ADR's silence.
- **Where:** in `main`, after `resolve_batch` and `require_clean_tree`, **before** ADR-013's
  configuration-hash refusal, and before `load_dgraph`. The order is fixed here so an operator with
  two problems sees the environment one first: it is the cheaper to fix and the likelier to occur.
- **The pin is ADR-008's**, `experiments/extract_reference_env.txt`. No new pin is captured: one taken
  now would record today's stack rather than the pilot's while looking like a measurement of the
  latter. `scripts/assemble_gate_dgf1_0.py` already reuses this pin for DGF-1.
- **The compared fields.** The authority is a single module constant, pinned literally in a test —
  the pattern `REPRO_METRIC_KEYS` already uses. This list is its content, and the check prints it:
  1. the five ADR-008 packages: `torch`, `torch-geometric`, `pyg-lib`, `numpy`, `scikit-learn`;
  2. the GPU name, the CUDA runtime and `compute_capability`, **compared unconditionally** — absence
     of CUDA where the pin names a GPU is drift, not a skipped check;
  3. **the resolved device.** The runner resolves it once and compares `str(resolve_device(...))`
     against the pin's GPU line. **No signature change is needed:** `str(resolve_device(x))`
     round-trips losslessly through `resolve_device`, so passing the string the check inspected is
     enough to guarantee the run uses what was checked;
  4. the Python version and the cuDNN version.
- **`deterministic` and `CUBLAS_WORKSPACE_CONFIG` are *not* pre-run fields.** The second draft
  required them and was wrong: `deterministic` is `False` until `seed_everything` runs inside
  `RunSession.__enter__`, so the comparison would refuse on every stack forever. Both are **already
  compared post-run** in `REPRO_METRIC_KEYS`, which is the correct place for them, and they stay
  there unchanged. This bullet exists so the next reader does not re-derive the same bad idea.
- **The criterion, and its honest limits.** A field is compared if a change in it can alter the
  arithmetic or select different kernels. `compute_capability` and the five packages meet that
  squarely; the resolved device is the case this ADR exists for. **`platform`, Python and cuDNN do
  not meet it**, and are treated as follows: `platform` is **excluded** (a Windows build bump moves
  the string without touching the arithmetic); Python and cuDNN are **included as stack-move
  signals**, a weaker justification, stated as such. If either causes a spurious refusal in practice,
  drop that field by dated amendment — never by loosening the outcome rules.
- **Two gaps, recorded rather than papered over.** The **driver** is named by ADR-008 clause 4 but is
  absent from the pin and from every check; only its downstream symptom (CUDA disappearing) is
  detectable. The pin's other `pip freeze` entries (`scipy`, `pandas`, `networkx`, …) are not read.
  The check therefore reports **"matches the pin on the fields listed above"**, never "the
  environment".
- **New code, named as a deliverable.** `_pinned_versions` parses only the pin's `[pip freeze]`
  section, so items 2–4 need a `[runtime]`-block parser. It lives in `scripts/run_dgf1_gnn.py`,
  which already imports from `check_extract_regression` as `assemble_gate_dgf1_0.py` does; extracting
  the shared code is clause 6's follow-up, not this ADR's. **Implementation note:** the pin's
  `cublas_workspace_config` line carries a trailing prose comment, so a naive `partition("=")` parser
  reads it as part of the value; the `[runtime]` block is also CRLF. A parser written without
  allowing for both refuses a stack that has not moved.

### Clause 2 — Three outcomes, only two of which follow a run

| stage | condition | outcome | exit |
|---|---|---|---|
| before the run | any compared field drifts | **REFUSED** — no training, no row, nothing certified | non-zero |
| after the run | every compared field equal | **PASS** — reproduced *on a stack matching ADR-008's pin, on the fields clause 1 lists*; never unconditional | 0 |
| after the run | any compared field differs | **FAIL** — did not reproduce; clause 4 governs | non-zero |

- **There is no post-run "inconclusive".** Drift refuses before the run, so a drifted comparison
  cannot occur and no clause of this ADR is unreachable.
- **A refusal is not a run**, so re-running after restoring the stack is not a retry. This states what
  ADR-008 clause 2 governs — the re-running of a *completed, failed comparison* — and grants nothing
  further. Clause 4 forbids the case that rule does reach, without exception.
- **A REFUSED must be recorded even though it writes no row.** The refusal output goes verbatim into
  `notebooks/lab/` (tracked), with the date. Without this, a refusal leaves no trace anywhere while a
  FAIL leaves a permanent append-only row — an asymmetry that would reward inducing a refusal over
  running the test. **Neither outcome certifies anything, so there is nothing to gain by preferring
  one**; the record requirement removes even the appearance of an incentive.
- **Exit codes follow the only precedent**, `0` on pass and non-zero otherwise
  (`scripts/check_extract_regression.py`). No third code is invented.
- **A departure from ADR-008 clause 4, stated.** That clause meets drift by running anyway and
  degrading to "every arm within its recorded seed band", explicitly labelled. This check has one row
  and no band, so degradation has nothing to degrade to, and it refuses instead. The header lists
  clause 4 as inherited; this is the one place this ADR departs from it, and it is named here rather
  than left for a reader to find.
- **A run that raises** is neither PASS nor FAIL: `RunSession` writes its row marked `ERRORED` and the
  comparison is never reached. **This ADR does not decide whether such an attempt may be repeated.**
  The second draft granted that permission untested and unbounded; it is left open here, to be
  decided if it happens, with the aborted row in hand.
- **The PASS wording is normative.** Any document quoting it carries the qualifier in the same
  sentence; bare "re-certified" is not an available phrasing.

### Clause 3 — The run records its own environment, through a channel that is not hashed

*This clause is unchanged from the second draft, and is the only part of it that survived review: an
implementer prototyped it end to end — config hash unchanged and still equal to the stored pilot
row's, all four keys reaching `metrics_json`, suite 238 → 251 with no regressions.*

- **What is recorded** in the `repro_check` row's `metrics_json`: `env_pin` (repo-relative path),
  `env_pin_sha256` (the pin file's content hash, so a later edit to the pin is detectable from the
  row alone), `env_fields` (what clause 1 compared), and `env_state` (the observed values).
- **The mechanism.** A **new non-hashed channel**: a keyword argument on `run_dgf1` (e.g.
  `extra_metrics: dict | None = None`) whose contents are passed to `run.log_metrics` and **never**
  to `run_config_values`. The first draft's `base_cfg` route is wrong twice: `base_cfg` is inside the
  hashed config, so the pre-run guard would refuse forever; and `base_cfg` keys do not reach
  `metrics_json` at all, only `PROVENANCE_KEYS` do. **Adding these names to `PROVENANCE_KEYS` is not
  a fix** — that allow-list reads the hashed `cfg_values`.
- **This is a code change to `adapters/dgf1/train_gnn.py`**, stated plainly because the first draft
  denied needing one. It is outside `gbe/` and outside ADR-008's certified paths, so it does **not**
  re-trigger the 49-row reproduction bar.
- **Why in the row.** `docs/timeline.md` and `CLAUDE.md` are gitignored, so a record kept only there
  is absent from the versioned repo — the reasoning ADR-013 clause 2 used to create
  `gates/ERRATUM-DGF1-1.md`. A row is append-only and cannot be quietly restated. ADR-005 named this
  gap: "the environment is the piece it does not pin." This closes it for one run kind, by the
  cheapest means that touches no shared code.
- **Comparison handling.** The reference pilot row predates these keys, so they join
  `DELIBERATELY_UNCOMPARED` with that reason. **The coverage test must exercise the env-logging path**
  — it currently calls `run_dgf1` without these keys and would never see them — and **the
  `main`-driven fakes must stop swallowing the new keyword**: `fake_run_dgf1`'s `**kw` absorbs and
  discards `extra_metrics`, so every such test would stay green while no key ever reached a row.
  Build the expected key set in the test from a literal, not from the production helper.

### Clause 4 — What a FAIL means, and what follows

- **The no-retry rule is absolute and unchanged.** A completed comparison that failed is never re-run
  in the hope of a different answer (ADR-008 clause 2). This ADR adds no exception.
- **A FAIL has two possible causes:** ADR-013's code change altered the gated computation; or the
  stack moved, unrecorded, between the pin's capture and the pilot. The pilot's row records no
  version, so the second cannot be excluded from the row alone.
- **It is recorded as "did not reproduce; cause not established"** — never as "regression", never as
  "environment". The investigation ADR-013 clause 3 requires proceeds; this ADR neither replaces it
  with a route nor promises it will succeed.
- **What it costs:** ADR-013 clause 3 applies in full. No later DGF-1 run, Gate 3's included, relies
  on the trainer. **There is no alternative certification, and this ADR offers none.**

### Clause 5 — Tests, written in the implementation session

An implementer has already prototyped items 1–6 against the current code: **13 tests, all passing,
suite 238 → 251, no regressions.** They are achievable as specified.

1. **Each of the three outcomes is produced and asserted**, driving `main`. PASS and FAIL are already
   driven through `main` as of `cd4912d`; **REFUSED is new**, as are the env fields.
2. **Drift refuses before the dataset is loaded**, asserted by observing the loader was never called.
3. **A refusal writes no registry row**, which is what makes clause 2's "a refusal is not a run" true
   in code and not only in prose.
4. **The CPU-fallback case gets its own test:** pin naming a CUDA GPU, resolved device CPU → REFUSED,
   not FAIL. **The trap, and it has been demonstrated:** the test must monkeypatch
   `torch.cuda.is_available`, which also silences `environment_drift`'s own GPU branch — a
   conditional implementation makes the test pass with no refusal at all. The mutation must be run
   and recorded, not merely described.
5. **The `config.yaml` route is covered:** `device: cpu` in the adapter config with the pin naming a
   GPU yields REFUSED. This route bypasses the hash guard entirely, so it needs its own case.
6. **The compared-field list is pinned literally**, and each field is shown to be compared — the
   failure mode that survived seven mutations in `ccb986f`.
7. **The env-provenance keys do not change the config hash.** A test asserts the rebuilt reference
   hash still equals the stored pilot row's with `extra_metrics` populated. This is the first draft's
   blocking defect, converted into a standing guard.

### Clause 6 — Scope, and what is deferred

- **This check alone compares one run against one stored row at bit-for-bit equality.** That is the
  only place in DGF-1 where an environment move is indistinguishable from the effect being measured.
- **No claim is made about other run kinds, and no exemption is implied.** The second draft asserted
  that gate batches "compare arms run against each other" so a stack move moves both; that premise is
  withdrawn. Gate 1's 24 rows share commit `09c6e726` but came from **at least two** invocations
  (`run_dgf1_floor.py` loops over `feature_sets`, so both XGBoost arms may be one run), and nothing
  in ADR-011 or ADR-012 requires a shared session or stack.
- **Gate 3's ADR must decide its own case and cite this clause.** If it compares new rows against
  Gate 1's **stored** test rows, it inherits this precondition and says so; if it re-runs all arms in
  one batch, it states that instead.
- **Deferred to its own ADR:** that registry rows record no environment at all is a `gbe/run` change
  affecting all four GBE models. So is extracting the shared environment check out of
  `scripts/check_extract_regression.py`, which pulls ELL-1 adapters into any importer transitively —
  as `assemble_gate_dgf1_0.py` already does, accepted at Gate 0.

### Clause 7 — If the stack ever drifts unrecoverably

Not decided here, deliberately. Three drafts tried to decide this in advance and produced three
unsound answers; the environment matches the pin today, so the question is hypothetical.

**If it becomes real, it is decided in a new ADR with the refusal output in hand**, and the following
is on the record as a starting point rather than a decision:

- **The informative comparison is old code against the stored row**, not old code against new code.
  Running the pre-ADR-013 trainer (`e47723f`, the parent of `ccb986f`) on the current stack and
  comparing *it* to `dgf1-20260913T232506Z-1acd5067` attributes the cause: if it reproduces the row,
  the stack is equivalent and the code change is implicated; if it does not, the stack moved.
- **Old-code-versus-new-code certifies nothing**, which is why the second draft failed: the removed
  loop runs after the gated scoring and before `save_scores`, so those two agree by construction.
- **Any such procedure needs what the second draft lacked:** the old runner has no `repro-check` mode
  and its `pilot` mode runs five seeds; HEAD's runner cannot drive the old trainer
  (`run_config_values` does not exist there); `compare_repro` pins `experiment` to `pilot`/
  `repro_check` on both sides and would reject new tags; `data/` is gitignored, so a worktree has no
  snapshot; and a worktree's `experiments/registry.csv` is not the canonical one.
- **Whatever is decided, it does not restore comparability with rows recorded on the old stack.** If
  the stack has moved, Gate 3 comparing against Gate 1's stored rows is the separate and larger
  problem, and clause 6 routes it to Gate 3's ADR.

## Alternatives rejected

- **The second draft's re-baseline as a certification route** (`3a21d3b` clause 5). Rejected: the test
  cannot fail for the reason the check exists, so it certified on a known fact while the required
  proposition went untested; and it was not executable. Its diagnostic value survives, in clause 7.
- **The first draft's "unobtainable certification" disclosure** (`d7c453e` clause 5). Rejected: it
  relaxed ADR-013 clause 3 while claiming to inherit it, was reachable after a FAIL, was adjudicated
  by its own author, and rested on console text no artifact preserved.
- **Amend ADR-013 clause 3 in place** (`bd422c3`). Withdrawn and struck; see *Draft history*.
- **No environment check at all.** Rejected: the ADR-005 overclaim on a pass, and two live routes to
  condemning a sound trainer.
- **Capture a fresh DGF-1 pin now.** Rejected: it records the stack of 2026-09-15, not the pilot's,
  while looking like a measurement of the latter.
- **Let a drifted run proceed, labelled uninterpretable.** Rejected: ten minutes spent to write a row
  that certifies nothing, and a row in the registry is a standing invitation to be cited later.
- **Pre-run comparison of `deterministic`** (`3a21d3b` clause 1 item 4). Rejected on measurement: it
  refuses on every stack. Kept post-run, where it already works.

## Consequences

- **The check becomes refusable for reasons unrelated to the code**, which is the intent. A spurious
  refusal costs a second and a decision, never a run or a false verdict.
- **A drifted stack blocks the certification, and therefore Gate 3, with no route around it.** That
  is ADR-013 clause 3 operating as accepted. It is the price of not inventing a fourth way around it,
  and it is accepted here explicitly rather than engineered away.
- **The pin comparison is textually brittle** and inherited as such: `environment_drift` matches the
  GPU by substring against the whole pin file and the CUDA runtime by exact spacing, and the pin's
  `cublas_workspace_config` line carries a prose comment. Recorded as known properties.
- **The July-to-September gap is not closed and cannot be.** Clause 2's PASS wording and clause 4's
  refusal to attribute are how the programme lives with it honestly.
- **A pass is weaker than it sounds and is written that way:** "reproduced on a stack matching
  ADR-008's pin, on the fields ADR-014 clause 1 lists".
- **Implementation is a later session's** (`CLAUDE.md`, doc-first). Until this is accepted and
  implemented, no `repro_check` run is treated as certifying — and none has been run.

## Revisit when

- **The stack drifts and the check refuses.** Clause 7's question becomes real and gets its own ADR,
  with the refusal output quoted.
- **A run raises.** Clause 2 leaves the repeat question open deliberately; decide it then.
- **The follow-up ADR is drafted** — rows recording their own environment, and the extraction of the
  shared environment check.
- **Gate 3's pre-registration is drafted.** It resolves clause 6 for its own arms.
- **A field in clause 1 causes a spurious refusal.** Drop that field by dated amendment, with the
  refusal quoted; never by loosening the outcome rules.
- **Never, to convert a REFUSED or a FAIL into a pass**, and never to reintroduce a route that lets an
  uncertified trainer carry a later gate. Three drafts tried; all three were wrong.
