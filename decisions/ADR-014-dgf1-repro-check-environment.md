# ADR-014 — The DGF-1 re-certification's environment precondition, and what a failed or unobtainable certification costs

**Status:** proposed
**Date:** 2026-09-15
**Deciders:** coderback
**Replaces the withdrawn in-place amendment to ADR-013 clause 3** (commit `bd422c3`, proposed
2026-09-15, never accepted). See *Draft history*.
**Inherits, does not re-open:** the bit-for-bit bar, the reference run and the hard-wiring of
`repro-check` (**ADR-013** clause 3); the no-retry rule for a reproduction check (**ADR-008**
clause 2) — this ADR adds no exception to it; the environment pin and its five packages
(**ADR-008** clause 4); same-environment reproducibility (**ADR-005**).
**Changes no gated quantity, no metric and no claim, and schedules no test-window run.** Both signed
gates stand exactly as recorded. **Nothing rests on this ADR yet:** the registry contains **zero**
`repro_check` rows (verified 2026-09-15), so no result is being reinterpreted.

**Docs affected — to apply on acceptance:**
- `decisions/ADR-013-dgf1-sensitivity-view-inputs.md`: the proposed amendment under clause 3 is
  **struck**, replaced by a dated pointer to this ADR. Clause 3's text is otherwise unchanged. This
  is the pattern ADR-013 used on ADR-011 — a dated pointer, not an in-place rewrite.
- `CLAUDE.md` (line 39) and `docs/timeline.md` (lines 23, 204), which state the re-certification's
  bar, gain the environment precondition. **Both are gitignored**, so neither carries the record:
  this ADR and `notebooks/lab/` are the versioned account.
- Gate 3's pre-registration ADR, when drafted, cites clause 5 below.

## Draft history — why the in-place amendment was withdrawn

`bd422c3` proposed this as a dated amendment inside ADR-013 clause 3. An adversarial pass rejected
it, and the rejection holds. Four defects, each verified against the repo:

1. **It re-opened a rule ADR-013 declares out of scope.** ADR-013's header lists "the no-retry rule
   for a reproduction check (**ADR-008** clause 2)" under *Inherits, does not re-open*. The
   amendment's item 4 created a licensed second run — a relaxation — and then asserted of itself
   that "nothing is loosened". By the amendment's own stated test ("a new ADR would be right if the
   bar were being *relaxed*"), it belonged in a new ADR.
2. **Its own items contradicted each other.** Refusing on drift *before* training means no
   comparison and no row, so its table row "drifted + every field equal → INCONCLUSIVE" was
   unreachable, its requirement to cite "both rows" was impossible, and its mandated test could only
   have been satisfied by asserting a symptom the real flow cannot produce — the defect class
   ADR-013 clause 1 exists to record.
3. **Its escape hatch does not exist here.** It borrowed ADR-008's "re-establish the reference on the
   new stack". ADR-008 can offer that because its reference is a re-freezable 49-row manifest;
   DGF-1's is one frozen pilot row, and regenerating it *is* the check. An unrestorable drift would
   have deadlocked Gate 3 permanently.
4. **It misattributed its central rule.** ADR-008 clause 4 says equality is void and "the check
   degrades to 'every arm within its recorded seed band', the result is explicitly labelled as
   such". That is degraded-but-interpretable. "Certifies nothing" was a new and stricter decision
   presented as a citation.

Two further errors: the amendment claimed its approach was "the same correction-of-omission this ADR
performed on ADR-011" — ADR-013 gave ADR-011 *dated pointers* and decided in a new document, so the
precedent argued the opposite; and it specified exit code 2, diverging silently from
`scripts/check_extract_regression.py`, whose only precedent is `return 0 if ok else 1`.

**The withdrawal is cheap and total:** no run was performed under it, and the amendment was never
accepted.

## Disclosure — what had been seen when this was written

- **No test-window data.** Nothing under `data/`, no score file, and no label from steps 482–821 was
  read, by me or by any reviewing agent, in the sessions that produced ADR-013 or this ADR.
- **Measured 2026-09-15, read-only:** `environment_drift` against ADR-008's pin returns `[]` — the
  live stack matches on all five packages, the GPU and the CUDA runtime. The registry holds zero
  `repro_check` rows. The pilot row `dgf1-20260913T232506Z-1acd5067` records `deterministic` and
  `cublas_workspace_config` and **no package version at all**.
- **Known before drafting:** that the gated path is bit-identical across `ccb986f` on the synthetic
  fixture (config hash, `proba`, and `scores_sha256` all equal, the 16 withdrawn keys removed). That
  is the result this ADR governs the real-data confirmation of.

## Context

ADR-013 clause 3 removed the defective reported-view scoring from `run_dgf1` and required a
real-data re-certification: re-run the validation pilot's seed 0, and require its gated metrics and
score-file hash to equal the pilot row's bit-for-bit. Clause 3 fixes that bar and says nothing about
the stack the bar is measured on.

**ADR-005 already settles the principle** (*Consequences*): a claim "must claim 'reproducible on the
recorded environment', never 'reproducible anywhere'". ADR-008 clause 4 applies it to a reproduction
check and names the five packages that must not move. The DGF-1 check inherited neither.

**Two concrete harms follow, in opposite directions.**

- **On a pass:** the check prints `REPRODUCED … bit-for-bit` with no statement of the stack. Recorded
  in the notebook and cited by Gate 3's ADR, that reads as unconditional reproduction — the claim
  ADR-005 forbids.
- **On a failure:** a torch, PyG or driver move changes the arithmetic for reasons unrelated to
  ADR-013's code change. Clause 3's no-retry rule then blocks every later DGF-1 run, Gate 3 included,
  on a trainer that is sound.

**A third harm, found while drafting, is live today and invisible.** `environment_drift` guards its
GPU and CUDA checks behind `if torch.cuda.is_available()`
(`scripts/check_extract_regression.py`), and `resolve_device("auto")` silently falls back to CPU
(`gbe/run/device.py:11`). On a session where CUDA is unavailable — a driver break, a replaced
machine — drift reports nothing, the run trains on CPU, scatter aggregation sums in a different
order, every float differs, and the outcome is recorded as a regression. The environment check as
inherited would not catch the single most likely environment failure.

**What the pin can and cannot establish.** ADR-008's pin was captured 2026-07-26 at `3fb0632`. The
DGF-1 pilot ran 2026-09-13 at `2c8f484`, and its row records no version. So "matches the pin"
establishes *the live stack equals the July pin*, never *the live stack equals the pilot's stack*.
That gap is unclosable retrospectively and is what clause 4 below is built around.

## Decision

### Clause 1 — The environment is a precondition of the check, not a caveat on its result

- **Checked before anything else runs**, beside ADR-013's configuration-hash refusal and before
  `load_dgraph`. A stack that cannot certify costs a second, not a training run.
- **The pin is ADR-008's**, `experiments/extract_reference_env.txt`. No new pin is captured: one
  taken now would record today's stack rather than the pilot's, and would dress an assumption as a
  measurement. `scripts/assemble_gate_dgf1_0.py` already reuses this pin for DGF-1.
- **The compared fields, written out** — the check states them in its own output, and this list is
  the authority:
  1. the five ADR-008 packages: `torch`, `torch-geometric`, `pyg-lib`, `numpy`, `scikit-learn`;
  2. the GPU name and the CUDA runtime, **compared unconditionally**;
  3. **the resolved device.** If the pin names a CUDA GPU and `resolve_device` yields CPU, that is
     drift. This is the case the inherited function silently passes, and it is the point of this
     clause;
  4. the Python version and the cuDNN version, both recorded in the pin and previously unchecked.
- **`platform` is deliberately not compared.** The pin records a Windows build string; an OS update
  moves it without touching the arithmetic, and comparing it would refuse sound runs. Stated here so
  the omission is a decision rather than an oversight.
- **The check reports what it compared, never "the environment".** "Matches the pin on the fields
  listed above" is the strongest available phrasing, because the pin's other `pip freeze` entries
  (`scipy`, `pandas`, `networkx`, …) are not read.

### Clause 2 — Three outcomes, only two of which follow a run

| stage | condition | outcome | exit |
|---|---|---|---|
| before the run | any compared field drifts | **REFUSED** — no training, no row, nothing certified | non-zero |
| after the run | every compared field equal | **PASS** — reproduced *on a stack matching ADR-008's pin, on the fields clause 1 lists*; never unconditional | 0 |
| after the run | any compared field differs | **FAIL** — did not reproduce; clause 4 governs what that means | non-zero |

- **There is no post-run "inconclusive".** Drift refuses before the run, so a drifted comparison
  cannot occur. This is what keeps ADR-008 clause 2 intact: a refusal is not a run, so re-running
  after restoring the stack is not a retry, and **no exception to the no-retry rule is created**.
- **Exit codes follow the precedent**, `0` on pass and non-zero otherwise
  (`scripts/check_extract_regression.py`). No third code is invented; the printed outcome word and
  the registry row are the record, not the exit status.
- **The PASS wording is normative.** Any document quoting it carries the qualifier in the same
  sentence. "Re-certified", bare, is not an available phrasing.

### Clause 3 — The run records its own environment, so the record is not prose

The row written by a `repro_check` run carries, through `base_cfg` into `metrics_json`:

- `env_pin` — the pin's repo-relative path;
- `env_pin_sha256` — the hash of the pin file's content, so a later edit to the pin is detectable
  from the row alone;
- `env_fields` — the fields clause 1 compared;
- `env_drift` — `"none"`, which is the only value a row can carry, because drift refuses before the
  run.

**Why in the row.** `docs/timeline.md` and `CLAUDE.md` are gitignored, so a record kept only there is
absent from the versioned repo and from any open release — the reasoning ADR-013 clause 2 used to
create `gates/ERRATUM-DGF1-1.md`. A row is append-only, tracked, and cannot be quietly restated.

**ADR-005 named this gap when it set the determinism policy** (*Consequences*): "The registry already
pins config hash, commit, and data snapshot; **the environment is the piece it does not pin**." This
clause closes it for one run kind, by the cheapest means that touches no shared code. Closing it for
every row is the follow-up in clause 6.

**This needs no `gbe/` change** — `base_cfg` already carries `experiment` this way — and therefore
does not trigger ADR-008's re-run bar. The pilot row predates these keys, so they are added to the
check's `DELIBERATELY_UNCOMPARED` set with that reason stated.

### Clause 4 — What a FAIL means, and what it does not license

- **The no-retry rule is absolute and unchanged.** A FAIL is never re-run in the hope of a different
  answer (ADR-008 clause 2). This ADR adds no exception, in any circumstance.
- **A FAIL has two possible causes, and the investigation names which:**
  1. ADR-013's code change altered the gated computation — a regression;
  2. the stack moved, unrecorded, between the pin's capture (2026-07-26) and the pilot
     (2026-09-13). The pilot's row cannot rule this out, because it records no version.
- **Until the cause is established, the outcome is recorded as "did not reproduce; cause not yet
  established"**, never as "regression" and never as "environment". Clause 3 of ADR-013 continues to
  apply: no later DGF-1 run, Gate 3's included, relies on the trainer.
- **Why this is not a softening.** It constrains the *conclusion drawn*, not the consequence: the
  trainer stays blocked either way. What it forbids is the overclaim of naming a cause the evidence
  cannot distinguish — the same discipline clause 2 applies to the pass.

### Clause 5 — The unobtainable certification, and the deadlock's exit

Drift may be unrestorable: a yanked wheel, replaced hardware, a forced driver or CUDA update. Clause
1 then refuses forever, and ADR-013 clause 3 blocks Gate 3 forever. That is not an acceptable
resting state, so it has a named exit, and the exit is a disclosure rather than a pass:

- **The researcher may declare the re-certification unobtainable**, in a dated decision recorded as
  an amendment here, which **quotes the refusal output verbatim** and states why the drift cannot be
  restored.
- **The stated cost, which travels with every downstream claim:** ADR-013's clause-3 change is then
  certified by its test suite alone — clause 6's tests and the mutation checks recorded in the lab
  notebook — and **never described as reproduced**. Any document citing the re-certification says
  "not obtained; drift recorded in ADR-014".
- **Gate 3 may then proceed only if its ADR states this explicitly** and carries the qualification in
  its own text. It is not inherited silently.
- **It is never available while the check could be run.** The refusal output must show drift that
  restoration cannot clear. A check not yet attempted is not an unobtainable one.

### Clause 6 — Scope: why this check and not every run

- **This check alone compares one run against one stored row at bit-for-bit equality.** That is the
  only place in DGF-1 where an environment move is indistinguishable from the effect being measured.
  Gate and pilot batches compare arms run against each other with variance, and a stack move moves
  both arms.
- **Gate 3's ADR must decide its own case**, and this clause is what it cites: if Gate 3 compares new
  rows against Gate 1's **stored** test rows rather than re-running both arms in one batch, it
  inherits this ADR's precondition, and must say so. If it runs all arms in one batch, it does not.
- **The general question is routed, not answered here:** that registry rows record no environment at
  all is a `gbe/run` change affecting all four GBE models, and belongs in its own ADR. Clause 3
  fixes only the one instance it can fix without touching the core.
- **Not restructured here:** `run_dgf1_gnn.py` importing `environment_drift` from
  `scripts/check_extract_regression.py` pulls ELL-1 adapters in transitively. `assemble_gate_dgf1_0.py`
  already does exactly this and was accepted at Gate 0. Extracting the shared function is a
  code-organisation change to a script Gate 0 certified, so it goes to the follow-up ADR rather than
  being smuggled in beside a correctness fix.

### Clause 7 — Tests, written in the implementation session

1. **Each of the three outcomes is produced and asserted**, driving the runner's entry point, not
   only its helpers — the gap the adversarial pass found in `ccb986f`.
2. **Drift refuses before the dataset is loaded**, asserted by observing that the loader was never
   called.
3. **A refusal writes no registry row**, which is what makes clause 2's "a refusal is not a run"
   true in the code and not only in this document.
4. **The CPU-fallback case is covered explicitly:** with the pin naming a CUDA GPU and the resolved
   device CPU, the outcome is REFUSED and not FAIL. This is the live hole clause 1 exists to close,
   so it gets its own test rather than relying on the general drift path.
5. **The compared-field list is pinned literally**, and each field is shown to be compared, so a
   field cannot leave the list silently — the failure mode that survived seven mutations in
   `ccb986f` and is now guarded for the metric comparison.

## Alternatives rejected

- **Amend ADR-013 clause 3 in place (`bd422c3`).** Withdrawn; see *Draft history*. Its central defect
  is not fixable by editing: a licensed second run cannot live in a document whose header declares
  the no-retry rule out of scope.
- **Leave clause 3 as accepted, with no environment check.** Rejected: it produces the ADR-005
  overclaim on a pass and blocks Gate 3 on a sound trainer when the stack moves. The CPU-fallback
  case makes this concrete rather than hypothetical.
- **Capture a fresh DGF-1 environment pin now.** Rejected: it would record the stack of 2026-09-15,
  not of the pilot on 2026-09-13, while *looking* like a measurement of the latter. ADR-008's pin is
  no better anchored to the pilot, but it does not pretend to be, and it is already the programme's
  single environment anchor.
- **Keep a post-run INCONCLUSIVE outcome** (the withdrawn draft's design). Rejected: unreachable once
  drift refuses before the run, and reachable only by allowing drifted runs, which then requires an
  exception to ADR-008 clause 2.
- **Let a drifted run proceed for information, labelled uninterpretable.** Rejected: it spends ten
  minutes and writes a row that certifies nothing, and a row in the registry is a standing invitation
  to be cited later. The refusal output plus clause 5's disclosure carries the same information at no
  cost.
- **Move `environment_drift` into `gbe/run/`.** It is domain-agnostic and would satisfy the core
  inclusion rule. Rejected *for now*: any change under `gbe/` re-triggers ADR-008's 49-row
  reproduction bar, which is a large price for relocating a function during a correctness fix.

## Consequences

- **The check becomes refusable for reasons unrelated to the code.** That is the intent. A spurious
  refusal — the pin's text reformatted, a package legitimately upgraded — costs a second and a
  decision, never a run or a false verdict.
- **The pin comparison is textually brittle** and inherited as such: `environment_drift` matches the
  GPU by substring against the whole pin file and the CUDA runtime by exact spacing. Reformatting
  `experiments/extract_reference_env.txt` will refuse an unmoved stack. Recorded so the next person
  meets it as a known property.
- **The July-to-September gap is not closed and cannot be.** Clause 2's PASS wording and clause 4's
  two-branch reading are how the programme lives with it honestly.
- **A pass is weaker than it sounds, and is written that way.** "Reproduced on a stack matching
  ADR-008's pin, on the fields ADR-014 clause 1 lists" is the whole claim.
- **Implementation is a later session's** (`CLAUDE.md`, doc-first): this ADR was conceived in the
  session that found the omission. Until it is accepted and implemented, no `repro_check` run should
  be treated as certifying — and none has been run.

## Revisit when

- **The follow-up ADR is drafted** — registry rows recording their own environment, and the
  extraction of the shared environment check out of `scripts/check_extract_regression.py`.
- **Gate 3's pre-registration is drafted.** It resolves clause 6's question for its own arms, in its
  own text.
- **Drift is declared unrestorable.** Clause 5's disclosure is added here as a dated amendment, with
  the refusal output quoted.
- **Never, to convert a REFUSED or a FAIL into a pass.** A certification that was not obtained is
  recorded as not obtained.
