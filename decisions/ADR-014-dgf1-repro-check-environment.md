# ADR-014 — Withdrawn: the DGF-1 re-certification's environment rule

**Status:** withdrawn · erratum accepted 2026-10-08 (15:30) by coderback, at the end of this file
**Date:** proposed 2026-09-15 · **withdrawn 2026-09-15**, undecided, before any acceptance
**Deciders:** coderback
**Decides nothing.** No clause of this ADR is in force, and no run, number, gate or document depends
on it. It is kept because four drafts failed for a reason worth recording, and because the
measurements they produced are real and should not have to be rediscovered.
**Supersedes nothing, and is superseded by nothing.** `ADR-013` clause 3 stands exactly as accepted:
the re-certification is the stored-row comparison it specifies, and the environment is not part of
its bar. The in-place amendment once proposed under that clause (commit `bd422c3`) was struck on
2026-09-15 and stays struck.

## Why it is withdrawn

The question was real: ADR-013 clause 3 fixes a bit-for-bit bar and says nothing about the stack the
bar is measured on, while ADR-005 requires reproducibility to be claimed as same-environment. Four
drafts tried to answer it. Each was rejected by adversarial review, and each failed the same way.

| draft | commit | what it added | why it was rejected |
|---|---|---|---|
| 1 | `bd422c3` (in ADR-013) | an INCONCLUSIVE outcome and a licensed second run | created an exception to ADR-008 clause 2's no-retry rule inside a document whose header declares that rule inherited; its own items made each other unreachable; the ADR-008 escape it borrowed does not exist for a one-row reference; misattributed "certifies nothing" to ADR-008 clause 4 |
| 2 | `d7c453e` | an "unobtainable certification" disclosure | let Gate 3 proceed on a certification never obtained; routed provenance through `base_cfg`, which is hashed, so the check would have refused forever |
| 3 | `3a21d3b` | a "re-baseline": old trainer vs new trainer on the current stack | that comparison cannot fail — the removed loop runs after the gated scoring and before `save_scores` — so it certified on a known fact while the required proposition went untested; mandated a pre-run `deterministic` check that refuses on every stack; not executable (no `repro-check` mode at `e47723f`, `compare_repro` rejects its tags, `data/` and the registry unreachable from a worktree) |
| 4 | `6bbfbe1` | the escape removed, deadlock accepted | still carried three routes: *Revisit when* pre-authorised dropping a compared field by self-adjudicated amendment; the pin is an unprotected file and the claimed defence (`env_pin_sha256` making an edit "detectable from the row alone") is false, since nothing anywhere hashes the pin; and clause 2's outcome table decided PASS on "every compared field equal", where clause 1 defines those as the *environment* fields, omitting the stored-row comparison |

**The common cause.** Every draft invented a deadlock — *what if the stack has drifted and the check
can never pass?* — and then, because that deadlock was intolerable, smuggled in a route around it.
The deadlock was speculative in all four: **the environment matches the pin on every field today, and
the check is runnable now.** Four documents were written about the failure of a check nobody had run,
in a situation where it would pass.

**Two of my own failures are on the record here**, because they are the kind this programme exists to
catch. Draft 4 asserted that `scripts/run_dgf1_gnn.py` "already imports from
`check_extract_regression`" — it does not, and the claim would have justified pulling ELL-1 adapters
into the DGF-1 gate runner. It also cited "suite 238 → 251" twice as feasibility evidence for work
preserved in no branch, stash or commit, in the same document that rejected draft 1 for resting on
"console text no artifact preserved". The suite is 238.

**What happens instead:** the re-certification runs under ADR-013 clause 3 as accepted, and the
environment it ran under is recorded by hand in `notebooks/lab/`. That satisfies ADR-005's
same-environment requirement — the only thing all four drafts were actually needed for — without
deciding anything in advance about failures that have not occurred.

## What was established, and remains true

These were measured, not argued, and are recorded so the next attempt starts from them.

- **A live defect, latent today.** `environment_drift` guards its GPU and CUDA checks behind
  `if torch.cuda.is_available()` (`scripts/check_extract_regression.py`), and
  `resolve_device("auto")` falls back to CPU silently (`gbe/run/device.py:11`). If CUDA becomes
  unavailable, the check reports no drift, trains on CPU, fails every float comparison, and records a
  sound trainer as a regression under a rule that forbids retrying. A second route reaches the same
  place with no driver failure: `device: auto` lives in `adapters/dgf1/config.yaml` but is returned by
  `dgf1_hparams()`, not `dgf1_base_config()`, so it is in neither the hashed config nor the row, and
  editing that line to `cpu` passes the pre-run hash guard untouched. **This is a code defect in a
  guard, not a decision**, and can be fixed on its own merits.
- **The pin is unprotected.** `experiments/extract_reference_manifest.json` records the pin's path and
  **no hash**; nothing in `scripts/` or `tests/` hashes the file. Editing it is undetectable.
- **Environment provenance can reach a row without touching the config hash**, via a non-hashed
  keyword on `run_dgf1` passed to `log_metrics` and never to `run_config_values`. Verified: the hash
  stays `d825a002d014…`, equal to the stored pilot row's, and the keys reach `metrics_json`.
  `base_cfg` is *not* that channel — it is hashed, and its keys reach a row only through
  `PROVENANCE_KEYS`.
- **`deterministic` cannot be checked before a run.** It is `False` until `seed_everything` runs
  inside `RunSession.__enter__`. It and `cublas_workspace_config` are already compared post-run in
  `REPRO_METRIC_KEYS`, which is the right place.
- **`_pinned_versions` parses only the pin's `[pip freeze]` section.** Any check of `python`, `cudnn`,
  `compute_capability` or the GPU needs a new `[runtime]` parser, and that block is CRLF with a
  trailing prose comment on the `cublas_workspace_config` line.
- **The driver is unpinned.** ADR-008 clause 4 names it; the pin does not record it and no check sees
  it. Only its downstream symptom — CUDA disappearing — is detectable.
- **ADR-008 clause 4's stated remedy is not implemented.** The clause prescribes degrading to "every
  arm within its recorded seed band"; `scripts/check_extract_regression.py` instead refuses to report
  a pass. Any future rule should reckon with the text and the code separately.
- **The July-to-September gap is unclosable.** The pin was captured 2026-07-26 at `3fb0632`; the pilot
  ran 2026-09-13 at `2c8f484` and its row records no package version. "Matches the pin" can never
  mean "matches the pilot's stack".

## Revisit when

- **The re-certification has actually been run.** Whatever rule is worth having should be written by
  someone who has done it once, not in advance of it.
- **The stack drifts and a check refuses**, with the refusal output in hand. The informative
  comparison, if it ever matters, is the **old trainer against the stored row** — not old against
  new, which agrees by construction.
- **Rows recording their own environment** is a `gbe/run` change affecting all four GBE models and
  belongs in its own ADR (ADR-005: "the environment is the piece it does not pin").
- **Never** by reviving any of the four drafts' routes. Each was rejected on its merits, and the
  reasons are in the table above.

## Erratum, 2026-10-08 (accepted 2026-10-08 (15:30) by coderback)

**Source:** research audit 2026-10-07, §2
(`notebooks/audit/2026-10-07/README.md`; finding A S5).
The text above is not edited, so citations of its lines stay valid. **No clause changes.**

**`:6-7`, "no run, number, gate or document depends on it", stopped being true on 2026-10-07.**
ADR-015, accepted that day, quotes `:60-61` as evidence for its unhashed metrics channel
(ADR-015:791-797). That check ran on code that no commit contains (ADR-015's erratum).
Nothing else here changes: this ADR stays withdrawn and decides nothing.
