# ADR-016 — Close Gate 1's test-window access; the pre-registration guard checks which document

**Status:** proposed
**Date:** 2026-10-07 (fourth draft; drafts 1–3 reviewed the same day, see *Draft history*)
**Deciders:** coderback
**Docs affected:** none in `docs/`. This ADR does not edit ADR-012 or ADR-015. It decides the question
ADR-015 clause 6 left open, and it discloses one consequence for ADR-012 clause 10 (see *Consequences*).
**Opens no gated window.** This document carries no `Stage-1 seeds:` line, and its subject is closing
one.

## Context

DGF-1's two runners open the test window 482–821 in `--mode gate` through one shared guard,
`require_accepted_preregistration` (`scripts/preregistration.py:28-57`). The guard reads `**Status:**`
and the seed-count line from *whatever path it is handed*. Two things follow from that:

- **Gate 1's batch can be re-run.** ADR-012 is accepted and carries its count, so gate mode still
  opens 482–821 today (`tests/test_dgf1_runner.py:92`). ADR-012 clause 8 provided for one batch:

  > 5. Only then the single test
  > batch.
  > — `decisions/ADR-012-dgf1-gate1-preregistration.md:334-335`

  Clause 5 added one conditional top-up, a stage 2 "triggered only if a gated clause is unresolvable
  at stage 1" (`ADR-012:262-264`). It was not triggered (`gates/GATE-DGF1-1.md:84`). All 24 gate rows
  ran at commit `09c6e72` on 2026-09-14, and the gate file is signed. Closing gate mode therefore
  leaves no provision of ADR-012 unrun, and any further test-window number from gate mode would fall
  outside a pre-registration.
- **Any accepted file opens the window.** An accepted ADR, or a temporary file, carrying a seed-count
  line unlocks it. The tests do exactly that (`tests/test_dgf1_floor.py:167-170`). ADR-015 clause 6
  recorded this as a code defect, owed before Gate 3's pre-registration is accepted, and left one
  question open:

  > - Whether Gate 1's gate mode should still unlock 482–821 now that Gate 1 is signed is a separate
  >   question, not decided here.
  > — `decisions/ADR-015-dgf1-official-split-positioning.md:899-900`

The research audit of 2026-10-07 listed the guard's identity fix as a prerequisite for any further
gated run (`notebooks/audit/2026-10-07/README.md` §1 item 7). It did not raise whether gate mode
should stay open; its one gate-mode finding (`notebooks/audit/2026-10-07/A-dgf1-integrity.md:109`,
S9) concerns unchecked hyperparameter flags. The researcher answered ADR-015's open question on
2026-10-07: close it.

## Decision

1. **One shared map and one shared guard.**
   - **The map.** `scripts/preregistration.py` holds a single map from each gated mode name to the one
     document that may open it. A mode name means one batch in every runner.
   - **The map starts empty.** After this ADR, no mode maps to any document.
   - **The guard.** It takes the mode, the `--preregistration` path and the dirty-tree override. It
     refuses, in this order:
     1. a mode that is not in the map. A closed mode, which today is only `gate`, has its own record,
        so its refusal names the document that closed it;
     2. a dirty tree, or any override of the clean-tree requirement. The guard checks the tree itself
        (`git_dirty`), rather than relying on each runner's `main` to call `require_clean_tree`
        (`scripts/run_dgf1_floor.py:89`, `scripts/run_dgf1_gnn.py:276`);
     3. a path whose resolved form is not the mapped document's;
     4. a status other than `accepted`;
     5. a missing `Stage-1 seeds:` line, matched with the guard's existing pattern.
   - **Why it is shared.** Both runners call this one guard for every test-window mode, so they cannot
     drift apart (`scripts/preregistration.py:3-4`).
2. **No dirty tree in a gated mode.** A path match is only as good as the file at that path. An
   uncommitted edit, or a file placed there before the real document is accepted, would match. The
   check sits in the shared guard (1.2), so it applies to every mapped mode, as repro-check already
   refuses the override for itself (`scripts/run_dgf1_gnn.py:95-104`).
3. **Gate 1's gate mode is refused.**
   - **How.** `--mode gate` in both runners calls the shared guard first, before any hyperparameter
     check. `gate` is a closed mode, so it is refused with a message naming this ADR and
     `gates/GATE-DGF1-1.md`, whatever `--preregistration` says, ADR-012 included.
   - **Why the mode stays.** It is kept, not deleted, so the refusal says why. The code as it ran for
     Gate 1 is the code at `09c6e72`.
4. **The resolvers dispatch modes explicitly.** Today the test-window branch is what runs when no other
   mode matches (`scripts/run_dgf1_floor.py:71-74`, `scripts/run_dgf1_gnn.py:114-118`), and only
   argparse's `choices` keeps an unknown mode out. Each resolver names every mode it serves and refuses
   any other.
5. **A future gated batch adds its own map entry.**
   - **Who.** Gate 3's ablations, or a matched-time batch (ADR-013 clause 5), open the test window only
     through a mode added to the map, mapped to that batch's own accepted pre-registration. Adding the
     entry is part of implementing that ADR.
   - **What may never be mapped.** No mode may ever map to ADR-012 or ADR-015, and `gate` may never
     become a key of the map.

**What this does not do.**
- **ADR-012 is not revisited.** Gate 1 ran under clause 8 as written, and its verdict, rows and gate
  file stand unchanged. One property clause 10 relied on does change, and is disclosed under
  *Consequences*.
- **No other change.** No registry row, gate file or document in `docs/` changes.
- **ADR-015's official runner is unaffected.** It is specified to import neither
  `scripts/run_dgf1_gnn.py` nor `scripts/preregistration.py` (`ADR-015:569`). The floor runner imports
  the guard (`scripts/run_dgf1_floor.py:41`), so the official runner cannot import that either.
- **Library entry points stay outside the guard.** The guard governs the runners only. `run_floor` and
  `run_dgf1` accept any split, so code that calls them directly with the test split bypasses it. That
  route predates this ADR, which does not claim to close it (see *Revisit when*).
- **Old checkouts are not closed.** Running gate mode from a checkout of `09c6e72`, or any commit
  before this ADR's implementation, still opens 482–821 under the old guard. Code cannot close that;
  the detection signal under *Revisit when* is how it would be noticed.

## Alternatives rejected

- **Keep ADR-012, matched by path.** This is ADR-015 clause 6's minimal fix. It closes the
  any-accepted-file hole but leaves a signed gate's batch re-runnable on the test window, with no
  pre-registered purpose.
- **Delete gate mode.** An unknown-mode error says less than a refusal that names its reason, and the
  diff is larger.
- **Let each runner name its own document.** The two runners could then drift apart for one batch.
- **Leave the guard until Gate 3's pre-registration.** ADR-015 allows that timing, but the fix is
  small, and every week it waits is a week in which any accepted file opens the window.

## Consequences

- **The Gate-1 score files can no longer be regenerated at will.**
  - **What changes.** ADR-012 clause 10 keeps them out of git as "regenerable from a deterministic
    config" (`ADR-012:364-366`). Regenerating them means re-running gate mode, which this ADR closes.
  - **What rests on them.** The clause-10 paired bootstrap is "reproducible from the stored score
    files" (`ADR-012:379-380`), and GATE-DGF1-1 states the rows are "bit-for-bit reproducible on the
    recorded environment" (`gates/GATE-DGF1-1.md:6`). Both claims stay true, but after this ADR they
    can be exercised only from the stored files.
  - **What remains.** The provenance is unchanged: each gate row's `scores_sha256`.
  - **What to do, before the implementing commit.** Back up the score files of the 24 gate rows
    (run ids at `gates/GATE-DGF1-1.md:9`) outside the repo, and verify each copy's SHA-256 against its
    row's `scores_sha256`.
  - **If they are ever needed and lost.** Regenerating them needs its own pre-registration.
    - **Its only permitted use** is restoring files whose SHA-256 equals the row's `scores_sha256`.
    - **It logs no metric.**
    - **On a mismatch,** the files are discarded unread and the cause is found, never retried.
- **Code, in a later session after acceptance:**
  - `scripts/preregistration.py`: the empty map and the shared guard;
  - both runners' resolvers: explicit dispatch, and the guard called first in every gated mode;
  - the module docstrings that still describe gate mode as opening 482–821
    (`scripts/run_dgf1_floor.py:3-21`, `scripts/run_dgf1_gnn.py:5-18`, `scripts/preregistration.py:1-13`).
- **Tests, rewritten so they fail if the decision is undone:**
  - gate mode in both runners is refused for ADR-012 and for a temporary accepted file with a
    `Stage-1 seeds:` line, and the refusal comes before the hyperparameter check;
  - the guard, given a fixture map entry, accepts the mapped document when the tree is clean
    (`git_dirty` monkeypatched), the document is accepted and it carries a count. It refuses another
    path, a dirty tree, an override, a non-accepted status and a missing count;
  - the real map maps no mode to ADR-012 or ADR-015, `gate` is not one of its keys, and the refusal
    of `gate` names this ADR;
  - an unknown mode is refused at both resolvers;
  - both runners' gated branches call the shared guard.

  The tests to rewrite are:
  - `tests/test_dgf1_floor.py:139-177`, and its import at `:34`, which reaches the guard through the
    floor runner's re-export (`scripts/run_dgf1_floor.py:39-41`);
  - `tests/test_dgf1_runner.py:68-100`;
  - `tests/test_dgf1_runner.py:542-560`.

  ADR-015's clause-7 test 13 ("**The shared guard refuses this ADR**", `ADR-015:1093`) is written
  against the new guard whenever it is written, before or after this ADR's code. It also matches
  ADR-015's text with the guard's own `Stage-1 seeds:` pattern, not a substring, because ADR-015
  mentions that string in prose (`:11`, `:886`, `:889`, `:1236`) and carries `**Positioning seeds:**`
  instead. ADR-015's header cites the pattern's current line (`scripts/preregistration.py:25`), which
  the rewrite will move.
- **Mutation checks.** Each of these must make a test fail:
  - a guard that ignores the mapped document;
  - a guard that skips the clean-tree check;
  - a guard that honours the dirty override;
  - a gate mode that opens the window again;
  - `gate` added to the map, mapped to any document;
  - a resolver that falls through to the test window;
  - a map entry pointing at ADR-012.
- **The change touches only `scripts/` and `tests/`.** Nothing under `gbe/` or `adapters/ell1/`
  changes, so ADR-008's regression check is not triggered, and no new number is produced.

## Revisit when

- **A pre-registered batch needs Gate 1's arms on the test window again,** for example a replication
  study. It gets its own pre-registration and its own map entry; this ADR is not reopened.
- **The path comparison misfires,** for example when files move between platforms or are reached
  through symlinks. Then the comparison is fixed in code, and the decision stands.
- **A test-window number appears outside a mapped mode.**
  - **How it would be noticed.** A registry row whose window overlaps 482–821 and whose experiment tag
    is not the tag a mapped mode writes. Tags differ from mode names, for example
    `scripts/run_dgf1_gnn.py:53,105`.
  - **The one exemption.** The 24 Gate-1 rows, identified by run id (`gates/GATE-DGF1-1.md:9`) rather
    than by tag or commit, since a re-run from an old checkout would carry the same tag and commit.
  - **What then.** If such a row appears, a lint that only mapped modes may select the test split
    becomes necessary.

## Draft history

- **Draft 1, diff-only review.** One blocking finding: Context said the audit listed both problems,
  but its §1 item 7 lists only the guard fix. Corrected. Should-fix items folded in:
  - the shared map;
  - the dirty-override refusal;
  - refusal before the hyperparameter checks;
  - explicit dispatch;
  - the limit on library entry points;
  - the floor runner's tests;
  - stage 2 accounted for.
- **Draft 2, diff-only review of the changes.** Two blocking findings, both corrected:
  - **Audit A's gate-mode finding.** Draft 2 said the audit "did not raise" gate mode, but audit A's S9
    is a gate-mode finding about flags.
  - **ADR-012 clause 10.** Draft 2 did not disclose that closing gate mode ends clause 10's
    "regenerable" property for the Gate-1 score files.

  Should-fix items and nits folded in:
  - the map is stated as empty, with the guard and the dirty refusal moved into shared code, so the
    tests have code to act on;
  - no mode may map to ADR-012 or ADR-015;
  - test 13's direct seed-line check, and "`Stage-1 seeds:`" instead of "seed-count";
  - a detection signal for the third *Revisit* bullet;
  - the floor runner's exclusion from ADR-015's official runner;
  - the Gate-1 commit `09c6e72`, and the docstrings to update;
  - one mode name means one batch;
  - the test lines corrected to 542-560.
- **Draft 3, diff-only review of the changes. Nothing blocking**, so under the stopping rule the
  should-fix items and nits were folded in without another round, and the ADR goes to the
  researcher's decision. Folded in:
  - **Score files:**
    - the regeneration remedy now restores only hash-matching files and logs no metric;
    - the backup is hash-verified before the implementing commit;
    - the claims that rest on the files are disclosed.
  - **The guard:**
    - the guard checks the tree itself;
    - `gate` has a closed-mode record and can never be a map key;
    - test 13 uses the guard's pattern;
    - the floor test's import is added to the list.
  - **Scope and detection:**
    - old checkouts are disclosed as not closable in code;
    - the detection signal is keyed to the 24 run ids, to window overlap and to tags rather than mode
      names.
  - **Wording:** "outside the guard"; test 13 quoted exactly.
