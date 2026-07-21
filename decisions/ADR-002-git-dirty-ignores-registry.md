# ADR-002 — git_dirty ignores the append-only registry

**Status:** accepted
**Date:** 2026-07-21
**Deciders:** coderback
**Docs affected:** none (infra semantics only; `gbe/run/config.py`)

## Context

Every run records `git_dirty` (from `git status --porcelain`) so a reported number can be
traced to whether its *inputs* — code, config, data — were committed. `experiments/registry.csv`
is tracked (papers read it; it is append-only). But it is a run **output**: the first run in
a batch appends its row, which dirties the working tree, so every subsequent run in the same
batch records `git_dirty=true` even though no code changed. Observed on the ELL-1 Gate-0
batch: with clean, committed code, 5 of 6 baseline rows still came back dirty purely because
the registry grew mid-batch. `dirty` was conflating "uncommitted code" with "registry is
being appended," making the flag alarming and useless for its actual purpose.

## Decision

`git_dirty()` ignores run-output artifacts — by default `experiments/registry.csv` — so the
flag reflects only uncommitted **code/config**. Implemented via a `_dirty_paths()` helper that
parses porcelain output and drops ignored paths; `IGNORED_DIRTY_PATHS` holds the default set.

## Alternatives rejected

- **Gitignore the live registry.** Rejected: the registry is meant to be committed and
  reviewed (append-only, papers assembled from it); untracking it loses that provenance.
- **Commit the registry after every run.** Rejected: six commits per batch, violates
  one-logical-change-per-commit, and still races the next run.
- **Accept dirty=true on batch runs 2..n.** Rejected: makes the single most useful
  provenance flag noise; a reader can no longer trust `dirty` to mean "uncommitted code."

## Consequences

- `dirty` now answers "were the run's inputs committed?" — its intended meaning. A batch of
  runs on committed code yields all `dirty=false`, regardless of registry growth.
- A hand-edit of `registry.csv` (forbidden anyway — append-only) would not show as dirty;
  that is not `git_dirty`'s job (the append-only guard/test is).
- The 6 pre-ADR Gate-0 rows remain `dirty=true` in the registry (append-only, never
  rewritten); the re-run adds 6 clean rows, and `GATE-ELL1-0.md` cites the clean batch.

## Revisit when

A run starts writing another tracked output *during* the run (e.g. a checkpoint index or the
gate file mid-run) — add it to `IGNORED_DIRTY_PATHS` at that point.
