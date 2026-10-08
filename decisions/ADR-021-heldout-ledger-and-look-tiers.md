# ADR-021 — The held-out ledger, the three tiers of a look, and the leakage rule's wording

**Status:** proposed
**Date:** proposed 2026-10-08
**Tier:** A. It sets the held-out and leakage rules (CLAUDE.md *Process*, ADR-017).
**Review rounds:** 0/4
**Deciders:** coderback
**Implemented by:** plan row 5a, at least 12 hours after acceptance and not on the same calendar
day. It must be accepted before EDR-1 starts (ADR-020 condition 6), and before the next look at a
held-out set (CLAUDE.md, *Next*).

**Docs affected:**
- `CLAUDE.md:166-167` (held-out data), `:187-189` (the leakage-test rule), DGF-1's qualifications
  (`:101-109`) and *Next* item 4 (`:70-71`);
- `docs/timeline.md`, rows 5 and 5a;
- `docs/subagent-preamble.md`: an agent's report of a held-out figure becomes a ledger row. This is
  an added duty, not a narrowed ban, so it needs no tier of its own.

## Context

**"Peek" has no definition, and the rule learned from a real breach is only in ADR-011** (audit D8,
`notebooks/audit/2026-10-07/D-governing-docs.md:52`). CLAUDE.md says "never … peek at any held-out
eval set" (`CLAUDE.md:166`). Yet DGF-1's split design computed the test window's prevalence on
purpose (ADR-011:78-84), and a reviewing agent computed test-window label statistics by mistake
(ADR-011:92-95). After that slip, ADR-011 made a rule (ADR-011:106-110). Any later change had to make
Gate 1 harder for the GNN, or rest entirely on label-free evidence. That rule never reached
CLAUDE.md.

**GATE-DGF1-1's disclosure was incomplete.** It named two of the looks before its batch. The
corrected list has five kinds, and two of them no ADR discloses (`gates/ERRATUM-DGF1-1.md`,
addendum). Disclosure written from memory misses things.

**CLAUDE.md's leakage-test rule cannot always be met** (audit D7). DGF-1's labels are undated, and
its features are a snapshot. No test can show that no training tensor touches post-cutoff data, so
the honest output is a documented limitation (GATE-DGF1-1:215-217). CLAUDE.md also lacks that gate's
qualification 4.

**Decided by the researcher on 2026-10-07:**
- **The ledger, the "peek" tiers and the leakage-rule rewording, now; Kapoor's sheet at EDR-1 Phase
  0** (`notebooks/lab/2026-10-07.md:14`). The three tiers, structure, label statistics and scores,
  are the audit's (`notebooks/audit/2026-10-07/D-governing-docs.md:99`).
- **The guard writes the ledger, and DGF-1's looks are backfilled** (the dated plan, row 5a,
  `docs/timeline.md:72`).
- **Qualification 4 goes with the rewording.** D7's remedy pairs them
  (`notebooks/audit/2026-10-07/D-governing-docs.md:51`). The audit also listed it for the
  consistency ADR (`notebooks/audit/2026-10-07/README.md:78`), which drops it once this ADR is
  accepted.

**From the measurement and the prototype** (ADR-017 item 3;
`notebooks/measurements/2026-10-08-heldout-ledger/`):
- **The guard sees a third of the paths.** Six committed scripts can read a held-out DGF-1 set, and
  four of them do so outside the guard. These include the two whose looks GATE-DGF1-1 missed
  (`notebooks/measurements/2026-10-08-heldout-ledger/output.txt`). A guard-written ledger alone
  would have recorded none of DGF-1's looks before Gate 1.
- **The prototype found two traps.** It has 34 tests, and 25 of 25 mutants are killed
  (`notebooks/measurements/2026-10-08-heldout-ledger/prototype/mutation-output.txt`).
  - **Two of the four scripts load the data through PyG's `DGraphFin`,** not the adapter, so a lint
    that looked only for the adapter's loader missed them.
  - **A ledger row written at the start of a batch makes the tree dirty.** Every row of that batch
    would record `git_dirty=true`, and the next runner's guard would refuse it. Both the problem
    and its fix are tested.

**Decided by the researcher on 2026-10-08,** after the measurement (`notebooks/lab/2026-10-08.md`):
- every held-out reader writes the ledger, enforced by a test;
- `git_dirty` ignores the ledger, as it ignores the registry;
- a labels look needs an accepted document that names it first;
- the prototype is kept in `notebooks/measurements/`, as ADR-018's was.

## Decision

### 1. Held-out sets

A held-out set is data that a pre-registration or a reported-only ADR reserves for its batch. There
are two today:
- `dgf1-temporal-test-482-821`, DGF-1's temporal test window (ADR-011);
- `dgraph-official-test`, DGraph's official test mask (ADR-010, ADR-015).

A later document that reserves a set adds its name to the ledger module's list, in the commit that
accepts it.

ELL-1's test steps are out of scope. ELL-1 is closed, and its window is never reused (ADR-008).

### 2. A look, in three tiers

**A look is either of these:**
- any computation from a held-out set's data;
- a person or agent reading a figure computed from one, which no signed gate file prints, while
  working on a pre-registration, review or batch for that set.

Re-reading a signed gate file, or any figure it prints, is not a look.

**A look's tier is the highest it reaches:**
- **structure:** label-free facts about the set's inputs, such as sizes, dates, degrees, feature
  ranges and graph views;
- **labels:** any statistic that uses the set's labels without a model, such as a count, a
  prevalence, or anything conditioned on the label;
- **scores:** any model output or metric on the set.

### 3. What each tier allows

- **At every tier, never** train on a held-out set, or tune or select anything against it.
  CLAUDE.md is unchanged on this.
- **Structure:** allowed, once it is recorded.
- **Labels:** only when an accepted ADR or pre-registration names the look first, and only once it
  is recorded. A draft cannot authorise its own look.
- **Scores:** only in a batch that an accepted pre-registration (gated) or an accepted reported-only
  ADR names, opened through the pre-registration guard.
- **An unauthorised look,** such as an agent's slip or an incidental sighting, is recorded as soon as
  it is known. Its authority reads "none", with the reason. It does not lift clause 4.

### 4. The post-look rule

Once a set has had a labels or scores look, every later choice in a pre-registration on that set
must either make its gate harder to pass or rest entirely on label-free evidence. That covers a
change to an existing pre-registration and a choice in a new one. Each such pre-registration
discloses the set's ledger rows (clause 6).

This is ADR-011's rule (ADR-011:106-110), generalised from one gate to every pre-registration on a
set that has been looked at. It already binds DGF-1's test window, which has had both kinds of look.

### 5. The ledger

**Where and what.** The ledger is `experiments/heldout_ledger.csv`. It is append-only like the
registry, and committed. One row per look, with these columns:
- `timestamp_utc`, `held_out_set`, `tier`;
- `what`: the look named, never its value. The repository is public, and a figure is refused;
- `by`, `via`;
- `authority`: the accepted document, or "none: …";
- `git_commit`.

**One function writes every row,** before the held-out data is read, and flushes it to disk:
`record_look` in `scripts/heldout_ledger.py`. It refuses:
- an unknown set or tier;
- an empty field, or a figure in `what`;
- a labels or scores look whose authority is not an accepted document.

**Three writers:**
- **the pre-registration guard:** as its last step, once every refusal check has passed, it records
  a scores look naming its document. Every batch that reads a held-out set opens it through the
  guard, whether gated or reported-only;
- **every committed script that reads a held-out set outside the guard:** each records its look
  before its first held-out read, taking its authority as an argument;
- **a person, by hand,** through the same module's command line: for an agent's look or an
  incidental one. Only this path may record a look as incidental.

**`git_dirty` ignores the ledger** (`gbe/run/config.py`, `IGNORED_DIRTY_PATHS`), as ADR-002 made it
ignore the registry. This changes `gbe/`, so ADR-008's regression check is re-run after it. No batch
that relies on the trainer runs before that re-run passes.

### 6. Disclosure comes from the ledger

Every gate or reported-only file assembled after this ADR lists what was seen before its batch from
the ledger. Its list is the set's rows dated before the batch's own guard row, and nothing typed by
hand. Gate 3's assembler is the first, and its test fails if a ledger row is missing from the
disclosure.

### 7. Tests (implementation, plan row 5a)

| invariant | test | prototype evidence |
|---|---|---|
| `record_look`'s refusals; one header; rows appended, never rewritten | `tests/test_heldout_ledger.py` | 6 of the 25 mutants |
| a labels or scores look needs an accepted document; an incidental one says "none" | same | 4 mutants |
| the committed ledger is a prefix of the working one | same, against `git show HEAD:` | 2 mutants |
| the guard records one scores row naming its document, and nothing when it refuses | the guard's tests | 4 mutants |
| every committed script that reads a held-out set records its look or passes the guard; only the hand path records an incidental look | `tests/test_heldout_readers.py` | 6 mutants |
| the disclosure lists earlier rows for its set only, in time order | `tests/test_heldout_ledger.py` | 3 mutants |
| a ledger append leaves `git_dirty` false | `tests/test_git_dirty.py` | tested in the prototype |

The prototype's 25 mutants are re-run on the real module, and the output is cited in the
implementing commit (`notebooks/measurements/2026-10-08-heldout-ledger/prototype/`).

The readers' lint finds a script that loads the DGraph data (`load_dgraph`, `prepare_dgraph`, or
PyG's `DGraphFin(`) and names a held-out set (`splits["test"]`, `test_mask`), or one that reads gate
score files (`load_scores(`). Today it flags the four scripts in the inventory. Row 5a makes each one
record its look:
- `scripts/measure_dgf1_temporal_split.py` and `scripts/audit_dgf1_datasource.py`, at the labels
  tier;
- `scripts/verify_dgraph_snapshot.py`, at structure;
- `scripts/assemble_gate_dgf1_1.py`, at scores.

### 8. Backfill (plan row 5a)

DGF-1's looks before this ADR are entered from the records. Each row has `via` set to "backfill"
plus its source, and a timestamp from git or the lab, marked approximate where only a date exists.
The sources:
- the corrected disclosure list in `gates/ERRATUM-DGF1-1.md`;
- the Gate-1 batch (`09c6e72`) and its assembly;
- the looks at the official test mask (`scripts/verify_dgraph_snapshot.py`,
  `scripts/measure_dgf1_temporal_split.py`; ADR-010, ADR-011);
- agents' exposures recorded in the audit (`notebooks/audit/2026-10-07/README.md` §6) and in later
  reviews (each added preamble location).

### 9. The leakage rule's wording, and a leakage sheet

Every DataSource keeps its automated leakage test. Where the data cannot support one (labels without
dates, features snapshotted after the cutoff), the DataSource documents the limitation instead, and
every gate file on that DataSource carries it.

Each new DataSource's Phase-0 ADR has a leakage sheet. It answers Kapoor & Narayanan's eight leakage
types (`notebooks/audit/2026-10-07/literature-recheck.md` §4) with a test or a documented limitation
for each. EDR-1's Phase-0 ADRs are the first (ADR-020).

### 10. DGF-1's qualification 4

CLAUDE.md's list of Gate-1 qualifications gains the one the gate file states
(GATE-DGF1-1:215-217): the two clauses share one floor arm; the estimand is seed variability on one
window; undated labels and snapshot features make both arms' absolute numbers optimistic.

## Alternatives rejected

- **A ledger written only by the guard** (the 2026-10-07 decision). It would have missed four of
  six readers (`notebooks/measurements/2026-10-08-heldout-ledger/output.txt`), and every DGF-1 look
  before Gate 1. The researcher widened it on 2026-10-08.
- **Hand rows only.** This is what produced GATE-DGF1-1's incomplete disclosure.
- **Two other fixes for the dirty tree:**
  - the guard writes the row and stops, and the batch restarts once the row is committed: one extra
    step per batch, and no `gbe/` change;
  - rows go to an ignored file that is folded into the ledger later: more moving parts, and a row can
    be left unfolded.

  The researcher chose ADR-002's precedent instead.
- **Label looks allowed whenever they are recorded.** This is weaker than CLAUDE.md's "never peek",
  and ADR-011's own prevalence look would still have been free.
- **The post-look rule only for changes to an existing pre-registration,** as ADR-011 worded it.
  Gate 3 and matched-time's stage 2 are new pre-registrations on a window that Gate 1 has already
  scored. Under the narrow rule, nothing would bind them.
- **A hash chain over the rows.** The ledger is committed, so git already shows any rewrite, and the
  append-only test refuses one.
- **Recording every re-read of a signed gate file.** Those figures are public record, and the rows
  would bury the looks that matter.

## Consequences

**On acceptance, these amendments are applied and committed to the governance repository:**
- **`CLAUDE.md:166-167`** becomes the existing sentence, then: "A look at a held-out set has three
  tiers (ADR-021): **structure** (label-free facts), **labels** (any statistic using its labels) and
  **scores** (any model output on it). Every look is recorded in `experiments/heldout_ledger.csv`
  before it happens. A labels look needs an accepted document that names it first; a scores look
  runs only through the pre-registration guard. After a labels or scores look at a set, every later
  choice in a pre-registration on it must make its gate harder to pass or rest entirely on
  label-free evidence."
- **`CLAUDE.md:187-189`** gains: "Where the data cannot support such a test (labels without dates,
  features snapshotted after the cutoff), the DataSource documents the limitation, and every gate
  file on it carries it. Each new DataSource's Phase-0 ADR answers Kapoor & Narayanan's eight leakage
  types, each with a test or a documented limitation (ADR-021)."
- **DGF-1's qualifications** gain: "**Seed-only, on one window.** The two clauses share one floor arm,
  so they are not two confirmations. The estimand is seed variability on this fixed window, and
  undated labels and snapshot features make both arms' absolute numbers optimistic."
- ***Next* item 4** records the acceptance and row 5a.
- **`docs/timeline.md`:** row 5 gets its actual date. Row 5a reads "every held-out reader writes the
  ledger, the guard included; `git_dirty` ignores it, with ADR-008's re-run; DGF-1's looks
  backfilled".
- **`docs/subagent-preamble.md`:** its "report its `path:line`" sentence adds "so the main session
  can record it in the ledger".

**What gets harder:**
- every new held-out reader needs an authority and a ledger call;
- a labels look needs an accepted document first;
- a `gbe/` change costs one ADR-008 re-run.

**What gets easier:** a gate's disclosure is generated, not remembered.

## Revisit when

- a held-out set appears that is not a fixed slice of one dataset (for example, EDR-1's rolling
  cutoffs);
- or the readers' lint misses a reader that is found later. The lint then gains that loader, with a
  mutant that shows the gap.

## Draft history

- **Draft 1** (2026-10-08).
