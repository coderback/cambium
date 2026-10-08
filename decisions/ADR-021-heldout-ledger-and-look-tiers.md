# ADR-021 — The held-out ledger, the three tiers of a look, and the leakage rule's wording

**Status:** proposed
**Date:** proposed 2026-10-08 · draft 2 2026-10-08
**Tier:** A. It sets the held-out and leakage rules (CLAUDE.md *Process*, ADR-017).
**Review rounds:** 1/4
**Deciders:** coderback
**Implemented by:** plan row 5a, in the order clause 11 gives. The work starts at least 12 hours
after acceptance and not on the same calendar day. This ADR must be accepted before EDR-1 starts
(ADR-020 condition 6), and before the next look at a held-out set (CLAUDE.md, *Next*).

**Docs affected:**
- `CLAUDE.md`:
  - `:166-167` (held-out data);
  - `:187-189` (the leakage-test rule);
  - DGF-1's qualifications (`:101-109`);
  - *Next* item 4 (`:70-71`).
- `docs/timeline.md`, rows 5 and 5a.
- `docs/subagent-preamble.md`: an agent's report of a held-out figure becomes a ledger row. This adds
  a duty and narrows no ban.

**Accepted ADRs, unchanged:**
- **ADR-016:** its mode → document map is untouched. Its admitted gap, that library entry points
  accept the test split (ADR-016:92-94), is closed by clause 5's accessors.
- **ADR-015:** its batch opens the official test mask through clause 5's accessor, under its own
  authority. It already names its runner and assembler (ADR-015:435, :552), so it passes clause 3's
  check without amendment.

## Context

**"Peek" has no definition, and the rule learned from a real breach is only in ADR-011** (audit D8,
`notebooks/audit/2026-10-07/D-governing-docs.md:52`). CLAUDE.md says "never … peek at any held-out
eval set" (`CLAUDE.md:166`). Yet DGF-1's split design computed the test window's prevalence on
purpose (ADR-011:78-84), and a reviewing agent computed test-window label statistics by mistake
(ADR-011:92-95). After that slip, ADR-011 made a rule (ADR-011:106-110). Any later change had to make
Gate 1 harder for the GNN, or rest entirely on label-free evidence. That rule never reached
CLAUDE.md.

**GATE-DGF1-1's disclosure was incomplete.** It named two kinds of look before its batch, and the
corrected list has five. One of them, which holds two facts, no ADR discloses
(`gates/ERRATUM-DGF1-1.md`, addendum). Disclosure written from memory misses things.

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
  four of them do so outside the guard (`notebooks/measurements/2026-10-08-heldout-ledger/output.txt`).
- **Round 1 found more routes than that.**
  - The runners open the window as `dgf1_splits()[window]`.
  - The adapter exposes the mask as `official_test_mask`.
  - The library functions accept the test split from any caller (ADR-016:92-94).

  So per-caller recording cannot be complete, and recording moves to where the data is handed out.
- **Prototype v2** has 41 tests, and 36 of 36 mutants are killed
  (`notebooks/measurements/2026-10-08-heldout-ledger/prototype/mutation-output.txt`). It covers the
  ledger, the accessors, the authority check, overlapping sets, the append-only dirty check and the
  backstop lint. v1 is kept in git at `012a023`. The prototype found three traps:
  - **Two scripts load through PyG's `DGraphFin`,** not the adapter.
  - **A ledger row written at a batch's start makes the tree dirty**, under today's `git_dirty`.
  - **A text lint flags its own fixtures,** so the backstop parses code instead.

**Decided by the researcher on 2026-10-08** (`notebooks/lab/2026-10-08.md`):
- **Before draft 1:**
  - every held-out reader writes the ledger, enforced by test;
  - `git_dirty` ignores the ledger;
  - a labels look needs an accepted document that names it first;
  - the prototype is kept in `notebooks/measurements/`.
- **After round 1:**
  - looks are recorded at the adapter's accessors;
  - the held-out parts of the two scripts that looked at test labels before any document
    authorised it are retired.

## Decision

### 1. Held-out sets

A held-out set is data that a pre-registration or a reported-only ADR reserves for its batch. There
are two today:
- `dgf1-temporal-test-482-821`, DGF-1's temporal test window (ADR-011);
- `dgraph-official-test`, DGraph's official test mask (ADR-010, ADR-015).

**Overlapping sets count together.** Each set lists the sets it shares units with. These two share
users (ADR-015:234), so a look at either counts, for clauses 4 and 6, as a look at both. A later
document that reserves a set adds it, with its overlaps, in the commit that accepts it.

ELL-1's test steps are out of scope. ELL-1 is closed, and its window is never reused (ADR-008).

### 2. A look, in three tiers

**A look is either of these:**
- any computation from a held-out set's data;
- anyone reading a figure computed from one that no signed gate file prints. That includes reading
  a held-out row of `experiments/registry.csv`, whose metrics are scores.

**Not looks:**
- re-reading a signed gate file, or a figure it prints;
- the loader's integrity checks over the whole snapshot, which compare whole-dataset totals with
  the figures the dataset's own publication reports (ADR-010:24).

**A look's tier is the highest it reaches:**
- **structure:** label-free facts about the set's inputs, such as sizes, dates, degrees, feature
  ranges and graph views;
- **labels:** any statistic that uses the set's labels without a model. A statistic pooled over
  several sets is a labels look at every held-out set it includes;
- **scores:** any model output or metric on the set.

### 3. What each tier allows

- **At every tier, never** train on a held-out set, or tune or select anything against it.
  CLAUDE.md is unchanged on this.
- **Structure:** allowed, once it is recorded. Its authority is a stated reason.
- **Labels and scores:** only under an accepted document that named the look first. That means the
  document is accepted, its acceptance date is on or before the look, and it names the script making
  the look. A draft cannot authorise its own look. Scores are produced only in a batch that an
  accepted pre-registration (gated) or an accepted reported-only ADR names.
- **An assembler reads under its batch's authority.** Its look is recorded like the batch's, and
  that document must name the assembler too.
- **An unauthorised look,** such as an agent's slip or an incidental sighting, is recorded as soon as
  it is known. Its authority reads "none", with the reason, and it goes through the hand path. It
  does not lift clause 4.

### 4. The post-look rule

Once a set, or a set it overlaps, has had a labels or scores look, every later choice in a
pre-registration or reported-only ADR on it must do one of these:
- use no held-out label or score from any set. Validation evidence, such as a pilot that sizes a
  seed count, is allowed;
- or only remove claims (ADR-019:96's framing).

**In reported-only work, an informed choice is allowed,** provided it is disclosed as informed (as
in ADR-013:346-348).

Each such document discloses the overlapping sets' ledger rows (clause 6). This generalises
ADR-011's rule (ADR-011:106-110) from one gate to every document on a set that has been looked at.
It already binds DGF-1's test window and the official test mask.

### 5. The ledger, written at the data's chokepoint

**Where and what.** The ledger is `experiments/heldout_ledger.csv`. It is append-only like the
registry, and committed. One row per look, with these columns:
- `timestamp_utc`, `held_out_set`, `tier`;
- `what`: the look named, never its value. The repository is public, so a standalone number is
  refused, and a digit inside a name such as `gate3` or `ADR-015` is not;
- `by`, `via`;
- `authority`: the accepted document, a structure reason, or "none: …";
- `git_commit`.

**One function writes every row,** `record_look` in `scripts/heldout_ledger.py`. It flushes and
syncs the row to disk before returning. It refuses an unknown set or tier, an empty field, a figure
in `what`, and an authority that fails clause 3.

**The chokepoint.** The DGF-1 adapter stops handing out held-out parts by default:
- `dgf1_splits()` returns the validation window only;
- `load_dgraph()` drops the official test mask;
- the raw snapshot loader for the verifier also drops it.

Each part comes only from an accessor that records the look and then returns it:
`open_test_window(…)` and `open_official_test_mask(…)`. Each takes the tier, authority and `via`.
- **The gated runners** call the accessor after the pre-registration guard has passed, with the
  guard's document as the authority. ADR-016's map is unchanged.
- **ADR-015's runner** calls `open_official_test_mask` under ADR-015.
- **An assembler** calls the accessor under its batch's document.

A failed attempt leaves its row, which records an open that may have read nothing. Clause 6 handles
it.

**The backstop.** A test parses every committed Python file under `scripts/`, `adapters/dgf1/` and
`notebooks/`. It skips the accessor modules, `test_*.py` files (synthetic fixtures) and
`…/prototype/` folders. It fails on any of these:
- a call to PyG's `DGraphFin(` or to `TemporalSplit(`;
- an attribute or subscript ending in `test_mask`;
- a call to `load_scores(` in a file that calls no accessor;
- an `incidental=` or `backfill=` keyword outside the hand path.

Its limit, stated rather than hidden: it is per file. A file that opens a window somewhere and reads
validation score files elsewhere passes it, so the accessor, not the backstop, is the guarantee.

**The hand path** (`scripts/heldout_ledger.py`'s command line) records an incidental look as "none:
…". It also records a backfill row with `backfill=True`, which keeps the given authority, checks
nothing, and starts `via` with "backfill:". Only this path may use either.

**What `git_dirty` ignores.** `git_dirty` ignores the ledger only while the working ledger is HEAD's
ledger with rows appended (`gbe/run/config.py`). Any other edit makes the tree dirty. This changes
`gbe/`, so ADR-008's regression check is re-run after it, and no batch that relies on the trainer
runs before that re-run passes. It compounds CLAUDE.md's open question of whether DGF-1's repro-check
is re-run (`CLAUDE.md:96-97`). The researcher decides both together.

### 6. Disclosure comes from the ledger

Every gate or reported-only file assembled after this ADR builds its "seen before this batch" list
mechanically, from two sources:
- the ledger's rows for its set and the sets it overlaps, dated before its batch's first opener row
  at the batch's commit;
- its validation runs, from the registry.

Nothing is typed by hand. The assembler also refuses in two cases:
- a held-out registry row of its batch predates that opener row;
- that opener row is not in HEAD's ledger.

The first assembler written after this ADR, matched-time's or Gate 3's, is the first to do this.
Its test fails if a ledger row is missing from the disclosure.

### 7. Tests (implementation, plan row 5a)

| invariant | test | prototype mutants |
|---|---|---|
| the ledger file: appended, one header, synced before return | `tests/test_heldout_ledger.py` | 3 |
| `record_look`'s field checks, the figure filter included | same | 6 |
| clause 3's authority check: missing, proposed, accepted after the look, naming nothing; labels without a document; an incidental look claiming one | same | 6 |
| append-only, and `git_dirty` ignoring only an append to HEAD's ledger | same, and `tests/test_git_dirty.py` | 3 |
| the ordinary paths carry no held-out part; each accessor records before it returns or loads | `tests/test_dgf1_heldout_access.py` | 4 |
| disclosure covers overlapping sets' earlier rows, in time order | `tests/test_heldout_ledger.py` | 3 |
| the backstop's routes and its exclusions | `tests/test_heldout_readers.py` | 11 |

The prototype's 36 mutants are re-run on the real modules, and the output is cited in the
implementing commit.

### 8. The scripts that read held-out data today

The inventory found four, plus the two runners:
- **Retired from held-out use:** the held-out parts of `scripts/measure_dgf1_temporal_split.py` and
  `scripts/audit_dgf1_datasource.py` refuse. Their validation parts stay. No accepted document named
  their label looks first, and none will.
- **Retired whole:** `scripts/assemble_gate_dgf1_1.py` refuses at its start. GATE-DGF1-1 is signed,
  and ADR-016 closed its window.
- **`scripts/verify_dgraph_snapshot.py`** loads through the adapter. It opens the official test mask
  at the structure tier, for snapshot verification (ADR-010).
- **`scripts/run_dgf1_floor.py` and `scripts/run_dgf1_gnn.py`:** their gate branches call the
  accessor after the guard. Gate mode stays closed (ADR-016).

### 9. Backfill (plan row 5a)

DGF-1's looks before this ADR are entered through the hand path as backfill rows. Each row's `via`
names its source, and its authority is one of two:
- the document the look ran under, as for the Gate-1 batch under ADR-012;
- or "none: before any authorising document", as for ADR-011's measurement script and the review
  agent's slip.

**Times.** Exact times come from run ids or git. Where a source gives only a date, the row is written
at midnight UTC, and `via` says so.

The sources:
- the corrected disclosure list in `gates/ERRATUM-DGF1-1.md`;
- the Gate-1 batch (`09c6e72`) and its assembly;
- the looks at the official test mask (ADR-010, ADR-011);
- agents' exposures recorded in the audit (`notebooks/audit/2026-10-07/README.md` §6) and in later
  reviews (each added preamble location).

The backfill lands before the matched-time pre-registration (plan row 6a) is accepted, because that
document's disclosure comes from the ledger.

### 10. The leakage rule's wording, a leakage sheet, and qualification 4

**The leakage-test rule.** Every DataSource keeps its automated leakage test. Where the data cannot
support one (labels without dates, features snapshotted after the cutoff), the DataSource documents
the limitation instead, and every gate file on that DataSource carries it.

**The leakage sheet.** Each new DataSource's Phase-0 ADR has one. It answers Kapoor & Narayanan's
eight leakage types (`notebooks/audit/2026-10-07/literature-recheck.md` §4) with a test or a
documented limitation for each. EDR-1's Phase-0 ADRs are the first (ADR-020).

**Qualification 4.** CLAUDE.md's Gate-1 list gains the qualification the gate file states
(GATE-DGF1-1:215-217).

### 11. Implementation order (plan row 5a)

1. **The `git_dirty` change and its test.** Then ADR-008's re-run in the researcher's terminal, with
   the DGF-1 repro-check question decided alongside it.
2. **`scripts/heldout_ledger.py`** and its tests.
3. **The adapter's accessors** and their tests.
4. **The runners, the four scripts and the backstop test.**
5. **The backfill**, committed before plan row 6a is accepted.

The prototype's mutants are re-run at step 4.

## Alternatives rejected

- **A ledger written only by the guard** (the 2026-10-07 decision). It would have missed four of six
  readers (`notebooks/measurements/2026-10-08-heldout-ledger/output.txt`), and round 1 found more.
- **Draft 1's design:** the guard records, each script records, and a regex lint catches the rest.
  Round 1 showed the runners' `dgf1_splits()[window]` and the adapter's `official_test_mask` slip
  past it. It also sent ADR-015's batch through a guard ADR-016 forbids it to use.
- **Hand rows only.** This is what produced GATE-DGF1-1's incomplete disclosure.
- **Two other fixes for the dirty tree:**
  - the guard writes the row and stops, and the batch restarts once the row is committed;
  - rows go to an ignored file, folded into the ledger later.

  The researcher chose ADR-002's precedent, narrowed here to appends only.
- **Label looks allowed whenever they are recorded.** This is weaker than CLAUDE.md's "never peek".
- **An authority that need only be accepted.** Any accepted ADR would then authorise any look,
  ADR-012 included.
- **The post-look rule as ADR-011 worded it,** "harder for the GNN, or label-free". It cannot be met
  by validation-sized seed counts, and it means nothing for reported-only work.
- **Keeping the two label scripts runnable** under a future document. The researcher retired their
  held-out parts.
- **A hash chain over the rows.** The ledger is committed, and the append-only check refuses a
  rewrite.

## Consequences

**On acceptance, these amendments are applied and committed to the governance repository.**

**`CLAUDE.md:166-167`** becomes:

> - Never train on, tune against, or peek at any held-out eval set (temporal-holdout test periods,
>   CRUXEval, SWE-bench Verified, etc.). **A look** at a held-out set has three tiers (ADR-021):
>   - **structure:** label-free facts;
>   - **labels:** any statistic using its labels;
>   - **scores:** any model output on it.
>
>   **Recording.** Every look is recorded in `experiments/heldout_ledger.csv`, where the adapter
>   hands the data out, or by hand as soon as an unplanned look is known. Re-reading a signed gate
>   file is not a look.
>
>   **Labels and scores** need an accepted document that named the look first, and scores come only
>   from a gated or reported-only batch it names.
>
>   **A peek** is any look that is unrecorded or unauthorised.
>
>   **After a labels or scores look at a set,** a later choice in a document on it uses no held-out
>   label or score, or only removes claims, or (in reported-only work) is disclosed as informed.

**`CLAUDE.md:187-189`** gains:

> Where the data cannot support such a test (labels without dates, features snapshotted after the
> cutoff), the DataSource documents the limitation, and every gate file on it carries it. Each new
> DataSource's Phase-0 ADR answers Kapoor & Narayanan's eight leakage types, each with a test or a
> documented limitation (ADR-021).

**DGF-1's qualifications** gain:

> **Seed-only, on one window.** The two clauses share one floor arm, so they are not two
> confirmations. The estimand is seed variability on this fixed window, and undated labels and
> snapshot features make both arms' absolute numbers optimistic.

***Next* item 4** records the acceptance and row 5a, in the same two lines, so no line the preamble
cites moves. The qualification and the edits at `:166` and `:187` all fall after `:102`. If any
line the preamble cites does move, the preamble is renumbered in the same commit.

**`docs/timeline.md`:** row 5 gets its actual date, and row 5a reads "accessors and ledger; the
`git_dirty` change with ADR-008's re-run; the scripts; DGF-1's backfill, before row 6a".

**`docs/subagent-preamble.md`:** its "report its `path:line`" sentence adds "so the main session can
record it in the ledger".

**What gets harder:**
- every held-out read needs an accessor and an authority that named it;
- the two label scripts lose their held-out parts;
- a `gbe/` change costs one ADR-008 re-run.

**What gets easier:** a gate's disclosure is generated, not remembered.

## Revisit when

- A held-out set appears that is not a fixed slice of one dataset, for example EDR-1's rolling
  cutoffs, or one held in a new adapter. That adapter gets its own accessors before any look.
- Or a route around the accessors is found. The backstop then gains it, with a mutant.

## Draft history

- **Draft 1** (2026-10-08, `0e8aaf1`).
- **Round 1** (2026-10-08): a full Tier-A review by a fresh subagent with the standing preamble.
  - **Tier A confirmed.**
  - **4 blocking findings:**
    - ADR-015's batch and the assemblers could not meet the scores rule;
    - the lint missed the runners' `dgf1_splits()[window]`, `official_test_mask` and library calls;
    - the post-look rule could not be met (validation evidence, reported-only work);
    - the two held-out sets overlap.
  - **8 should-fix findings and 4 nits.**
- **The researcher's decisions** (2026-10-08): record at the adapter's accessors, and retire the
  two label scripts' held-out parts.
- **Prototype v2** (2026-10-08): 41 tests, and 36 of 36 mutants killed.
- **Draft 2** (2026-10-08):
  - **Blocking:**
    - the accessors are the chokepoint, ADR-015 opens through one, and assemblers read under their
      batch's authority (clauses 3 and 5);
    - an AST backstop with its limit stated (clause 5);
    - clause 4 reworded to "no held-out label or score, or only removes claims, or disclosed as
      informed in reported-only work";
    - overlapping sets (clauses 1, 4 and 6).
  - **Should-fix:**
    - loader integrity checks and pooled statistics tiered, and the purpose qualifier dropped
      (clause 2);
    - an authority must have named the look first, and the backfill authority rule is stated
      (clauses 3 and 9);
    - `git_dirty` ignores appends only (clause 5);
    - the opener row and the assembler's refusals (clause 6);
    - validation runs in the disclosure, and "the first assembler" corrected (clause 6);
    - the CLAUDE.md texts define "peek", carry both recording routes and the signed-gate exemption,
      and keep every cited line in place;
    - missing mutants added (fsync, the figure filter, load order, the backstop's routes);
    - an implementation order (clause 11).
  - **Nits:**
    - the erratum's undisclosed look is one kind;
    - the hand path is named;
    - date-only backfill times are written at midnight UTC;
    - `incidental=` is caught whatever its value.
