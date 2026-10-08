# ADR-021 — The held-out ledger, the three tiers of a look, and the leakage rule's wording

**Status:** proposed
**Date:** proposed 2026-10-08 · draft 5 2026-10-08
**Tier:** A. It sets the held-out and leakage rules, and it amends a Tier-A ADR (CLAUDE.md
*Process*, ADR-017).
**Review rounds:** 4/4, spent. Round 4's one blocking finding was escalated, and the researcher moved
it into tests (2026-10-08). Its test and mutant are in prototype v5 (I6, I7).
**Deciders:** coderback
**Implemented by:** plan row 5a, in the order clause 11 gives. The work starts at least 12 hours
after acceptance and not on the same calendar day. This ADR must be accepted before EDR-1 starts
(ADR-020 condition 6), and before the next look at a held-out set (CLAUDE.md, *Next*).
**Amends ADR-015, narrowly** (clause 12). It changes:
- the identity exclusions;
- the "HEAD equals the working tree" sentence;
- where the mask check and `data_identity` get their tensors;
- four "Held-out look" lines;
- one disclosure.

ADR-015 is unimplemented, and it needs row 5a's accessors anyway, so its restarted wait costs
nothing.

**Docs affected:**
- `CLAUDE.md`:
  - `:166-167` (held-out data);
  - `:187-189` (the leakage-test rule);
  - DGF-1's qualifications (a bullet after `:108-109`);
  - *Next* item 4 (`:70-71`).
- `decisions/ADR-015-dgf1-official-split-positioning.md` (clause 12).
- `docs/timeline.md`, rows 5 and 5a.
- `docs/subagent-preamble.md`: an agent's report of a held-out figure becomes a ledger row. This adds
  a duty and narrows no ban.

**ADR-016 is unchanged.** Its mode → document map is untouched. Its admitted gap, that library entry
points accept the test split (ADR-016:92-94), is closed by clause 5.

## Context

**"Peek" has no definition, and the rule learned from a real breach is only in ADR-011** (audit D8,
`notebooks/audit/2026-10-07/D-governing-docs.md:52`). CLAUDE.md says "never … peek at any held-out
eval set" (`CLAUDE.md:166`). Yet DGF-1's split design computed the test window's prevalence on
purpose (ADR-011:78-81), and a reviewing agent computed test-window label statistics by mistake
(ADR-011:92). After that slip, ADR-011 made a rule (ADR-011:106-110). Any later change had to make
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

**From the measurement, the prototypes and three review rounds** (ADR-017 item 3;
`notebooks/measurements/2026-10-08-heldout-ledger/`):
- **The guard sees a third of the paths.** Six committed scripts can read a held-out DGF-1 set, and
  four of them do so outside the guard (`notebooks/measurements/2026-10-08-heldout-ledger/output.txt`).
- **Each review round found routes the previous design left open:**
  - **Round 1:** `dgf1_splits()[window]`, `official_test_mask`, and library calls with the test
    split.
  - **Round 2:** every user's label is loaded for every caller, so `data.y[data.node_time >= 482]`
    needs no window object.
  - **Round 3:** the official train and validation masks together mark exactly the labelled users,
    so with node times they say which hidden users are labelled. Hiding labels by node time also
    broke ADR-015's official track.
  - **Round 4:** the adapter builds `labelled_mask` from the raw labels
    (`adapters/dgf1/datasource_dgraph.py:120`), so it too marks which hidden users are labelled.

  The guarantee is therefore what the default load leaves out, not who calls what.
- **Prototype v5** has 68 tests, and 79 of 79 mutants are killed
  (`notebooks/measurements/2026-10-08-heldout-ledger/prototype/mutation-output.txt`). Each mutant is
  named for the invariant it breaks. v1 is at `012a023`, v2 at `7f15bce`, v3 at `91d2031`, and v4 at
  `175d6ab`.

**Decided by the researcher on 2026-10-08** (`notebooks/lab/2026-10-08.md`):
- **Before draft 1:**
  - every held-out reader writes the ledger, enforced by test;
  - `git_dirty` ignores the ledger;
  - a labels look needs an accepted document that names it first;
  - the prototype is kept in `notebooks/measurements/`.
- **After round 1:**
  - looks are recorded at the adapter's accessors;
  - the two label scripts' held-out parts are retired.
- **After round 2:**
  - held-out labels are hidden by default;
  - ADR-015 is amended narrowly, here;
  - sets are scoped by track.
- **After round 3:**
  - both round-3 blockers are fixed;
  - the mechanism is stated as invariants, each pinned by a named test and its mutants (clause 5).
- **After round 4,** with the review budget spent: its blocking finding, `labelled_mask`, is moved
  into tests (CLAUDE.md *Process*, escalation), and its should-fix items and nits are folded in.

## Decision

### 1. Held-out sets, scoped by track

A held-out set is data that a pre-registration or a reported-only ADR reserves, and it is held out
from **the track that reserves it**:
- `dgf1-temporal-test-482-821`: DGF-1's temporal test window, held out from the temporal track's
  fitting, selection and design (ADR-011);
- `dgraph-official-test`: DGraph's official test mask, held out from the official track's fitting
  and selection (ADR-015).

**Overlaps.** The two sets share users (ADR-015:234). So a look at either counts, for clauses 4 and
6, as a look at both.

**Two overlaps are design, not looks.** Each track's use of the other track's held-out users, inside
its own training and validation data, is that track's design. Each is disclosed once in the
reserving ADR:
- **the temporal track's training and validation windows** contain official-test users
  (ADR-015:430);
- **the official track's training and validation masks** contain temporal-test users. Clause 12 adds
  this disclosure to ADR-015.

A later document that reserves a set adds it, with its track and overlaps, in the commit that accepts
it. ELL-1's test steps are out of scope. ELL-1 is closed, and its window is never reused (ADR-008).

### 2. A look, in three tiers

**A look is either of these:**
- any computation from a held-out set's data;
- anyone reading a figure computed from one that no signed gate file prints. That includes reading
  a held-out row of `experiments/registry.csv`, whose metrics are scores.

**Not looks:**
- re-reading a signed gate file, or a figure it prints;
- reading the dataset's own publication, including its whole-snapshot class totals (ADR-010:24);
- the adapter's whole-snapshot integrity check (invariant I9), which compares totals with that
  publication and returns only pass or fail.

These two exemptions take precedence over the pooled rule below. **But a difference is a look:** the
published totals less the labels the default load shows are the hidden users' totals. Computing
that is a labels look at `dgf1-temporal-test-482-821`, and it is recorded. I6 hides each user's
label; it does not hide aggregates someone else published.

**A look's tier is the highest it reaches:**
- **structure:** label-free facts about the set's inputs, such as sizes, dates, degrees, feature
  ranges and graph views;
- **labels:** any statistic that uses the set's labels without a model, or a mask derived from them.
  A statistic pooled over several sets is a labels look at each held-out set it includes, apart
  from clause 1's design overlaps;
- **scores:** any model output or metric on the set.

### 3. What each tier allows

- **At every tier, never** train on a held-out set, or tune or select anything against it.
  CLAUDE.md is unchanged on this.
- **Structure:** allowed, once it is recorded. Its authority is a stated reason.
- **Labels and scores: only under an accepted document that named the look first.** That means all
  of these:
  - the document is accepted in HEAD, and it waited a night: its latest change is on an earlier
    calendar day than the look, and at least 12 hours earlier. The latest change is the latest date
    on its Status and Date lines: an acceptance, an amendment or an erratum, so every restarted wait
    is honoured. A date with no recorded time counts as 23:59 that day. The calendar is the
    researcher's local time zone, a named constant in the module;
  - it carries a line `**Held-out look:** <set> · <tier> · <script> · <batch>` naming this set,
    tier, script and batch label. A scores line covers labels;
  - that script is the running `__main__`, taken from the process;
  - the look's time is the clock's. Only the hand path may give a time, and only an earlier one.

  A prose mention authorises nothing, and a draft cannot authorise its own look. A batch label
  belongs to one authority: a row whose label already appears under another authority is refused. Scores are produced
  only in a batch that an accepted pre-registration (gated) or an accepted reported-only ADR names.
- **An assembler reads under its batch's authority.** That document carries a line for the assembler
  too. An assembler that reads held-out registry rows records its look directly with `record_look`.
- **An unauthorised look,** such as an agent's slip or an incidental sighting, is recorded as soon as
  it is known. Its authority reads "none", with the reason, and it goes through the hand path. It
  does not lift clause 4.

### 4. The post-look rule

Once a set, or a set it overlaps, has had a labels or scores look, every later choice in a
pre-registration or reported-only ADR on it must do one of these:
- **use no held-out label or score.** "Use" means the value could change the choice. Validation
  evidence, such as a pilot that sizes a seed count, is allowed. So is a figure a signed gate file
  prints, when it is cited as a result and not used to tune;
- **or only remove claims.** That includes making the gated claim harder to earn, ADR-011's branch
  (ADR-011:106-107), and ADR-019:96's framing.

**In reported-only work, an informed choice is allowed,** provided it is disclosed as informed (as
in ADR-013:346-348).

Each such document discloses the overlapping sets' ledger rows (clause 6). This generalises
ADR-011's rule (ADR-011:106-110) from one gate to every document on a set that has been looked at.
It already binds DGF-1's test window and the official test mask.

### 5. The mechanism, as invariants

**The ledger** is `experiments/heldout_ledger.csv`: append-only like the registry, and committed. Its
columns are:
- `timestamp_utc`, `held_out_set`, `tier`;
- `what`: the look named, never its value. A standalone number is refused, and a digit inside a name
  such as `gate3` is not;
- `by`, `via`;
- `authority`: the accepted document, a structure reason, or "none: …";
- `batch`: the label of the batch the look belongs to;
- `git_commit`.

**Where the code lives.** `record_look` and the accessors live in `adapters/dgf1/heldout.py`, and the
hand path's command line in `scripts/heldout_ledger.py`. The adapter imports nothing from
`scripts/`.

**Each invariant below is pinned by a named test and by the prototype's mutants, which
`mutation-output.txt` lists under their invariant's number.** Row 5a writes the tests on the real
modules and re-runs those mutants.

| # | invariant | test |
|---|---|---|
| I1 | `record_look` appends one row under one header, and syncs it to disk before returning | `tests/test_heldout_ledger.py` |
| I2 | it refuses an unknown set or tier, an empty field (`batch` included), and a figure in `what` | same |
| I3 | a labels or scores look needs clause 3's authority: in HEAD; its latest change a calendar day and 12 hours before the look; a "Held-out look" line matching the set, tier, script and batch; and `via` equal to the running `__main__`. The time is the clock's, and a given time may not be later | same |
| I4 | an incidental look says "none". A backfill row skips only I3, keeps I2, and is refused from the cutoff | same |
| I5 | the committed ledger is a prefix of the working one, and `git_dirty` ignores the ledger only then | same, and `tests/test_git_dirty.py` |
| I6 | the default paths carry no held-out label and nothing derived from one. `dgf1_splits()` gives validation only; `load_dgraph()` drops all three official masks, hides every label after the validation window with a hidden value that is no real label, and builds `labelled_mask` from the hidden labels, so it marks no later user | `tests/test_dgf1_heldout_access.py` |
| I7 | `open_test_window` records, then returns the window, plus the true labels at the labels or scores tier only. Putting them back sets `y` and rebuilds `labelled_mask` from them | same |
| I8 | `open_official_track` refuses the structure tier. Otherwise it records, then returns every label and all three official masks | same |
| I9 | the integrity check takes no argument, compares with pinned published totals, and returns pass or fail only | same |
| I10 | disclosure lists earlier rows for the batch's set and its overlaps, before the batch's first row, in time order. A batch with no opener row cannot be disclosed. Two batches under one document stay apart, and a batch label belongs to one authority | `tests/test_heldout_ledger.py` |
| I11 | the backstop finds raw routes in committed code under `scripts/`, `adapters/dgf1/` and `notebooks/`, outside the accessor modules and `notebooks/measurements/*/prototype/`: PyG's `DGraphFin`, the adapter's raw loader `_load_raw` and `TemporalSplit(`, called or imported, through an alias too; any attribute or subscript naming an official mask; a literal path into `data/dgraph` or `experiments/scores`; `load_scores(` in a file with no accessor; and `incidental=`, `backfill=` or `now=` outside the hand path | `tests/test_heldout_readers.py` |

**The backstop's limits, stated rather than hidden.** It is per file, so a path or loader imported
under another name from a module it does not flag escapes it. It cannot tell a held-out registry
read from a validation one, since the registry holds both. So I6 to I8, what the default load leaves
out, are the guarantee. The backstop only catches raw routes, and a route found later is added to it
with a mutant (*Revisit when*). Row 5a moves today's store paths into the accessor modules; the
backstop finds them in six scripts today (`test_the_backstop_on_committed_code_finds_the_known_routes`).

**The batch label.** Each opener passes a `batch` label that names one batch: a gated pre-registration
and its stage, or one of ADR-015's modes. The authorising document names it on its "Held-out look"
line. Two batches under one document therefore never merge, and a label cannot be reused to move a
batch's start.

**What `git_dirty` ignores.** This is a generic append-only parameter in `gbe/run/config.py`, so
ADR-008's regression check is re-run after it. No batch that relies on the trainer runs before that
re-run passes. The researcher decides it together with CLAUDE.md's open DGF-1 repro-check question
(`CLAUDE.md:96-97`). Hiding labels after the validation window cannot change a training or validation
row, but that repro-check would show it.

### 6. Disclosure comes from the ledger

Every gate or reported-only file assembled after this ADR builds its "seen before this batch" list
mechanically (I10), from two sources:
- the ledger;
- its validation runs, from the registry.

Nothing is typed by hand.

**The assembler refuses in two cases:**
- a held-out registry row of its batch predates the batch's first opener row;
- that row is not in HEAD's ledger.

The first assembler written after this ADR, matched-time's or Gate 3's, is the first to do this.
Its tests fail on a missing ledger row and on each refusal.

### 7. (Merged into clause 5's table.)

### 8. What changes in the code that exists today

- **Retired from held-out use:** the held-out parts of `scripts/measure_dgf1_temporal_split.py` and
  `scripts/audit_dgf1_datasource.py` refuse. Their validation parts stay.
- **Retired whole:** `scripts/assemble_gate_dgf1_1.py` refuses at its start. GATE-DGF1-1 is signed,
  and ADR-016 closed its window.
- **`scripts/verify_dgraph_snapshot.py`** loads through the adapter. It uses the integrity check
  (I9) for its label totals, and opens no held-out set.
- **`load_dgraph()`** hides `y` before `prepare_dgraph` runs, because `prepare_dgraph` builds
  `labelled_mask` from the labels it is given (`adapters/dgf1/datasource_dgraph.py:120`).
- **`scripts/run_dgf1_floor.py` and `scripts/run_dgf1_gnn.py`:** their gate branches open the window
  after the guard, and put the returned labels back before scoring, setting both `data.y` and
  `data.labelled_mask` (I7). Scoring needs both: it picks its targets from `labelled_mask`
  (`adapters/dgf1/train_gnn.py:123`, `adapters/dgf1/baselines_tabular.py:139`, through
  `window_target_mask` at `adapters/dgf1/datasource_dgraph.py:269`), and reads `data.y`
  (`adapters/dgf1/train_gnn.py:142`, `adapters/dgf1/baselines_tabular.py:140`). Gate mode stays
  closed (ADR-016).
- **Tests that read the test split:**
  - `tests/test_dgf1_datasource.py:197-201` checks the windows through a bounds reader for the
    public config;
  - `tests/test_dgf1_runner.py:202-208` builds its test-window object from those bounds, with
    synthetic data.
- **ADR-015's line citations into `adapters/dgf1/datasource_dgraph.py`** will move. Row 5a's commit
  lists them, and ADR-015's implementation re-checks them.

### 9. Backfill (plan row 5a)

DGF-1's looks before this ADR are entered through the hand path as backfill rows (I4). Each row's
`via` names its source, and its authority is one of two:
- the document the look ran under, as for the Gate-1 batch under ADR-012;
- or "none: before any authorising document", as for ADR-011's measurement script and the review
  agent's slip.

**Times.** Exact times come from run ids or git. Where a source gives only a date, the row is
written at midnight UTC, and `via` says so. The hand path is the only code that gives a time (I3,
I11). **Batch labels.** Each backfill row carries one for the look it records, such as `gate1`
under ADR-012, and no label is shared between authorities (I10).

**The sources:**
- the corrected disclosure list in `gates/ERRATUM-DGF1-1.md`;
- the Gate-1 batch (`09c6e72`) and its assembly;
- the looks at the official test mask (ADR-010, ADR-011);
- agents' exposures recorded in the audit (`notebooks/audit/2026-10-07/README.md` §6) and in later
  reviews (each added preamble location).

The backfill lands before the matched-time pre-registration (plan row 6a) is accepted. Then the
backfill flag is removed.

### 10. The leakage rule's wording, a leakage sheet, and qualification 4

**The leakage-test rule.** Every DataSource keeps its automated leakage test. Where the data cannot
support one (labels without dates, features snapshotted after the cutoff), the DataSource documents
the limitation instead, and every gate file on that DataSource carries it.

**The leakage sheet.** Each new DataSource's Phase-0 ADR has one. It answers Kapoor & Narayanan's
eight leakage types (`notebooks/audit/2026-10-07/literature-recheck.md` §4) with a test or a
documented limitation for each. EDR-1's Phase-0 ADRs are the first (ADR-020).

**Hidden labels in a new adapter.** A new adapter hides its held-out labels by default, and hands
them out only through accessors of its own, before any look.

**Qualification 4.** CLAUDE.md's Gate-1 list gains the qualification the gate file states
(GATE-DGF1-1:215-217), as a bullet after the engine-gate bullet.

### 11. Implementation order (plan row 5a)

1. **The `git_dirty` change and its test.** Then ADR-008's re-run, with the DGF-1 repro-check
   question decided alongside it.
2. **`adapters/dgf1/heldout.py`** (I1 to I5, I10), and the hand path's command line.
3. **The adapter's default paths and accessors** (I6 to I9).
4. **The runners, the scripts and tests in clause 8, and the backstop** (I11). The prototype's
   mutants are re-run here.
5. **The backfill**, committed before plan row 6a is accepted. Then the backfill flag is removed.

### 12. The amendment to ADR-015

**Why it is needed:**
- **ADR-015's runner opens the official track (I8) and writes the ledger.** Its identity test
  requires every repo file the runner opens to be in `code_identity` or among its exclusions
  (ADR-015:938-943), and the ledger is in neither (ADR-015:580-581).
- **Its mask check and `data_identity` need all labels and all three masks** (ADR-015:565-567,
  :648-650). Under I6 the default load has neither.

**On acceptance, these edits are applied to ADR-015. Lines `:3`, `:580` and `:582` change within
themselves, each staying one line, so the preamble's `:202` and `:208-209` do not move:**
- **`:580`:** "the registry;" becomes "the registry; the held-out ledger
  (`experiments/heldout_ledger.csv`, ADR-021);".
- **`:582`:** "Because a dirty tree is refused, HEAD equals the working tree." becomes "Because a
  dirty tree is refused, HEAD equals the working tree, apart from rows appended to the ledger
  (ADR-021)."
- **The Status line** gains "· amended <date> by ADR-021 clause 12, at the end of this file".
- **A section appended at the end,** "Amendment, <date> (ADR-021 clause 12)". It says:
  - every mode of `scripts/run_dgf1_official.py` loads through `open_official_track` under this ADR;
  - the mask check (`:565-567`) and `data_identity` (`:648-650`) run on the tensors it returns: every
    label, and all three masks. The mask check compares the masks with a `labelled_mask` rebuilt
    from those labels, never the default load's;
  - each mode is its own batch, labelled `official-certify`, `official-retune` or
    `official-positioning`;
  - the official track's training and validation masks contain temporal-test users. That is this
    track's design, and it is disclosed here (ADR-021 clause 1);
  - the identity test's fixture runs (`:937-941`) never write the real ledger. The test's wrapper
    sets the ledger path and repository root on `adapters.dgf1.heldout` and starts the runner with
    `runpy.run_path(<runner>, run_name="__main__")`, so the running script is the runner. No
    environment variable or command-line option redirects the ledger;
  - four lines:
    - `**Held-out look:** dgraph-official-test · scores · scripts/run_dgf1_official.py · official-certify`
    - `**Held-out look:** dgraph-official-test · scores · scripts/run_dgf1_official.py · official-retune`
    - `**Held-out look:** dgraph-official-test · scores · scripts/run_dgf1_official.py · official-positioning`
    - `**Held-out look:** dgraph-official-test · scores · scripts/assemble_dgf1_positioning.py · official-positioning`

No other clause changes.

## Alternatives rejected

- **A ledger written only by the guard** (the 2026-10-07 decision). It would have missed four of six
  readers (`notebooks/measurements/2026-10-08-heldout-ledger/output.txt`), and the reviews found
  more.
- **Drafts 1 to 3's designs:**
  - per-caller recording and a regex lint;
  - accessors that guarded the window and the mask but not the labels;
  - hidden labels with the official masks left in the default load.

  Each round showed a route past the design before it.
- **Documenting the label gap instead of closing it.** The researcher chose to hide the labels.
- **Every overlap counts.** Every retune and pilot would write rows for a use the reserving ADR
  discloses. The researcher scoped sets by track.
- **A separate amendment to ADR-015,** or **splitting this ADR into rules and mechanism.** That would
  mean a second Tier-A document and budget. The researcher kept one ADR, with the mechanism stated
  as tested invariants.
- **Authority by prose mention.** It is trivially satisfied, since ADR-015 mentions `run_dgf1_gnn.py`.
- **Hand rows only.** This is what produced GATE-DGF1-1's incomplete disclosure.
- **Two other fixes for the dirty tree:** the guard writes the row and stops, or rows go to a pending
  file. The researcher chose ADR-002's precedent, narrowed here to appends.
- **Label looks allowed whenever they are recorded.** This is weaker than CLAUDE.md's "never peek".
- **The post-look rule as ADR-011 worded it.** It cannot be met by validation-sized seed counts, and
  it means nothing for reported-only work.
- **Keeping the two label scripts runnable.** The researcher retired their held-out parts.
- **A hash chain over the rows.** The ledger is committed, and I5 refuses a rewrite.

## Consequences

**On acceptance, these amendments are applied and committed:** ADR-015 (clause 12) in the public
repository, and the rest in the governance repository.

**`CLAUDE.md:166-167`** becomes:

> - Never train on, tune against, or peek at any held-out eval set (temporal-holdout test periods,
>   CRUXEval, SWE-bench Verified, etc.). **A look** at a held-out set has three tiers (ADR-021):
>   - **structure:** label-free facts;
>   - **labels:** any statistic using its labels;
>   - **scores:** any model output on it.
>
>   **Where the data comes from.** Held-out labels, and masks derived from them, are left out of the
>   default load. Only the adapter's accessors hand them out.
>
>   **Recording.** Every look is recorded in `experiments/heldout_ledger.csv` before the data is
>   read, or by hand as soon as an unplanned look is known. Re-reading a signed gate file is not a
>   look.
>
>   **Labels and scores** need an accepted document whose "Held-out look" line named the look first,
>   and scores come only from a gated or reported-only batch it names.
>
>   **A peek** is any look that is unrecorded or unauthorised.
>
>   **After a labels or scores look at a set,** a later choice in a document on it uses no held-out
>   label or score, or only removes claims, or (in reported-only work) is disclosed as informed.

**`CLAUDE.md:187-189`** gains:

> Where the data cannot support such a test (labels without dates, features snapshotted after the
> cutoff), the DataSource documents the limitation, and every gate file on it carries it. Each new
> DataSource's Phase-0 ADR answers Kapoor & Narayanan's eight leakage types, each with a test or a
> documented limitation, and its adapter hides held-out labels by default (ADR-021).

**DGF-1's qualifications** gain, after the engine-gate bullet:

> **Seed-only, on one window.** The two clauses share one floor arm, so they are not two
> confirmations. The estimand is seed variability on this fixed window, and undated labels and
> snapshot features make both arms' absolute numbers optimistic.

***Next* item 4** records the acceptance and row 5a, in the same two lines, so no line the preamble
cites moves. The qualification and the edits at `:166` and `:187` all fall after `:102`. If any
line the preamble cites does move, the preamble is renumbered in the same commit.

**`docs/timeline.md`:** row 5 gets its actual date, and row 5a reads "the default load without
held-out labels or masks, the accessors and ledger; the `git_dirty` change with ADR-008's re-run;
the scripts; DGF-1's backfill, before row 6a".

**`docs/subagent-preamble.md`:** its "report its `path:line`" sentence adds "so the main session can
record it in the ledger".

**What gets harder:**
- every held-out read needs an accessor and an authority line that named it;
- the default load carries no held-out label;
- the two label scripts lose their held-out parts;
- a `gbe/` change costs one ADR-008 re-run.

**What gets easier:** a gate's disclosure is generated, not remembered.

## Revisit when

- A held-out set appears that is not a fixed slice of one dataset, for example EDR-1's rolling
  cutoffs. Its adapter's default load must then leave the set's labels out, and the set is added
  with its track and overlaps.
- Or a route around the default load is found. The backstop then gains it, with a mutant.

## Draft history

- **Draft 1** (2026-10-08, `0e8aaf1`).
- **Round 1** (2026-10-08): a full Tier-A review by a fresh subagent with the standing preamble.
  - **4 blocking findings:**
    - the scores route could not serve ADR-015 or the assemblers;
    - the lint missed the runners, `official_test_mask` and library calls;
    - the post-look rule could not be met;
    - the sets overlap.
  - **8 should-fix findings and 4 nits.**
- **The researcher's decisions:** accessors, and retiring the two label scripts' held-out parts.
- **Prototype v2:** 41 tests, and 36 of 36 mutants killed.
- **Draft 2** (`c3f2d4f`).
- **Round 2** (2026-10-08): a diff-only pass.
  - **3 blocking findings:**
    - the labels were not behind the accessor;
    - ADR-015's identity test trips on the ledger;
    - the official test overlaps the temporal training and validation windows.
  - **6 should-fix findings and 4 nits.**
- **The researcher's decisions:** hide labels by default; amend ADR-015 here; scope sets by track.
- **Prototype v3:** 51 tests, and 47 of 48 mutants killed. The survivor is equivalent.
- **Draft 3** (`0209f64`).
- **Round 3** (2026-10-08): a diff-only pass.
  - **2 blocking findings:**
    - hiding labels by node time broke ADR-015's official track;
    - the official train and validation masks leak which hidden users are labelled.
  - **4 should-fix findings:**
    - the verifier's integrity check;
    - prose mentions authorising;
    - `via` from `__main__` unpinned;
    - two batches merging.
  - **6 nits.**
- **The researcher's decision:** fix both, with the mechanism stated as tested invariants.
- **Prototype v4:** 59 tests, and 62 of 62 mutants killed, after five weak tests were strengthened.
- **Draft 4** (`ff9be31`):
  - **Blocking:**
    - `open_official_track` (I8) and the official-track design overlap (clause 1);
    - clause 12 widened to the mask check, `data_identity`, the "Held-out look" lines and the
      disclosure;
    - the default load drops all three official masks (I6);
    - the backstop flags any official mask (I11).
  - **Should-fix:**
    - the integrity check (I9, clause 8);
    - the structured "Held-out look" line (I3);
    - `via` from `__main__`, pinned;
    - the batch label (I10).
  - **Nits:**
    - the code moves to `adapters/dgf1/heldout.py`;
    - the runners put the accessor's labels into `data.y`;
    - the hidden value is tested;
    - the latest amendment date counts;
    - stale ADR-015 citations are listed in row 5a;
    - the error messages are matched in tests;
    - ADR-015:208-209 is added to the preamble.
- **Round 4** (2026-10-08): a diff-only pass, the last in the budget.
  - **1 blocking finding:** `labelled_mask` is built from the raw labels, so the default load still
    marks which hidden users are labelled, and clause 8's runners would score no target.
  - **6 should-fix findings:**
    - the backstop's gaps: aliases, `test_*.py`, raw reads, the raw loader;
    - a caller's time;
    - batch labels free and reusable;
    - the wait only partly mechanised;
    - mutants not listed by invariant;
    - published totals as a route to aggregates.
  - **4 nits:**
    - ADR-015 `:647-649` should be `:648-650`;
    - ADR-015's Status line must stay one line;
    - the integrity check took caller totals;
    - the identity test's seam was unstated.
- **The researcher's decision:** move the blocking finding into tests.
- **Prototype v5:** 68 tests, and 79 of 79 mutants killed. The finding's
  test is `test_labelled_mask_marks_no_user_after_the_validation_window`, killed by the mutant "I6
  labelled_mask built from the raw labels (round 4's B1)"; the runners' half is
  `test_with_labels_rebuilds_labelled_mask_so_the_window_has_targets`.
- **Draft 5** (2026-10-08):
  - **Blocking, moved into tests:** `labelled_mask` in I6 and I7; clause 8's runners and
    `load_dgraph()`; clause 12's mask check.
  - **Should-fix:**
    - I11 widened: aliases, imports, the raw loader, store paths and `now=`, with no `test_*.py`
      exemption;
    - the time from the clock (clause 3, I3);
    - the batch label on the "Held-out look" line, one authority per label (clause 3, I10);
    - the wait from the latest Status or Date change, a calendar day and 12 hours, in the
      researcher's zone. This prototypes the latest-amendment rule draft 4 left untested;
    - mutants named by invariant;
    - published totals not a look, their difference with visible labels a labels look (clause 2).
  - **Nits:**
    - ADR-015 citations corrected;
    - clause 12 keeps `:3` on one line;
    - I9 takes no argument;
    - the identity test's seam (clause 12).
