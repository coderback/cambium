# ADR-021 — The held-out ledger, the three tiers of a look, and the leakage rule's wording

**Status:** proposed
**Date:** proposed 2026-10-08 · draft 3 2026-10-08
**Tier:** A. It sets the held-out and leakage rules, and it amends a Tier-A ADR (CLAUDE.md
*Process*, ADR-017).
**Review rounds:** 2/4
**Deciders:** coderback
**Implemented by:** plan row 5a, in the order clause 11 gives. The work starts at least 12 hours
after acceptance and not on the same calendar day. This ADR must be accepted before EDR-1 starts
(ADR-020 condition 6), and before the next look at a held-out set (CLAUDE.md, *Next*).
**Amends ADR-015, narrowly** (clause 12). The ledger joins its identity exclusions beside the
registry, and its "HEAD equals the working tree" sentence allows ledger appends. No other clause of
ADR-015 changes. ADR-015 is unimplemented, so its restarted wait costs nothing.

**Docs affected:**
- `CLAUDE.md`:
  - `:166-167` (held-out data);
  - `:187-189` (the leakage-test rule);
  - DGF-1's qualifications (a bullet after `:108-109`);
  - *Next* item 4 (`:70-71`).
- `decisions/ADR-015-dgf1-official-split-positioning.md:580-582` (clause 12).
- `docs/timeline.md`, rows 5 and 5a.
- `docs/subagent-preamble.md`: an agent's report of a held-out figure becomes a ledger row. This adds
  a duty and narrows no ban.

**ADR-016 is unchanged.** Its mode → document map is untouched. Its admitted gap, that library entry
points accept the test split (ADR-016:92-94), is closed by clause 5's hidden labels and accessors.

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

**From the measurement and the prototype** (ADR-017 item 3;
`notebooks/measurements/2026-10-08-heldout-ledger/`):
- **The guard sees a third of the paths.** Six committed scripts can read a held-out DGF-1 set, and
  four of them do so outside the guard (`notebooks/measurements/2026-10-08-heldout-ledger/output.txt`).
- **Two rounds of review found more routes.**
  - **Round 1:** the runners open the window as `dgf1_splits()[window]`, the adapter exposes the
    mask as `official_test_mask`, and the library functions accept the test split from any caller
    (ADR-016:92-94).
  - **Round 2:** `load_dgraph()` hands every user's label and appearance time to every caller, so
    `data.y[data.node_time >= 482]` reads the window's labels with no window object at all. The
    window's bounds are public config, so guarding them guards nothing.

  The guard therefore sits on the labels themselves.
- **Prototype v3** is in `notebooks/measurements/2026-10-08-heldout-ledger/prototype/`. It has 51
  tests, and 47 of its 48 mutants are killed
  (`notebooks/measurements/2026-10-08-heldout-ledger/prototype/mutation-output.txt`). The survivor,
  "a document not in HEAD authorises", is equivalent: a missing document yields empty text, which
  the Status check then refuses. It covers:
  - the ledger and the hidden labels;
  - the accessors, and what each tier returns;
  - the authority check, read from HEAD;
  - the guarded backfill and overlapping sets;
  - the append-only dirty check;
  - the backstop lint.

  v1 is at `012a023`, and v2 at `7f15bce`.

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
- **After round 2:**
  - held-out labels are hidden by default;
  - ADR-015 is amended narrowly, here;
  - held-out sets are scoped by the track that reserves them.

## Decision

### 1. Held-out sets, scoped by track

A held-out set is data that a pre-registration or a reported-only ADR reserves, and it is held out
from **the track that reserves it**. There are two today:
- `dgf1-temporal-test-482-821`: DGF-1's temporal test window, held out from the temporal track's
  fitting, selection and design (ADR-011);
- `dgraph-official-test`: DGraph's official test mask, held out from the official track's fitting
  and selection (ADR-015).

**Overlaps.** The two sets share users (ADR-015:234). So a look at either counts, for clauses 4 and
6, as a look at both.

**One overlap is design, not a look.** The official test's users also sit in the temporal track's
training and validation windows. Their labels are used by design there, and ADR-015 discloses this
once (ADR-015:430). That use is the temporal track's design, not a look at the official set.

A later document that reserves a set adds it, with its track and overlaps, in the commit that accepts
it. ELL-1's test steps are out of scope. ELL-1 is closed, and its window is never reused (ADR-008).

### 2. A look, in three tiers

**A look is either of these:**
- any computation from a held-out set's data;
- anyone reading a figure computed from one that no signed gate file prints. That includes reading
  a held-out row of `experiments/registry.csv`, whose metrics are scores.

**Not looks:**
- re-reading a signed gate file, or a figure it prints;
- the loader's integrity checks over the whole snapshot, which compare whole-dataset totals with
  the figures the dataset's own publication reports (ADR-010:24). This exemption takes precedence
  over the pooled rule below.

**A look's tier is the highest it reaches:**
- **structure:** label-free facts about the set's inputs, such as sizes, dates, degrees, feature
  ranges and graph views;
- **labels:** any statistic that uses the set's labels without a model. A statistic pooled over
  several sets is a labels look at every held-out set it includes, apart from clause 1's
  design overlap;
- **scores:** any model output or metric on the set.

### 3. What each tier allows

- **At every tier, never** train on a held-out set, or tune or select anything against it.
  CLAUDE.md is unchanged on this.
- **Structure:** allowed, once it is recorded. Its authority is a stated reason.
- **Labels and scores: only under an accepted document that named the look first.** That means all
  of these:
  - the document is accepted in HEAD, on an earlier calendar day than the look;
  - it names the script making the look;
  - that script is the running `__main__`, taken from the process, never typed by the caller.

  A draft cannot authorise its own look. Scores are produced only in a batch that an accepted
  pre-registration (gated) or an accepted reported-only ADR names.
- **An assembler reads under its batch's authority.** That document must name the assembler too.
  An assembler that reads held-out registry rows records its look directly with `record_look`.
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

### 5. The ledger, written where the data is handed out

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

**What the adapter hands out by default:**
- `dgf1_splits()` returns the validation window only;
- `load_dgraph()` drops the official test mask;
- **`load_dgraph()` hides the label of every user who appears after the validation window**. They
  carry a "hidden" value that `labelled_mask` excludes, so no caller can count, condition on, train
  on or score against them;
- the raw snapshot loader for the verifier does the same.

**The accessors.** Each held-out part comes only from one, which records the look and then returns
it: `open_test_window(…)` and `open_official_test_mask(…)`.
- **What each tier returns.** A structure open returns the window, with labels still hidden. A labels
  or scores open also returns the true labels. So the tier is enforced by what comes back, not just
  declared.
- **The gated runners** open after the pre-registration guard has passed, with the guard's document
  as the authority. ADR-016's map is unchanged.
- **ADR-015's runner** opens the official test mask under ADR-015.
- **An assembler** opens under its batch's document.

The window's bounds stay public config. They are read through a separate bounds reader, and a bounds
object reaches no label.

**The backstop.** A test parses every committed Python file under `scripts/`, `adapters/dgf1/` and
`notebooks/`. It skips three kinds of file:
- the accessor modules;
- `test_*.py` files, which use synthetic fixtures;
- `notebooks/measurements/*/prototype/` folders.

It fails on any of these:
- a call to PyG's `DGraphFin(` or to `TemporalSplit(`;
- an attribute or subscript ending in `test_mask`;
- a call to `load_scores(` in a file that calls no accessor;
- an `incidental=` or `backfill=` keyword outside the hand path.

**Its limits, stated rather than hidden.** It is per file. It cannot tell a held-out registry read
from a validation one, since the registry holds both. So the hidden labels and the accessors are the
guarantee, and the backstop only catches the raw routes.

**The hand path** (`scripts/heldout_ledger.py`'s command line) records two kinds of row:
- **an incidental look,** whose authority reads "none: …";
- **a backfill row,** marked `backfill=True`. It skips only the authority check, so the figure filter
  still applies. It is refused for a look dated on or after this ADR's acceptance, and the flag is
  removed once step 5 of clause 11 is committed.

Only this path may record either kind.

**What `git_dirty` ignores.** `git_dirty` ignores the ledger only while the working ledger is HEAD's
ledger with rows appended. Any other edit makes the tree dirty. This is a generic append-only
parameter in `gbe/run/config.py`, so ADR-008's regression check is re-run after it, and no batch that
relies on the trainer runs before that re-run passes. It compounds CLAUDE.md's open question of
whether DGF-1's repro-check is re-run (`CLAUDE.md:96-97`). The researcher decides both together.

### 6. Disclosure comes from the ledger

Every gate or reported-only file assembled after this ADR builds its "seen before this batch" list
mechanically, from two sources:
- the ledger's rows for its set and the sets it overlaps, dated before the batch's first opener row;
- its validation runs, from the registry.

**The batch** is the set of opener rows that carry its authority and name its runner. It is not one
commit, because ADR-015 commits after every invocation (ADR-015:1104-1107). Nothing in the list is
typed by hand.

**The assembler refuses in two cases:**
- a held-out registry row of its batch predates the batch's first opener row;
- that row is not in HEAD's ledger.

The first assembler written after this ADR, matched-time's or Gate 3's, is the first to do this.
Its tests fail on a missing ledger row and on each of the two refusals.

### 7. Tests (implementation, plan row 5a)

| invariant | test |
|---|---|
| the ledger file: appended, one header, synced before return | `tests/test_heldout_ledger.py` |
| `record_look`'s field checks, the figure filter included | same |
| clause 3's authority check: not in HEAD, the working tree read, proposed, accepted after or on the day of the look, a `via` that is no script, naming nothing; labels without a document; an incidental look claiming one | same |
| the backfill path: authority kept, figure filter kept, refused from the cutoff | same |
| append-only, and `git_dirty` ignoring only an append to HEAD's ledger | same, and `tests/test_git_dirty.py` |
| the default paths: no test split, no official test mask, later labels hidden | `tests/test_dgf1_heldout_access.py` |
| each accessor records before it loads, and a structure open returns no labels | same |
| disclosure covers overlapping sets' earlier rows, in time order; the assembler's two refusals | `tests/test_heldout_ledger.py`, and the first assembler's tests |
| the backstop's routes and exclusions | `tests/test_heldout_readers.py` |

Each row is pinned by the prototype's mutants (`mutation-output.txt`). They are re-run on the real
modules at step 4 of clause 11, and the output is cited in that commit.

### 8. What changes in the code that exists today

- **Retired from held-out use:** the held-out parts of `scripts/measure_dgf1_temporal_split.py` and
  `scripts/audit_dgf1_datasource.py` refuse. Their validation parts stay. No accepted document named
  their label looks first, and none will.
- **Retired whole:** `scripts/assemble_gate_dgf1_1.py` refuses at its start. GATE-DGF1-1 is signed,
  and ADR-016 closed its window.
- **`scripts/verify_dgraph_snapshot.py`** loads through the adapter. It opens the official test mask
  at the structure tier, for snapshot verification (ADR-010).
- **`scripts/run_dgf1_floor.py` and `scripts/run_dgf1_gnn.py`:** their gate branches open the window
  after the guard. Gate mode stays closed (ADR-016).
- **Tests that read the test split:**
  - `tests/test_dgf1_datasource.py:197-201` checks the windows through the bounds reader;
  - `tests/test_dgf1_runner.py:202-208` builds its test-window object from the bounds, inside the
    test, with synthetic data.

### 9. Backfill (plan row 5a)

DGF-1's looks before this ADR are entered through the hand path as backfill rows. Each row's `via`
names its source, and its authority is one of two:
- the document the look ran under, as for the Gate-1 batch under ADR-012;
- or "none: before any authorising document", as for ADR-011's measurement script and the review
  agent's slip.

**Times and contents.** Exact times come from run ids or git. Where a source gives only a date, the
row is written at midnight UTC, and `via` says so. Because the figure filter applies, rows taken from
lab entries carry no figure.

**The sources:**
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
(GATE-DGF1-1:215-217), as a bullet after the engine-gate bullet.

### 11. Implementation order (plan row 5a)

1. **The `git_dirty` change and its test.** Then ADR-008's re-run in the researcher's terminal, with
   the DGF-1 repro-check question decided alongside it.
2. **`scripts/heldout_ledger.py`** and its tests.
3. **The adapter:** hidden labels, the accessors, the bounds reader, and their tests.
4. **The runners, the scripts and tests in clause 8, and the backstop test.** The prototype's
   mutants are re-run here.
5. **The backfill.** It is committed before plan row 6a is accepted. Then the backfill flag is
   removed.

### 12. The amendment to ADR-015

**What breaks without it.** ADR-015's runner opens the ledger through `open_official_test_mask`.
- **Its identity test fails.** The test requires every repo file the runner opens to be in
  `code_identity` or among its exclusions (ADR-015:938-943), and the ledger is in neither
  (ADR-015:580-581).
- **Adding the ledger to `code_identity` instead** would change the identity on every append, so the
  runner would refuse itself.

**On acceptance, these edits are applied to ADR-015, line for line, so no line moves:**
- **`:580`:** "the registry;" becomes "the registry; the held-out ledger
  (`experiments/heldout_ledger.csv`, ADR-021);".
- **`:582`:** "Because a dirty tree is refused, HEAD equals the working tree." becomes "Because a
  dirty tree is refused, HEAD equals the working tree, apart from rows appended to the ledger
  (ADR-021). *Amended <date> by ADR-021 clause 12.*"
- **The Status line** gains "· amended <date> by ADR-021 clause 12 (`:580-582`)".

No other clause changes. ADR-015's restarted wait (CLAUDE.md *Process*) costs nothing, because it is
unimplemented.

## Alternatives rejected

- **A ledger written only by the guard** (the 2026-10-07 decision). It would have missed four of six
  readers (`notebooks/measurements/2026-10-08-heldout-ledger/output.txt`), and the reviews found
  more.
- **Draft 1's design:** the guard records, each script records, and a regex lint catches the rest.
  Round 1 showed routes past it.
- **Draft 2's design:** accessors that guard the window and the mask, but not the labels. Round 2
  showed `data.y[data.node_time >= 482]` reads the labels with no accessor.
- **Documenting the label gap instead of closing it.** The researcher chose to hide the labels.
- **Every overlap counts.** Every retune and pilot would write official-set rows for a use ADR-015
  already discloses. The researcher scoped sets by track.
- **A separate amendment to ADR-015.** That would mean a second Tier-A document with its own rounds,
  for two lines this ADR already forces.
- **Hand rows only.** This is what produced GATE-DGF1-1's incomplete disclosure.
- **Two other fixes for the dirty tree:** the guard writes the row and stops, or rows go to a pending
  file. The researcher chose ADR-002's precedent, narrowed here to appends.
- **Label looks allowed whenever they are recorded.** This is weaker than CLAUDE.md's "never peek".
- **An authority that need only be accepted.** Any accepted ADR would then authorise any look.
- **The post-look rule as ADR-011 worded it.** It cannot be met by validation-sized seed counts, and
  it means nothing for reported-only work.
- **Keeping the two label scripts runnable.** The researcher retired their held-out parts.
- **A hash chain over the rows.** The ledger is committed, and the append-only check refuses a
  rewrite.

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
>   **Where the data comes from.** Held-out labels are hidden by default, and only the adapter's
>   accessors hand them out.
>
>   **Recording.** Every look is recorded in `experiments/heldout_ledger.csv` before the data is
>   read, or by hand as soon as an unplanned look is known. Re-reading a signed gate file is not a
>   look.
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

**DGF-1's qualifications** gain, after the engine-gate bullet:

> **Seed-only, on one window.** The two clauses share one floor arm, so they are not two
> confirmations. The estimand is seed variability on this fixed window, and undated labels and
> snapshot features make both arms' absolute numbers optimistic.

***Next* item 4** records the acceptance and row 5a, in the same two lines, so no line the preamble
cites moves. The qualification and the edits at `:166` and `:187` all fall after `:102`. If any
line the preamble cites does move, the preamble is renumbered in the same commit.

**`docs/timeline.md`:** row 5 gets its actual date, and row 5a reads "hidden labels, accessors and
ledger; the `git_dirty` change with ADR-008's re-run; the scripts; DGF-1's backfill, before row 6a".

**`docs/subagent-preamble.md`:** its "report its `path:line`" sentence adds "so the main session can
record it in the ledger".

**What gets harder:**
- every held-out read needs an accessor and an authority that named it;
- labels after the validation window are invisible by default;
- the two label scripts lose their held-out parts;
- a `gbe/` change costs one ADR-008 re-run.

**What gets easier:** a gate's disclosure is generated, not remembered.

## Revisit when

- A held-out set appears that is not a fixed slice of one dataset, for example EDR-1's rolling
  cutoffs, or one held in a new adapter. That adapter hides its held-out labels and gets its own
  accessors before any look.
- Or a route around the hidden labels is found. The backstop then gains it, with a mutant.

## Draft history

- **Draft 1** (2026-10-08, `0e8aaf1`).
- **Round 1** (2026-10-08): a full Tier-A review by a fresh subagent with the standing preamble.
  - **Tier A confirmed.**
  - **4 blocking findings:**
    - ADR-015's batch and the assemblers could not meet the scores rule;
    - the lint missed the runners' `dgf1_splits()[window]`, `official_test_mask` and library calls;
    - the post-look rule could not be met;
    - the two held-out sets overlap.
  - **8 should-fix findings and 4 nits.**
- **The researcher's decisions** (2026-10-08): accessors, and retiring the two label scripts'
  held-out parts.
- **Prototype v2:** 41 tests, and 36 of 36 mutants killed.
- **Draft 2** (2026-10-08, `c3f2d4f`). It answered every round-1 finding.
- **Round 2** (2026-10-08): a diff-only pass by a fresh subagent with the standing preamble.
  - **Tier A confirmed.**
  - **3 blocking findings:**
    - the labels were not behind the accessor;
    - ADR-015 was not unchanged, since its identity test trips on the ledger;
    - the official test overlaps the temporal training and validation windows.
  - **6 should-fix findings and 4 nits.**
- **The researcher's decisions** (2026-10-08): hide held-out labels by default; amend ADR-015
  narrowly here; scope sets by track.
- **Prototype v3:** 51 tests; 47 of 48 mutants killed, the survivor equivalent.
- **Draft 3** (2026-10-08):
  - **Blocking:**
    - hidden labels, with the tier enforced by what the accessor returns (clause 5);
    - clause 12's amendment to ADR-015;
    - track-scoped sets and the design overlap (clauses 1 and 2).
  - **Should-fix:**
    - the backfill keeps the figure filter, is refused from the cutoff, and its flag is removed
      after step 5;
    - registry-reading assemblers record directly, and the backstop's registry limit is stated;
    - the authority is read from HEAD, accepted on an earlier day, with `via` taken from the
      process;
    - the batch is defined by authority and runner, not by commit;
    - "use" is defined, and ADR-011's "harder to earn" branch is kept;
    - clause 7 covers the assembler's refusals and the backfill.
  - **Nits:**
    - the prototype exemption is narrowed to `notebooks/measurements/*/prototype/`;
    - the integrity exemption takes precedence over the pooled rule;
    - the mutant counts are left to `mutation-output.txt`;
    - qualification 4's placement is stated;
    - the ADR-011 citations are narrowed to `:78-81` and `:92`.
