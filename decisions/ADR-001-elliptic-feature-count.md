# ADR-001 — Elliptic feature count is 165 (time_step excluded from node features)

**Status:** accepted · erratum proposed 2026-10-08, at the end of this file
**Date:** 2026-07-21
**Deciders:** coderback
**Docs affected:** `docs/01-elliptic-embedding-model-BUILD.md` §2.1 (and §2.2)

## Context

Doc-01 §2.1 pins Elliptic at "**166 features per node** (≈94 local + ≈72 aggregated)".
`load_elliptic(strict=True)` raised a doc-first escalation on the real data instead of
silently accepting it. Empirical inspection of the raw file
(`data/elliptic/elliptic_txs_features.csv`, downloaded from the PyG mirror, last modified
2019-07-31):

- **167 columns**, 203,769 rows, no header.
- Column 0 = `txId`; column 1 = `time_step` (1..49); columns 2..166 = **165 feature columns**.
- Other counts match the doc exactly: 49 steps, 234,355 edges, 4,545 illicit / 42,019
  licit / 157,205 unknown.

The "166" comes from Weber et al. (2019), whose count **includes `time_step` among the 94
local features** (165 CSV features + `time_step` = 166). PyG's `EllipticBitcoinDataset`
takes columns `2:` → 165 features, matching our loader. So this is not a bad download or a
different dataset version — it is a counting convention, and the doc adopted the paper's.

There is also a leakage reason to keep them separate, not just a naming one: under a
temporal holdout, the *time index itself must not be a model input feature*. `time_step`
is precisely the axis along which train (≤34) and test (35–49) differ; feeding it into `x`
lets the model key on "which period," which is the distribution shift we are testing
generalisation across. It belongs on `data.time_step` (split machinery only), never in `x`.

## Decision

ELL-1 uses **165 node features**; `time_step` is carried separately on `data.time_step`
and used only by the temporal split, never as an input feature. Amend doc-01 §2.1 to state
the raw layout explicitly (167 cols = txId + time_step + 165 features), note that the
paper's "166" counts `time_step` among the local features, and set the feature contract to
165. Change `EXPECTED_FEATURES` from 166 to 165 in `adapters/ell1/datasource_elliptic.py`.

## Alternatives rejected

- **Include `time_step` in `x` to literally reach 166.** Rejected: makes the time index a
  model input under a temporal split — a distribution-shift/leakage trap, and the opposite
  of what ELL-1 exists to test.
- **Keep the doc at 166 and loosen the loader assert.** Rejected: weakening a guard to make
  data fit a wrong number is exactly the anti-pattern the escalation exists to prevent
  (CLAUDE.md: "do not edit this constant to make it pass"). The doc is wrong; fix the doc.
- **Treat 165 vs 166 as a dataset-version issue.** Rejected: the counts, edges, and label
  distribution all match the canonical Weber split; it is a counting convention, verified.

## Consequences

- `docs/01-elliptic-embedding-model-BUILD.md` §2.1 amended (166 → 165 + layout note); §2.2
  gains a one-line note that `time_step` is not a feature.
- `EXPECTED_FEATURES = 165`; `load_elliptic(strict=True)` then passes on the real data.
- Baselines (RF/LR, doc §5/§8) train on the **165** features — the "raw features" floor is
  165-dim, and the "94 local only" experiment (§2.2, §3) is 94 including or excluding
  `time_step`? → local features are columns 2..95 (94 cols), `time_step` still excluded.
- No change to `gbe/` — the core never saw this number (it lives in the adapter).

## Revisit when

Switching to Elliptic++ (different feature schema) or if a future re-download shows a
different column count — in which case re-verify against this ADR before trusting any run.

## Erratum, 2026-10-08 (proposed)

**Source:** research audit 2026-10-07, §2
(`notebooks/audit/2026-10-07/README.md`; finding A2 E-S1).
The text above is not edited, so citations of its lines stay valid. **No clause changes.**

- **`:57-58`, "local features are columns 2..95 (94 cols)", is off by one against `:21-22`.**
  - **The layout.** Weber et al.'s first 94 features are local and include `time_step`
    (`notebooks/audit/2026-10-07/literature-recheck.md` §2, quoting Weber p.2), and `time_step` is
    CSV column 1 (`:17`). So the local features without it are CSV columns 2..94 (93 columns), and
    the 72 aggregates are CSV columns 95..166, which are `x` columns 93..164.
  - **The assumption.** This holds if the CSV keeps the paper's column order. That was not checked
    against the data, though ADR-001's own layout (`:17`) is consistent with it.
  - **Local-94** (`x[:, :94]`, `scripts/run_ell1_local94.py:45`, `:69`) therefore held the 93 local
    features plus the first aggregate.
- **The same slip recurs as "features 94–164"**, a range of 71 columns:
  - ADR-003:61 and ADR-004:55;
  - GATE-ELL1-1:70, and GATE-ELL1-3:28 and :176, each with an erratum beside it;
  - doc-01 and `docs/document-amendments-v0.2.md`, corrected in place when this erratum is accepted;
  - code comments at `gbe/eval/ablations.py:112` and `scripts/run_ell1_local94.py:5`, `:10` and
    `:45`, left as they are. Editing `gbe/` requires ADR-008's re-run, so each is corrected with
    the next change to its file;
  - lab 2026-07-25:6 ("all 71 hand-built one-hop aggregates"), a dated log, which is not edited.
- **What it changes:** no verdict. The local-94 diagnostic removed 71 of the 72 aggregates, not all
  of them (`gates/ERRATUM-ELL1-1.md`). The 165-feature contract, `EXPECTED_FEATURES` and the
  `time_step` exclusion are unaffected.
