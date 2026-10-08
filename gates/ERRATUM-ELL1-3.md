# ERRATUM-ELL1-3 — GATE-ELL1-3's feature range

**Dated:** 2026-10-08 (proposed)
**Source:** research audit 2026-10-07, §2 (`notebooks/audit/2026-10-07/README.md`; finding A2 E-S1)
**Concerns:** `gates/GATE-ELL1-3.md`, signed PASSED 2026-07-25. That file is **not edited**.

**This is not a gate file and carries no verdict.** Gate 3's verdict, its tables and all three
clauses stand exactly as recorded.

**What is corrected.** GATE-ELL1-3 names the hand-built one-hop aggregates as "features 94–164" at
lines 28 and 176. That range holds 71 columns. Elliptic has 72 aggregate features, which are `x`
columns 93–164 if the CSV keeps Weber et al.'s order (ADR-001's erratum).

**What it changes:** nothing in the gate's inference. Both passages concern the aggregates as a
group: line 28 frames the question, and line 176 reads the reported `config` arm. Every arm in the
gate saw all 165 features.

Any write-up drawing on GATE-ELL1-3 reads this notice with it.
