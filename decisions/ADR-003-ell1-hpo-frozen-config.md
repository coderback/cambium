# ADR-003 — ELL-1 HPO winning config, frozen (the program's one sweep)

**Status:** accepted
**Date:** 2026-07-24
**Deciders:** coderback
**Docs affected:** none (freezes an HPO outcome; inherited by DGF-1/EDR-1 per doc-00 §6.4)

> **Transfer clause superseded by ADR-009 (2026-07-27).** Three places below — the *Transfer rule*,
> the *Per-model HPO* entry under *Alternatives rejected*, and *Revisit when* — say downstream
> models retune "**lr + fan-out**". doc-00 §6.4 now fixes **"lr + one graph-appropriate second
> knob"**, named per model: naming fan-out specifically hard-coded a property of ELL-1's graph into
> a program-wide rule, and it cannot bind on DGraph at average degree ~1.16. **DGF-1's named second
> knob is `batch_size`**; EDR-1 names its own. Read every "lr + fan-out" below as "lr + the model's
> named second knob"; the two-knob budget is unchanged.
>
> **The frozen config in the table below is untouched and remains authoritative** — backbone,
> layers, hidden, aggregator, dropout, lr, fan-out and the fixed budget all stand exactly as
> frozen, and `adapters/ell1/config.yaml` is unchanged. Only the *transfer* clause moved. The body
> is deliberately not rewritten: dated ADRs, like dated gate files, are not silently edited.

## Context

Doc-00 §6.4 mandates **one** Optuna + ASHA sweep, run **on ELL-1 only** (the cheapest graph),
≤50 trials over a small enumerated space, frozen in an ADR and transferred upward — no NAS,
no per-model search (a searched architecture *weakens* the controlled-study framing).

The sweep is implemented in `adapters/ell1/hpo.py` / `scripts/run_ell1_hpo.py`:
- **Space (doc §6.4):** layers {2,3}, hidden {128,256}, aggregator {mean,max}, dropout
  {0.2,0.5}, lr log-uniform [1e-4,1e-2], fan-out {[10,10],[15,10],[25,10]}. Fixed budget:
  encoder 2-layer MLP, LayerNorm, 40 epochs, batch 1024, weight-decay 5e-4, Adam.
- **Objective:** validation **illicit-F1** (the gate metric) on an **inner temporal split —
  train 1-29, validate 30-34, strict inductive**. The 35-49 test window is never touched
  (constitution: never tune against the held-out eval; enforced by `test_hpo_no_eval_peek.py`).
- **Search:** TPE sampler (seed 42) + SuccessiveHalving (ASHA) pruner; 50 trials, **4 completed
  / 46 pruned**. Persistent SQLite storage made the sweep resumable across intermittent CUDA
  faults on the 4 GB GPU. Full study: `experiments/ell1_hpo_study.csv`.

## Decision

**Freeze the winning config (trial 33, validation illicit-F1 = 0.9274 on steps 30-34):**

| Hyperparameter | Value | Source |
|---|---|---|
| backbone | GraphSAGE | doc §3.2 (homogeneous/inductive) |
| num_layers | **3** | searched (see multi-hop note) |
| hidden_dim | 128 | searched |
| aggregator | mean | searched |
| dropout | 0.2 | searched |
| lr | 6.636671097096978e-4 | searched (log-uniform) |
| fan_out | [25, 10] (→ [25,10,10] for the 3rd hop) | searched |
| encoder_layers | 2 | fixed budget |
| norm | LayerNorm | fixed (gbe/CLAUDE.md rule 2) |
| epochs | 40 | fixed budget |
| batch_size | 1024 | fixed budget |
| weight_decay | 5e-4 | fixed budget |

The top two configs (#33 F1=0.9274, #13 F1=0.9185) are identical but for lr — the winning
region is stable, not a single lucky trial.

**On the 3-layer depth (doc §3.2: "3 only with a stated multi-hop story"):** the story is
transaction chains — Elliptic's hand-built features 94-164 already encode *one* hop, so a
2-layer GNN largely recomputes them; the genuine structural signal is multi-hop money-flow
across chained transactions, which a 3rd layer reaches. The sweep chose 3 over 2 on the
validation window, consistent with this. This is the sanctioned multi-hop justification.

**GCN comparator inherits this config** (layers/hidden/dropout/lr/fan-out/budget); only the
convolution differs (aggregator is GraphSAGE-specific and ignored), so the Gate-1 comparison
is capacity- and compute-matched (doc-00 §8).

**Transfer rule (doc §6.4):** DGF-1 and EDR-1 inherit this winning region and tune only
**lr + fan-out**, not the full space.

## Alternatives rejected

- **Keep the doc's 2-layer default.** Rejected: the sweep is the doc-mandated mechanism for
  this choice, and it picked 3 (with a valid multi-hop story) on the leak-free inner split.
- **Search a wider / continuous space, or > 50 trials.** Rejected: doc §6.4 caps the budget;
  46/50 pruned shows the enumerated space was explored, and the winning region is stable.
- **Per-model HPO on DGF-1/EDR-1.** Rejected: doc forbids it — transfer the region, tune only
  lr + fan-out downstream.

## Consequences

- `adapters/ell1/config.yaml` gains a `gnn:` block holding this config; it flows into the
  `gbe.run` config hash, so every Gate-1 registry row is pinned to this exact setting.
- `val illicit-F1 = 0.9274` is a **30-34 validation** number, NOT a test number — it must
  never be reported as a result. The 35-49 gate numbers come only from the frozen 3-seed
  runs, judged by the pre-registered ADR-004 criterion.
- The `.db` is a gitignored working artifact; the committed provenance is the study CSV + this
  ADR.

## Revisit when

Switching to Elliptic++ (different feature schema); or if the transfer to DGF-1 (lr + fan-out
retune only) clearly underperforms, indicating the region did not transfer.
