# Literature re-check (audit 2026-10-07 §5; dated plan row 4a)

**Done 2026-10-07 by the main session.** The audit's §5 asked for five sources to be read against
their PDFs, because the agents read them through a summarising fetch tool: Weber, Maganti, GADBench,
Kapoor & Narayanan, and arXiv:2604.19514. Maganti 2026 *is* arXiv:2604.19514, so four papers were
read.

**Method.**
- Each PDF was downloaded from `arxiv.org/pdf/<id>` on 2026-10-07 and its text extracted with `pypdf`
  6.18.1.
- Quotes below are from that text, so ligatures and spacing are normalised. "p." is the PDF page.
- No summarising tool was used. The PDFs are not committed; the hashes identify the versions read.

| paper | version read | SHA-256 of the PDF |
|---|---|---|
| Weber et al. 2019, "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics" | arXiv:1908.02591v1 (31 Jul 2019) | `a20c8be5…74507` |
| Maganti 2026, "When Graph Structure Becomes a Liability: A Critical Re-Evaluation of Graph Neural Networks for Bitcoin Fraud Detection under Temporal Distribution Shift" | arXiv:2604.19514v1 (21 Apr 2026) | `5a54687d…6e16c` |
| Tang et al. 2023, "GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection" | arXiv:2306.12251v2 (16 Nov 2023) | `3dadd647…8ada7b` |
| Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science" | arXiv:2207.07048v1 (14 Jul 2022) | `c433bced…f70c42` |

## Verdicts

| # | claim, and where the repo makes it | verdict |
|---|---|---|
| 1 | "the GraphSAGE+RF hybrid (≈0.807 F1) **collapses to ≈0.12 F1** under a strict inductive protocol" (doc-01:115, :140, :175; doc-00:76; document-amendments-v0.2:42; ADR-004:55-56; timeline.md:134) | **Wrong.** See §1. |
| 2 | the high number came from full-graph message passing **and batch-norm statistics** (doc-01:115); the collapse was "partly BatchNorm" (document-amendments-v0.2:42; doc-00:76) | **Overstated.** BatchNorm is a channel the paper names, not a cause it measures. |
| 3 | ADR-004:55-56 ties the 0.807 "high-water mark" to the RF floor | **Wrong.** 0.807 is the hybrid's score in Maganti's earlier drafts. |
| 4 | Maganti says aggregate features 94–164 already encode one hop (doc-01:175) | **Confirmed**, but Maganti's own feature split is off by one (§2). |
| 5 | Weber's 166 features include `time_step` among the first 94 local ones (ADR-001:21) | **Confirmed.** |
| 6 | the local/aggregate split is off by one: 93 local + 72 aggregate once `time_step` is dropped (A2 E-S1; the ADR-001 erratum) | **Supported** by Weber's text. |
| 7 | Weber's split is steps 1–34 train, 35–49 test (doc-01:54; ADR-011:217); there are no edges between time steps | **Confirmed.** |
| 8 | Weber reports RF 0.788 and GCN 0.628 illicit F1; Maganti reports RF 0.821 and GraphSAGE 0.689 (B N4) | **Confirmed**, with a caveat on Maganti (§1). |
| 9 | GADBench reports AUPRC for DGraph-Fin, so ADR-007's "no published DGraph comparator" for AUPRC is false (A2 E-S4) | **Confirmed.** |
| 10 | GADBench has RF-Graph and XGB-Graph, tree ensembles with neighbour aggregation (B S1) | **Confirmed.** |
| 11 | GADBench's DGraph-Fin split: three earlier reports disagreed (literature inventory, item 4) | **Resolved:** the pre-existing split is kept. |
| 12 | "tree baselines match GNNs on DGraph" (ADR-010:106, :134) | **Holds only for XGB-Graph** (§3). |
| 13 | GADBench tunes by "random search over 10 trials" (audit D11) | **Not in v2's text** (§3). |
| 14 | Kapoor & Narayanan give eight leakage types; L3.3 is sampling bias in the test distribution (audit D6, D19, D-P2) | **Confirmed.** |

## 1. Maganti 2026 (arXiv:2604.19514v1)

**The 0.807 → ≈0.12 figure is a misreading.** The abstract (p.1):

> Earlier drafts of this work reported F1=0.807 for a concatenation hybrid of GraphSAGE embeddings
> and raw features; under the clean protocol the same hybrid falls to F1=0.699 ± 0.015, and the GNN
> contributes a statistically reliable but small +0.018 F1 lift over a matched-capacity MLP
> substitute (p = 0.015, d = +1.20) that is dwarfed by the 0.124 F1 gap to raw features alone.

So 0.807 falls to **0.699 ± 0.015**, and **0.124** is the hybrid's gap to raw-feature Random Forest,
which p.2 gives as "F1=0.823 ± 0.002". Nothing in the paper falls to ≈0.12 F1.

**BatchNorm is a named channel, not a measured cause.** p.2 says prior "inductive" setups that "run
the encoder on the full graph and simply mask the loss … still leak test-period feature statistics
through batch normalisation and neighbourhood aggregation". The paper's protocol rules out both
channels at once (p.11), and it attributes no part of the hybrid's drop to BatchNorm. The repo's BatchNorm
ban stands on the leakage argument, which the paper supports; it should not cite a "collapse" as
evidence.

**The direction of the transductive gap.** The paired experiment (p.1–2) finds GraphSAGE scoring
F1 = 0.294 ± 0.028 trained transductively and 0.689 ± 0.017 trained inductively, "with the inductive
model stronger than its transductive counterpart". So in this paper, training-time exposure to
test-period adjacency *lowered* GraphSAGE's score. doc-01:140's row "Transductive leakage inflates GNN
gains" is not what this experiment shows.

**Two cautions before citing Maganti for any number.**
- **Test-set early stopping.** Training uses "early stopping (patience 40 on test-period F1)" (p.11),
  so its headline numbers were selected on its own test steps.
- **It misreports Weber.** p.4 says Weber "established the founding GCN baseline (F1=0.70) and
  reported it alongside Logistic Regression (0.53) and Random Forest (0.64)". Weber's Table 1 (p.5)
  gives GCN 0.628, Skip-GCN 0.705, Logistic Regression 0.481 (0.533 with GCN embeddings) and Random
  Forest 0.788. Cite Weber directly.

**Protocol facts.** These match the 2026-09-11 literature report:
- the split is 1–34 / 35–49, "the standard temporal split of Pareja et al." (p.11);
- "At inference we re-instantiate the full graph and perform a single forward pass." (p.11).

## 2. Weber et al. 2019 (arXiv:1908.02591v1)

**Features (p.2).** "Each node has associated 166 features. The first 94 features represent local
information about the transaction – including the time step … The remaining 72 features, called
aggregated features, are obtained by aggregating transaction information one-hop backward/forward."
- **For ADR-001:** with `time_step` dropped, the 165-column table holds **93 local + 72 aggregate**,
  which supports A2's off-by-one erratum.
- **Maganti miscounts:** it gives "Node features 165 (94 local + 71 aggregate)" (p.7) and attributes
  "71 manually engineered aggregate features" to Weber (p.4), the same off-by-one.

**The rest checks out:**
- **Split (p.4):** "the first 34 time steps are used for training the model and the last 15 for test";
- **Edges (p.3):** "there are no edges connecting different time steps";
- **Table 1 (p.5), illicit F1:** Random Forest (all features) 0.788; GCN 0.628; Skip-GCN 0.705;
- **Dark market (p.5):** the shutdown is discussed;
- **Citation:** the title and the seven authors match doc-01:172.

## 3. GADBench (arXiv:2306.12251v2)

**Metrics (p.5).** AUROC, AUPRC "calculated by average precision", and Rec@K.
- **AUPRC on DGraph-Fin:** Table 4 (p.8) reports it for every model, with AUROC and Rec@K in Table 13
  (p.26). ADR-007's "no published DGraph comparator" for AUPRC is false, as A2 E-S4 found.

**Tree ensembles with neighbour aggregation (p.4).** "new tree ensemble baselines with neighbor
aggregation, referred to as RF-Graph and XGB-Graph". With default hyperparameters, "XGB-Graph and
RF-Graph consistently surpass other compared models across all metrics", and the gap is largest in
the fully supervised setting (p.6).

**DGraph-Fin's split (p.5).**
- **The rule:** "In a fully-supervised setting, we preserve pre-existing data splits when available.
  If such divisions are not provided, … 40% for training, 20% for validation, and the remaining 40%
  for testing."
- **For DGraph-Fin:** Table 3 lists a 70% training ratio, so the fully supervised numbers use its
  pre-existing split.
- **The three earlier reports:** "keeping existing splits" and "a 70% training split" were right;
  "random 40/20/40" applies only to datasets without a split.

**"Tree baselines match GNNs on DGraph" holds only for XGB-Graph.** Table 4 (p.8) gives AUPRC (%)
with tuned hyperparameters, on GADBench's protocol and DGraph-Fin's own split, so it is
cross-protocol to DGF-1:

| model | AUPRC on DGraph-Fin (%) |
|---|---|
| RF | 2.57 |
| XGBoost | 2.75 |
| GCN | 3.80 |
| GraphSAGE | 3.77 |
| GAT | 3.85 |
| BWGNN | 3.97 |
| RF-Graph | 2.15 |
| XGB-Graph | 3.79 |

- **Plain trees sit about one point below the GNNs.**
- **XGB-Graph, with neighbour aggregation, matches them.**
- **For DGF-1:** the parity floor has node-level edge statistics but no neighbour aggregates, so it
  sits between these two kinds of tree. That supports B S1's reported tree + neighbour-aggregates arm
  for Gate 3.
- **Lab 2026-09-11:284:** the floor's validation AUPRC (0.0323) "sits where GADBench's DGraph numbers
  live". That is true of Table 4's range (1.67–4.24%), across protocols.

**Tuning budget (p.6; p.25–26).** GADBench uses "random search", keeps "the model that yields the best
AUPRC score on the validation set", and marks models that "could not complete random search within a
day". No trial count appears in v2's text. The "10" is the number of runs behind the
default-hyperparameter tables ("Each model is executed 10 times", pp.24–25). Audit D11's "10 trials" is not
supported by the paper.

## 4. Kapoor & Narayanan (arXiv:2207.07048v1)

**The taxonomy.** "a fine-grained taxonomy of 8 types of leakage" (p.1–2), defined on p.5–6:

| label | type |
|---|---|
| L1.1 | no test set |
| L1.2 | pre-processing on training and test set |
| L1.3 | feature selection on training and test set |
| L1.4 | duplicates in datasets |
| L2 | model uses features that are not legitimate |
| L3.1 | temporal leakage |
| L3.2 | nonindependence between train and test samples |
| L3.3 | sampling bias in test distribution |

- **D19's "Kapoor L3.3"** is correctly labelled.
- **D6's four gaps for EDR-1** map onto the taxonomy: duplicates (L1.4); serial fraud and firm
  persistence (L3.2); pre-cutoff disclosing text (L2, or L3.1); pretrained-encoder lookahead (L3.1).
- **The "model info sheets" (p.29)** answer each type in turn. They are the model for D-P2's leakage
  sheet, so EDR-1's sheet should follow their structure.

## What this changes

**Row 5b, the errata, now has its content:**
1. **The Elliptic leakage citation.**
   - **The correct statement:** the hybrid's 0.807 (earlier drafts) falls to 0.699 ± 0.015 under the
     strict protocol; 0.124 is its gap to raw-feature RF (0.823 ± 0.002).
   - **The other corrections:** BatchNorm is a named channel, not a measured cause; the paper's own
     transductive-trained GraphSAGE scored lower; its numbers are early-stopped on test-period F1.
   - **Where:** doc-00:76, doc-01:115, :140, :175, document-amendments-v0.2:42, ADR-004:55-56,
     timeline.md:134.
2. **ADR-001:** Weber's text confirms 93 local + 72 aggregate once `time_step` is dropped, the
   off-by-one A2 found. B's draft correction to GATE-ELL1-1 ("71 of the 72 aggregates were removed;
   one remained") is consistent with it.
3. **ADR-007:** GADBench reports AUPRC on DGraph-Fin.
4. **ADR-010:106, :134:** "tree baselines match GNNs on DGraph" holds for XGB-Graph, not plain trees.

**No signed verdict changes.**
- **ELL-1:** Gate 1 failed against its pre-registered criterion, and Gate 3 passed on its own
  ablations. The citation only motivated the strict inductive protocol, which stands on its own
  grounds: zero edges cross the cutoff (GATE-ELL1-1).
- **DGF-1:** Gate 1's comparison never relied on GADBench.

**For audit D11:** its "10 trials" claim is unsupported; the rest of D11 is untouched.

**Citing from here on:** cite Weber, GADBench and Kapoor by the versions above. Cite Maganti only for
its protocol, and its numbers only with the test-set early-stopping caveat.

## 5. Added 2026-10-08, for the row-5b errata

Two more checks, by the same method. Davis & Goadrich has no arXiv version: the ICML 2006 paper was
downloaded from `ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf`, SHA-256
`94b4a112…3f908`, 8 pages.

**Davis & Goadrich (2006), "The Relationship Between Precision-Recall and ROC Curves".** The repo
says ROC and PR space are not order-equivalent under skew (ADR-007:47-48; doc-00:161; doc-02:98;
`gbe/eval/metrics.py:9-10`). The paper says the opposite about curves, and the repo's
point about areas:
- **Curves:** "one curve dominates a second curve in ROC space if and only if the first dominates
  the second in Precision-Recall space" (Theorem 3.2, p.3, "for a fixed number of positive and
  negative examples").
- **Areas:** "algorithms that optimize the area under the ROC curve are not guaranteed to optimize
  the area under the PR curve" (abstract, p.1). §5 (p.7) gives a counter-example with 20 positives
  and 2,000 negatives: AUC-ROC prefers curve II (0.875 vs 0.813), and AUC-PR prefers curve I (0.514
  vs 0.038).
- **Interpolation:** "in PR space it is incorrect to linearly interpolate between points" (abstract,
  p.1). ADR-007:131-132 cites this correctly.

So the citation holds for areas, not for orderings in general. The supported wording: ROC-AUC and
AUPRC can rank two models differently, although a curve that dominates in one space dominates in
the other.

**GADBench's AUROC table (Table 13, p.26, tuned by random search).** §3 gave AUPRC only. AUROC (%)
on DGraph-Fin:

| model | AUROC on DGraph-Fin (%) |
|---|---|
| RF | 70.37 |
| XGBoost | 72.43 |
| GCN | 75.51 |
| GraphSAGE | 75.60 |
| GAT | 75.53 |
| BWGNN | 76.30 |
| RF-Graph | 67.78 |
| XGB-Graph | 75.83 |

On both metrics the plain trees sit below the GNNs, and XGB-Graph matches them; RF-Graph does not.
So "tree baselines match GNNs on DGraph" (ADR-010:106, relied on at :134) and "trees nearly match
GNNs here" (document-amendments-v0.2:51) hold for XGB-Graph only. GADBench's own summary (p.8) is that "all
methods perform poorly on the DGraph-Fin dataset".
