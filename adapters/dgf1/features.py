"""DGF-1 model inputs: the raw 17 features + the view-derived statistics, standardised fit-on-train.

ADR-011 clause 3: every fitted feature statistic is fitted on users with ``node_time <= 369`` only,
and — because the view features change with the view — **from the training view** (the graph as of
``train_max``), never from a scoring view. The fitted :class:`gbe.features.Standardizer` is then
frozen and applied unchanged to every view: training, validation (as of 481), test (as of 821).

Two data facts shape this, both measured on training-window users only (label-free, 2026-09-11):

* **The raw ``-1`` values are a sentinel strictly below the observed range.** In every one of the
  17 columns the non-sentinel minimum is >= 0 and no value lies in (-1, 0), so a single threshold
  separates "missing" from any observed value. Standardisation is affine, so it keeps that
  separation exactly — no indicator columns and no imputation, i.e. no new modelling decision: this
  is ELL-1's transform (doc-01 §2.2.5) unchanged. Missingness comes in blocks (38.6% of training
  users miss exactly 14 columns, 10.4% miss 16), which the network can read from the pattern.
* **``edge_type_8`` is constant (zero) in the training view** — DGraph's type-8 edges are all dated
  529–821. The core Standardizer maps such a column to 0 everywhere, so a test-window type-8 count
  cannot enter the model as ~1e6. The column is reported by :func:`constant_input_features` so a
  gate file can state it.

The tree floor consumes :func:`node_inputs` **untransformed** — trees are scale-invariant, and
parity (ADR-011 clause 4) is about the information each arm sees, not its preprocessing.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data

from gbe.eval import TemporalSplit
from gbe.features.scaling import Standardizer
from adapters.dgf1.datasource_dgraph import (
    EXPECTED_FEATURES,
    VIEW_FEATURE_NAMES,
    GraphView,
    graph_view,
)

RAW_FEATURE_NAMES: tuple[str, ...] = tuple(f"x_{j}" for j in range(EXPECTED_FEATURES))
INPUT_FEATURE_NAMES: tuple[str, ...] = RAW_FEATURE_NAMES + VIEW_FEATURE_NAMES


def node_inputs(data: Data, view: GraphView) -> Tensor:
    """``[N, 17 + 13]``: raw features next to the statistics of *this* view (doc-02 §2.2.4b)."""
    return torch.cat([data.x, view.node_features], dim=1)


def fit_input_transform(data: Data, split: TemporalSplit) -> Standardizer:
    """Fit on users existing by ``train_max``, using the training view's statistics (clause 3)."""
    view = graph_view(data, split.train_max)
    return Standardizer.fit(node_inputs(data, view), data.node_time <= split.train_max)


def constant_input_features(transform: Standardizer) -> list[str]:
    """Names of the columns the transform zeroes because they never varied in training."""
    return [n for n, c in zip(INPUT_FEATURE_NAMES, transform.constant.tolist()) if c]
