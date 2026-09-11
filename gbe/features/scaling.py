"""Fit-on-train feature standardisation — the leakage-safe scaler shared by every model.

Extracted from `adapters/ell1/train_gnn.standardize_fit_on_train` (doc-01 §2.2.5) when DGF-1
became its second use (doc-00 §1: the second use case is what proves an abstraction). The rule is
the one both models need identically: statistics come **only from the fit rows** (users that exist
by the training cutoff), and the same frozen statistics are applied to every later tensor — a
scoring view, a validation window — never refitted on it. A statistic fitted over all users would
leak test-window information into every input (doc-01 §2.2.5, ADR-011 clause 3).

**One rule added for the second use case: a column with zero variance on the fit rows maps to 0,
everywhere.** The inherited formula divides by ``std.clamp_min(eps)``, so a column that is constant
in training would enter scoring as ``(value - mean) / eps`` — a factor of ~1e6. That is not
hypothetical: DGraph has **no** type-8 edge by step 369 and **84,338** of them dated 529–821, all
inside the test window, so the edge-type-8 count is constant (zero) for every training user and
non-zero for many test users. The model never saw the column vary, so it carries no learned
meaning; mapping it to 0 makes scoring match training. The rule is **inert for ELL-1** — no Elliptic
column has training std below 1e-6 at either cutoff (minimum 0.237 at 34, 0.160 at 29) — so ELL-1's
numbers are unchanged (ADR-008).

Domain-agnostic: no dataset values, no feature names — adapters name their columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Standardizer:
    """Per-column ``(x - mean) / std`` with statistics frozen at fit time.

    Attributes:
        mean: ``[1, F]`` column means over the fit rows.
        std: ``[1, F]`` sample std over the fit rows, clamped below at ``eps`` (as ELL-1 did).
        constant: ``[F]`` columns with **exactly zero** variance on the fit rows; transformed to 0.
    """

    mean: Tensor
    std: Tensor
    constant: Tensor

    @classmethod
    def fit(cls, x: Tensor, fit_mask: Tensor, eps: float = 1e-6) -> "Standardizer":
        """Fit on the rows of ``x`` selected by ``fit_mask`` — and on nothing else.

        Raises on fewer than two fit rows: a sample std over one row is undefined, and an empty fit
        set means the caller's cutoff selected nobody, which is a bug upstream, not a scaling choice.
        """
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D [N, F], got shape {tuple(x.shape)}")
        if fit_mask.dtype != torch.bool or fit_mask.shape != (x.size(0),):
            raise ValueError("fit_mask must be a bool tensor with one entry per row of x")
        n_fit = int(fit_mask.sum())
        if n_fit < 2:
            raise ValueError(f"need at least 2 fit rows to estimate a std, got {n_fit}")
        rows = x[fit_mask]
        mean = rows.mean(dim=0, keepdim=True)
        raw_std = rows.std(dim=0, keepdim=True)
        return cls(mean=mean, std=raw_std.clamp_min(eps), constant=(raw_std == 0).view(-1))

    def transform(self, x: Tensor) -> Tensor:
        """Apply the frozen statistics. Returns a new tensor; ``x`` is untouched."""
        if x.dim() != 2 or x.size(1) != self.mean.size(1):
            raise ValueError(
                f"expected [N, {self.mean.size(1)}] input, got shape {tuple(x.shape)}"
            )
        out = (x - self.mean) / self.std
        if bool(self.constant.any()):
            out[:, self.constant] = 0.0
        return out
