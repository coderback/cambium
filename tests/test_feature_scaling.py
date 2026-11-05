"""Guards for the core fit-on-train Standardizer (gbe.features.scaling).

Written in the same session as the extraction (CLAUDE.md: "untested guards don't exist"). The two
properties that matter: it is **fit on the fit rows and nothing else**, and it is **bit-identical to
ELL-1's original formula** wherever the new zero-variance rule does not fire — which is what keeps
the ADR-008 reference numbers where they are.
"""

from __future__ import annotations

import pytest
import torch

from gbe.features import Standardizer


def _ell1_original(x, fit_mask):
    """The pre-extraction body of adapters/ell1/train_gnn.standardize_fit_on_train, verbatim."""
    mu = x[fit_mask].mean(dim=0, keepdim=True)
    sigma = x[fit_mask].std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sigma


def _x(n=200, f=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, f, generator=g) * torch.tensor([1.0, 5.0, 0.1, 30.0, 2.0, 1e-3])[:f] + 3.0


def test_bit_identical_to_ell1s_original_formula():
    x = _x()
    mask = torch.arange(200) < 120
    assert torch.equal(Standardizer.fit(x, mask).transform(x), _ell1_original(x, mask))


def test_statistics_come_only_from_the_fit_rows():
    """Clause-5 test 7, generic form: whatever happens to non-fit rows, the statistics do not move."""
    x = _x()
    mask = torch.arange(200) < 120
    before = Standardizer.fit(x, mask)
    x2 = x.clone()
    x2[~mask] = x2[~mask] * 1000 + 77
    after = Standardizer.fit(x2, mask)
    assert torch.equal(before.mean, after.mean)
    assert torch.equal(before.std, after.std)
    assert torch.equal(before.constant, after.constant)


def test_statistics_are_frozen_and_applied_to_later_tensors_unrefitted():
    x = _x()
    mask = torch.arange(200) < 120
    s = Standardizer.fit(x, mask)
    later = _x(seed=1) * 4.0
    assert torch.equal(s.transform(later), (later - s.mean) / s.std)


def test_a_column_constant_in_training_maps_to_zero_everywhere():
    """The DGraph edge-type-8 case: zero in every training row, non-zero at scoring. Without the
    rule the scoring value would be (count - 0) / 1e-6."""
    x = _x()
    x[:, 2] = 0.0
    mask = torch.arange(200) < 120
    s = Standardizer.fit(x, mask)
    assert s.constant.tolist() == [False, False, True, False, False, False]
    scoring = x.clone()
    scoring[150:, 2] = 7.0                                   # the column comes alive after training
    out = s.transform(scoring)
    assert bool((out[:, 2] == 0).all())
    assert float(out.abs().max()) < 100, "a constant-in-training column leaked a ~1e6 value"


def test_tiny_but_nonzero_variance_keeps_the_ell1_clamp():
    """Only EXACTLY-zero variance is treated as constant; a tiny std follows the inherited clamp,
    so the rule cannot change a column ELL-1's formula would have handled."""
    x = _x()
    # Near 0, where float32 can represent 1e-7-scale steps (at 3.0 its resolution is ~2.4e-7, so a
    # 1e-9 ramp there would round to one value and be genuinely constant).
    x[:, 5] = torch.linspace(0, 1e-7, 200)
    mask = torch.arange(200) < 120
    s = Standardizer.fit(x, mask)
    assert 0 < float(x[mask, 5].std()) < 1e-6, "fixture must sit strictly inside the clamp"
    assert not bool(s.constant[5])
    assert torch.equal(s.transform(x), _ell1_original(x, mask))


def test_the_sentinel_stays_separable_after_scaling():
    """DGraph's -1 sits below every observed value; an affine map must keep it strictly below."""
    x = torch.tensor([[-1.0], [0.0], [0.5], [3.0], [-1.0], [10.0]])
    s = Standardizer.fit(x, torch.ones(6, dtype=torch.bool))
    out = s.transform(x).view(-1)
    assert float(out[x.view(-1) == -1].max()) < float(out[x.view(-1) >= 0].min())


@pytest.mark.parametrize("mask", [torch.zeros(10, dtype=torch.bool),
                                  torch.arange(10) == 3])
def test_fewer_than_two_fit_rows_is_an_error(mask):
    with pytest.raises(ValueError, match="at least 2 fit rows"):
        Standardizer.fit(torch.randn(10, 3), mask)


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        Standardizer.fit(torch.randn(10), torch.ones(10, dtype=torch.bool))
    with pytest.raises(ValueError):
        Standardizer.fit(torch.randn(10, 3), torch.ones(9, dtype=torch.bool))
    with pytest.raises(ValueError):
        Standardizer.fit(torch.randn(10, 3), torch.ones(10, dtype=torch.long))
    s = Standardizer.fit(torch.randn(10, 3), torch.ones(10, dtype=torch.bool))
    with pytest.raises(ValueError):
        s.transform(torch.randn(10, 4))
