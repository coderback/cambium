"""Guard: the resolvability criterion the DGF-1 Gate-1 assembler applies (ADR-006 clause 2, ADR-012).

ADR-006 pins the criterion so a verdict "must not depend on which script evaluates it": the sample
std (``ddof=1``), SE_diff = sqrt(s_a^2/n_a + s_b^2/n_b), and resolvable iff diff > 0 and
diff > 2·SE_diff. The audit of 2026-10-07 (A2 E-B4) found that no test imported any copy of it, so
switching to ``ddof=0`` or to 1·SE_diff left the suite green.

Each fixture below is computed by hand and chosen so that exactly one of those edits flips it. The
expected values are written as literals, never derived with numpy, so the test cannot share a bug
with the code. This pins the live copy in `scripts/assemble_gate_dgf1_1.py`. A shared, tested
criterion for later gates belongs with the ADR that fixes their statistics, and outside `gbe/`
(gbe/CLAUDE.md rule 3).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from assemble_gate_dgf1_1 import criterion  # noqa: E402

# Both arms have sample std exactly 0.1 at n = 3 (population std 0.0816…), so
# SE_diff = sqrt(0.01/3 + 0.01/3) = 0.08165 with ddof=1, and 0.06667 with ddof=0.
SE_DDOF1 = math.sqrt(0.01 / 3 + 0.01 / 3)
SE_DDOF0 = math.sqrt((0.02 / 3) / 3 + (0.02 / 3) / 3)
ARM = [1.0, 1.1, 1.2]


def test_sample_std_and_se_diff_are_the_pinned_ones():
    c = criterion(ARM, [0.9, 1.0, 1.1])
    assert c["g_sd"] == pytest.approx(0.1) and c["f_sd"] == pytest.approx(0.1)
    assert c["se"] == pytest.approx(SE_DDOF1)
    assert c["two_se"] == pytest.approx(2 * SE_DDOF1)


def test_a_gap_between_the_ddof0_and_ddof1_bars_is_not_resolvable():
    """diff = 0.15 sits between 2·SE with ddof=0 (0.133) and with ddof=1 (0.163)."""
    assert 2 * SE_DDOF0 < 0.15 < 2 * SE_DDOF1
    c = criterion(ARM, [0.85, 0.95, 1.05])
    assert c["diff"] == pytest.approx(0.15)
    assert c["resolvable"] is False, "the criterion used the population std (ddof=0)"


def test_a_gap_between_one_and_two_se_is_not_resolvable():
    """diff = 0.12 clears 1·SE_diff (0.082) but not 2·SE_diff (0.163), nor 2·SE with ddof=0."""
    assert SE_DDOF1 < 0.12 < 2 * SE_DDOF0
    c = criterion(ARM, [0.88, 0.98, 1.08])
    assert c["resolvable"] is False, "the criterion used a multiplier below 2"


def test_a_gap_above_two_se_is_resolvable():
    c = criterion(ARM, [0.6, 0.7, 0.8])
    assert c["diff"] == pytest.approx(0.4) and c["resolvable"] is True


def test_a_negative_gap_is_never_resolvable():
    c = criterion([0.6, 0.6000001, 0.6000002], [0.9, 0.9000001, 0.9000002])
    assert c["diff"] < 0 and c["resolvable"] is False
