"""Tests for the gate statistics v2 prototype (draft 2).

Anchors: textbook values where one exists (Student's t at df 4, two-sided 95%: 2.776), and
hand-computed values (Welch df 7.2 below). Sizing expectations are frozen from the measurement's
section G grid (`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt`), which is computed by
an independent copy of the power function.
"""

from __future__ import annotations

import math

import pytest

from gate_stats_proto import ALPHA, look_level, plan, resolvability, stage1_power

ARM = [1.0, 1.1, 1.2, 1.3, 1.4]  # mean 1.2, sample var 0.025 (ddof=1), population var 0.02
FIXED = [0.0] * 5                # a seed-deterministic comparator


def arm(mean: float, s: float) -> list[float]:
    """Five values with mean `mean` and sample standard deviation exactly `s` (up to rounding)."""
    h = s / math.sqrt(2.5)
    return [mean + k * h for k in (-2, -1, 0, 1, 2)]


# ------------------------------------------------------------------ the per-look level


def test_alpha_is_the_one_sided_level_of_two_se():
    assert ALPHA == pytest.approx(0.0227501, abs=1e-7)


def test_two_looks_split_alpha():
    assert look_level(ALPHA, 8, 20) == pytest.approx(0.011375, abs=1e-6)
    assert look_level(ALPHA, 16, 20) == pytest.approx(0.011375, abs=1e-6)


def test_a_single_look_keeps_alpha():
    assert look_level(ALPHA, 20, 20) == pytest.approx(ALPHA)


# ------------------------------------------------------------------ the criterion


def test_textbook_t_value_at_welch_df_of_a_deterministic_comparator():
    r = resolvability(ARM, [1.0] * 5, level=0.025)
    assert r["df"] == pytest.approx(4.0)
    assert r["t_crit"] == pytest.approx(2.776, abs=5e-4)
    assert r["se"] == pytest.approx(math.sqrt(0.025 / 5))


def test_welch_df_with_two_varying_arms():
    # var 2.5 and 5 at n = 5: v = 0.5 and 1.0, df = 1.5^2 / (0.25/4 + 1/4) = 7.2 exactly.
    r = resolvability([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 5.0], level=ALPHA)
    assert r["df"] == pytest.approx(7.2, abs=1e-12)
    assert r["se"] == pytest.approx(math.sqrt(1.5))


def test_welch_t_not_z_and_sample_not_population_variance():
    # diff 0.2, SE 0.070711, t 2.828: below t_crit(df 4) = 2.869 at level ALPHA. It would resolve
    # under a z critical value (2.0), under ddof=0 (t 3.162), or under pooled df 8 (t_crit 2.36).
    r = resolvability(ARM, [1.0] * 5, level=ALPHA)
    assert r["t_crit"] == pytest.approx(2.869, abs=5e-4)
    assert r["resolvable"] is False


def test_one_sided_level():
    # diff 0.23, t 3.253: above t_crit 2.869 at ALPHA, below 3.467 at ALPHA / 2.
    assert resolvability(ARM, [0.97] * 5, level=ALPHA)["resolvable"] is True
    assert resolvability(ARM, [0.97] * 5, level=ALPHA / 2)["resolvable"] is False


def test_p_value_is_one_sided_and_matches_the_level():
    # diff / SE set to the textbook 2.776445 at df 4, so the one-sided p is 0.025.
    b = [1.2 - 2.776445 * math.sqrt(0.025 / 5)] * 5
    r = resolvability(ARM, b, level=ALPHA)
    assert r["p_one_sided"] == pytest.approx(0.025, abs=1e-4)


def test_direction_matters():
    assert resolvability([0.97] * 5, ARM, level=ALPHA)["resolvable"] is False


# ------------------------------------------------------------------ refusals


def test_seed_floor_is_five_per_arm():
    with pytest.raises(ValueError, match="seed floor"):
        resolvability(ARM[:4], [1.0] * 4, level=ALPHA)


def test_unequal_seed_counts_are_refused():
    with pytest.raises(ValueError, match="unequal seed counts"):
        resolvability(ARM, [1.0] * 6, level=ALPHA)


def test_bit_identical_arms_do_not_vary_even_with_a_rounding_residue():
    # numpy gives [0.11] * 5 a ddof=1 variance of about 2.4e-34, not 0.
    with pytest.raises(ValueError, match="neither arm varies"):
        resolvability([0.11] * 5, [0.11] * 5, level=ALPHA)


def test_the_pilot_refuses_before_any_test_window_run():
    with pytest.raises(ValueError, match="neither arm varies"):
        plan({"c": (FIXED, FIXED)})
    with pytest.raises(ValueError, match="seed floor"):
        plan({"c": (arm(0.1, 0.01)[:4], FIXED[:4])})


# ------------------------------------------------------------------ power and stage-1 n


def test_planned_power_matches_the_measured_point():
    # Section G: delta / s = 1.3323 at 8 seeds, comparator fixed, level alpha / 2: planned 0.7793,
    # simulated 0.7795. A normal approximation gives about 0.80.
    assert stage1_power(1.0, 0.0, 8, 1.3323, ALPHA / 2) == pytest.approx(0.7793, abs=1e-3)


def test_sizing_uses_the_split_level_half_the_gap_and_eighty_percent():
    # Section G grid (Bonferroni column): s 0.005 at gap 0.010 -> 16 (12 at a single-look level,
    # or under the common boundary); s 0.010 at gap 0.020 -> 16; s 0.020 at gap 0.050 -> 12.
    assert plan({"c": (arm(0.010, 0.005), FIXED)})["n"] == 16
    assert plan({"c": (arm(0.020, 0.010), FIXED)})["n"] == 16
    assert plan({"c": (arm(0.050, 0.020), FIXED)})["n"] == 12
    assert plan({"c": (arm(0.100, 0.002), FIXED)})["n"] == 8


def test_sizing_needs_every_clause():
    p = plan({"easy": (arm(0.100, 0.002), FIXED), "hard": (arm(0.020, 0.010), FIXED)})
    assert p["n"] == 16


def test_no_positive_gap_gives_the_cap_and_is_not_powered():
    p = plan({"c": (arm(-0.010, 0.002), FIXED)})
    assert p["n"] == 20 and p["powered"] == {"c": False}


def test_under_powered_at_the_cap_is_reported():
    p = plan({"c": (arm(0.010, 0.010), FIXED)})  # section G: 20, still under-powered
    assert p["n"] == 20 and p["powered"] == {"c": False}


def test_joint_power_is_the_union_bound():
    p = plan({"one": (arm(0.020, 0.010), FIXED), "two": (arm(0.020, 0.010), FIXED)})
    row = next(r for r in p["rows"] if r["n"] == p["n"])
    single = row["power"]["one"]
    assert 0.8 <= single < 1.0
    assert row["joint_power_at_least"] == pytest.approx(2 * single - 1)
