"""The rule never sees the gap, every path has one outcome, and the reading is fixed in advance."""

from __future__ import annotations

import dataclasses
import inspect
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import go_no_go_proto as R  # noqa: E402

DELTA = {"fraud_auc": 0.04, "fraud_auprc": 0.008}
OFFSETS = (-0.002, -0.001, 0.0, 0.001, 0.002)


def arm(auc_mean, auprc_mean=0.03, spread=1.0, auprc_spread=1.0):
    """Five seeds around each mean. ``spread`` scales the offsets for both metrics' own sizes."""
    return {"fraud_auc": {s: auc_mean + spread * OFFSETS[s] for s in R.PILOT_SEEDS},
            "fraud_auprc": {s: auprc_mean + auprc_spread * 0.1 * OFFSETS[s] for s in R.PILOT_SEEDS}}


def run(dgf1, floor, delta=DELTA):
    return R.decide(R.rule_inputs(dgf1, floor), delta)


# ------------------------------------------------------------------ blind to direction


def test_the_rule_receives_no_mean_and_no_gap():
    assert {f.name for f in dataclasses.fields(R.RuleInputs)} == {"best_auc", "spreads"}
    assert list(inspect.signature(R.decide).parameters) == ["inputs", "delta_gated", "alpha"]


def test_swapping_the_arms_changes_nothing():
    a, b = arm(0.70, spread=2.0), arm(0.74, spread=0.5)
    swapped = run(b, a)
    original = run(a, b)
    assert original["go"] == swapped["go"] and original["sane"] == swapped["sane"]
    assert sorted(original["power"].values()) == pytest.approx(sorted(swapped["power"].values()))


@pytest.mark.parametrize("dgf1_mean", [0.60, 0.72, 0.76, 0.90])
def test_moving_an_arm_while_the_best_stays_above_the_bar_changes_nothing(dgf1_mean):
    assert run(arm(dgf1_mean), arm(0.74)) == run(arm(0.74), arm(0.74))


def test_a_gain_of_any_sign_gets_the_same_decision():
    assert run(arm(0.60), arm(0.74))["go"] is True
    assert run(arm(0.90), arm(0.74))["go"] is True


# ------------------------------------------------------------------ the sanity bar


def test_neither_arm_above_the_bar_is_a_no_go_however_precise():
    out = run(arm(0.54), arm(0.53))
    assert out["sane"] is False and out["go"] is False and out["informative"]


def test_exactly_at_the_bar_passes():
    assert R.decide(R.RuleInputs(best_auc=R.SANITY_BAR, spreads={m: (0.001, 0.001) for m in R.METRICS}), DELTA)["sane"]


def test_either_arm_can_clear_the_bar():
    assert run(arm(0.60), arm(0.50))["sane"] and run(arm(0.50), arm(0.60))["sane"]


def test_the_higher_mean_is_what_is_passed():
    assert R.rule_inputs(arm(0.60), arm(0.70)).best_auc == pytest.approx(0.70)
    assert R.rule_inputs(arm(0.70), arm(0.60)).best_auc == pytest.approx(0.70)


# ------------------------------------------------------------------ precision


def test_the_power_is_at_twenty_seeds_half_the_gap_and_the_two_look_level():
    s_a, s_b, d = 0.03, 0.02, 0.04
    want = R.welch_power(s_a, s_b, 20, 0.02, R.ALPHA / 2)
    assert R.margin_power(s_a, s_b, d) == pytest.approx(want)
    assert want != pytest.approx(R.welch_power(s_a, s_b, 12, 0.02, R.ALPHA / 2))
    assert want != pytest.approx(R.welch_power(s_a, s_b, 20, 0.04, R.ALPHA / 2))
    assert want != pytest.approx(R.welch_power(s_a, s_b, 20, 0.02, R.ALPHA))


def _spread_for_power(target, d):
    lo, hi = 1e-5, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if R.margin_power(mid, mid, d) > target else (lo, mid)
    return lo


def test_the_bar_is_eighty_percent_power():
    s = _spread_for_power(0.80, 0.04)
    inputs = lambda x: R.RuleInputs(best_auc=0.7, spreads={"fraud_auc": (x, x), "fraud_auprc": (9.0, 9.0)})  # noqa: E731
    assert R.decide(inputs(s * 0.99), DELTA)["go"] is True
    assert R.decide(inputs(s * 1.01), DELTA)["go"] is False


@pytest.mark.parametrize("auc_ok,auprc_ok,expected", [(True, False, ["fraud_auc"]), (False, True, ["fraud_auprc"]), (True, True, list(R.METRICS))])
def test_one_informative_metric_is_enough(auc_ok, auprc_ok, expected):
    spreads = {"fraud_auc": (0.001, 0.001) if auc_ok else (5.0, 5.0),
               "fraud_auprc": (0.0001, 0.0001) if auprc_ok else (5.0, 5.0)}
    out = R.decide(R.RuleInputs(best_auc=0.7, spreads=spreads), DELTA)
    assert out["go"] is True and out["informative"] == expected


def test_no_informative_metric_is_a_no_go():
    out = R.decide(R.RuleInputs(best_auc=0.7, spreads={m: (5.0, 5.0) for m in R.METRICS}), DELTA)
    assert out["go"] is False and out["informative"] == []


@pytest.mark.parametrize("gap", [0.0, -0.01])
def test_no_gated_gain_means_that_metric_can_never_be_informative(gap):
    assert R.margin(gap) is None
    assert R.margin_power(0.001, 0.001, gap) == 0.0


def test_neither_arm_varying_fails_precision_without_crashing():
    assert R.margin_power(0.0, 0.0, 0.04) == 0.0
    flat = {"fraud_auc": {s: 0.7 for s in R.PILOT_SEEDS}, "fraud_auprc": {s: 0.03 for s in R.PILOT_SEEDS}}
    assert run(flat, flat)["informative"] == []


def test_one_arm_varying_is_enough_to_judge():
    assert R.margin_power(0.001, 0.0, 0.04) > 0.8


# ------------------------------------------------------------------ a broken pilot has no decision


@pytest.mark.parametrize("seeds", [(0, 1, 2, 3), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 9)])
def test_only_the_fixed_seeds_are_read(seeds):
    bad = {m: {s: 0.5 + 0.001 * s for s in seeds} for m in R.METRICS}
    for a, b in ((bad, arm(0.7)), (arm(0.7), bad)):
        with pytest.raises(R.BrokenPilot, match="seeds"):
            R.rule_inputs(a, b)


def test_a_missing_metric_is_a_broken_pilot():
    with pytest.raises(R.BrokenPilot):
        R.rule_inputs({"fraud_auc": arm(0.7)["fraud_auc"]}, arm(0.7))


def test_a_non_finite_value_is_a_broken_pilot():
    bad = arm(0.7)
    bad["fraud_auc"][2] = math.nan
    with pytest.raises(R.BrokenPilot, match="non-finite"):
        R.rule_inputs(bad, arm(0.7))


# ------------------------------------------------------------------ the margin clause


FLOOR = [0.70 + 0.001 * i for i in range(-10, 10)]


def below(gain, m=0.02, spread=1.0):
    dgf1 = [v * 0 + 0.70 + gain + spread * 0.001 * i for v, i in zip(FLOOR, range(-10, 10))]
    return R.below_margin(dgf1, FLOOR, m, n1=20, n2=20, stage=1, pilot_s=(0.006, 0.006))


def test_a_gain_far_below_the_margin_resolves():
    assert below(0.0)["resolvable"] is True


def test_a_gain_at_or_above_the_margin_never_resolves():
    assert below(0.02)["resolvable"] is False
    assert below(0.05)["resolvable"] is False


def test_the_clause_tests_the_shifted_comparator_at_the_look_s_level():
    out = below(0.0)
    assert out["diff"] == pytest.approx(0.02)
    two_look = R.below_margin(FLOOR, FLOOR, 0.02, n1=8, n2=20, stage=2, pilot_s=(0.006, 0.006))
    assert out["level"] == pytest.approx(R.ALPHA) and two_look["level"] == pytest.approx(R.ALPHA / 2)


def test_the_clause_refuses_when_neither_arm_varies():
    flat = [0.7] * 20
    out = R.below_margin(flat, flat, 0.02, n1=20, n2=20, stage=1, pilot_s=(0.0, 0.0))
    assert out["refused"] and out["resolvable"] is False


def test_the_margin_is_half_the_gated_gap():
    assert R.margin(0.04) == pytest.approx(0.02)


def test_only_the_gain_clauses_trigger_a_second_look():
    assert R.second_look([{"resolvable": True}, {"resolvable": True}]) is False
    assert R.second_look([{"resolvable": True}, {"resolvable": False}]) is True


# ------------------------------------------------------------------ the reading


YES, NO = {"refused": None, "resolvable": True}, {"refused": None, "resolvable": False}
REFUSED = {"refused": "neither arm varies", "resolvable": False}


@pytest.mark.parametrize("gain,under,row", [
    (YES, NO, "resolvable"),
    (NO, YES, "below the margin"),
    (YES, YES, "resolvable and below the margin"),
    (NO, NO, "not resolvable"),
    (REFUSED, NO, "refused"),
    (REFUSED, YES, "refused"),
    (YES, None, "resolvable"),
    (NO, None, "not resolvable"),
    (NO, REFUSED, "not resolvable"),
    (YES, REFUSED, "resolvable"),
])
def test_each_result_has_one_row(gain, under, row):
    assert R.reading(gain, under) == row
