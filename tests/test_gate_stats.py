"""The shared gate statistics (ADR-018 clause 7, plan row 2a).

Expected values are literals, never derived with numpy or scipy, so a test cannot share a bug with
the code. Anchors: textbook values where one exists (Student's t at df 4, two-sided 95%: 2.776), and
hand-computed values (the Welch df of 7.2 below). Sizing expectations are frozen from section G of
`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt`, computed by an independent copy of the
power function and checked there by Monte Carlo. All inputs are synthetic.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gate_stats import (  # noqa: E402
    ALPHA,
    PilotClause,
    Refused,
    alpha_power_table,
    judge,
    look_level,
    pilot_arm,
    plan,
    powered_failure,
    resolvability,
    seed_correlation,
    stage2_triggered,
    welch_power,
)

ARM = [1.0, 1.1, 1.2, 1.3, 1.4]  # mean 1.2, sample var 0.025 (ddof=1), population var 0.02
FIXED = [0.0] * 5                # a seed-deterministic comparator


def arm(mean: float, s: float, n: int = 5) -> list[float]:
    """n evenly spaced values with mean `mean` and sample standard deviation exactly `s`."""
    h = s / math.sqrt(n * (n + 1) / 12)
    return [mean + (k - (n - 1) / 2) * h for k in range(n)]


def clause(a, b, direction: str = "a > b") -> PilotClause:
    return PilotClause(direction, dict(enumerate(a)), dict(enumerate(b)))


# ------------------------------------------------------------------ the per-look level


def test_alpha_is_the_one_sided_level_of_two_se():
    assert ALPHA == pytest.approx(0.0227501, abs=1e-7)


def test_two_looks_split_alpha():
    assert look_level(ALPHA, 8, 20) == pytest.approx(0.011375, abs=1e-6)
    assert look_level(ALPHA, 16, 20) == pytest.approx(0.011375, abs=1e-6)


def test_a_single_look_keeps_alpha():
    assert look_level(ALPHA, 20, 20) == pytest.approx(ALPHA)


def test_stage_one_cannot_exceed_stage_two():
    with pytest.raises(ValueError, match="cannot exceed"):
        look_level(ALPHA, 21, 20)


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
    # diff 0.23, t 3.253: above t_crit 2.869 at ALPHA, below 3.600 at ALPHA / 2.
    assert resolvability(ARM, [0.97] * 5, level=ALPHA)["resolvable"] is True
    assert resolvability(ARM, [0.97] * 5, level=ALPHA / 2)["resolvable"] is False


def test_p_value_is_one_sided_and_matches_the_level():
    # diff / SE set to the textbook 2.776445 at df 4, so the one-sided p is 0.025.
    b = [1.2 - 2.776445 * math.sqrt(0.025 / 5)] * 5
    assert resolvability(ARM, b, level=ALPHA)["p_one_sided"] == pytest.approx(0.025, abs=1e-4)


def test_direction_matters():
    assert resolvability([0.97] * 5, ARM, level=ALPHA)["resolvable"] is False


def test_a_non_finite_metric_is_an_error_not_a_verdict():
    with pytest.raises(ValueError, match="non-finite"):
        resolvability([1.0, 1.1, 1.2, 1.3, float("nan")], FIXED, level=ALPHA)


# ------------------------------------------------------------------ refusals


def test_seed_floor_is_five_per_arm():
    with pytest.raises(Refused, match="seed floor"):
        resolvability(ARM[:4], [1.0] * 4, level=ALPHA)


def test_unequal_seed_counts_are_refused():
    with pytest.raises(Refused, match="unequal seed counts"):
        resolvability(ARM, [1.0] * 6, level=ALPHA)


def test_bit_identical_arms_do_not_vary_even_with_a_rounding_residue():
    # numpy gives [0.11] * 5 a ddof=1 variance of about 2.4e-34, not 0.
    with pytest.raises(Refused, match="neither arm varies"):
        resolvability([0.11] * 5, [0.11] * 5, level=ALPHA)


def test_the_pilot_refuses_before_any_test_window_run():
    with pytest.raises(Refused, match="neither arm varies"):
        plan({"c": clause(FIXED, FIXED)})
    with pytest.raises(Refused, match="seed floor"):
        plan({"c": clause(arm(0.1, 0.01)[:4], FIXED[:4])})
    with pytest.raises(Refused, match="unequal seed counts"):
        plan({"c": clause(arm(0.1, 0.01), [0.0] * 6)})


# ------------------------------------------------------------------ one clause at one look


def test_judge_uses_the_design_s_level_at_each_look():
    two_looks = judge(ARM, [0.97] * 5, n1=5, n2=20, stage=1, pilot_s=(0.1, 0.0))
    single = judge(ARM, [0.97] * 5, n1=5, n2=5, stage=1, pilot_s=(0.1, 0.0))
    assert two_looks["level"] == pytest.approx(0.011375, abs=1e-6) and two_looks["resolvable"] is False
    assert single["level"] == pytest.approx(ALPHA) and single["resolvable"] is True


def test_judge_prints_realised_and_planned_se():
    # Realised sqrt(0.025 / 5); planned from the pilot's sds at the look's n: sqrt((0.04 + 0.01) / 5).
    r = judge(ARM, [1.0] * 5, n1=5, n2=20, stage=1, pilot_s=(0.2, 0.1))
    assert r["se"] == pytest.approx(math.sqrt(0.005))
    assert r["planned_se"] == pytest.approx(0.1)
    assert {"diff", "df", "level", "t_crit", "p_one_sided", "verdict"} <= set(r)


def test_stage_two_is_cumulative():
    a, b = arm(1.0, 0.1, 20), [0.0] * 20
    assert judge(a, b, n1=8, n2=20, stage=2, pilot_s=(0.1, 0.0))["n"] == 20
    with pytest.raises(ValueError, match="judges 20 seeds"):
        judge(arm(1.0, 0.1, 12), [0.0] * 12, n1=8, n2=20, stage=2, pilot_s=(0.1, 0.0))
    with pytest.raises(ValueError, match="judges 8 seeds"):
        judge(arm(1.0, 0.1, 20), [0.0] * 20, n1=8, n2=20, stage=1, pilot_s=(0.1, 0.0))


def test_a_single_look_design_has_no_stage_two():
    with pytest.raises(ValueError, match="no stage 2"):
        judge(arm(1.0, 0.1, 20), [0.0] * 20, n1=20, n2=20, stage=2, pilot_s=(0.1, 0.0))


def test_a_refused_clause_is_reported_cannot_pass_and_triggers_stage_two():
    r = judge([0.11] * 8, [0.11] * 8, n1=8, n2=20, stage=1, pilot_s=(0.0, 0.0))
    assert r["refused"] and r["verdict"].startswith("refused: neither arm varies")
    assert r["resolvable"] is False
    assert stage2_triggered([r]) is True


def test_stage_two_runs_iff_any_clause_is_unresolved():
    passed = judge(arm(1.0, 0.1, 8), [0.0] * 8, n1=8, n2=20, stage=1, pilot_s=(0.1, 0.0))
    failed = judge(arm(0.0, 0.1, 8), [0.0] * 8, n1=8, n2=20, stage=1, pilot_s=(0.1, 0.0))
    assert passed["resolvable"] is True and failed["resolvable"] is False
    assert stage2_triggered([passed, passed]) is False
    assert stage2_triggered([passed, failed]) is True


# ------------------------------------------------------------------ power and stage-1 n


def test_planned_power_matches_the_measured_point():
    # Section G: delta / s = 1.3323 at 8 seeds, comparator fixed, level alpha / 2: planned 0.7793,
    # simulated 0.7795. A normal approximation gives about 0.80.
    assert welch_power(1.0, 0.0, 8, 1.3323, ALPHA / 2) == pytest.approx(0.7793, abs=1e-3)


def test_sizing_uses_the_split_level_half_the_gap_and_eighty_percent():
    # Section G grid (Bonferroni column): s 0.005 at gap 0.010 -> 16 (12 at a single-look level,
    # or under the common boundary); s 0.010 at gap 0.020 -> 16; s 0.020 at gap 0.050 -> 12.
    assert plan({"c": clause(arm(0.010, 0.005), FIXED)})["n"] == 16
    assert plan({"c": clause(arm(0.020, 0.010), FIXED)})["n"] == 16
    assert plan({"c": clause(arm(0.050, 0.020), FIXED)})["n"] == 12
    assert plan({"c": clause(arm(0.100, 0.002), FIXED)})["n"] == 8


def test_sizing_needs_every_clause():
    p = plan({"easy": clause(arm(0.100, 0.002), FIXED), "hard": clause(arm(0.020, 0.010), FIXED)})
    assert p["n"] == 16


def test_a_pre_registered_planning_factor_changes_the_size():
    # s 0.020 at gap 0.050 needs 12 at the default half; at the full gap it needs the smallest, 8.
    assert plan({"c": clause(arm(0.050, 0.020), FIXED)}, factor=1.0)["n"] == 8


def test_a_clause_below_positive_gap_gives_the_cap_and_is_not_powered():
    p = plan({"bad": clause(arm(-0.010, 0.002), FIXED), "good": clause(arm(0.100, 0.002), FIXED)})
    assert p["n"] == 20 and p["powered"] == {"bad": False, "good": True}


def test_under_powered_at_the_cap_is_reported():
    p = plan({"c": clause(arm(0.010, 0.010), FIXED)})  # section G: 20, still under-powered
    assert p["n"] == 20 and p["powered"] == {"c": False}


def test_the_cap_is_a_single_look_at_alpha():
    p = plan({"c": clause(arm(0.010, 0.010), FIXED)})
    assert p["rows"][-1]["n"] == 20 and p["rows"][-1]["level"] == pytest.approx(ALPHA)
    assert p["rows"][0]["level"] == pytest.approx(0.011375, abs=1e-6)


def test_both_gate_bounds_are_the_same_sum():
    p = plan({"one": clause(arm(0.020, 0.010), FIXED), "two": clause(arm(0.020, 0.010), FIXED)})
    row = next(r for r in p["rows"] if r["n"] == p["n"])
    single = row["power"]["one"]
    assert 0.8 <= single < 1.0
    assert row["joint_power_at_least"] == pytest.approx(2 * single - 1)
    assert row["stage2_at_most"] == pytest.approx(2 * (1 - single))


def test_the_bounds_stay_probabilities():
    weak = clause(arm(0.002, 0.010), FIXED)
    row = plan({"one": weak, "two": weak, "three": weak})["rows"][0]
    assert row["joint_power_at_least"] == 0.0 and row["stage2_at_most"] == 1.0


@pytest.mark.parametrize("kw", [dict(candidates=(4, 20)), dict(candidates=(12, 8, 20)),
                                dict(candidates=(8, 12), cap=20)])
def test_a_departure_keeps_the_floor_and_a_rising_set_ending_at_the_cap(kw):
    with pytest.raises(ValueError, match="candidates must rise"):
        plan({"c": clause(arm(0.1, 0.01), FIXED)}, **kw)


def test_a_departure_from_the_default_design_is_sized_on_its_own_set():
    p = plan({"c": clause(arm(0.100, 0.002), FIXED)}, candidates=(6, 10), cap=10)
    assert p["n"] == 6 and p["cap"] == 10 and p["simulated"] is False


# ------------------------------------------------------------------ powered failure


FAILED = arm(0.001, 0.010, 16)    # the gap is far below the planning effect, so it does not resolve
DELTA = 0.010                     # sizing s 0.010 at gap 0.020: chosen n is 16, power above 0.80


def _failed(a=FAILED, n1=16):
    return judge(a, [0.0] * len(a), n1=n1, n2=20, stage=1, pilot_s=(0.010, 0.0))


def test_a_powered_failure_needs_planned_and_realised_power():
    assert _failed()["resolvable"] is False
    assert powered_failure(_failed(), FAILED, [0.0] * 16, planned_power=0.85, delta=DELTA) is True


def test_low_planned_power_is_not_a_powered_failure():
    assert powered_failure(_failed(), FAILED, [0.0] * 16, planned_power=0.79, delta=DELTA) is False


def test_realised_noise_above_the_pilot_s_is_not_a_powered_failure():
    noisy = arm(0.001, 0.030, 16)
    assert powered_failure(_failed(noisy), noisy, [0.0] * 16, planned_power=0.85, delta=DELTA) is False


def test_the_realised_gap_is_never_used():
    # Same realised sds, a gap in the wrong direction: the answer turns on delta and the sds alone.
    below = arm(-0.050, 0.010, 16)
    assert powered_failure(_failed(below), below, [0.0] * 16, planned_power=0.85, delta=DELTA) is True


def test_power_is_recomputed_at_the_deciding_look():
    # Both looks of a two-look design test at alpha / 2. At s 0.010 and delta 0.0078, power is 0.715 at
    # 16 seeds and 0.833 at 20. A single look at 20 tests at alpha: at delta 0.007, power is 0.832
    # there, and would be 0.738 at alpha / 2.
    a16, a20 = arm(0.001, 0.010, 16), arm(0.001, 0.010, 20)
    at_16 = judge(a16, [0.0] * 16, n1=16, n2=20, stage=1, pilot_s=(0.010, 0.0))
    at_20 = judge(a20, [0.0] * 20, n1=16, n2=20, stage=2, pilot_s=(0.010, 0.0))
    single = judge(a20, [0.0] * 20, n1=20, n2=20, stage=1, pilot_s=(0.010, 0.0))
    assert powered_failure(at_16, a16, [0.0] * 16, planned_power=0.85, delta=0.0078) is False
    assert powered_failure(at_20, a20, [0.0] * 20, planned_power=0.85, delta=0.0078) is True
    assert powered_failure(single, a20, [0.0] * 20, planned_power=0.85, delta=0.007) is True


def test_a_refused_or_resolved_clause_is_never_a_powered_failure():
    same = [0.11] * 16
    refused = judge(same, same, n1=16, n2=20, stage=1, pilot_s=(0.010, 0.0))
    assert powered_failure(refused, same, same, planned_power=0.85, delta=DELTA) is False
    high = arm(1.0, 0.010, 16)
    assert powered_failure(_failed(high), high, [0.0] * 16, planned_power=0.85, delta=DELTA) is False


def test_no_planning_effect_is_not_a_powered_failure():
    assert powered_failure(_failed(), FAILED, [0.0] * 16, planned_power=float("nan"), delta=-0.001) is False


# ------------------------------------------------------------------ the correlation clause 6 prints


def test_per_seed_correlation_on_shared_seed_ids():
    a = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}
    assert seed_correlation(a, {s: 2 * v for s, v in a.items()}) == pytest.approx(1.0)
    assert seed_correlation(a, {s: -v for s, v in a.items()}) == pytest.approx(-1.0)
    # Pairs by seed id, never by position: the same values in another key order are still +1.
    assert seed_correlation(a, {4: 10.0, 3: 8.0, 2: 6.0, 1: 4.0, 0: 2.0}) == pytest.approx(1.0)


def test_correlation_is_undefined_without_shared_seeds_or_variation():
    a = {0: 1.0, 1: 2.0, 2: 3.0}
    assert seed_correlation(a, {7: 1.0, 8: 2.0, 9: 3.0}) is None
    assert seed_correlation(a, {0: 5.0, 1: 5.0, 2: 5.0}) is None


# ------------------------------------------------------------------ the table


def test_the_table_prints_every_item_of_clause_six():
    text = alpha_power_table(plan({"gnn-vs-floor": PilotClause(
        "GNN above floor", dict(enumerate(arm(0.020, 0.010))), dict(enumerate(FIXED)))}))
    for item in ("Stage-1 seeds: 16", "Design: 16 -> 20", "α = 0.02275", "16: 0.011375", "20: 0.022750",
                 "Clause gnn-vs-floor: GNN above floor", "pilot s: 0.010000 (varied) vs 0.000000 (did not vary)",
                 "Δ_val: 0.020000; δ: 0.010000", "per-seed correlation: n/a", "planned stage-1 power: 8: ",
                 "powered: yes", "Joint stage-1 power at least: ", "Stage 2 triggered at most: ",
                 "per-look levels sum to α (ADR-018 clause 3)", "0.0165–0.0215", "0.0201–0.0230"):
        assert item in text, item
    assert "not simulated" not in text


def test_the_table_says_when_a_design_was_not_simulated():
    p = plan({"c": clause(arm(0.100, 0.002), FIXED)}, candidates=(6, 10), cap=10)
    assert alpha_power_table(p).endswith("; not simulated")


def test_the_table_prints_no_power_for_a_non_positive_gap():
    text = alpha_power_table(plan({"c": clause(arm(-0.010, 0.002), FIXED)}))
    assert "8: n/a" in text and "powered: no" in text and "Joint stage-1 power at least: n/a" in text


# ------------------------------------------------------------------ from registry rows


def row(seed=0, *, arm_name="gnn", window="370-481", value=0.7, dirty="false", deterministic=True, notes="",
        commit="abc", model="dgf1"):
    metrics = {"arm": arm_name, "window": window, "fraud_auc": value, "deterministic": deterministic}
    return {"run_id": f"r{seed}", "model": model, "git_commit": commit, "git_dirty": dirty, "seed": str(seed),
            "metrics_json": json.dumps(metrics), "notes": notes}


ARGS = dict(model="dgf1", arm="gnn", metric="fraud_auc", window="370-481")


def test_pilot_values_come_keyed_by_seed_from_the_matching_rows_only():
    rows = [row(0, value=0.7), row(1, value=0.8), row(2, arm_name="floor"), row(3, window="1-369"),
            row(4, model="ell1")]
    assert pilot_arm(rows, **ARGS) == {0: 0.7, 1: 0.8}


def test_a_pilot_never_reads_a_test_window():
    with pytest.raises(ValueError, match="never touches a test window"):
        pilot_arm([], **{**ARGS, "window": "482-821"})


@pytest.mark.parametrize("bad", [dict(dirty="true"), dict(deterministic=False), dict(notes="ERRORED: oom")])
def test_a_pilot_row_must_be_clean_and_deterministic(bad):
    with pytest.raises(ValueError, match="not a clean, deterministic"):
        pilot_arm([row(0), row(1, **bad)], **ARGS)


def test_a_pilot_has_no_retries_and_one_commit():
    with pytest.raises(ValueError, match="twice"):
        pilot_arm([row(0), row(0)], **ARGS)
    with pytest.raises(ValueError, match="spans commits"):
        pilot_arm([row(0), row(1, commit="def")], **ARGS)
