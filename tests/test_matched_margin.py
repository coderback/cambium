"""ADR-023's margin reads ADR-012's validation pilot and nothing else, and says where the rule starts to fail."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import matched_margin as M  # noqa: E402

OFFSETS = (-0.002, -0.001, 0.0, 0.001, 0.002)


def row(arm, seed, auc, auprc, tag="pilot", window="370-481", commit="c1", model="dgf1"):
    m = {"experiment": tag, "arm": arm, "window": window, "deterministic": True, "fraud_auc": auc, "fraud_auprc": auprc}
    return {"run_id": f"{arm}-{seed}-{tag}", "model": model, "seed": str(seed), "git_commit": commit, "git_dirty": "false",
            "notes": "", "metrics_json": json.dumps(m)}


def pilot(dgf1=(0.75, 0.04), floor=(0.71, 0.03), commit="c1"):
    rows = []
    for arm, (auc, auprc) in ((M.DGF1_ARM, dgf1), (M.FLOOR_ARM, floor)):
        rows += [row(arm, s, auc + OFFSETS[s], auprc + 0.1 * OFFSETS[s], commit=commit) for s in range(5)]
    return rows


def test_the_gap_is_dgf1_minus_the_floor_and_the_margin_is_half_of_it():
    arms = M.gated_pilot(pilot())
    out = M.margin_row(arms[M.DGF1_ARM]["fraud_auc"], arms[M.FLOOR_ARM]["fraud_auc"])
    assert out["gap"] == pytest.approx(0.04) and out["margin"] == pytest.approx(0.02)


def test_a_gap_at_or_below_zero_has_no_margin():
    arms = M.gated_pilot(pilot(dgf1=(0.70, 0.04), floor=(0.71, 0.03)))
    out = M.margin_row(arms[M.DGF1_ARM]["fraud_auc"], arms[M.FLOOR_ARM]["fraud_auc"])
    assert out["gap"] < 0 and math.isnan(out["margin"]) and "fails_both" not in out


@pytest.mark.parametrize("extra", [
    row(M.DGF1_ARM, 0, 0.1, 0.1, tag="repro_check", commit="c9"),      # a repro-check shares its arm and window
    row(M.DGF1_ARM, 0, 0.1, 0.1, tag="retune", commit="c8"),
    row(M.DGF1_ARM, 0, 0.1, 0.1, tag="gate", window="482-821", commit="c7"),
    row(M.DGF1_ARM, 0, 0.1, 0.1, tag="pilot", window="482-821", commit="c6"),   # even tagged pilot, another window
    row(M.DGF1_ARM, 0, 0.1, 0.1, tag="pilot", model="ell1", commit="c5"),
])
def test_only_the_pilot_s_rows_are_read(extra):
    assert M.gated_pilot([extra, *pilot()]) == M.gated_pilot(pilot())


def test_the_report_reads_the_same_with_a_row_it_must_ignore():
    extra = row(M.DGF1_ARM, 0, 0.1, 0.1, tag="pilot", window="482-821", commit="c6")
    assert M.report([extra, *pilot()]) == M.report(pilot())


@pytest.mark.parametrize("drop", [0, 4])
def test_a_missing_seed_is_refused(drop):
    rows = [r for r in pilot() if not (r["seed"] == str(drop) and M.FLOOR_ARM in r["run_id"])]
    with pytest.raises(ValueError, match="not exactly seeds"):
        M.gated_pilot(rows)


def test_an_extra_seed_is_refused():
    with pytest.raises(ValueError, match="not exactly seeds"):
        M.gated_pilot([*pilot(), row(M.DGF1_ARM, 5, 0.75, 0.04)])


def test_arms_on_two_commits_are_refused():
    rows = [r if M.FLOOR_ARM not in r["run_id"] else {**r, "git_commit": "c2"} for r in pilot()]
    with pytest.raises(ValueError, match="span commits"):
        M.gated_pilot(rows)


def test_the_spread_at_the_bar_is_where_the_rule_s_power_falls_to_the_target():
    arms = M.gated_pilot(pilot())
    out = M.margin_row(arms[M.DGF1_ARM]["fraud_auc"], arms[M.FLOOR_ARM]["fraud_auc"])
    m, s_floor = out["margin"], out["s_floor"]
    assert M.welch_power(out["fails_dgf1"], s_floor, 20, m, M.ALPHA / 2) == pytest.approx(0.8, abs=1e-6)
    assert M.welch_power(out["fails_both"], out["fails_both"], 20, m, M.ALPHA / 2) == pytest.approx(0.8, abs=1e-6)
    assert M.welch_power(out["fails_dgf1"] * 1.1, s_floor, 20, m, M.ALPHA / 2) < 0.8
    assert out["fails_both"] < out["fails_dgf1"]


def test_the_power_at_half_the_margin_is_at_half_the_margin():
    arms = M.gated_pilot(pilot())
    out = M.margin_row(arms[M.DGF1_ARM]["fraud_auprc"], arms[M.FLOOR_ARM]["fraud_auprc"])
    assert out["power_half"] == pytest.approx(M.welch_power(out["s_dgf1"], out["s_floor"], 20, out["margin"] / 2, M.ALPHA / 2))
    assert "at a true gain of m/2" in M.report(pilot())


def test_the_rule_s_level_is_the_two_look_one_at_twenty_seeds():
    assert M.LEVEL == pytest.approx(M.ALPHA / 2) and M.FINAL_N == 20


def test_the_report_prints_both_metrics_and_where_each_fails():
    text = M.report(pilot())
    assert "fraud_auc: Δ_gated +0.040000, m 0.020000" in text and "fraud_auprc: Δ_gated +0.010000, m 0.005000" in text
    assert text.count("the rule starts to fail at") == 2
