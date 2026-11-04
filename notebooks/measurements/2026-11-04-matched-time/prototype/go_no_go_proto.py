"""Prototype: matched-time's go/no-go rule, its margin clause and its reading (ADR-023, ADR-017 item 3).

The rule decides whether the batch is worth its look without seeing the gap between the arms. So the
pilot is first reduced to what the rule may see: the higher ROC-AUC mean, with no word of whose it is,
and each arm's spread. ``decide`` receives nothing else.

The margin clause is ADR-018 clause 1's one-sided absence claim: clause 2 with the arms swapped and
the comparator shifted up by the margin, at the look's level. It does not size the batch, and its
non-resolution does not trigger a second look.

Reads no data and computes no model metric: its inputs are values already in registry rows.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from gate_stats import ALPHA, POWER_TARGET, judge, sample_var, stage2_triggered, welch_power  # noqa: E402

METRICS = ("fraud_auc", "fraud_auprc")
SANITY_BAR = 0.55          # the researcher's number (ADR-023 clause 5)
PILOT_SEEDS = (0, 1, 2, 3, 4)   # fixed at acceptance: the researcher's number
FINAL_N = 20
MARGIN_FACTOR = 0.5        # ADR-018 clause 4's planning factor


class BrokenPilot(ValueError):
    """A pilot the rule may not read. Root cause, then every seed again on one commit; no rule is computed."""


@dataclass(frozen=True)
class RuleInputs:
    """Everything the rule sees. It has no field that holds either arm's mean, or the gap."""

    best_auc: float
    spreads: Mapping[str, tuple[float, float]]


def rule_inputs(dgf1: Mapping[str, Mapping[int, float]], floor: Mapping[str, Mapping[int, float]]) -> RuleInputs:
    """Reduce the pilot to what the rule may see, refusing a pilot that is not exactly the fixed seeds."""
    for arm, values in (("dgf1", dgf1), ("floor", floor)):
        for metric in METRICS:
            if metric not in values or set(values[metric]) != set(PILOT_SEEDS):
                raise BrokenPilot(f"{arm} {metric} is not exactly seeds {PILOT_SEEDS}")
            if not all(math.isfinite(v) for v in values[metric].values()):
                raise BrokenPilot(f"{arm} {metric} holds a non-finite value")
    spreads = {m: (math.sqrt(sample_var(list(dgf1[m].values()))), math.sqrt(sample_var(list(floor[m].values()))))
               for m in METRICS}
    best = max(float(np.mean(list(dgf1["fraud_auc"].values()))), float(np.mean(list(floor["fraud_auc"].values()))))
    return RuleInputs(best_auc=best, spreads=spreads)


def margin(delta_gated: float) -> float | None:
    """The clause's margin, half the gated validation gap; None where the gated view had no gain."""
    return MARGIN_FACTOR * delta_gated if delta_gated > 0 else None


def margin_power(s_dgf1: float, s_floor: float, delta_gated: float, alpha: float = ALPHA) -> float:
    """Power at 20 seeds, at the two-look level, to resolve a gap the size of the margin."""
    m = margin(delta_gated)
    if m is None or (s_dgf1 == 0 and s_floor == 0):
        return 0.0
    return welch_power(s_dgf1, s_floor, FINAL_N, m, alpha / 2)


def decide(inputs: RuleInputs, delta_gated: Mapping[str, float], alpha: float = ALPHA) -> dict:
    """Go needs the sanity bar, and enough precision on at least one metric."""
    sane = inputs.best_auc >= SANITY_BAR
    power = {m: margin_power(*inputs.spreads[m], delta_gated[m], alpha) for m in METRICS}
    informative = [m for m in METRICS if power[m] >= POWER_TARGET]
    return {"go": sane and bool(informative), "sane": sane, "power": power, "informative": informative}


def below_margin(dgf1: Sequence[float], floor: Sequence[float], m: float, *, n1: int, n2: int, stage: int,
                 pilot_s: tuple[float, float], alpha: float = ALPHA) -> dict:
    """Is the gain below ``m``? Clause 2 with the arms swapped and the comparator shifted up by ``m``."""
    return judge(np.asarray(floor, dtype=float) + m, dgf1, n1=n1, n2=n2, stage=stage, pilot_s=pilot_s, alpha=alpha)


def second_look(gain_results: Sequence[Mapping]) -> bool:
    """Only the gain clauses can trigger the second look. The margin clauses are never passed here."""
    return stage2_triggered(gain_results)


def reading(gain: Mapping, below: Mapping | None) -> str:
    """Which row of the report's table a metric reads."""
    if gain["refused"]:
        return "refused"
    resolved = bool(gain["resolvable"])
    under = below is not None and not below["refused"] and bool(below["resolvable"])
    if resolved and under:
        return "resolvable and below the margin"
    if resolved:
        return "resolvable"
    return "below the margin" if under else "not resolvable"
