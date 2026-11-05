"""Prototype of the gate statistics v2 functions (scratch, written before ADR-018 is accepted).

Label-free arithmetic: no data, no registry, no model metric (ADR-017 item 3). Draft 2: the two looks
split alpha (Bonferroni), refusals happen at the pilot, and an arm "varies" unless its values are
bit-identical.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

# The one-sided level of "diff > 2 x SE_diff" under a normal statistic.
ALPHA = float(stats.norm.sf(2.0))  # 0.0227501...
SEED_FLOOR = 5
POWER_TARGET = 0.8
CANDIDATES = (8, 12, 16, 20)
CAP = 20


def look_level(alpha: float, n1: int, n2: int) -> float:
    """Per-look one-sided level: alpha / 2 when a second look is possible (n1 < n2), else alpha."""
    return alpha / 2 if n1 < n2 else alpha


def varies(values) -> bool:
    """An arm varies over seeds unless every value is bit-identical."""
    v = np.asarray(values, dtype=float)
    return not bool(np.all(v == v[0]))


def sample_var(values) -> float:
    """ddof=1 variance, exactly 0.0 for an arm that does not vary (rounding cannot leave a residue)."""
    v = np.asarray(values, dtype=float)
    return float(v.var(ddof=1)) if varies(v) else 0.0


def welch_df(var_a: float, n_a: int, var_b: float, n_b: int) -> float:
    """Welch-Satterthwaite df, not rounded; a zero-variance arm drops out of both sums."""
    va, vb = var_a / n_a, var_b / n_b
    den = (va * va / (n_a - 1) if va > 0 else 0.0) + (vb * vb / (n_b - 1) if vb > 0 else 0.0)
    return (va + vb) ** 2 / den


def _check_arms(a, b) -> None:
    if len(a) < SEED_FLOOR or len(b) < SEED_FLOOR:
        raise ValueError(f"seed floor: every arm needs at least {SEED_FLOOR} seeds")
    if len(a) != len(b):
        raise ValueError("unequal seed counts: every arm in a comparison runs the same seeds")
    if not (varies(a) or varies(b)):
        raise ValueError("neither arm varies over seeds; a seed test cannot judge this comparison")


def resolvability(a, b, level: float) -> dict:
    """One-sided Welch test of mean(a) > mean(b) at the per-look level from look_level()."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    _check_arms(a, b)
    var_a, var_b = sample_var(a), sample_var(b)
    se = math.sqrt(var_a / len(a) + var_b / len(b))
    df = welch_df(var_a, len(a), var_b, len(b))
    t_crit = float(stats.t.isf(level, df))
    diff = float(a.mean() - b.mean())
    return {"diff": diff, "se": se, "df": df, "level": level, "t_crit": t_crit,
            "p_one_sided": float(stats.t.sf(diff / se, df)), "resolvable": diff > t_crit * se}


def stage1_power(s_a: float, s_b: float, n: int, delta: float, level: float) -> float:
    """Planned power of the one-sided Welch test at n seeds per arm, true gap delta, sds s_a, s_b."""
    var_a, var_b = s_a * s_a, s_b * s_b
    se = math.sqrt(var_a / n + var_b / n)
    df = welch_df(var_a, n, var_b, n)
    return float(stats.nct.sf(stats.t.isf(level, df), df, delta / se))


def plan(pilot: dict, alpha: float = ALPHA, target: float = POWER_TARGET) -> dict:
    """The alpha and power table, and stage-1 n, from validation pilot values.

    pilot maps clause -> (a_values, b_values), in the clause's direction. Refuses before any test-window
    run if an arm has fewer than 5 pilot seeds, counts differ, or neither arm varied.
    """
    clauses = {}
    for name, (a, b) in pilot.items():
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        _check_arms(a, b)
        clauses[name] = (math.sqrt(sample_var(a)), math.sqrt(sample_var(b)), float(a.mean() - b.mean()))
    rows, chosen = [], None
    for n in CANDIDATES:
        level = look_level(alpha, n, CAP)
        power = {k: stage1_power(sa, sb, n, gap / 2, level) if gap > 0 else float("nan")
                 for k, (sa, sb, gap) in clauses.items()}
        shortfall = sum(1 - p for p in power.values()) if all(p == p for p in power.values()) else float("nan")
        rows.append({"n": n, "level": level, "power": power,
                     "joint_power_at_least": max(0.0, 1 - shortfall) if shortfall == shortfall else float("nan")})
        if chosen is None and all(p >= target for p in power.values()):  # nan >= target is False
            chosen = n
    n = chosen if chosen is not None else CAP
    final = next(r for r in rows if r["n"] == n)
    return {"n": n, "rows": rows, "clauses": clauses,
            "powered": {k: bool(p >= target) for k, p in final["power"].items()}}
