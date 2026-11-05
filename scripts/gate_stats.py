"""Gate statistics v2: the shared functions every gate assembler imports (ADR-018 clause 7).

Scope is ADR-018's: the per-look level (clause 3), the one-sided Welch criterion and its refusals
(clause 2), planned power, the stage-1 seed count and its bounds (clause 4), and the alpha and power
table (clause 6).

Absence and equivalence claims (ADR-018 clause 1, ADR-019 clause 7) are clause 2 with the arms swapped
and the comparator shifted by the margin, so the caller builds them from ``judge``. Clause 4 does not
size them, and ``plan`` does not take them. The Gate-1 ``criterion`` in ``assemble_gate_dgf1_1.py``
stays as it is and is never used for a new gate.

Reads no data and computes no model metric: its inputs are values already in registry rows.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

# The one-sided level of "diff > 2 x SE_diff" under a normal statistic (clause 1).
ALPHA = float(stats.norm.sf(2.0))
SEED_FLOOR = 5
POWER_TARGET = 0.8
CANDIDATES = (8, 12, 16, 20)
CAP = 20
# A pilot never touches a test window (clause 4). A later gate's test window is added here.
TEST_WINDOWS = frozenset({"482-821"})
# Designs the measurement simulated (`notebooks/measurements/2026-10-07-gate-stats-v2/output.txt`):
# section G's two looks, and section C's single looks.
SIMULATED_DESIGNS = frozenset({(5, 20), (8, 20), (12, 20), (16, 20), (5, 5), (20, 20)})
ALPHA_BASIS = ("per-look levels sum to α (ADR-018 clause 3); measured 0.0165–0.0215 on §G's two-look "
               "designs and 0.0201–0.0230 on §C's single looks; Welch is approximate where both arms vary")


class Refused(ValueError):
    """Clause 2's three refusals. At assembly the clause is reported as refused and cannot pass."""


# ------------------------------------------------------------------ the per-look level


def look_level(alpha: float, n1: int, n2: int) -> float:
    """Per-look one-sided level: alpha / 2 when a second look is possible (n1 < n2), else alpha."""
    if n1 > n2:
        raise ValueError(f"stage 1 ({n1} seeds) cannot exceed stage 2 ({n2} seeds)")
    return alpha / 2 if n1 < n2 else alpha


# ------------------------------------------------------------------ the criterion


def _values(values: Sequence[float]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(v)):
        raise ValueError("an arm holds a non-finite metric value")
    return v


def varies(values: Sequence[float]) -> bool:
    """An arm varies over seeds unless every value is bit-identical."""
    v = _values(values)
    return not bool(np.all(v == v[0]))


def sample_var(values: Sequence[float]) -> float:
    """ddof=1 variance, exactly 0.0 for an arm that does not vary (rounding cannot leave a residue)."""
    v = _values(values)
    return float(v.var(ddof=1)) if varies(v) else 0.0


def welch_df(var_a: float, n_a: int, var_b: float, n_b: int) -> float:
    """Welch-Satterthwaite df, not rounded; a zero-variance arm drops out of both sums."""
    va, vb = var_a / n_a, var_b / n_b
    den = (va * va / (n_a - 1) if va > 0 else 0.0) + (vb * vb / (n_b - 1) if vb > 0 else 0.0)
    return (va + vb) ** 2 / den


def _check_arms(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) < SEED_FLOOR or len(b) < SEED_FLOOR:
        raise Refused(f"seed floor: every arm needs at least {SEED_FLOOR} seeds")
    if len(a) != len(b):
        raise Refused("unequal seed counts: every arm in a comparison runs the same seeds")
    if not (varies(a) or varies(b)):
        raise Refused("neither arm varies over seeds; a seed test cannot judge this comparison")


def resolvability(a: Sequence[float], b: Sequence[float], level: float) -> dict:
    """One-sided Welch test of mean(a) > mean(b) at a per-look level. Raises ``Refused``."""
    a, b = _values(a), _values(b)
    _check_arms(a, b)
    var_a, var_b = sample_var(a), sample_var(b)
    se = math.sqrt(var_a / len(a) + var_b / len(b))
    df = welch_df(var_a, len(a), var_b, len(b))
    t_crit = float(stats.t.isf(level, df))
    diff = float(a.mean() - b.mean())
    return {"diff": diff, "se": se, "df": df, "level": level, "t_crit": t_crit,
            "p_one_sided": float(stats.t.sf(diff / se, df)), "resolvable": diff > t_crit * se}


def judge(a: Sequence[float], b: Sequence[float], *, n1: int, n2: int, stage: int,
          pilot_s: tuple[float, float], alpha: float = ALPHA) -> dict:
    """One clause at one look of a design, with everything the gate file prints (clause 2).

    Stage 2 is cumulative, so its arms hold all ``n2`` seeds. A refusal comes back as a result that
    cannot pass. An arm count that fits neither the refusals nor the design is a broken batch, and it
    raises: no verdict is written over it.
    """
    if stage not in (1, 2) or (stage == 2 and n1 == n2):
        raise ValueError(f"design {n1} -> {n2} has no stage {stage}")
    n = n1 if stage == 1 else n2
    level = look_level(alpha, n1, n2)
    planned_se = math.sqrt((pilot_s[0] ** 2 + pilot_s[1] ** 2) / n)
    try:
        result = resolvability(a, b, level)
    except Refused as why:
        return {"stage": stage, "n": n, "level": level, "planned_se": planned_se, "refused": str(why),
                "resolvable": False, "verdict": f"refused: {why}"}
    if len(a) != n:
        raise ValueError(f"stage {stage} of design {n1} -> {n2} judges {n} seeds per arm, not {len(a)}")
    verdict = "resolvable" if result["resolvable"] else f"not resolvable at this look's level {level:.6f}"
    return {**result, "stage": stage, "n": n, "planned_se": planned_se, "refused": None, "verdict": verdict}


def stage2_triggered(results: Iterable[Mapping]) -> bool:
    """Stage 2 runs iff any gated clause is unresolvable or refused at stage 1 (clause 3)."""
    return any(not r["resolvable"] for r in results)


# ------------------------------------------------------------------ power and stage-1 n


def welch_power(s_a: float, s_b: float, n: int, delta: float, level: float) -> float:
    """Power of the one-sided Welch test at n seeds per arm, true gap delta, sds s_a and s_b."""
    var_a, var_b = s_a * s_a, s_b * s_b
    se = math.sqrt(var_a / n + var_b / n)
    df = welch_df(var_a, n, var_b, n)
    return float(stats.nct.sf(stats.t.isf(level, df), df, delta / se))


def powered_failure(result: Mapping, a: Sequence[float], b: Sequence[float], *, planned_power: float,
                    delta: float, target: float = POWER_TARGET) -> bool:
    """A failed clause is a powered failure only if planned and realised power both reach the target.

    The realised power uses the realised sds at the n and level of the look that decided the clause,
    at the pre-registered delta. The realised gap is never used. A refused or resolved clause is never
    a powered failure (clause 4).
    """
    if result["refused"] or result["resolvable"] or not planned_power >= target:
        return False
    s_a, s_b = math.sqrt(sample_var(a)), math.sqrt(sample_var(b))
    return welch_power(s_a, s_b, result["n"], delta, result["level"]) >= target


@dataclass(frozen=True)
class PilotClause:
    """One gated clause's validation pilot: ``a`` is expected above ``b``. Values are keyed by seed."""

    direction: str
    a: Mapping[int, float]
    b: Mapping[int, float]


def seed_correlation(a: Mapping[int, float], b: Mapping[int, float]) -> float | None:
    """The per-seed correlation on shared seed ids; None where it is undefined (clause 2)."""
    shared = sorted(a.keys() & b.keys())
    if len(shared) < 2:
        return None
    x, y = [a[s] for s in shared], [b[s] for s in shared]
    if not (varies(x) and varies(y)):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def plan(pilot: Mapping[str, PilotClause], *, alpha: float = ALPHA, target: float = POWER_TARGET,
         candidates: Sequence[int] = CANDIDATES, cap: int = CAP, factor: float = 0.5) -> dict:
    """Clause 4's stage-1 seed count, and everything clause 6's table prints, from the pilot's values.

    Refuses, before any test-window run, an arm below the floor, unequal counts and an arm pair that
    does not vary. ``candidates``, ``cap`` and ``factor`` depart from the defaults only under a
    pre-registration's stated reason (clauses 4 and 5).
    """
    if list(candidates) != sorted(set(candidates)) or candidates[0] < SEED_FLOOR or candidates[-1] != cap:
        raise ValueError(f"candidates must rise from at least {SEED_FLOOR} to the cap {cap}")
    clauses = {}
    for name, c in pilot.items():
        a, b = _values(list(c.a.values())), _values(list(c.b.values()))
        _check_arms(a, b)
        gap = float(a.mean() - b.mean())
        clauses[name] = {"direction": c.direction, "s_a": math.sqrt(sample_var(a)), "s_b": math.sqrt(sample_var(b)),
                         "a_varied": varies(a), "b_varied": varies(b), "gap": gap, "delta": factor * gap,
                         "correlation": seed_correlation(c.a, c.b)}
    rows, chosen = [], None
    for n in candidates:
        level = look_level(alpha, n, cap)
        power = {k: welch_power(c["s_a"], c["s_b"], n, c["delta"], level) if c["gap"] > 0 else float("nan")
                 for k, c in clauses.items()}
        shortfall = sum(1 - p for p in power.values())   # nan where any gap is not positive
        rows.append({"n": n, "level": level, "power": power,
                     "joint_power_at_least": max(0.0, 1 - shortfall) if shortfall == shortfall else float("nan"),
                     "stage2_at_most": min(1.0, shortfall) if shortfall == shortfall else float("nan")})
        if chosen is None and all(p >= target for p in power.values()):  # nan >= target is False
            chosen = n
    n = chosen if chosen is not None else cap
    final = next(r for r in rows if r["n"] == n)
    return {"n": n, "cap": cap, "alpha": alpha, "target": target, "rows": rows, "clauses": clauses,
            "powered": {k: bool(p >= target) for k, p in final["power"].items()},
            "simulated": (n, cap) in SIMULATED_DESIGNS}


# ------------------------------------------------------------------ the table


def _num(x: float | None, places: int = 4) -> str:
    return "n/a" if x is None or x != x else f"{x:.{places}f}"


def alpha_power_table(p: Mapping) -> str:
    """Clause 6's table, as text. It is produced from ``plan``'s output and never typed by hand."""
    lines = [f"Stage-1 seeds: {p['n']}", f"Design: {p['n']} -> {p['cap']}, α = {p['alpha']:.5f}",
             "Each look's ℓ, by stage-1 candidate: "
             + ", ".join(f"{r['n']}: {r['level']:.6f}" for r in p["rows"]), ""]
    for name, c in p["clauses"].items():
        lines += [f"Clause {name}: {c['direction']}",
                  f"  pilot s: {c['s_a']:.6f} ({'varied' if c['a_varied'] else 'did not vary'}) vs "
                  f"{c['s_b']:.6f} ({'varied' if c['b_varied'] else 'did not vary'})",
                  f"  Δ_val: {c['gap']:.6f}; δ: {c['delta']:.6f}; per-seed correlation: {_num(c['correlation'])}",
                  "  planned stage-1 power: "
                  + ", ".join(f"{r['n']}: {_num(r['power'][name])}" for r in p["rows"]),
                  f"  powered: {'yes' if p['powered'][name] else 'no'}"]
    final = next(r for r in p["rows"] if r["n"] == p["n"])
    lines += ["", f"Joint stage-1 power at least: {_num(final['joint_power_at_least'])}",
              f"Stage 2 triggered at most: {_num(final['stage2_at_most'])}",
              f"Basis for α: {ALPHA_BASIS}" + ("" if p["simulated"] else "; not simulated")]
    return "\n".join(lines)


# ------------------------------------------------------------------ from registry rows


def pilot_arm(rows: Iterable[Mapping], *, model: str, arm: str, metric: str, window: str) -> dict[int, float]:
    """One arm's pilot values by seed, from registry rows: clean, deterministic, one commit (clause 4)."""
    if window in TEST_WINDOWS:
        raise ValueError(f"a pilot never touches a test window, and {window} is one")
    values: dict[int, float] = {}
    commits = set()
    for r in rows:
        if r["model"] != model:
            continue
        m = json.loads(r["metrics_json"])
        if m.get("arm") != arm or m.get("window") != window:
            continue
        if r["git_dirty"] != "false" or m.get("deterministic") is not True or "ERRORED" in r["notes"]:
            raise ValueError(f"run {r['run_id']} is not a clean, deterministic pilot row")
        if int(r["seed"]) in values:
            raise ValueError(f"arm {arm} has seed {r['seed']} twice; a pilot has no retries")
        values[int(r["seed"])] = float(m[metric])
        commits.add(r["git_commit"])
    if len(commits) > 1:
        raise ValueError(f"arm {arm}'s pilot spans commits {sorted(commits)}")
    return values
