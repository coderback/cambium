"""Gate statistics v2: label-free arithmetic and Monte Carlo behind the ADR (ADR-017 item 3).

No data, no registry, no model metric: every number printed is a property of a decision rule,
computed on synthetic normal draws. Self-contained, so it does not import the prototype.
ADR-018 adopts the split-alpha (Bonferroni) rule of section G; "common c" is the rejected
Pocock-type alternative, kept for comparison.
Run from this folder: python measure_gate_stats.py > output.txt  (about 3 minutes)
"""

from __future__ import annotations

import math
import sys

import numpy as np
from scipy import integrate, optimize, stats

# The one-sided level that "diff > 2 x SE_diff" has always meant under a normal statistic.
ALPHA = float(stats.norm.sf(2.0))  # 0.0227501...
POWER_TARGET = 0.8


def _both_below(c: float, rho: float) -> float:
    """P(Z1 <= c, Z2 <= c) for a standard bivariate normal with correlation rho, by 1-D quadrature."""
    s = math.sqrt(1.0 - rho * rho)
    val, _ = integrate.quad(lambda z: stats.norm.pdf(z) * stats.norm.cdf((c - rho * z) / s),
                            -np.inf, c, epsabs=1e-13, epsrel=1e-12)
    return val


def two_look_boundary(alpha: float, n1: int, n2: int) -> float:
    """Common z-scale boundary c with P(Z1 > c or Z2 > c) = alpha, looks at n1 and n2 seeds per arm.

    Z1 and Z2 are the cumulative statistics at the two looks, so corr = sqrt(n1 / n2).
    A single look (n1 >= n2) returns the plain one-sided quantile.
    """
    if n1 >= n2:
        return float(stats.norm.isf(alpha))
    rho = math.sqrt(n1 / n2)
    return float(optimize.brentq(lambda c: 1.0 - _both_below(c, rho) - alpha, 1.0, 5.0, xtol=1e-12))


def welch_df(var_a: float, n_a: int, var_b: float, n_b: int) -> float:
    """Welch-Satterthwaite df from per-arm sample variances; a zero-variance arm drops out."""
    va, vb = var_a / n_a, var_b / n_b
    den = (va * va / (n_a - 1) if va > 0 else 0.0) + (vb * vb / (n_b - 1) if vb > 0 else 0.0)
    return (va + vb) ** 2 / den


def stage1_power(s_a: float, s_b: float, n: int, delta: float, boundary_z: float) -> float:
    """Planned power of the one-sided Welch test at n seeds per arm, true gap delta, pilot sds s_a, s_b."""
    var_a, var_b = s_a * s_a, s_b * s_b
    se = math.sqrt(var_a / n + var_b / n)
    df = welch_df(var_a, n, var_b, n)
    t_crit = float(stats.t.isf(stats.norm.sf(boundary_z), df))
    return float(stats.nct.sf(t_crit, df, delta / se))


def size_stage1(pilot: dict, candidates=(8, 12, 16, 20), cap: int = 20,
                alpha: float = ALPHA, target: float = POWER_TARGET) -> dict:
    """Smallest candidate n whose stage-1 power at delta = val_gap / 2 reaches target on every metric.

    pilot maps metric -> (s_a, s_b, val_gap) from the validation pilot. No candidate qualifies, or a
    val_gap <= 0, gives the cap.
    """
    rows = []
    chosen = None
    for n in candidates:
        c = two_look_boundary(alpha, n, cap)
        powers = {}
        for metric, (s_a, s_b, gap) in pilot.items():
            powers[metric] = stage1_power(s_a, s_b, n, gap / 2, c) if gap > 0 else float("nan")
        rows.append({"n": n, "boundary_z": c, "power": powers})
        if chosen is None and all(p >= target for p in powers.values()):  # nan >= target is False
            chosen = n
    return {"n": chosen if chosen is not None else cap, "rows": rows}


RNG = np.random.default_rng(20261007)
REPS, CHUNK = 400_000, 100_000


def welch_stats(g, f):
    n = g.shape[1]
    vg, vf = g.var(axis=1, ddof=1) / n, f.var(axis=1, ddof=1) / n
    se = np.sqrt(vg + vf)
    den = np.where(vg > 0, vg * vg / (n - 1), 0.0) + np.where(vf > 0, vf * vf / (n - 1), 0.0)
    df = (vg + vf) ** 2 / den
    return g.mean(axis=1) - f.mean(axis=1), se, df


def simulate(n1, n2, s_g, s_f, delta, rule):
    """Fraction of REPS resolving at stage 1, and at either look (stage 2 runs iff stage 1 misses)."""
    hit1 = hit_any = 0
    for _ in range(REPS // CHUNK):
        g = RNG.normal(delta, s_g, size=(CHUNK, n2))
        f = RNG.normal(0.0, s_f, size=(CHUNK, n2)) if s_f > 0 else np.zeros((CHUNK, n2))
        r = []
        for n in ((n1, n2) if n1 < n2 else (n1,)):
            d, se, df = welch_stats(g[:, :n], f[:, :n])
            r.append(d > rule(df, n1 < n2) * se)
        hit1 += int(r[0].sum())
        hit_any += int(np.logical_or.reduce(r).sum())
    return hit1 / REPS, hit_any / REPS


def v2_rule(n1, n2):
    level = stats.norm.sf(two_look_boundary(ALPHA, n1, n2))
    return lambda df, _two: stats.t.isf(level, df)


def old_rule(df, _two):
    return 2.0


def main():
    out = sys.stdout
    mc_se = math.sqrt(ALPHA * (1 - ALPHA) / REPS)
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, scipy {__import__('scipy').__version__}", file=out)
    print(f"alpha (one-sided level of 'diff > 2 SE' under a normal statistic) = {ALPHA:.5f}", file=out)
    print(f"Monte Carlo: {REPS:,} draws per cell, seed 20261007; SE of an alpha estimate near "
          f"{ALPHA:.4f} is {mc_se:.5f}\n", file=out)

    print("A. Known-variance (z) arithmetic, exact by quadrature", file=out)
    rho = math.sqrt(8 / 20)
    print(f"  single look, z > 2:                         alpha = {ALPHA:.4f}", file=out)
    print(f"  two looks (8 then 20 seeds), z > 2 at both: alpha = {1 - _both_below(2.0, rho):.4f}",
          file=out)
    for n1 in (5, 8, 12, 16):
        c = two_look_boundary(ALPHA, n1, 20)
        print(f"  common boundary, looks at {n1:2d} and 20 seeds (rho = {math.sqrt(n1 / 20):.3f}): "
              f"c = {c:.4f}; per-look level {stats.norm.sf(c):.5f}", file=out)
    print(f"  Bonferroni alternative (per-look level alpha / 2): c = {stats.norm.isf(ALPHA / 2):.4f}", file=out)
    print(f"  check against Pocock's K=2 constant (equal spacing, one-sided 0.025): "
          f"c = {two_look_boundary(0.025, 10, 20):.4f} (published 2.178)\n", file=out)

    print("B. Per-look one-sided Welch critical values t_crit(df)", file=out)
    c8 = two_look_boundary(ALPHA, 8, 20)
    print(f"  {'df':>4} {'single look (c = 2)':>22} {'two looks 8/20 (c = %.4f)' % c8:>30}", file=out)
    for df in (2, 4, 7, 11, 15, 19, 38, 1e9):
        print(f"  {df:>4.0f} {stats.t.isf(ALPHA, df):>22.3f} {stats.t.isf(stats.norm.sf(c8), df):>30.3f}",
              file=out)
    print("", file=out)

    print("C. False-pass rate under no true gap (normal seeds); nominal alpha = %.4f" % ALPHA, file=out)
    print("  s_f / s_g = 0 is a seed-deterministic comparator (Welch df = n - 1).", file=out)
    print(f"  {'design':>14} {'s_f/s_g':>8} {'old rule (2 SE at each look)':>30} {'Welch, common c':>22}",
          file=out)
    for n1, n2 in ((3, 3), (5, 5), (8, 20), (5, 20), (12, 20), (16, 20), (20, 20)):
        for ratio in (0.0, 0.5, 1.0, 2.0):
            _, old_any = simulate(n1, n2, 1.0, ratio, 0.0, old_rule)
            v2 = "refused (floor)" if n1 < 5 else f"{simulate(n1, n2, 1.0, ratio, 0.0, v2_rule(n1, n2))[1]:.4f}"
            label = f"{n1}" if n1 == n2 else f"{n1} then {n2}"
            print(f"  {label:>14} {ratio:>8.1f} {old_any:>30.4f} {v2:>22}", file=out)
    print("", file=out)

    print("D. Power at the planning effect where the planned stage-1 power is exactly 0.80 (common c)", file=out)
    print(f"  {'design':>10} {'s_f/s_g':>8} {'delta/s_g':>10} {'planned':>8} {'simulated stage 1':>18} "
          f"{'simulated both stages':>22}", file=out)
    for n1 in (8, 12, 16):
        c = two_look_boundary(ALPHA, n1, 20)
        for ratio in (0.0, 1.0):
            delta = optimize.brentq(lambda d: stage1_power(1.0, ratio, n1, d, c) - 0.8, 1e-6, 50.0)
            p1, p_any = simulate(n1, 20, 1.0, ratio, delta, v2_rule(n1, 20))
            print(f"  {n1:>4} / 20 {ratio:>8.1f} {delta:>10.4f} {0.8:>8.2f} {p1:>18.4f} {p_any:>22.4f}",
                  file=out)
    print("  The old rule at its own sizing boundary (2 SE_diff = delta, known variance): stage-1 power = "
          f"{stats.norm.sf(2.0 - 2.0):.2f}\n", file=out)

    print("E. Satisfiability: stage-1 n chosen from {8, 12, 16, 20}, comparator seed-deterministic", file=out)
    print("  (old = ADR-012 clause 5: 2 SE_diff(n) <= val_gap / 2; common c = planned power >= 0.80 at "
          "delta = val_gap / 2)", file=out)
    gaps = (0.010, 0.020, 0.050, 0.100)
    print(f"  {'s_g':>6} | " + " | ".join(f"gap {g:.3f} old/com." for g in gaps), file=out)
    for s_g in (0.002, 0.005, 0.010, 0.020, 0.050):
        cells = []
        for gap in gaps:
            old = next((n for n in (8, 12, 16, 20) if 2 * s_g / math.sqrt(n) <= gap / 2), 20)
            v2 = size_stage1({"m": (s_g, 0.0, gap)})["n"]
            power20 = stage1_power(s_g, 0.0, 20, gap / 2, 2.0)
            flag = "" if power20 >= 0.8 or v2 < 20 else "*"
            cells.append(f"{old:>8} / {v2:<2}{flag:1}")
        print(f"  {s_g:>6.3f} | " + " | ".join(f"{c:>17}" for c in cells), file=out)
    print("  * = n = 20 and planned power at 20 is still below 0.80 (the clause is under-powered)", file=out)

    print("", file=out)
    print("F. Planning on an upper confidence bound for the pilot sd instead of the pilot sd", file=out)
    factor = math.sqrt(4 / stats.chi2.ppf(0.20, 4))
    print(f"  80% upper bound from 5 pilot seeds (df 4): s x {factor:.3f}; seeds needed scale by about "
          f"{factor ** 2:.2f}x", file=out)
    for s_g, gap in ((0.005, 0.020), (0.010, 0.050), (0.002, 0.010)):
        a = size_stage1({"m": (s_g, 0.0, gap)})["n"]
        b = size_stage1({"m": (s_g * factor, 0.0, gap)})["n"]
        print(f"  s {s_g:.3f}, gap {gap:.3f}: stage-1 n {a} with the pilot sd, {b} with the bound", file=out)

    print("", file=out)
    print("G. Bonferroni across the looks (per-look level alpha / 2, z = %.4f), beside the common boundary"
          % stats.norm.isf(ALPHA / 2), file=out)

    def bonf_rule(n1, n2):
        level = ALPHA / 2 if n1 < n2 else ALPHA
        return lambda df, _two: stats.t.isf(level, df)

    print("  False-pass rate under no true gap (fresh draws; compare section C's common-c column):", file=out)
    print(f"  {'design':>14} {'s_f/s_g':>8} {'Bonferroni':>11}", file=out)
    for n1, n2 in ((5, 20), (8, 20), (12, 20), (16, 20)):
        for ratio in (0.0, 0.5, 1.0, 2.0):
            bf = simulate(n1, n2, 1.0, ratio, 0.0, bonf_rule(n1, n2))[1]
            print(f"  {f'{n1} then {n2}':>14} {ratio:>8.1f} {bf:>11.4f}", file=out)

    print("  Power at section D's planning effects (comparator fixed), common boundary vs Bonferroni:", file=out)
    print(f"  {'design':>10} {'delta/s_g':>10} {'planned common':>15} {'planned Bonf.':>14} "
          f"{'sim. stage 1 Bonf.':>19} {'sim. both Bonf.':>16}", file=out)
    for n1 in (8, 12, 16):
        c = two_look_boundary(ALPHA, n1, 20)
        delta = optimize.brentq(lambda d: stage1_power(1.0, 0.0, n1, d, c) - 0.8, 1e-6, 50.0)
        planned_b = stage1_power(1.0, 0.0, n1, delta, float(stats.norm.isf(ALPHA / 2)))
        p1, p_any = simulate(n1, 20, 1.0, 0.0, delta, bonf_rule(n1, 20))
        print(f"  {n1:>4} / 20 {delta:>10.4f} {0.8:>15.2f} {planned_b:>14.4f} {p1:>19.4f} {p_any:>16.4f}",
              file=out)

    def size_bonf(s_g, gap):
        for n in (8, 12, 16, 20):
            c = float(stats.norm.isf(ALPHA / 2)) if n < 20 else 2.0
            if gap > 0 and stage1_power(s_g, 0.0, n, gap / 2, c) >= 0.8:
                return n
        return 20

    print("  Stage-1 n (section E's grid), common boundary / Bonferroni:", file=out)
    print(f"  {'s_g':>6} | " + " | ".join(f"gap {g:.3f}" for g in gaps), file=out)
    for s_g in (0.002, 0.005, 0.010, 0.020, 0.050):
        cells = [f"{size_stage1({'m': (s_g, 0.0, gap)})['n']:>2} / {size_bonf(s_g, gap):<2}" for gap in gaps]
        print(f"  {s_g:>6.3f} | " + " | ".join(f"{c:>9}" for c in cells), file=out)

    print("", file=out)
    print("H. Arms that share seed ids: per-seed correlation r between the two arms' scores, no true gap,",
          file=out)
    print("   8 then 20 seeds, equal spread (Welch treats the arms as independent)", file=out)
    print(f"  {'r':>5} {'common boundary':>16} {'Bonferroni':>11}", file=out)
    for r in (0.0, 0.5, 0.9):
        rates = []
        for rule in (v2_rule(8, 20), bonf_rule(8, 20)):
            hit = 0
            for _ in range(REPS // CHUNK):
                zg = RNG.standard_normal((CHUNK, 20))
                f = r * zg + math.sqrt(1 - r * r) * RNG.standard_normal((CHUNK, 20))
                looks = []
                for n in (8, 20):
                    d, se, df = welch_stats(zg[:, :n], f[:, :n])
                    looks.append(d > rule(df, True) * se)
                hit += int(np.logical_or(*looks).sum())
            rates.append(hit / REPS)
        print(f"  {r:>5.1f} {rates[0]:>16.4f} {rates[1]:>11.4f}", file=out)


if __name__ == "__main__":
    main()
