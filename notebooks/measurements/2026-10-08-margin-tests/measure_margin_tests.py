"""Margin tests under ADR-018's two-look design: does ADR-019 need ADR-018's seed rules changed?

Label-free: synthetic normal seed scores, no data, no registry, no model metric (ADR-017 item 3).
Both arms vary (sd 1 each, equal seeds). Each look runs ADR-018's one-sided Welch test at the
per-look level (alpha / 2 with two looks, alpha with one):
  superiority     gain > 0          (the gate's own clause)
  non-superiority gain < m          (ADR-019's "no" test: the comparator shifted up by m beats the arm)
  lower bound     gain > -m         (the second half of a two-one-sided-tests equivalence claim)
Stage 2 (20 seeds, cumulative) is final for every clause, as ADR-018 clause 3 says.

  A  "ride along": only superiority triggers stage 2 (ADR-018 unchanged).
  B  "gated": an unresolved margin test also triggers stage 2 (what draft 1 implied).
Run from this folder: python measure_margin_tests.py > output.txt  (a few minutes)
"""

from __future__ import annotations

import math
import sys

import numpy as np
from scipy import optimize, stats

ALPHA = float(stats.norm.sf(2.0))
CAP = 20
RNG = np.random.default_rng(20261008)
REPS, CHUNK = 400_000, 100_000


def level(n1: int) -> float:
    return ALPHA / 2 if n1 < CAP else ALPHA


def welch(a, b):
    n = a.shape[1]
    va, vb = a.var(axis=1, ddof=1) / n, b.var(axis=1, ddof=1) / n
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va * va / (n - 1) + vb * vb / (n - 1))
    return a.mean(axis=1) - b.mean(axis=1), se, df


def planned_power_at(n: int, delta: float, lvl: float) -> float:
    var = 1.0 / n + 1.0 / n
    df = (var ** 2) / (2 * (1.0 / n) ** 2 / (n - 1))
    return float(stats.nct.sf(stats.t.isf(lvl, df), df, delta / math.sqrt(var)))


def run(n1: int, gain: float, m: float) -> dict:
    counts = {k: 0 for k in ("A_sup", "A_no", "A_stage2", "B_sup", "B_no", "B_stage2", "A_no_tost")}
    lvl = level(n1)
    for _ in range(REPS // CHUNK):
        a = RNG.normal(gain, 1.0, size=(CHUNK, CAP))
        b = RNG.normal(0.0, 1.0, size=(CHUNK, CAP))
        looks = []
        for n in ((n1, CAP) if n1 < CAP else (CAP,)):
            d, se, df = welch(a[:, :n], b[:, :n])
            t = stats.t.isf(lvl, df) * se
            looks.append({"sup": d > t, "mar": d < m - t, "low": d > -m + t})
        first, last = looks[0], looks[-1]
        two = len(looks) == 2
        # A: stage 2 runs iff superiority is unresolved at stage 1.
        a2 = ~first["sup"] if two else np.zeros(CHUNK, bool)
        a_sup = np.where(a2, last["sup"], first["sup"])
        a_mar = np.where(a2, last["mar"], first["mar"])
        a_low = np.where(a2, last["low"], first["low"])
        # B: stage 2 runs iff superiority or the margin test is unresolved at stage 1.
        b2 = (~first["sup"] | ~first["mar"]) if two else np.zeros(CHUNK, bool)
        b_sup = np.where(b2, last["sup"], first["sup"])
        b_mar = np.where(b2, last["mar"], first["mar"])
        counts["A_sup"] += int(a_sup.sum())
        counts["A_no"] += int((~a_sup & a_mar).sum())
        counts["A_no_tost"] += int((~a_sup & a_mar & a_low).sum())
        counts["A_stage2"] += int(a2.sum())
        counts["B_sup"] += int(b_sup.sum())
        counts["B_no"] += int((~b_sup & b_mar).sum())
        counts["B_stage2"] += int(b2.sum())
    return {k: v / REPS for k, v in counts.items()}


def main():
    out = sys.stdout
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, scipy {__import__('scipy').__version__}",
          file=out)
    print(f"alpha {ALPHA:.5f}; per-look level {ALPHA / 2:.6f} with two looks; {REPS:,} draws per cell, "
          f"seed 20261008; Monte Carlo SE at 0.02 is {math.sqrt(0.02 * 0.98 / REPS):.5f}", file=out)
    m = optimize.brentq(lambda d: planned_power_at(CAP, d, ALPHA / 2) - 0.8, 1e-6, 10.0)
    print(f"margin m = {m:.4f} arm sds: the margin test's planned power at a true gain of 0 is 0.80 "
          f"at 20 seeds and level alpha / 2\n", file=out)

    print("1. Outcomes by true gain (in units of m). sup = gain resolved above 0 (the gate passes);", file=out)
    print("   no = superiority not resolved and gain resolved below m; stage2 = share of runs that took", file=out)
    print("   stage 2; no(TOST) = 'no' if the claim also had to resolve gain > -m.", file=out)
    hdr = (f"  {'n1':>3} {'gain/m':>7} | {'A sup':>7} {'A no':>7} {'A stage2':>9} {'A no(TOST)':>11} | "
           f"{'B sup':>7} {'B no':>7} {'B stage2':>9}")
    print(hdr, file=out)
    for n1 in (8, 12, 16, 20):
        for g in (-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0):
            r = run(n1, g * m, m)
            print(f"  {n1:>3} {g:>7.1f} | {r['A_sup']:>7.4f} {r['A_no']:>7.4f} {r['A_stage2']:>9.4f} "
                  f"{r['A_no_tost']:>11.4f} | {r['B_sup']:>7.4f} {r['B_no']:>7.4f} {r['B_stage2']:>9.4f}",
                  file=out)
        print("", file=out)

    print("2. ADR-019 draft 2's four categories under A (judged at the last look that runs), and the", file=out)
    print("   alternative 'yes needs the gain shown above m' (d > m + t SE at that look).", file=out)
    print(f"  {'n1':>3} {'gain/m':>7} | {'above':>7} {'small':>7} {'below':>7} {'unres.':>7} | "
          f"{'shown > m':>10}", file=out)
    for n1 in (8, 20):
        for g in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            c = categories(n1, g * m, m)
            print(f"  {n1:>3} {g:>7.2f} | {c['above']:>7.4f} {c['small']:>7.4f} {c['below']:>7.4f} "
                  f"{c['unresolved']:>7.4f} | {c['above_m']:>10.4f}", file=out)
        print("", file=out)

    section3(out)


def gate_category(n1: int, gain: float, m: float, size: int) -> np.ndarray:
    """One gate, one metric, procedure A: 0 = above, 1 = below the bar (small or below), 2 = unresolved."""
    lvl = level(n1)
    a = RNG.normal(gain, 1.0, size=(size, CAP))
    b = RNG.normal(0.0, 1.0, size=(size, CAP))
    looks = []
    for n in ((n1, CAP) if n1 < CAP else (CAP,)):
        d, se, df = welch(a[:, :n], b[:, :n])
        t = stats.t.isf(lvl, df) * se
        looks.append({"sup": d > t, "mar": d < m - t})
    first, last = looks[0], looks[-1]
    stage2 = ~first["sup"] if len(looks) == 2 else np.zeros(size, bool)
    sup = np.where(stage2, last["sup"], first["sup"])
    mar = np.where(stage2, last["mar"], first["mar"])
    return np.where(sup & ~mar, 0, np.where(mar, 1, 2))


def section3(out) -> None:
    print("3. When would GDE's start need a written justification? Gates 1 and 3 run independently,", file=out)
    print("   stage 1 at 8 seeds, cap 20. 'Strong' design: the 'no' test has 0.80 planned power at a true", file=out)
    print("   gain of 0; 'weak': 0.50. Truths are (Gate-1 gain, Gate-3 gain) in units of each design's m.", file=out)
    print("   Columns: chance of each answer; then the chance a justification is required under", file=out)
    print("   rule NO (only after 'no'), A' (also 'not answered' where 'no' was still possible, i.e.", file=out)
    print("   Gate 1 not above), A (also any 'not answered'), B (any answer but 'yes'). A', A and B add to", file=out)
    print("   'no' only for the weak design. A'' (added after ADR-019's round 3): 'not answered' only where", file=out)
    print("   neither gate is above, the exact condition for a hidden 'no'.", file=out)
    print(f"  {'design':>6} {'truth':>11} | {'yes':>6} {'notsup':>6} {'2-part':>6} {'no':>6} {'NA,G1up':>7} "
          f"{'NA,other':>8} | {'NO':>6} {'A_':>6} {'A':>6} {'B':>6} {'A__':>6}", file=out)
    for name, target in (("strong", 0.80), ("weak", 0.50)):
        m = optimize.brentq(lambda d: planned_power_at(CAP, d, ALPHA / 2) - target, 1e-6, 10.0)
        for g1, g3 in ((0.0, 0.0), (0.0, 2.0), (2.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0)):
            g1c = gate_category(8, g1 * m, m, REPS)
            g3c = gate_category(8, g3 * m, m, REPS)
            yes = (g1c == 0) & (g3c == 0)
            notsup = (g1c == 0) & (g3c == 1)
            two = (g1c == 1) & (g3c == 0)
            no = (g1c == 1) & (g3c == 1)
            na = ~(yes | notsup | two | no)
            na_up = na & (g1c == 0)
            na_other = na & (g1c != 0)
            weak = name == "weak"
            rules = {"NO": no, "A_": no | (weak & na_other), "A": no | (weak & na), "B": no | (weak & ~yes),
                     "A__": no | (weak & na & (g1c != 0) & (g3c != 0))}
            row = [x.mean() for x in (yes, notsup, two, no, na_up, na_other)]
            print(f"  {name:>6} {f'({g1:g}, {g3:g})':>11} | " + " ".join(f"{v:>6.3f}" for v in row[:4]) +
                  f" {row[4]:>7.3f} {row[5]:>8.3f} | " + " ".join(f"{rules[k].mean():>6.3f}" for k in rules),
                  file=out)
        print("", file=out)


def categories(n1: int, gain: float, m: float) -> dict:
    counts = {k: 0 for k in ("above", "small", "below", "unresolved", "above_m")}
    lvl = level(n1)
    for _ in range(REPS // CHUNK):
        a = RNG.normal(gain, 1.0, size=(CHUNK, CAP))
        b = RNG.normal(0.0, 1.0, size=(CHUNK, CAP))
        looks = []
        for n in ((n1, CAP) if n1 < CAP else (CAP,)):
            d, se, df = welch(a[:, :n], b[:, :n])
            t = stats.t.isf(lvl, df) * se
            looks.append({"sup": d > t, "mar": d < m - t, "above_m": d > m + t})
        first, last = looks[0], looks[-1]
        stage2 = ~first["sup"] if len(looks) == 2 else np.zeros(CHUNK, bool)
        sup = np.where(stage2, last["sup"], first["sup"])
        mar = np.where(stage2, last["mar"], first["mar"])
        above_m = np.where(stage2, last["above_m"], first["above_m"])
        counts["above"] += int((sup & ~mar).sum())
        counts["small"] += int((sup & mar).sum())
        counts["below"] += int((~sup & mar).sum())
        counts["unresolved"] += int((~sup & ~mar).sum())
        counts["above_m"] += int(above_m.sum())
    return {k: v / REPS for k, v in counts.items()}


if __name__ == "__main__":
    main()
