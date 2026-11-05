"""What the two clauses can say at the precision bar, by arithmetic. No data and no model metric.

At 20 seeds per arm and the two-look level, with equal spreads chosen so the margin clause has exactly
80% power at a true gain of zero: how often does each clause resolve, as the true gain moves from 0 to m?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from gate_stats import ALPHA, welch_power  # noqa: E402

m, level, n = 1.0, ALPHA / 2, 20
lo, hi = 1e-4, 10.0
for _ in range(100):
    s = (lo + hi) / 2
    lo, hi = (s, hi) if welch_power(s, s, n, m, level) > 0.8 else (lo, s)
se = (2 * s * s / n) ** 0.5
print(f"spread at the bar: {s:.4f} x m, so m is {m / se:.2f} standard errors of the difference")
print("true gain | gain clause resolves | margin clause resolves")
for g in (0.0, 0.25, 0.5, 0.75, 1.0):
    gain = welch_power(s, s, n, g, level) if g > 0 else level
    below = welch_power(s, s, n, m - g, level) if g < 1 else level
    print(f"   {g:.2f} m  |        {gain:.3f}         |        {below:.3f}")
