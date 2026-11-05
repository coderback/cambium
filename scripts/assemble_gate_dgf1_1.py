"""Retired: this script assembled gates/GATE-DGF1-1.md (ADR-012). It now refuses (ADR-021 clause 8).

GATE-DGF1-1 is signed and ADR-016 closed its window, so nothing may read that batch's score files again.
The full script is in the repository's history, at ``cb3ace6``.

``criterion`` stays because it records the rule the signed verdict used, and
``tests/test_gate_criteria.py`` pins it. New gates use ``scripts/gate_stats.py``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def mean_sd(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1))


def criterion(g_vals, f_vals) -> dict:
    g_mean, g_sd = mean_sd(g_vals)
    f_mean, f_sd = mean_sd(f_vals)
    n_g, n_f = len(g_vals), len(f_vals)
    se = math.sqrt(g_sd ** 2 / n_g + f_sd ** 2 / n_f)
    diff = g_mean - f_mean
    welch = stats.ttest_ind(g_vals, f_vals, equal_var=False)
    # Welch–Satterthwaite degrees of freedom -> exact two-sided 95% multiplier, for context only
    num = (g_sd ** 2 / n_g + f_sd ** 2 / n_f) ** 2
    den = (g_sd ** 2 / n_g) ** 2 / (n_g - 1) + (f_sd ** 2 / n_f) ** 2 / (n_f - 1)
    df = num / den if den > 0 else float("inf")
    t_crit = float(stats.t.ppf(0.975, df))
    return {"g_mean": g_mean, "g_sd": g_sd, "f_mean": f_mean, "f_sd": f_sd, "diff": diff,
            "se": se, "two_se": 2 * se, "resolvable": diff > 0 and diff > 2 * se,
            "welch_p": float(welch.pvalue), "df": df, "t_crit": t_crit,
            "resolvable_exact": diff > 0 and diff > t_crit * se}


def main() -> None:
    raise SystemExit("refusing: retired by ADR-021 clause 8. GATE-DGF1-1 is signed, and ADR-016 closed its window.")


if __name__ == "__main__":
    main()
