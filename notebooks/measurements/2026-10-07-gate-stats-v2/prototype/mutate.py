"""Mutation check for the gate statistics v2 prototype (draft 2).

Each mutant reproduces a defect; the tests must fail on every one. Each mutant is written to a fresh
temporary directory beside a copy of the tests, never in place. Run: python mutate.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
SRC = (HERE / "gate_stats_proto.py").read_text(encoding="utf-8")

MUTANTS = {
    "M1 population variance (ddof=0)": ("float(v.var(ddof=1)) if varies(v)", "float(v.var(ddof=0)) if varies(v)"),
    "M2 z critical value, not Welch t": ("t_crit = float(stats.t.isf(level, df))", "t_crit = float(stats.norm.isf(level))"),
    "M3 pooled df, not Welch df": ("df = welch_df(var_a, len(a), var_b, len(b))", "df = len(a) + len(b) - 2"),
    "M4 two-sided level inside the test": ("t_crit = float(stats.t.isf(level, df))", "t_crit = float(stats.t.isf(level / 2, df))"),
    "M5 no split across looks": ("return alpha / 2 if n1 < n2 else alpha", "return alpha"),
    "M6 split applied to a single look too": ("return alpha / 2 if n1 < n2 else alpha", "return alpha / 2"),
    "M7 seed floor 3": ("SEED_FLOOR = 5", "SEED_FLOOR = 3"),
    "M8 no zero-variance refusal": ("if not (varies(a) or varies(b)):", "if False:"),
    "M9 direction ignored": ("diff = float(a.mean() - b.mean())", "diff = abs(float(a.mean() - b.mean()))"),
    "M10 plans power at the full validation gap": ("stage1_power(sa, sb, n, gap / 2, level)", "stage1_power(sa, sb, n, gap, level)"),
    "M11 power target 0.5": ("POWER_TARGET = 0.8", "POWER_TARGET = 0.5"),
    "M12 any clause suffices": ("if chosen is None and all(p >= target", "if chosen is None and any(p >= target"),
    "M13 power from a z approximation": ("return float(stats.nct.sf(stats.t.isf(level, df), df, delta / se))",
                                         "return float(stats.norm.sf(stats.t.isf(level, df) - delta / se))"),
    "M14 plans at the single-look level": ("level = look_level(alpha, n, CAP)", "level = look_level(alpha, n, n)"),
    "M15 Welch df numerator va^2 + vb^2": ("return (va + vb) ** 2 / den", "return (va * va + vb * vb) / den"),
    "M16 'varies' judged by computed variance": ("return not bool(np.all(v == v[0]))", "return float(v.var(ddof=1)) > 0"),
    "M17 unequal seed counts accepted": ("if len(a) != len(b):", "if False:"),
    "M18 pilot not checked before planning": ("        _check_arms(a, b)\n        clauses[name]", "        clauses[name]"),
    "M19 joint power as a product": ("max(0.0, 1 - shortfall) if shortfall == shortfall else float(\"nan\")",
                                     "float(np.prod(list(power.values())))"),
    "M20 two-sided p-value": ("\"p_one_sided\": float(stats.t.sf(diff / se, df))",
                              "\"p_one_sided\": float(2 * stats.t.sf(abs(diff) / se, df))"),
}


def run_tests(directory: Path) -> int:
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(directory)],
                          cwd=directory, capture_output=True, text=True).returncode


def main() -> int:
    problems = 0
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "unmutated"
        base.mkdir()
        shutil.copy(HERE / "gate_stats_proto.py", base)
        shutil.copy(HERE / "test_gate_stats_proto.py", base)
        rc = run_tests(base)
        print(f"unmutated: {'PASS' if rc == 0 else 'FAIL'} (exit {rc})")
        problems += rc != 0
        for i, (name, (old, new)) in enumerate(MUTANTS.items()):
            assert SRC.count(old) == 1, f"{name}: anchor not unique"
            d = Path(tmp) / f"m{i:02d}"
            d.mkdir()
            (d / "gate_stats_proto.py").write_text(SRC.replace(old, new), encoding="utf-8")
            shutil.copy(HERE / "test_gate_stats_proto.py", d)
            killed = run_tests(d) != 0
            print(f"{name}: {'killed' if killed else 'SURVIVED'}")
            problems += not killed
    print(f"{len(MUTANTS)} mutants, {problems} problem(s)")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
