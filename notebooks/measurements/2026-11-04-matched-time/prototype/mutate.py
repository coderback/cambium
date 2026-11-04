"""Mutation check for the go/no-go prototype. It works on temporary copies and reads only gate_stats from the repo."""
import ctypes
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[3] / "scripts"
# The copies live in a temporary folder, so they are pointed at the repo's scripts by absolute path.
SRC = (HERE / "go_no_go_proto.py").read_text(encoding="utf-8").replace(
    'Path(__file__).resolve().parents[4] / "scripts"', f'Path(r"{SCRIPTS}")')

MUTANTS = [
    ("the rule receives the arms' means", "    best_auc: float\n", "    best_auc: float\n    gap: float = 0.0\n"),
    ("the bar reads the lower mean", "    best = max(float(", "    best = min(float("),
    ("the bar reads DGF-1's mean only, so it sees direction", 'float(np.mean(list(floor["fraud_auc"].values()))))', "-1.0)"),
    ("the bar is strict", "sane = inputs.best_auc >= SANITY_BAR", "sane = inputs.best_auc > SANITY_BAR"),
    ("the bar is lower", "SANITY_BAR = 0.55 ", "SANITY_BAR = 0.50 "),
    ("the bar is ignored", 'return {"go": sane and bool(informative)', 'return {"go": bool(informative)'),
    ("precision is ignored", 'return {"go": sane and bool(informative)', 'return {"go": sane'),
    ("both metrics must be informative", "and bool(informative)", "and len(informative) == 2"),
    ("the power bar is lower", "if power[m] >= POWER_TARGET", "if power[m] >= 0.5"),
    ("the power uses the single-look level", "m, alpha / 2)", "m, alpha)"),
    ("the power is at fewer seeds", "FINAL_N = 20", "FINAL_N = 12"),
    ("the margin is the whole gap", "MARGIN_FACTOR = 0.5 ", "MARGIN_FACTOR = 1.0 "),
    ("no gated gain still has a margin", "if delta_gated > 0 else None", "if delta_gated >= 0 else None"),
    ("a negative gated gain has a margin", "return MARGIN_FACTOR * delta_gated if delta_gated > 0 else None", "return MARGIN_FACTOR * abs(delta_gated) or None"),
    ("neither arm varying crashes", "if m is None or (s_dgf1 == 0 and s_floor == 0):", "if m is None:"),
    ("one arm not varying fails precision", "(s_dgf1 == 0 and s_floor == 0)", "(s_dgf1 == 0 or s_floor == 0)"),
    ("the spreads are swapped with the wrong metric", "sample_var(list(floor[m].values()))", "sample_var(list(floor[METRICS[0]].values()))"),
    ("extra seeds are read", "set(values[metric]) != set(PILOT_SEEDS)", "not set(PILOT_SEEDS) <= set(values[metric])"),
    ("missing seeds are read", "set(values[metric]) != set(PILOT_SEEDS)", "not set(values[metric]) <= set(PILOT_SEEDS)"),
    ("a missing metric is not refused", "if metric not in values or set(", "if metric in values and set("),
    ("non-finite values are read", "if not all(math.isfinite(v) for v in values[metric].values()):", "if False:"),
    ("only one arm is checked", 'for arm, values in (("dgf1", dgf1), ("floor", floor)):', 'for arm, values in (("dgf1", dgf1),):'),
    ("the margin clause does not shift", "np.asarray(floor, dtype=float) + m, dgf1", "np.asarray(floor, dtype=float), dgf1"),
    ("the margin clause shifts the wrong way", "np.asarray(floor, dtype=float) + m, dgf1", "np.asarray(floor, dtype=float) - m, dgf1"),
    ("the margin clause does not swap the arms", "judge(np.asarray(floor, dtype=float) + m, dgf1,", "judge(dgf1, np.asarray(floor, dtype=float) + m,"),
    ("a refused gain clause still reads", '    if gain["refused"]:\n        return "refused"\n', ""),
    ("a refused margin clause counts", 'under = below is not None and not below["refused"] and bool(below["resolvable"])',
     'under = below is not None and (bool(below["refused"]) or bool(below["resolvable"]))'),
    ("a resolvable gain hides the margin", '    if resolved and under:\n        return "resolvable and below the margin"\n', ""),
    ("below the margin reads as nothing", 'return "below the margin" if under else "not resolvable"', 'return "not resolvable"'),
    ("a second look is never triggered", "return stage2_triggered(gain_results)", "return False"),
]


def run(d: Path, src: str) -> tuple[bool, str]:
    (d / "go_no_go_proto.py").write_text(src, encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "test_go_no_go_proto.py"],
                       cwd=d, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    return r.returncode != 0, (lines[-1] if lines else r.stderr[-200:])


def main() -> None:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # below normal
    for name, old, _ in MUTANTS:
        assert SRC.count(old) == 1, f"{name}: anchor found {SRC.count(old)}x: {old!r}"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        d = Path(tmp)
        shutil.copy(HERE / "test_go_no_go_proto.py", d)
        dead, last = run(d, SRC)
        print(f"unmutated: {'FAILS' if dead else 'passes'} - {last}", flush=True)
        if dead:
            raise SystemExit(1)
        survived = 0
        for name, old, new in MUTANTS:
            dead, last = run(d, SRC.replace(old, new))
            survived += not dead
            print(f"{'KILLED  ' if dead else 'SURVIVED'} {name} - {last[:60]}", flush=True)
    print(f"\n{len(MUTANTS) - survived}/{len(MUTANTS)} mutants killed", flush=True)


main()
