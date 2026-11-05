"""ADR-023's margin: half the gated view's validation gap, fixed before acceptance (clause 5).

    python scripts/matched_margin.py > decisions/ADR-023-margin.txt

It reads ADR-012's validation pilot, selected by the `pilot` tag, and nothing else. A row on any other
window or with any other tag is never used, and no Gate-1 test-window figure enters. It prints each metric's
gap and margin, and the spread at which the go/no-go rule starts to fail, so the rule is known not to be
vacuous (as ADR-012:270-285). This is arithmetic on rows already in the registry. It runs no model.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gate_stats import ALPHA, POWER_TARGET, pilot_arm, sample_var, welch_power  # noqa: E402

REGISTRY = REPO_ROOT / "experiments" / "registry.csv"
MODEL, WINDOW, TAG = "dgf1", "370-481", "pilot"
DGF1_ARM, FLOOR_ARM = "dgf1-parity", "xgboost-parity"
METRICS = ("fraud_auc", "fraud_auprc")
SEEDS = {0, 1, 2, 3, 4}
FINAL_N = 20
LEVEL = ALPHA / 2
MARGIN_FACTOR = 0.5   # ADR-018 clause 4's planning factor


def gated_pilot(rows: Iterable[Mapping[str, str]]) -> dict[str, dict[str, dict[int, float]]]:
    """Each arm's pilot values by metric and seed. Refuses anything but exactly seeds 0 to 4, on one commit."""
    pilot = [r for r in rows if r["model"] == MODEL and json.loads(r["metrics_json"]).get("experiment") == TAG]
    arms = {a: {m: pilot_arm(pilot, model=MODEL, arm=a, metric=m, window=WINDOW) for m in METRICS}
            for a in (DGF1_ARM, FLOOR_ARM)}
    for a, values in arms.items():
        for m in METRICS:
            if set(values[m]) != SEEDS:
                raise ValueError(f"{a} {m} is not exactly seeds {sorted(SEEDS)}")
    used = {r["git_commit"] for r in pilot if json.loads(r["metrics_json"]).get("window") == WINDOW
            and json.loads(r["metrics_json"]).get("arm") in (DGF1_ARM, FLOOR_ARM)}
    if len(used) != 1:
        raise ValueError(f"the pilot's arms span commits {sorted(used)}")
    return arms


def spread_at_the_bar(power_of_spread) -> float:
    """The spread at which power falls to the target, by bisection. Power falls as the spread grows."""
    lo, hi = 1e-9, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if power_of_spread(mid) > POWER_TARGET else (lo, mid)
    return (lo + hi) / 2


def margin_row(dgf1: Mapping[int, float], floor: Mapping[int, float]) -> dict[str, float]:
    """One metric: the gap, the margin, and where the rule starts to fail."""
    gap = sum(dgf1.values()) / len(dgf1) - sum(floor.values()) / len(floor)
    if gap <= 0:
        return {"gap": gap, "margin": float("nan")}
    m = MARGIN_FACTOR * gap
    s_dgf1, s_floor = math.sqrt(sample_var(list(dgf1.values()))), math.sqrt(sample_var(list(floor.values())))
    return {"gap": gap, "margin": m, "s_dgf1": s_dgf1, "s_floor": s_floor,
            "power": welch_power(s_dgf1, s_floor, FINAL_N, m, LEVEL),
            "power_half": welch_power(s_dgf1, s_floor, FINAL_N, m / 2, LEVEL),
            "fails_dgf1": spread_at_the_bar(lambda s: welch_power(s, s_floor, FINAL_N, m, LEVEL)),
            "fails_both": spread_at_the_bar(lambda s: welch_power(s, s, FINAL_N, m, LEVEL))}


def report(rows: Iterable[Mapping[str, str]]) -> str:
    arms = gated_pilot(rows)
    lines = [f"ADR-023 clause 5: the margin, from ADR-012's validation pilot (tag `{TAG}`, window {WINDOW}, seeds "
             f"{min(SEEDS)} to {max(SEEDS)} per arm).", f"The rule's power is at {FINAL_N} seeds per arm and level {LEVEL:.6f}.", ""]
    for metric in METRICS:
        r = margin_row(arms[DGF1_ARM][metric], arms[FLOOR_ARM][metric])
        lines.append(f"{metric}: Δ_gated {r['gap']:+.6f}, m {r['margin']:.6f}")
        if r["margin"] == r["margin"]:
            lines += [f"  the pilot's spreads: DGF-1 {r['s_dgf1']:.6f}, floor {r['s_floor']:.6f}; power at them {r['power']:.4f} at a true gain of 0, "
                      f"{r['power_half']:.4f} at a true gain of m/2",
                      f"  the rule starts to fail at: DGF-1's spread {r['fails_dgf1']:.6f}, with the floor's as above; "
                      f"or both spreads at {r['fails_both']:.6f}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    csv.field_size_limit(10_000_000)
    with REGISTRY.open(newline="", encoding="utf-8") as f:
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        sys.stdout.write(report(csv.DictReader(f)))


if __name__ == "__main__":
    main()
