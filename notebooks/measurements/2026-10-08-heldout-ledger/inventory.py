"""Which committed scripts can read a held-out DGF-1 set, and which of them pass the guard?

    python notebooks/measurements/2026-10-08-heldout-ledger/inventory.py > .../output.txt

Static and label-free: it parses source text and never imports repository code, loads data or
reads the registry (ADR-017 item 3). It measures one thing the ledger ADR needs: how many paths to
held-out data exist outside the pre-registration guard (`scripts/preregistration.py`), so the ADR
can say whether a guard-written ledger alone would be complete.

The held-out sets are DGF-1's temporal test window (482-821, ADR-011) and DGraph's official test
mask (ADR-010, ADR-015).
- **Detection is mechanical.** A script can read a held-out set when it loads the DGraph data and
  names the test split or the official test mask, or when it reads gate rows and score files from
  the registry.
- **The tier is not.** What a script computes *on the held-out part* was read from its code, and
  `TIER` below records that reading with the lines it rests on. A detected script missing from
  `TIER` is printed as "unread", so the table cannot silently go stale.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

LOADS_DGRAPH = re.compile(r"\b(load_dgraph|prepare_dgraph|DGraphFin)\b")
TEST_WINDOW = re.compile(r"""splits\[["']test["']\]|_window\(\s*["']test["']""")
OFFICIAL_TEST = re.compile(r"\btest_mask\b")
GATE_ROWS = re.compile(r"""load_scores|["']gate["']""")
GUARD = re.compile(r"require_gated_preregistration\(")

# What each script computes on the held-out part, read from its code on 2026-10-08.
TIER = {
    "scripts/assemble_gate_dgf1_1.py": ("scores", "reads the gate rows and their score files after the batch (:167, :179, :187)"),
    "scripts/audit_dgf1_datasource.py": ("labels", "labelled and fraud counts of 482-821 (:97-100)"),
    "scripts/measure_dgf1_temporal_split.py": ("labels", "support, fraud count and prevalence of 482-821 and of the official test mask (:64-71, :103, :110)"),
    "scripts/run_dgf1_floor.py": ("scores", "the gated floor batch, only through the guard (:71)"),
    "scripts/run_dgf1_gnn.py": ("scores", "the gated GNN batch, only through the guard (:119)"),
    "scripts/verify_dgraph_snapshot.py": ("structure", "the official test mask's size and node-time distribution (:178-192); its prevalence (:161-162) is over all labelled nodes"),
}


def committed_scripts() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "scripts/*.py"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout.split()
    return [REPO / p for p in out]


def detect(path: Path) -> dict | None:
    src = path.read_text(encoding="utf-8")
    loads = bool(LOADS_DGRAPH.search(src))
    window, official = bool(TEST_WINDOW.search(src)), bool(OFFICIAL_TEST.search(src))
    gate_rows = "dgf1" in path.name and "registry" in src and bool(GATE_ROWS.search(src))
    if not ((loads and (window or official)) or gate_rows):
        return None
    sets = [s for s, hit in (("482-821", window or gate_rows), ("official test", official)) if hit]
    rel = path.relative_to(REPO).as_posix()
    tier, why = TIER.get(rel, ("unread", "not yet read"))
    return {"script": rel, "sets": ", ".join(sets), "tier": tier, "why": why,
            "guard": "yes" if GUARD.search(src) else "no"}


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> None:
    rows = [r for p in committed_scripts() if (r := detect(p))]
    print(f"# Held-out readers among committed scripts at {git_head()}\n")
    print("| script | held-out set | tier on the held-out part | what, read from the code | through the guard |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| `{r['script']}` | {r['sets']} | {r['tier']} | {r['why']} | {r['guard']} |")
    outside = [r for r in rows if r["guard"] == "no"]
    print(f"\n{len(rows)} committed scripts can read a held-out DGF-1 set; {len(outside)} do so outside "
          "the guard:")
    for r in outside:
        print(f"- `{r['script']}` ({r['tier']})")
    unread = [r["script"] for r in rows if r["tier"] == "unread"]
    print(f"\nDetected but not read: {len(unread)}" + (f" ({', '.join(unread)})" if unread else ""))
    stale = sorted(set(TIER) - {r["script"] for r in rows})
    print(f"Read but no longer detected: {len(stale)}" + (f" ({', '.join(stale)})" if stale else ""))


if __name__ == "__main__":
    main()
