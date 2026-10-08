"""Mutation check for the ledger prototype: each edit breaks one property; some test must fail.

    python mutate.py > mutation-output.txt
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "ledger_proto.py").read_text(encoding="utf-8")

MUTANTS = [
    ("append becomes overwrite",
     'with path.open("a", newline=""', 'with path.open("w", newline=""'),
    ("header written on every append",
     "        if new_file:\n            w.writeheader()", "        w.writeheader()"),
    ("unknown held-out set accepted",
     "    if held_out_set not in HELD_OUT_SETS:\n        raise", "    if False:\n        raise"),
    ("unknown tier accepted",
     "    if tier not in TIERS:\n        raise", "    if False:\n        raise"),
    ("empty fields accepted",
     "        if not value.strip():\n            raise", "        if False:\n            raise"),
    ("figures allowed in 'what'",
     'if re.search(r"\\d\\.\\d|\\d{3,}", what):', "if False:"),
    ("authority never checked",
     "    check_authority(tier, authority, repo_root, incidental)\n", ""),
    ("a proposed document authorises",
     '    if not status or status.group(1).lower() != "accepted":\n        raise',
     "    if False:\n        raise"),
    ("labels need no accepted document",
     '    if tier == "structure":\n        return', '    if tier != "scores":\n        return'),
    ("an incidental look may claim authority",
     '        if not authority.startswith("none"):\n            raise', "        if False:\n            raise"),
    ("append-only check always passes",
     "    return working.startswith(committed)", "    return True"),
    ("append-only check only compares lengths",
     "    return working.startswith(committed)", "    return len(working) >= len(committed)"),
    ("guard skips the ledger",
     '    record_look(ledger, held_out_set=m.held_out_set, tier="scores",',
     '    if False: record_look(ledger, held_out_set=m.held_out_set, tier="scores",'),
    ("guard records the wrong tier",
     'tier="scores",\n                what=f"gated batch', 'tier="labels",\n                what=f"gated batch'),
    ("guard records the wrong authority",
     'via="scripts/preregistration.py", authority=m.document,',
     'via="scripts/preregistration.py", authority=mode,'),
    ("guard records before refusing",
     "    if mode not in modes:\n        raise SystemExit",
     "    if mode not in modes:\n        record_look(ledger, held_out_set=HELD_OUT_SETS[0], "
     "tier='structure', what='x', by='x', via='x', authority='x', git_commit='x', "
     "repo_root=repo_root)\n        raise SystemExit"),
    ("lint misses the official test mask",
     "|\\btest_mask\\b|", "|"),
    ("lint misses the PyG loader",
     "|\\bDGraphFin\\(", ""),
    ("lint misses gate score files",
     "or READS_GATE_SCORES.search(src)", "or False"),
    ("lint accepts any mention of the ledger",
     'RECORDS = re.compile(r"\\brecord_look\\(|\\brequire_gated_preregistration\\(")',
     'RECORDS = re.compile(r"record_look|ledger|preregistration")'),
    ("lint misses a script claiming incidental",
     'INCIDENTAL = re.compile(r"\\bincidental\\s*=\\s*True\\b")',
     'INCIDENTAL = re.compile(r"(?!x)x")'),
    ("lint flags the hand path too",
     "if n != HAND_PATH and INCIDENTAL.search(src)", "if INCIDENTAL.search(src)"),
    ("disclosure includes the batch's own row",
     'and r["timestamp_utc"] < before]', 'and r["timestamp_utc"] <= before]'),
    ("disclosure ignores the set",
     'if r["held_out_set"] == held_out_set and', "if"),
    ("disclosure keeps file order",
     '    seen.sort(key=lambda r: r["timestamp_utc"])\n', ""),
]


def run(mutated: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "ledger_proto.py").write_text(mutated, encoding="utf-8")
        shutil.copy(HERE / "test_ledger_proto.py", d / "test_ledger_proto.py")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                            "test_ledger_proto.py"], cwd=d, capture_output=True, text=True,
                           env={**os.environ, "CAMBIUM_REPO": str(HERE.parents[3])})
        last = [ln for ln in r.stdout.splitlines() if ln.strip()][-1]
        return r.returncode != 0, last


def main() -> None:
    ok, last = run(SRC)
    print(f"unmutated: {'FAILS (harness broken)' if ok else 'passes'} — {last}")
    killed = 0
    for name, old, new in MUTANTS:
        assert SRC.count(old) == 1, f"mutant {name!r}: target found {SRC.count(old)}x"
        dead, last = run(SRC.replace(old, new))
        killed += dead
        print(f"{'KILLED  ' if dead else 'SURVIVED'} {name} — {last}")
    print(f"\n{killed}/{len(MUTANTS)} mutants killed")


if __name__ == "__main__":
    main()
