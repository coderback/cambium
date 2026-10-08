"""Mutation check for the ledger prototype v2: each edit breaks one property; some test must fail.

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
REPO = os.environ.get("CAMBIUM_REPO") or str(HERE.parents[3])

OPEN_WINDOW_RECORD = ('    record_look(ledger, held_out_set="dgf1-temporal-test-482-821", tier=tier, what=what, by=by,\n'
                      '                via=via, authority=authority, git_commit=git_commit, repo_root=repo_root, now=now)\n')
OPEN_MASK = ('    record_look(ledger, held_out_set="dgraph-official-test", tier=tier, what=what, by=by, via=via,\n'
             '                authority=authority, git_commit=git_commit, repo_root=repo_root, now=now)\n'
             '    return load_raw()["test_mask"]')

MUTANTS = [
    # the ledger file
    ("append becomes overwrite", 'with path.open("a", newline=""', 'with path.open("w", newline=""'),
    ("header written on every append", "        if new_file:\n            w.writeheader()", "        w.writeheader()"),
    ("fsync removed", "        os.fsync(f.fileno())\n", ""),
    # field checks
    ("unknown held-out set accepted", "    if held_out_set not in HELD_OUT_SETS:\n        raise", "    if False:\n        raise"),
    ("unknown tier accepted", "    if tier not in TIERS:\n        raise", "    if False:\n        raise"),
    ("empty fields accepted", "        if not value.strip():\n            raise", "        if False:\n            raise"),
    ("figure check removed", "    if _FIGURE.search(what):", "    if False:"),
    ("figure check back to draft 1's pattern", '_FIGURE = re.compile(r"(?<![\\w-])\\d+(?:[.,]\\d+)*%?(?![\\w-])")',
     '_FIGURE = re.compile(r"\\d\\.\\d|\\d{3,}")'),
    ("figure check flags digits inside names", '_FIGURE = re.compile(r"(?<![\\w-])\\d+(?:[.,]\\d+)*%?(?![\\w-])")',
     '_FIGURE = re.compile(r"\\d")'),
    # authority
    ("authority never checked", "    check_authority(tier, authority, via, when, repo_root, incidental)\n", ""),
    ("a missing document authorises", "    if not doc.is_file():\n        raise", "    if False:\n        raise"),
    ("a proposed document authorises", '    if not status or status.group(1).lower() != "accepted":\n        raise', "    if False:\n        raise"),
    ("a document accepted after the look authorises", "date.fromisoformat(on.group(1)) > when.date()", "False"),
    ("a document that names nothing authorises", "    if via not in text:\n        raise", "    if False:\n        raise"),
    ("labels need no document", '    if tier == "structure":\n        return', '    if tier != "scores":\n        return'),
    ("an incidental look may claim authority", '        if not authority.startswith("none"):\n            raise', "        if False:\n            raise"),
    # append-only and git_dirty
    ("append-only check always passes", "    return working.startswith(committed)", "    return True"),
    ("append-only check only compares lengths", "    return working.startswith(committed)", "    return len(working) >= len(committed)"),
    ("dirty check ignores HEAD", '    committed = head.stdout if head.returncode == 0 else ""', '    committed = ""'),
    # the chokepoint
    ("ordinary splits hand out the test window", '    return {"val": Window(config["val_min"], config["val_max"])}',
     '    return {"val": Window(config["val_min"], config["val_max"]), "test": Window(config["test_min"], config["test_max"])}'),
    ("ordinary load keeps the official test mask", '    return {k: v for k, v in raw.items() if k != "test_mask"}', "    return dict(raw)"),
    ("opening the window records nothing", OPEN_WINDOW_RECORD, ""),
    ("the mask is loaded before the look is recorded", OPEN_MASK,
     '    m = load_raw()["test_mask"]\n'
     '    record_look(ledger, held_out_set="dgraph-official-test", tier=tier, what=what, by=by, via=via,\n'
     '                authority=authority, git_commit=git_commit, repo_root=repo_root, now=now)\n'
     '    return m'),
    # disclosure
    ("disclosure ignores overlapping sets", "    sets = {held_out_set} | HELD_OUT_SETS[held_out_set]", "    sets = {held_out_set}"),
    ("disclosure includes the batch's own row", 'and r["timestamp_utc"] < before]', 'and r["timestamp_utc"] <= before]'),
    ("disclosure keeps file order", '    seen.sort(key=lambda r: r["timestamp_utc"])\n', ""),
    # the backstop
    ("backstop misses the PyG loader", '"DGraphFin": "PyG loader", ', ""),
    ("backstop misses hand-built windows", ', "TemporalSplit": "hand-built window"', ""),
    ("backstop misses mask attributes", 'isinstance(node, ast.Attribute) and node.attr.endswith("test_mask")', "False"),
    ("backstop misses mask subscripts", 'isinstance(node.slice.value, str) and node.slice.value.endswith("test_mask")', "False"),
    ("backstop misses score files", "        if reads_scores and not opens:", "        if False:"),
    ("backstop misses incidental claims", 'any(k.arg == "incidental" for k in node.keywords)', "False"),
    ("backstop flags the hand path", "if name != HAND_PATH and any(", "if any("),
    ("backstop scans test files", '                or Path(name).name.startswith("test_")', "                or False"),
    ("backstop scans prototypes", '                or "/prototype/" in name):', "                or False):"),
    ("backstop skips notebooks", 'SCOPE = ("scripts/", "adapters/dgf1/", "notebooks/")', 'SCOPE = ("scripts/", "adapters/dgf1/")'),
]


def run(mutated: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "ledger_proto.py").write_text(mutated, encoding="utf-8")
        shutil.copy(HERE / "test_ledger_proto.py", d / "test_ledger_proto.py")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                            "test_ledger_proto.py"], cwd=d, capture_output=True, text=True,
                           env={**os.environ, "CAMBIUM_REPO": REPO})
        last = [ln for ln in r.stdout.splitlines() if ln.strip()][-1]
        return r.returncode != 0, last


def main() -> None:
    ok, last = run(SRC)
    print(f"unmutated: {'FAILS (harness broken)' if ok else 'passes'} — {last}")
    if ok:
        raise SystemExit(1)
    killed = 0
    for name, old, new in MUTANTS:
        assert SRC.count(old) == 1, f"mutant {name!r}: target found {SRC.count(old)}x"
        dead, last = run(SRC.replace(old, new))
        killed += dead
        print(f"{'KILLED  ' if dead else 'SURVIVED'} {name} — {last}")
    print(f"\n{killed}/{len(MUTANTS)} mutants killed")


if __name__ == "__main__":
    main()
