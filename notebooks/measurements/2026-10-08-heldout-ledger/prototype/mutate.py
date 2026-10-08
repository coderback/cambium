"""Mutation check for the ledger prototype v4: each edit breaks one property; some test must fail.

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

HIDE = '    data["y"] = [HIDDEN if t > config["val_max"] else y for y, t in zip(raw["y"], raw["node_time"])]'
EXEMPT = '                or re.match(r"notebooks/measurements/[^/]+/prototype/", name)):   # ADR evidence'
WINDOW_OPEN = ('    _open(ledger, "dgf1-temporal-test-482-821", tier=tier, authority=authority, by=by, what=what,\n'
               '          batch=batch, repo_root=repo_root, git_commit=git_commit, now=now)\n')
OFFICIAL_OPEN = ('    _open(ledger, "dgraph-official-test", tier=tier, authority=authority, by=by, what=what,\n'
                 '          batch=batch, repo_root=repo_root, git_commit=git_commit, now=now)\n'
                 '    raw = load_raw()\n')
FIGURE = '_FIGURE = re.compile(r"(?<![\\w-])\\d+(?:[.,]\\d+)*%?(?![\\w-])")'

MUTANTS = [
    # the ledger file
    ("append becomes overwrite", 'with path.open("a", newline=""', 'with path.open("w", newline=""'),
    ("header written on every append", "        if new_file:\n            w.writeheader()", "        w.writeheader()"),
    ("fsync removed", "        os.fsync(f.fileno())\n", ""),
    # field checks
    ("unknown held-out set accepted", "    if held_out_set not in HELD_OUT_SETS:\n        raise", "    if False:\n        raise"),
    ("unknown tier accepted", "    if tier not in TIERS:\n        raise", "    if False:\n        raise"),
    ("empty fields accepted", "        if not value.strip():\n            raise", "        if False:\n            raise"),
    ("an empty batch accepted", '("authority", authority),\n                        ("batch", batch)):', '("authority", authority)):'),
    ("figure check removed", "    if _FIGURE.search(what):", "    if False:"),
    ("figure check back to draft 1's pattern", FIGURE, '_FIGURE = re.compile(r"\\d\\.\\d|\\d{3,}")'),
    ("figure check flags digits inside names", FIGURE, '_FIGURE = re.compile(r"\\d")'),
    # authority
    ("authority never checked", "    if not backfill:      # a backfill row", "    if False:      # a backfill row"),
    ("a document not in HEAD authorises", "    if head.returncode != 0:\n        raise", "    if False:\n        raise"),
    ("the working-tree document is read", "    text = head.stdout",
     '    text = (repo_root / authority).read_text(encoding="utf-8") if (repo_root / authority).exists() else ""'),
    ("a proposed document authorises", '    if not status or status.group(1).lower() != "accepted":\n        raise', "    if False:\n        raise"),
    ("a document accepted after the look authorises", "date.fromisoformat(on.group(1)) >= when.date()", "False"),
    ("a document accepted the same day authorises", "date.fromisoformat(on.group(1)) >= when.date()",
     "date.fromisoformat(on.group(1)) > when.date()"),
    ("a caller's via is trusted", "    if via != current_script(repo_root):\n        raise", "    if False:\n        raise"),
    ("a prose mention authorises", "    if not any((held_out_set, t, via) in named for t in allowed):",
     "    if via not in text:"),
    ("the set is not matched", "    if not any((held_out_set, t, via) in named for t in allowed):",
     "    if not any(t2 in allowed and v == via for _, t2, v in named):"),
    ("a scores line does not cover labels", '    allowed = {tier} | ({"scores"} if tier == "labels" else set())',
     "    allowed = {tier}"),
    ("labels need no document", '    if tier == "structure":\n        return\n    head', '    if tier != "scores":\n        return\n    head'),
    ("an incidental look may claim authority", '        if not authority.startswith("none"):\n            raise', "        if False:\n            raise"),
    ("an interactive session may look", '    if not f:\n        raise LedgerError("a look must be made by a script',
     '    if False:\n        raise LedgerError("a look must be made by a script'),
    # backfill
    ("backfill accepts looks after the cutoff", "    if backfill and when >= BACKFILL_BEFORE:", "    if False:"),
    ("backfill skips the figure filter", "    if _FIGURE.search(what):", "    if _FIGURE.search(what) and not backfill:"),
    ("backfill runs the authority check", "    if not backfill:      # a backfill row", "    if True:      # a backfill row"),
    # append-only and git_dirty
    ("append-only check always passes", "    return working.startswith(committed)", "    return True"),
    ("append-only check only compares lengths", "    return working.startswith(committed)", "    return len(working) >= len(committed)"),
    ("dirty check ignores HEAD", '    committed = head.stdout if head.returncode == 0 else ""', '    committed = ""'),
    # the default paths
    ("default splits hand out the test window", '    return {"val": Window(config["val_min"], config["val_max"])}',
     '    return {"val": Window(config["val_min"], config["val_max"]), "test": Window(config["test_min"], config["test_max"])}'),
    ("default load keeps the official masks", "    data = {k: v for k, v in raw.items() if k not in OFFICIAL_MASKS}", "    data = dict(raw)"),
    ("default load keeps the train and val masks", 'OFFICIAL_MASKS = ("train_mask", "val_mask", "test_mask")',
     'OFFICIAL_MASKS = ("test_mask",)'),
    ("default load keeps later labels", HIDE, '    data["y"] = list(raw["y"])'),
    ("labels hidden only after the test window", HIDE, HIDE.replace('config["val_max"]', 'config["test_max"]')),
    ("the hidden value is a real label", "HIDDEN = -1 ", "HIDDEN = 2 "),
    ("integrity check always passes", "    return all(sum(1 for v in y if v == k) == n for k, n in expected.items())", "    return True"),
    # the accessors
    ("opening the window records nothing", WINDOW_OPEN, ""),
    ("window labels loaded before the look is recorded", WINDOW_OPEN, "    _y = load_raw()\n" + WINDOW_OPEN),
    ("a structure open of the window returns labels", '    if tier == "structure":\n        return window, None', "    if False:\n        return window, None"),
    ("the official track opens at structure", '    if tier == "structure":\n        raise LedgerError("the official masks', '    if False:\n        raise LedgerError("the official masks'),
    ("the official track loads before recording", OFFICIAL_OPEN,
     "    raw = load_raw()\n" + OFFICIAL_OPEN.replace("    raw = load_raw()\n", "")),
    ("the official track returns hidden labels", '    return {"y": list(raw["y"]), **{m: raw[m] for m in OFFICIAL_MASKS}}',
     '    return {"y": [HIDDEN] * len(raw["y"]), **{m: raw[m] for m in OFFICIAL_MASKS}}'),
    ("an opener takes the caller's via", "                via=current_script(repo_root), authority=authority, batch=batch,",
     '                via="scripts/run_gate3.py", authority=authority, batch=batch,'),
    # disclosure
    ("disclosure ignores overlapping sets", "    sets = {held_out_set} | HELD_OUT_SETS[held_out_set]", "    sets = {held_out_set}"),
    ("disclosure includes the batch's own row", 'and r["timestamp_utc"] < before]', 'and r["timestamp_utc"] <= before]'),
    ("disclosure keeps file order", '    seen.sort(key=lambda r: r["timestamp_utc"])\n', ""),
    ("disclosure starts at the last batch row", "    before = min(starts)", "    before = max(starts)"),
    ("disclosure of an unopened batch is empty", "    if not starts:\n        raise", "    if not starts:\n        return []\n        raise"),
    # the backstop
    ("backstop misses the PyG loader", '"DGraphFin": "PyG loader", ', ""),
    ("backstop misses hand-built windows", ', "TemporalSplit": "hand-built window"', ""),
    ("backstop misses mask attributes", "isinstance(node, ast.Attribute) and _mask_name(node.attr)", "False"),
    ("backstop misses mask subscripts", "isinstance(node.slice.value, str) and _mask_name(node.slice.value)", "False"),
    ("backstop sees only the test mask", '    return s.endswith("_mask") and s not in', '    return s.endswith("test_mask") and s not in'),
    ("backstop flags derived masks", 'and s not in ("labelled_mask", "train_seed_mask", "window_target_mask")', "and True"),
    ("backstop misses score files", "        if reads_scores and not opens:", "        if False:"),
    ("backstop misses incidental and backfill claims", 'any(k.arg in ("incidental", "backfill") for k in node.keywords)', "False"),
    ("backstop misses backfill claims", 'k.arg in ("incidental", "backfill")', 'k.arg == "incidental"'),
    ("backstop flags the hand path", "if name != HAND_PATH and any(", "if any("),
    ("backstop scans test files", '                or Path(name).name.startswith("test_")', "                or False"),
    ("backstop scans prototypes", EXEMPT, "                or False):"),
    ("backstop exempts any prototype folder", EXEMPT, '                or "/prototype/" in name):'),
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
    for name, old, _ in MUTANTS:
        assert SRC.count(old) == 1, f"mutant {name!r}: target found {SRC.count(old)}x"
    killed = 0
    for name, old, new in MUTANTS:
        dead, last = run(SRC.replace(old, new))
        killed += dead
        print(f"{'KILLED  ' if dead else 'SURVIVED'} {name} — {last}", flush=True)
    print(f"\n{killed}/{len(MUTANTS)} mutants killed")


if __name__ == "__main__":
    main()
