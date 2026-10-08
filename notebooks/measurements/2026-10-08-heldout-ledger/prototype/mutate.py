"""Mutation check for the ledger prototype v5: each edit breaks one property; some test must fail.

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
               '          batch=batch, repo_root=repo_root, git_commit=git_commit)\n')
OFFICIAL_OPEN = ('    _open(ledger, "dgraph-official-test", tier=tier, authority=authority, by=by, what=what,\n'
                 '          batch=batch, repo_root=repo_root, git_commit=git_commit)\n'
                 '    raw = load_raw()\n')
FIGURE = '_FIGURE = re.compile(r"(?<![\\w-])\\d+(?:[.,]\\d+)*%?(?![\\w-])")'
WAIT_CHECK = "    if on is None or on.date() >= local.date() or local - on < WAIT:"
LINE_CHECK = "    if not any((held_out_set, t, via, batch) in named for t in allowed):"
HAND = 'HAND_KEYWORDS = ("incidental", "backfill", "now")'

# Each mutant names the invariant it breaks (ADR-021's table, I1-I11).
MUTANTS = [
    ("I1 append becomes overwrite", 'with path.open("a", newline=""', 'with path.open("w", newline=""'),
    ("I1 header written on every append", "        if new_file:\n            w.writeheader()", "        w.writeheader()"),
    ("I1 fsync removed", "        os.fsync(f.fileno())\n", ""),
    ("I2 unknown held-out set accepted", "    if held_out_set not in HELD_OUT_SETS:\n        raise", "    if False:\n        raise"),
    ("I2 unknown tier accepted", "    if tier not in TIERS:\n        raise", "    if False:\n        raise"),
    ("I2 empty fields accepted", "        if not value.strip():\n            raise", "        if False:\n            raise"),
    ("I2 an empty batch accepted", '("authority", authority),\n                        ("batch", batch)):', '("authority", authority)):'),
    ("I2 figure check removed", "    if _FIGURE.search(what):", "    if False:"),
    ("I2 figure check back to draft 1's pattern", FIGURE, '_FIGURE = re.compile(r"\\d\\.\\d|\\d{3,}")'),
    ("I2 figure check flags digits inside names", FIGURE, '_FIGURE = re.compile(r"\\d")'),
    ("I3 authority never checked", "    if not backfill:      # a backfill row", "    if False:      # a backfill row"),
    ("I3 a document not in HEAD authorises", "    if head.returncode != 0:\n        raise", "    if False:\n        raise"),
    ("I3 the working-tree document is read", "    text = head.stdout",
     '    text = (repo_root / authority).read_text(encoding="utf-8") if (repo_root / authority).exists() else ""'),
    ("I3 a proposed document authorises", '    if not status or status.group(1).lower() != "accepted":\n        raise', "    if False:\n        raise"),
    ("I3 no wait at all", WAIT_CHECK, "    if on is None:"),
    ("I3 a document accepted the same day authorises", WAIT_CHECK,
     WAIT_CHECK.replace("on.date() >= local.date()", "on.date() > local.date()")),
    ("I3 the twelve hours are not checked", WAIT_CHECK, WAIT_CHECK.replace(" or local - on < WAIT", "")),
    ("I3 the Status line's amendments are ignored", "((_DATE_LINE, _ACCEPTED_ON), (_STATUS_LINE, _DATED))",
     "((_DATE_LINE, _ACCEPTED_ON),)"),
    ("I3 the first date counts, not the latest", "    return max(found) if found else None", "    return found[0] if found else None"),
    ("I3 a date without a time counts from midnight", "t or '23:59'", "t or '00:00'"),
    ("I3 a caller's via is trusted", "    if via != current_script(repo_root):\n        raise", "    if False:\n        raise"),
    ("I3 a prose mention authorises", LINE_CHECK, "    if via not in text:"),
    ("I3 the set is not matched", LINE_CHECK,
     "    if not any(t2 in allowed and v == via and b == batch for _, t2, v, b in named):"),
    ("I3 the batch is not matched", LINE_CHECK,
     "    if not any(s2 == held_out_set and t2 in allowed and v == via for s2, t2, v, _ in named):"),
    ("I3 a scores line does not cover labels", '    allowed = {tier} | ({"scores"} if tier == "labels" else set())',
     "    allowed = {tier}"),
    ("I3 labels need no document", '    if tier == "structure":\n        return\n    head', '    if tier != "scores":\n        return\n    head'),
    ("I3 an interactive session may look", '    if not f:\n        raise LedgerError("a look must be made by a script',
     '    if False:\n        raise LedgerError("a look must be made by a script'),
    ("I3 a look may be dated after the clock", "    if now is not None and now > clock:", "    if False:"),
    ("I4 an incidental look may claim authority", '        if not authority.startswith("none"):\n            raise', "        if False:\n            raise"),
    ("I4 backfill accepts looks after the cutoff", "    if backfill and when >= BACKFILL_BEFORE:", "    if False:"),
    ("I4 backfill skips the figure filter", "    if _FIGURE.search(what):", "    if _FIGURE.search(what) and not backfill:"),
    ("I4 backfill runs the authority check", "    if not backfill:      # a backfill row", "    if True:      # a backfill row"),
    ("I5 append-only check always passes", "    return working.startswith(committed)", "    return True"),
    ("I5 append-only check only compares lengths", "    return working.startswith(committed)", "    return len(working) >= len(committed)"),
    ("I5 dirty check ignores HEAD", '    committed = head.stdout if head.returncode == 0 else ""', '    committed = ""'),
    ("I6 default splits hand out the test window", '    return {"val": Window(config["val_min"], config["val_max"])}',
     '    return {"val": Window(config["val_min"], config["val_max"]), "test": Window(config["test_min"], config["test_max"])}'),
    ("I6 default load keeps the official masks", "    data = {k: v for k, v in raw.items() if k not in OFFICIAL_MASKS}", "    data = dict(raw)"),
    ("I6 default load keeps the train and val masks", 'OFFICIAL_MASKS = ("train_mask", "val_mask", "test_mask")',
     'OFFICIAL_MASKS = ("test_mask",)'),
    ("I6 default load keeps later labels", HIDE, '    data["y"] = list(raw["y"])'),
    ("I6 labels hidden only after the test window", HIDE, HIDE.replace('config["val_max"]', 'config["test_max"]')),
    ("I6 the hidden value is a real label", "HIDDEN = -1 ", "HIDDEN = 2 "),
    ("I6 labelled_mask built from the raw labels (round 4's B1)", '    data["labelled_mask"] = labelled(data["y"])',
     '    data["labelled_mask"] = labelled(raw["y"])'),
    ("I7 opening the window records nothing", WINDOW_OPEN, ""),
    ("I7 window labels loaded before the look is recorded", WINDOW_OPEN, "    _y = load_raw()\n" + WINDOW_OPEN),
    ("I7 a structure open of the window returns labels", '    if tier == "structure":\n        return window, None',
     "    if False:\n        return window, None"),
    ("I7 putting labels back leaves labelled_mask hidden", '    return {**data, "y": list(y), "labelled_mask": labelled(y)}',
     '    return {**data, "y": list(y)}'),
    ("I8 the official track opens at structure", '    if tier == "structure":\n        raise LedgerError("the official masks',
     '    if False:\n        raise LedgerError("the official masks'),
    ("I8 the official track loads before recording", OFFICIAL_OPEN,
     "    raw = load_raw()\n" + OFFICIAL_OPEN.replace("    raw = load_raw()\n", "")),
    ("I8 the official track returns hidden labels", '    return {"y": list(raw["y"]), **{m: raw[m] for m in OFFICIAL_MASKS}}',
     '    return {"y": [HIDDEN] * len(raw["y"]), **{m: raw[m] for m in OFFICIAL_MASKS}}'),
    ("I8 an opener takes the caller's via", "                via=current_script(repo_root), authority=authority, batch=batch,",
     '                via="scripts/run_gate3.py", authority=authority, batch=batch,'),
    ("I9 integrity check always passes", "    return bool(PUBLISHED) and all(", "    return True or all("),
    ("I9 no pinned totals passes", "    return bool(PUBLISHED) and all(", "    return all("),
    ("I10 disclosure ignores overlapping sets", "    sets = {held_out_set} | HELD_OUT_SETS[held_out_set]", "    sets = {held_out_set}"),
    ("I10 disclosure includes the batch's own row", 'and r["timestamp_utc"] < before]', 'and r["timestamp_utc"] <= before]'),
    ("I10 disclosure keeps file order", '    seen.sort(key=lambda r: r["timestamp_utc"])\n', ""),
    ("I10 disclosure starts at the last batch row", "    before = min(starts)", "    before = max(starts)"),
    ("I10 disclosure of an unopened batch is empty", "    if not starts:\n        raise", "    if not starts:\n        return []\n        raise"),
    ("I10 a batch label may pass between authorities", 'any(r["batch"] == batch and r["authority"] != authority',
     'any(False and r["batch"] == batch'),
    ("I11 backstop misses the PyG loader", 'RAW_CALLS = {"DGraphFin": "PyG loader", ', "RAW_CALLS = {"),
    ("I11 backstop misses hand-built windows", ' "TemporalSplit": "hand-built window",', ""),
    ("I11 backstop misses the raw loader", '             "_load_raw": "the raw loader"}', "             }"),
    ("I11 backstop ignores imports", 'RAW_IMPORTS = {"DGraphFin": "PyG loader", "_load_raw": "the raw loader"}', "RAW_IMPORTS = {}"),
    ("I11 backstop ignores aliases", "                fn = alias.get(fn, fn)\n", ""),
    ("I11 backstop misses store paths", "            elif _store_path(node):", "            elif False:"),
    ("I11 backstop misses joined store paths", "    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):", "    if False:"),
    ("I11 backstop joins paths in walk order", "sorted(strs, key=lambda n: (n.lineno, n.col_offset))", "strs"),
    ("I11 backstop misses mask attributes", "isinstance(node, ast.Attribute) and _mask_name(node.attr)", "False"),
    ("I11 backstop misses mask subscripts", "isinstance(node.slice.value, str) and _mask_name(node.slice.value)", "False"),
    ("I11 backstop sees only the test mask", '    return s.endswith("_mask") and s not in', '    return s.endswith("test_mask") and s not in'),
    ("I11 backstop flags derived masks", 'and s not in ("labelled_mask", "train_seed_mask", "window_target_mask")', "and True"),
    ("I11 backstop misses score files", "        if reads_scores and not opens:", "        if False:"),
    ("I11 backstop misses hand-set authority and time", "any(k.arg in HAND_KEYWORDS for k in node.keywords)", "False"),
    ("I11 backstop misses now=", HAND, 'HAND_KEYWORDS = ("incidental", "backfill")'),
    ("I11 backstop misses backfill claims", HAND, 'HAND_KEYWORDS = ("incidental", "now")'),
    ("I11 backstop flags the hand path", "if name != HAND_PATH and any(", "if any("),
    ("I11 backstop exempts test-named files again", EXEMPT, '                or Path(name).name.startswith("test_")\n' + EXEMPT),
    ("I11 backstop scans prototypes", EXEMPT, "                or False):"),
    ("I11 backstop exempts any prototype folder", EXEMPT, '                or "/prototype/" in name):'),
    ("I11 backstop skips notebooks", 'SCOPE = ("scripts/", "adapters/dgf1/", "notebooks/")', 'SCOPE = ("scripts/", "adapters/dgf1/")'),
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
