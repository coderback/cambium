"""The ledger's hand path (ADR-021): unplanned looks, written as soon as they are known.

    python scripts/heldout_ledger.py record --set <set> --tier <tier> --what "<what>" --by <who> \\
        --authority "none: <why>" --batch <label> --incidental [--when <ISO time with offset>]
    python scripts/heldout_ledger.py disclose --set <set> [--batch <label>]

Without --batch, disclose lists every look so far, for a document whose own batch has not opened.
A planned look goes through the adapter's accessors, never through here. DGF-1's looks from before
ADR-021 were entered through this path too, and its backfill flag was then removed (clause 9).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from adapters.dgf1.heldout import LedgerError, disclosure, read_ledger, record_look, seen_so_far  # noqa: E402


def _when(text: str) -> datetime:
    when = datetime.fromisoformat(text)
    if when.tzinfo is None:
        raise argparse.ArgumentTypeError(f"{text} has no UTC offset")
    return when


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="append one look to the ledger")
    for flag in ("--set", "--tier", "--what", "--by", "--authority", "--batch"):
        rec.add_argument(flag, required=True)
    rec.add_argument("--incidental", action="store_true", help="a look with no authority: 'none: <why>'")
    rec.add_argument("--when", type=_when, help="when the look happened, if earlier than now")
    dis = sub.add_parser("disclose", help="print what was seen, before a batch or so far")
    dis.add_argument("--set", required=True)
    dis.add_argument("--batch", help="list only what came before this batch's first row")
    args = ap.parse_args(argv)
    try:
        if args.command == "disclose":
            rows = read_ledger()
            lines = disclosure(rows, args.set, args.batch) if args.batch else seen_so_far(rows, args.set)
            print("\n".join(lines) or "nothing seen")
        else:
            row = record_look(args.set, args.tier, args.what, args.by, args.authority, args.batch,
                              incidental=args.incidental, now=args.when)
            print(f"recorded: {row['timestamp_utc']} · {row['held_out_set']} · {row['tier']} · {row['what']}")
    except LedgerError as why:
        print(f"refused: {why}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
