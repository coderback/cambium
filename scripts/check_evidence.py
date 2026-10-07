"""Mechanical evidence check for a document before it goes to the researcher (audit 2026-10-07 §4b
item 2; ADR-017).

    python scripts/check_evidence.py decisions/ADR-018-something.md   # the documents named
    python scripts/check_evidence.py --all                             # every tracked Markdown file
    python scripts/check_evidence.py --all --no-ignore                 # plus docs/ and CLAUDE.md files

Five checks. Each finding is printed as ``path:line: [check] reason``, never with the text it is
about: cited lines can be held-out statistics, so the checker reads them and the report does not
repeat them.

* **citations** — every ``path:N`` or ``path:N-M`` names a file that exists and lines inside it.
  Shorthands are resolved too: ``ADR-012:334``, ``GATE-DGF1-1.md:9``, ``doc-02:60``,
  ``research-plan:128``, and a bare ``:277`` after a file cited earlier in the same paragraph.
* **quotes** — a block quote closed by ``> — `path:N-M``` matches those lines, ignoring whitespace,
  ``>`` markers and emphasis; ``…`` marks an elision.
* **commits** — every backticked hex token of 7 to 40 characters names a commit in this repository
  (or in one passed with ``--extra-git-dir``), or is the 8-character suffix of a run id written in a
  tracked file.
* **verified** — a sentence asserting that something was verified or measured names a commit, a
  run id, or a saved measurement under ``notebooks/measurements/`` (ADR-017 item 3), and every
  backticked code identifier in it exists in some committed ``.py`` file.
* **statuses** — a phrase like "ADR-014 is accepted" matches that ADR's ``**Status:**`` line.

The verified and status checks skip ``notebooks/lab/`` (dated logs, true when written), quotations
(a block quote closed by an attribution line; the quoted document is checked on its own) and text
inside double quotes. A ``>`` block with no attribution is the document's own text and is checked.
Files under ``data/`` and ``experiments/`` are never opened, however a citation reaches them, and
citations into them are not checked. A bare file name shared by several files is not checked either.
Exit status: 0 when nothing is found, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

NEVER_OPEN = ("data/", "experiments/")
HISTORICAL = ("notebooks/lab/",)
MEASUREMENTS = "notebooks/measurements/"
CHECKS = ("citations", "quotes", "commits", "verified", "statuses")
STATUS_WORDS = ("accepted", "proposed", "withdrawn", "rejected", "superseded")

_BACKTICK = re.compile(r"`([^`\n]+)`")
_PATH_CITE = re.compile(r"^(?P<path>[\w./-]+\.(?:md|py|yaml|yml|toml|txt|json|cfg|ini)):(?P<a>\d+)(?:-(?P<b>\d+))?$")
_BARE_CITE = re.compile(r"^:(?P<a>\d+)(?:-(?P<b>\d+))?$")
_SHORTHANDS = (
    (re.compile(r"\bADR-(?P<key>\d{3}):(?P<a>\d+)(?:-(?P<b>\d+))?"), "adr"),
    (re.compile(r"\b(?P<key>GATE-[A-Z0-9]+-\d+)(?:\.md)?:(?P<a>\d+)(?:-(?P<b>\d+))?"), "gate"),
    (re.compile(r"\bdoc-(?P<key>\d{2}):(?P<a>\d+)(?:-(?P<b>\d+))?"), "doc"),
    (re.compile(r"\b(?P<key>research-plan):(?P<a>\d+)(?:-(?P<b>\d+))?"), "plan"),
)
_ATTRIBUTION = re.compile(r"^\s*>\s*—\s*`(?P<cite>[^`]+)`\s*$")
UNCHECKED = "unchecked"   # a bare file name several files share, or a file under data/ or experiments/

_HEX = re.compile(r"^[0-9a-f]{7,40}$")
_RUN_ID = re.compile(r"\b[a-z0-9]+-\d{8}T\d{6}Z-[0-9a-f]{8}\b")
_CLAIM = re.compile(r"\b(verified|measured)\b", re.IGNORECASE)
_OBLIGATION = re.compile(
    r"\b(must|should|will|would|can|cannot|could|to|never|not|until|once|if|when|being)\s+(?:be\s+|been\s+)?"
    r"(?:\w+\s+)?(verified|measured)\b", re.IGNORECASE)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?$")
_STATUS_PHRASE = re.compile(
    r"\bADR-(?P<num>\d{3})(?:\s+(?:is|remains|stays)|,)?\s+\(?(?P<status>" + "|".join(STATUS_WORDS) + r")\b")
_QUOTED = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z*`(\"“])")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    check: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.check}] {self.reason}"


# -- repository access -------------------------------------------------------------------------------
def _git(root: Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(["git", "-C", str(root), *args], input=stdin, capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout


def tracked_markdown(root: Path) -> list[str]:
    return [p for p in _git(root, "ls-files", "*.md").splitlines() if p]


def ignored_markdown(root: Path) -> list[str]:
    """The governing documents the public repository ignores: ``docs/`` and the CLAUDE.md files."""
    found = sorted(str(p.relative_to(root)).replace("\\", "/") for p in (root / "docs").glob("*.md"))
    found += [name for name in ("CLAUDE.md", "gbe/CLAUDE.md") if (root / name).is_file()]
    return found


def existing_commits(git_args: list[str], tokens: set[str]) -> set[str]:
    """The tokens that name a commit, checked in one ``git cat-file --batch-check`` call.

    ``git_args`` selects the repository: ``["-C", root]``, or ``["--git-dir", path]`` for one whose
    work tree is elsewhere (the private governance repository).
    """
    if not tokens:
        return set()
    ordered = sorted(tokens)
    out = subprocess.run(["git", *git_args, "cat-file", "--batch-check"], capture_output=True, text=True,
                         encoding="utf-8", check=True,
                         input="".join(f"{t}^{{commit}}\n" for t in ordered)).stdout
    return {t for t, line in zip(ordered, out.splitlines()) if not line.endswith("missing")
            and "ambiguous" not in line}


def run_id_suffixes(root: Path) -> set[str]:
    """The 8-hex suffix of every run id written in a tracked file outside ``data/`` and ``experiments/``.

    Documents abbreviate a run id to that suffix (for example `1acd5067`), which looks like a commit.
    """
    proc = subprocess.run(["git", "-C", str(root), "grep", "-h", "-o", "-E", r"T[0-9]{6}Z-[0-9a-f]{8}",
                           "--", ":!data", ":!experiments"], capture_output=True, text=True, encoding="utf-8")
    return {m[-8:] for m in proc.stdout.split()}


def tracked_files(root: Path) -> list[str]:
    return [p for p in _git(root, "ls-files").splitlines() if p]


def committed_python_text(root: Path) -> str:
    """Every line ever added to a committed ``.py`` file, as one string, for identifier lookups."""
    return _git(root, "log", "--all", "-p", "--format=", "--", "*.py")


def adr_statuses(root: Path) -> dict[str, str]:
    statuses = {}
    for p in sorted((root / "decisions").glob("ADR-[0-9][0-9][0-9]-*.md")):
        m = re.search(r"^\*\*Status:\*\*\s*(\w+)", p.read_text(encoding="utf-8"), re.MULTILINE)
        if m:
            statuses[p.name[4:7]] = m.group(1).lower()
    return statuses


def _resolve_shorthand(root: Path, kind: str, key: str) -> Path | None:
    patterns = {"adr": ("decisions", f"ADR-{key}-*.md"), "gate": ("gates", f"{key}.md"),
                "doc": ("docs", f"{key}-*.md"), "plan": ("docs", "research-plan-*.md")}
    folder, pattern = patterns[kind]
    hits = sorted((root / folder).glob(pattern))
    return hits[0] if len(hits) == 1 else None


def _never_opened(rel: str) -> bool:
    return rel.replace("\\", "/").startswith(NEVER_OPEN)


# -- document structure ------------------------------------------------------------------------------
def _lines_outside_code(text: str) -> list[tuple[int, str]]:
    """``(line number, line)`` for every line outside fenced code blocks."""
    kept, fenced = [], False
    for n, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            kept.append((n, line))
    return kept


def _is_quote(line: str) -> bool:
    return line.lstrip().startswith(">")


def _split_quotes(lines: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """``(prose, quotation)`` lines.

    A block quote closed by an attribution line (``> — `path:N` ``) quotes another document: its
    lines go to ``quotation``. Any other ``>`` block is the document's own text (a note or a
    callout), so it stays in ``prose`` with its ``>`` markers removed.
    """
    prose, quotation, block = [], [], []

    def flush() -> None:
        attributed = any(_ATTRIBUTION.match(l) for _, l in block)
        for n, l in block:
            if attributed:
                quotation.append((n, l))
            else:
                prose.append((n, re.sub(r"^\s*>\s?", "", l)))
        block.clear()

    for n, line in lines:
        if _is_quote(line):
            block.append((n, line))
            continue
        flush()
        prose.append((n, line))
    flush()
    return sorted(prose), sorted(quotation)


def _paragraphs(lines: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Prose paragraphs: split at blank lines, headings, list items and table rows."""
    paras: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for n, line in lines:
        s = line.strip()
        starts_new = not s or s.startswith("#") or s.startswith("|") or re.match(r"^([-*+]|\d+\.)\s", s)
        if starts_new and current:
            paras.append(current)
            current = []
        if s:
            current.append((n, line))
            if s.startswith("#") or s.startswith("|"):
                paras.append(current)
                current = []
    if current:
        paras.append(current)
    return paras


def _sentences(para: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """``(line number where the sentence starts, sentence)`` for one paragraph."""
    text, starts = "", []
    for n, line in para:
        starts.append((len(text), n))
        text += line.strip() + " "
    out, pos = [], 0
    for piece in _SENTENCE_END.split(text):
        line_no = max(n for offset, n in starts if offset <= pos)
        out.append((line_no, piece))
        pos = text.find(piece, pos) + len(piece)
    return out


def _normalise(text: str) -> str:
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    text = re.sub(r"[*_`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# -- the checks --------------------------------------------------------------------------------------
class Checker:
    def __init__(self, root: Path, extra_git_dirs: tuple[str, ...] = ()):
        self.root = root
        self.extra_git_dirs = extra_git_dirs
        self._line_counts: dict[Path, int] = {}
        self._python_text: str | None = None
        self._suffixes: set[str] | None = None
        self.statuses = adr_statuses(root)
        self.by_basename: dict[str, list[str]] = {}
        for rel in tracked_files(root) + ignored_markdown(root):     # names only; nothing is opened here
            self.by_basename.setdefault(rel.rsplit("/", 1)[-1], []).append(rel)

    def _line_count(self, path: Path) -> int:
        if path not in self._line_counts:
            self._line_counts[path] = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        return self._line_counts[path]

    def _python_history(self) -> str:
        if self._python_text is None:
            self._python_text = committed_python_text(self.root)
        return self._python_text

    def _cited_file(self, doc: Path, rel: str) -> Path | None | str:
        """The cited file; ``None`` when nothing matches; ``UNCHECKED`` for a bare file name that
        several files share, or for any file under ``data/`` or ``experiments/`` (never opened)."""
        for base in (self.root, doc.parent):
            candidate = (base / rel).resolve()
            if candidate.is_file():
                return self._openable(candidate)
        if "/" not in rel:
            matches = self.by_basename.get(rel, [])
            if len(matches) == 1:
                return self._openable((self.root / matches[0]).resolve())
            if len(matches) > 1:
                return UNCHECKED
        return None

    def _openable(self, path: Path) -> Path | str:
        try:
            rel = path.relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path                                     # outside the repository
        return UNCHECKED if _never_opened(rel) else path

    def citations(self, rel_doc: str, lines: list[tuple[int, str]]) -> list[Finding]:
        doc = self.root / rel_doc
        findings = []
        prose, quotation = _split_quotes(lines)
        for para in _paragraphs(prose) + [[q] for q in quotation]:
            last: Path | None = None
            for n, line in para:
                cites: list[tuple[Path | None, str, int, int | None]] = []
                # In reading order, so a bare `:N` refers to the file cited just before it.
                for token in _BACKTICK.findall(line):
                    m = _PATH_CITE.match(token)
                    if m:
                        if _never_opened(m["path"]) or "\\" in token:
                            continue
                        path = self._cited_file(doc, m["path"])
                        if path == UNCHECKED:
                            last = None
                            continue
                        cites.append((path, m["path"], int(m["a"]), int(m["b"]) if m["b"] else None))
                        last = path or last
                        continue
                    m = _BARE_CITE.match(token)
                    if m and last is not None:
                        cites.append((last, last.relative_to(self.root.resolve()).as_posix(), int(m["a"]),
                                      int(m["b"]) if m["b"] else None))
                        continue
                    shorthand = self._shorthand_cites(token)
                    cites += shorthand
                    last = next((c[0] for c in reversed(shorthand) if c[0] is not None), last)
                cites += self._shorthand_cites(_BACKTICK.sub(" ", line))
                for path, label, a, b in cites:
                    if path is None:
                        findings.append(Finding(rel_doc, n, "citations", f"cites {label}, which does not exist"))
                        continue
                    count = self._line_count(path)
                    hi = b if b is not None else a
                    if not (1 <= a <= hi <= count):
                        findings.append(Finding(rel_doc, n, "citations",
                                                f"cites {label}:{a}{'-' + str(b) if b else ''}, "
                                                f"but the file has {count} lines"))
        return findings

    def _shorthand_cites(self, text: str) -> list[tuple[Path | None, str, int, int | None]]:
        cites = []
        for pattern, kind in _SHORTHANDS:
            for m in pattern.finditer(text):
                cites.append((_resolve_shorthand(self.root, kind, m["key"]), m.group(0).split(":")[0],
                              int(m["a"]), int(m["b"]) if m["b"] else None))
        return cites

    def quotes(self, rel_doc: str, lines: list[tuple[int, str]]) -> list[Finding]:
        doc = self.root / rel_doc
        findings, block = [], []
        for n, line in lines:
            if not _is_quote(line):
                block = []
                continue
            m = _ATTRIBUTION.match(line)
            if not m:
                block.append(line)
                continue
            quoted, block = "\n".join(block), []
            cite = _PATH_CITE.match(m["cite"])
            if not cite:
                findings.append(Finding(rel_doc, n, "quotes", "attribution is not a path:line citation"))
                continue
            if _never_opened(cite["path"]):
                continue
            path = self._cited_file(doc, cite["path"])
            if path is None or path == UNCHECKED:
                continue                                    # reported, or deliberately unchecked
            a = int(cite["a"])
            b = int(cite["b"]) if cite["b"] else a
            source = _normalise("\n".join(path.read_text(encoding="utf-8").splitlines()[a - 1:b]))
            if not _fragments_in_order(_normalise(quoted), source):
                findings.append(Finding(rel_doc, n, "quotes",
                                        f"the quote above does not match {cite['path']}:{cite['a']}"
                                        f"{'-' + cite['b'] if cite['b'] else ''}"))
        return findings

    def commits(self, rel_doc: str, lines: list[tuple[int, str]]) -> list[Finding]:
        tokens = [(n, t) for n, line in lines for t in _BACKTICK.findall(line)
                  if _HEX.match(t) and re.search(r"[a-f]", t)]
        wanted = {t for _, t in tokens}
        known = existing_commits(["-C", str(self.root)], wanted)
        for git_dir in self.extra_git_dirs:
            known |= existing_commits(["--git-dir", git_dir], wanted - known)
        if wanted - known:
            if self._suffixes is None:
                self._suffixes = run_id_suffixes(self.root)
            known |= wanted & self._suffixes
        return [Finding(rel_doc, n, "commits", f"`{t}` is neither a commit nor a recorded run id")
                for n, t in tokens if t not in known]

    def verified(self, rel_doc: str, lines: list[tuple[int, str]]) -> list[Finding]:
        if rel_doc.startswith(HISTORICAL):
            return []
        findings = []
        prose, _ = _split_quotes(lines)
        for para in _paragraphs(prose):
            for n, sentence in _sentences(para):
                plain = _QUOTED.sub(" ", sentence)
                claims = [m for m in _CLAIM.finditer(plain)]
                asserted = [m for m in claims if not _OBLIGATION.search(plain[:m.end()][-60:])
                            and not _proper_noun(plain, m)]
                if not asserted:
                    continue
                ticks = _BACKTICK.findall(plain)
                has_commit = any(_HEX.match(t) and re.search(r"[a-f]", t) for t in ticks)
                # ADR-017 item 3: a measurement is evidenced by its saved script and output.
                has_measurement = any(t.startswith(MEASUREMENTS) for t in ticks)
                if not (has_commit or has_measurement or _RUN_ID.search(plain)):
                    findings.append(Finding(rel_doc, n, "verified",
                                            f"says '{asserted[0].group(1).lower()}' but names no commit or run id"))
                for t in ticks:
                    ident = t.removesuffix("()")
                    if (_IDENTIFIER.match(t) and ("_" in ident or "." in ident)
                            and not _HEX.match(t) and ident not in self._python_history()):
                        findings.append(Finding(rel_doc, n, "verified",
                                                f"`{t}` appears in no committed .py file"))
        return findings

    def statuses_check(self, rel_doc: str, lines: list[tuple[int, str]]) -> list[Finding]:
        if rel_doc.startswith(HISTORICAL):
            return []
        findings = []
        prose, _ = _split_quotes(lines)
        for n, line in prose:
            if line.lstrip().startswith("**Status:**"):
                continue
            for m in _STATUS_PHRASE.finditer(_QUOTED.sub(" ", line)):
                actual = self.statuses.get(m["num"])
                if actual is not None and actual != m["status"].lower():
                    findings.append(Finding(rel_doc, n, "statuses",
                                            f"says ADR-{m['num']} is {m['status']}, but its Status is {actual}"))
        return findings

    def check(self, rel_doc: str, checks: tuple[str, ...] = ()) -> list[Finding]:
        text = (self.root / rel_doc).read_text(encoding="utf-8")
        lines = _lines_outside_code(text)
        run = {"citations": self.citations, "quotes": self.quotes, "commits": self.commits,
               "verified": self.verified, "statuses": self.statuses_check}
        return [f for name in (checks or CHECKS) for f in run[name](rel_doc, lines)]


def _proper_noun(text: str, m: re.Match) -> bool:
    """A capitalised word straight after another word is a name ("SWE-bench Verified"), not a claim."""
    before = text[:m.start()].rstrip()
    return m.group(1)[0].isupper() and bool(before) and (before[-1].isalnum() or before[-1] == "-")


def _fragments_in_order(quote: str, source: str) -> bool:
    """True when every piece of ``quote`` between elisions appears in ``source``, in order."""
    pos = 0
    for fragment in (f.strip(" .") for f in re.split(r"…|\.\.\.", quote)):
        if not fragment:
            continue
        found = source.find(fragment, pos)
        if found < 0:
            return False
        pos = found + len(fragment)
    return True


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("documents", nargs="*", help="Markdown files, relative to the repository root")
    ap.add_argument("--all", action="store_true", help="check every tracked Markdown file")
    ap.add_argument("--no-ignore", action="store_true", help="also check docs/ and the CLAUDE.md files")
    ap.add_argument("--checks", default=",".join(CHECKS),
                    help=f"comma-separated subset of: {', '.join(CHECKS)} (default: all)")
    ap.add_argument("--extra-git-dir", action="append", default=[],
                    help="another repository whose commits may be cited (the governance repository)")
    args = ap.parse_args(argv)
    checks = tuple(c.strip() for c in args.checks.split(",") if c.strip())
    unknown = [c for c in checks if c not in CHECKS]
    if unknown:
        ap.error(f"unknown check(s): {', '.join(unknown)}")

    docs = list(args.documents)
    if args.all:
        docs += tracked_markdown(root)
    if args.no_ignore:
        docs += ignored_markdown(root)
    docs = sorted({d.replace("\\", "/") for d in docs if not _never_opened(d)})
    if not docs:
        ap.error("name at least one document, or pass --all")

    checker = Checker(root, tuple(args.extra_git_dir))
    findings = [f for d in docs for f in checker.check(d, checks)]
    for f in findings:
        print(f)
    print(f"{len(findings)} finding(s) in {len(docs)} document(s).")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
