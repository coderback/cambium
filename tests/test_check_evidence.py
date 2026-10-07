"""The evidence checker (audit 2026-10-07 §4b item 2; ADR-017) on a temporary repository.

Each check is shown to fire on the defect it exists for and to stay quiet on the correct form. The
two passages the audit named (ADR-013:298, a status phrase for a withdrawn ADR; ADR-015:793, a
"verified" claim with no commit behind it) are reproduced here as fixtures, so the tests do not
depend on those documents staying unfixed. Two guards are tested directly: files under ``data/``
and ``experiments/`` are never opened, and a report never repeats the text it is about.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from check_evidence import Checker, main  # noqa: E402

RUN_ID = "dgf1-20260913T232506Z-1acd5067"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "core.autocrlf=false", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A committed repository with two ADRs (one accepted, one withdrawn), code, and cited files."""
    _write(tmp_path, "decisions/ADR-001-kept.md", "# ADR-001\n\n**Status:** accepted\n")
    _write(tmp_path, "decisions/ADR-002-dropped.md", "# ADR-002\n\n**Status:** withdrawn\n")
    _write(tmp_path, "pkg/mod.py", "def log_metrics():\n    return 1\n")
    _write(tmp_path, "notes/source.md", "line one\n   5. Only then the single test\n      batch.\nline four\n")
    _write(tmp_path, "pkg/a/shared.md", "one\n")
    _write(tmp_path, "pkg/b/shared.md", "one\n")
    _write(tmp_path, "experiments/held.md", "SECRET-STAT-0.123\n")
    _write(tmp_path, "notes/runs.md", f"The pilot row is {RUN_ID}.\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _check(repo: Path, text: str, rel: str = "decisions/ADR-009-new.md", checks=()) -> list:
    _write(repo, rel, text)
    return Checker(repo).check(rel, tuple(checks))


def _kinds(findings) -> list[str]:
    return [f.check for f in findings]


# -- citations -----------------------------------------------------------------------------------
def test_a_citation_inside_the_file_passes(repo):
    assert _check(repo, "See `notes/source.md:2-3`.\n", checks=["citations"]) == []


def test_a_citation_past_the_end_of_the_file_is_flagged(repo):
    (f,) = _check(repo, "See `notes/source.md:3-9`.\n", checks=["citations"])
    assert f.line == 1 and "has 4 lines" in f.reason


def test_a_citation_of_a_missing_file_is_flagged(repo):
    (f,) = _check(repo, "See `notes/missing.md:1`.\n", checks=["citations"])
    assert "does not exist" in f.reason


def test_adr_shorthand_resolves_and_is_range_checked(repo):
    assert _check(repo, "As ADR-001:3 says.\n", checks=["citations"]) == []
    (f,) = _check(repo, "As ADR-001:30 says.\n", checks=["citations"])
    assert "ADR-001:30" in f.reason


def test_a_bare_line_refers_to_the_file_cited_just_before_it_on_the_same_line(repo):
    # Regression: `:3` once resolved against the previous line's file.
    text = "See `decisions/ADR-001-kept.md:1`\nand `pkg/mod.py:1` then `notes/source.md:2` and `:3`.\n"
    assert _check(repo, text, checks=["citations"]) == []
    (f,) = _check(repo, "See `pkg/mod.py:1` then `notes/source.md:2` and `:9`.\n", checks=["citations"])
    assert "notes/source.md:9" in f.reason


def test_a_bare_file_name_resolves_when_unique_and_is_skipped_when_shared(repo):
    (f,) = _check(repo, "See `mod.py:5`.\n", checks=["citations"])
    assert "has 2 lines" in f.reason
    assert _check(repo, "See `shared.md:99`.\n", checks=["citations"]) == []


def test_files_under_experiments_are_never_opened(repo, monkeypatch):
    real_read = Path.read_text

    def guarded(self, *args, **kwargs):
        assert "experiments" not in self.parts, f"opened {self}"
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    text = "See `experiments/held.md:99`, `held.md:99` and `../experiments/held.md:99`.\n"
    assert _check(repo, text, rel="notes/new.md", checks=["citations", "quotes"]) == []


# -- quotes --------------------------------------------------------------------------------------
QUOTE_OK = "> 5. **Only** then the single test\n> batch.\n> — `notes/source.md:2-3`\n"


def test_a_quote_matching_its_lines_passes_despite_whitespace_and_emphasis(repo):
    assert _check(repo, QUOTE_OK, checks=["quotes"]) == []


def test_a_quote_with_an_elision_passes(repo):
    text = "> 5. Only … batch.\n> — `notes/source.md:2-3`\n"
    assert _check(repo, text, checks=["quotes"]) == []


def test_a_misquote_is_flagged(repo):
    (f,) = _check(repo, "> 5. Only then a double test\n> — `notes/source.md:2-3`\n", checks=["quotes"])
    assert f.line == 2 and "does not match notes/source.md:2-3" in f.reason


# -- commits -------------------------------------------------------------------------------------
def test_a_real_commit_passes_and_an_invented_one_is_flagged(repo):
    head = _git(repo, "rev-parse", "--short=7", "HEAD").strip()
    assert _check(repo, f"Committed at `{head}`.\n", checks=["commits"]) == []
    (f,) = _check(repo, "Committed at `abc1234`.\n", checks=["commits"])
    assert "`abc1234`" in f.reason


def test_a_run_id_suffix_and_non_hashes_are_not_flagged(repo):
    text = "Pilot `1acd5067`; hash `d825a002…`; count `2048`; word `deadline`.\n"
    assert _check(repo, text, checks=["commits"]) == []


def test_a_commit_in_another_repository_passes_with_extra_git_dir(repo, tmp_path_factory):
    other = tmp_path_factory.mktemp("governance")
    _write(other, "x.md", "x\n")
    _git(other, "init", "-q")
    _git(other, "add", ".")
    _git(other, "commit", "-q", "-m", "x")
    sha = _git(other, "rev-parse", "--short=7", "HEAD").strip()
    _write(repo, "notes/new.md", f"Baseline `{sha}`.\n")
    assert _kinds(Checker(repo).check("notes/new.md", ("commits",))) == ["commits"]
    assert Checker(repo, (str(other / ".git"),)).check("notes/new.md", ("commits",)) == []


# -- verified ------------------------------------------------------------------------------------
def test_the_adr015_pattern_a_verified_claim_with_no_commit_is_flagged(repo):
    # Mirrors ADR-015:793, which rests a verification on a withdrawn ADR and names no commit.
    text = ("The trainers gain a keyword passed to `log_metrics`. ADR-002 verified that channel leaves "
            "the config hash unchanged:\n")
    (f,) = _check(repo, text, checks=["verified"])
    assert f.line == 1 and "names no commit or run id" in f.reason


def test_a_verified_claim_with_a_commit_run_id_or_measurement_passes(repo):
    head = _git(repo, "rev-parse", "--short=7", "HEAD").strip()
    for evidence in (f"`{head}`", RUN_ID, "`notebooks/measurements/timing.txt`"):
        assert _check(repo, f"The hash was verified at {evidence}.\n", checks=["verified"]) == []


def test_obligations_progressives_and_names_are_not_claims(repo):
    text = ("The backup must be verified first. The effects being measured are small. "
            "Never peek at SWE-bench Verified.\n")
    assert _check(repo, text, checks=["verified"]) == []


def test_an_identifier_in_a_verified_claim_must_exist_in_committed_code(repo):
    head = _git(repo, "rev-parse", "--short=7", "HEAD").strip()
    assert _check(repo, f"Measured at `{head}`: `log_metrics` keeps it.\n", checks=["verified"]) == []
    (f,) = _check(repo, f"Measured at `{head}`: `extra_metrics` keeps it.\n", checks=["verified"])
    assert "`extra_metrics`" in f.reason


def test_quotations_and_lab_entries_are_skipped_but_own_notes_are_checked(repo):
    quotation = "> The channel was verified.\n> — `notes/source.md:1`\n"
    assert _check(repo, quotation, checks=["verified"]) == []
    assert _check(repo, "The channel was verified.\n", rel="notebooks/lab/2026-01-01.md",
                  checks=["verified"]) == []
    (f,) = _check(repo, "Intro.\n\n> **Note.** The channel was verified.\n", checks=["verified"])
    assert f.line == 3


# -- statuses ------------------------------------------------------------------------------------
def test_the_adr013_pattern_a_withdrawn_adr_called_accepted_is_flagged(repo):
    # Mirrors ADR-013:298, a note saying "until ADR-014 is accepted" after ADR-014 was withdrawn.
    (f,) = _check(repo, "> **Note.** Until ADR-002 is accepted the bar is unchanged.\n", checks=["statuses"])
    assert f.line == 1 and "ADR-002 is accepted, but its Status is withdrawn" in f.reason


def test_a_true_status_quoted_text_and_the_status_line_are_not_flagged(repo):
    text = ('**Status:** proposed\n\nADR-001 is accepted. The audit said "ADR-002 is accepted" then.\n'
            "> ADR-002 remains accepted.\n> — `notes/source.md:1`\n")
    assert _check(repo, text, checks=["statuses"]) == []


# -- the command ---------------------------------------------------------------------------------
def test_main_exits_one_with_findings_and_zero_without(repo, capsys):
    _write(repo, "notes/bad.md", "See `notes/source.md:40`.\n")
    _write(repo, "notes/good.md", "See `notes/source.md:4`.\n")
    assert main(["notes/bad.md"], root=repo) == 1
    assert "notes/bad.md:1: [citations]" in capsys.readouterr().out
    assert main(["notes/good.md"], root=repo) == 0


def test_main_runs_only_the_named_checks_and_refuses_unknown_ones(repo, capsys):
    _write(repo, "notes/bad.md", "See `notes/source.md:40`. It was verified.\n")
    assert main(["notes/bad.md", "--checks", "verified"], root=repo) == 1
    out = capsys.readouterr().out
    assert "[verified]" in out and "[citations]" not in out
    with pytest.raises(SystemExit):
        main(["notes/bad.md", "--checks", "spelling"], root=repo)


def test_a_report_never_repeats_the_text_it_is_about(repo, capsys):
    _write(repo, "notes/secret.md", "SECRET-STAT-0.456\n")
    _write(repo, "notes/new.md", "> SECRET-STAT-0.999 was the value\n> — `notes/secret.md:1`\n")
    assert main(["notes/new.md"], root=repo) == 1
    out = capsys.readouterr().out
    assert "SECRET-STAT" not in out
