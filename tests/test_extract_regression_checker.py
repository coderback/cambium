"""Guards for the ADR-008 regression checker.

The checker is EXTRACT's acceptance test, so a checker that cannot *fail* is worse than no checker
at all — it would license the refactor while proving nothing. Each test below is written so that
breaking the corresponding behaviour makes it fail.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_extract_regression import (  # noqa: E402
    CRITICAL_PACKAGES,
    _canonical,
    compare,
    environment_drift,
    observed_rows,
)


def _ref(arm="real", seed=0, **metrics):
    return {"arm": arm, "seed": seed, "metrics": metrics or {"illicit_f1": 0.5}}


# --------------------------------------------------------------------------- compare()


def test_identical_metrics_produce_no_mismatch():
    reference = [_ref("real", 0, illicit_f1=0.6629, illicit_auc=0.9197)]
    observed = {("real", 0): {"illicit_f1": 0.6629, "illicit_auc": 0.9197}}
    mismatches, missing = compare(reference, observed)
    assert mismatches == [] and missing == []


def test_one_ulp_drift_is_a_mismatch():
    """ADR-008's bar is exact equality. A 1-ULP change passes `math.isclose` and is still a
    regression — the arithmetic moved. This is the test that pins the strictness."""
    value = 0.6923937360178971
    drifted = math.nextafter(value, math.inf)
    assert math.isclose(value, drifted)  # an approximate bar would accept this

    reference = [_ref("real", 4, illicit_f1=value)]
    observed = {("real", 4): {"illicit_f1": drifted}}
    mismatches, missing = compare(reference, observed)

    assert missing == []
    assert len(mismatches) == 1
    assert mismatches[0].metric == "illicit_f1"
    assert mismatches[0].reference == value and mismatches[0].observed == drifted


def test_absent_observed_row_is_reported_missing_not_passed():
    """A reference row with no counterpart must never silently count as agreement."""
    reference = [_ref("scrambled", 3, illicit_f1=0.5148)]
    mismatches, missing = compare(reference, observed={})
    assert mismatches == []
    assert missing == ["scrambled/seed3"]


def test_absent_metric_on_a_present_row_is_reported_missing():
    reference = [_ref("real", 0, illicit_f1=0.5, illicit_auc=0.9)]
    observed = {("real", 0): {"illicit_f1": 0.5}}          # auc not reported
    mismatches, missing = compare(reference, observed)
    assert mismatches == []
    assert missing == ["real/seed0:illicit_auc"]


def test_extra_observed_metric_is_not_a_regression():
    """ADR-008: post-refactor rows carry `illicit_auprc` that pre-ADR-007 references lack.
    Comparison is over the metrics the manifest recorded, so a new key must pass."""
    reference = [_ref("real", 0, illicit_f1=0.5)]
    observed = {("real", 0): {"illicit_f1": 0.5, "illicit_auprc": 0.31}}
    mismatches, missing = compare(reference, observed)
    assert mismatches == [] and missing == []


def test_mismatch_renders_both_values_at_full_precision():
    """The report has to be actionable: a rounded diff hides exactly the drift being reported."""
    reference = [_ref("real", 0, illicit_f1=0.6923937360178971)]
    observed = {("real", 0): {"illicit_f1": 0.6923937360178972}}
    (mm,), _ = compare(reference, observed)
    assert "0.6923937360178971" in str(mm) and "0.6923937360178972" in str(mm)


# --------------------------------------------------------------------------- environment


def test_canonical_normalises_pyg_lib_spelling():
    """Regression test for a real false positive: `pip freeze` writes `pyg_lib` while the docs and
    this module say `pyg-lib`. Comparing raw strings reported drift on a package that had not
    moved — and because drift makes the check inconclusive by design, the acceptance test could
    never have passed."""
    assert _canonical("pyg_lib") == _canonical("pyg-lib") == "pyg-lib"
    assert _canonical("scikit_learn") == _canonical("scikit-learn") == "scikit-learn"
    assert _canonical("torch-geometric") == "torch-geometric"


def test_live_environment_matches_the_committed_pin():
    """The pin is the reference stack; if this fails, either a library moved or the pin is stale."""
    pin = REPO_ROOT / "experiments" / "extract_reference_env.txt"
    assert pin.exists()
    assert environment_drift(pin) == []


def test_environment_drift_detects_a_changed_version(tmp_path):
    """Teeth: fabricate a pin claiming a different torch and confirm it is flagged."""
    pin = REPO_ROOT / "experiments" / "extract_reference_env.txt"
    fake = tmp_path / "pin.txt"
    fake.write_text(
        pin.read_text(encoding="utf-8").replace("torch==2.13.0+cu126", "torch==1.0.0"),
        encoding="utf-8",
    )
    drift = environment_drift(fake)
    assert any(d.startswith("torch:") and "1.0.0" in d for d in drift), drift


def test_every_critical_package_is_present_in_the_pin():
    """A package missing from the pin cannot be verified, which is itself reported as drift."""
    pin = REPO_ROOT / "experiments" / "extract_reference_env.txt"
    unverifiable = [d for d in environment_drift(pin) if "absent from the pin" in d]
    assert unverifiable == [], f"pin does not cover {CRITICAL_PACKAGES}: {unverifiable}"


# --------------------------------------------------------------------------- registry selection

HEAD = "c" * 40
OLD = "0" * 40


def _row(arm="gcn", seed=0, f1=0.5, experiment="extract_check", commit=HEAD, dirty="false",
         notes="", deterministic=True, run_id=None):
    metrics = {"illicit_f1": f1, "arm": arm, "deterministic": deterministic}
    if experiment is not None:
        metrics["experiment"] = experiment
    return {"run_id": run_id or f"r-{arm}-{seed}-{commit[:2]}-{f1}", "seed": str(seed),
            "git_commit": commit, "git_dirty": dirty, "notes": notes,
            "metrics_json": json.dumps(metrics)}


def _registry(path, rows):
    from gbe.run.registry import REGISTRY_COLUMNS

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: "" for c in REGISTRY_COLUMNS} | r)
    return path


def test_observed_rows_ignores_everything_but_the_check_tag(tmp_path):
    """The reference rows share (arm, seed) with the check rows, so selecting on the tag is what
    keeps the checker from comparing the reference against itself and always 'passing'."""
    path = _registry(tmp_path / "registry.csv", [
        _row(experiment="extract_reference", f1=0.111),
        _row(experiment="extract_check", f1=0.222),
        _row(experiment=None, f1=0.333),
    ])
    assert observed_rows(path, HEAD).observed == {("gcn", 0): {"illicit_f1": 0.222}}


def test_observed_rows_skips_rows_without_an_arm(tmp_path):
    """Untagged historical rows must not be mistaken for check rows."""
    row = _row() | {"metrics_json": json.dumps({"experiment": "extract_check", "illicit_f1": 0.9,
                                                 "deterministic": True})}
    assert observed_rows(_registry(tmp_path / "registry.csv", [row]), HEAD).observed == {}


def test_observed_rows_counts_only_rows_at_the_commit_being_certified(tmp_path):
    """The defect the audit of 2026-10-07 found (A2 E-B3): the latest row per identity was taken
    from any commit, so an older batch could stand in for one never run at HEAD."""
    path = _registry(tmp_path / "registry.csv", [_row(commit=OLD, f1=0.1), _row(commit=HEAD, f1=0.2)])
    assert observed_rows(path, HEAD).observed == {("gcn", 0): {"illicit_f1": 0.2}}
    assert observed_rows(path, OLD).observed == {("gcn", 0): {"illicit_f1": 0.1}}
    assert observed_rows(path, "f" * 40).observed == {}


@pytest.mark.parametrize("flaw", [{"dirty": "true"}, {"notes": "extract | ERRORED"},
                                  {"deterministic": False}])
def test_rows_that_cannot_certify_are_set_aside(tmp_path, flaw):
    path = _registry(tmp_path / "registry.csv", [_row(**flaw)])
    selection = observed_rows(path, HEAD)
    assert selection.observed == {} and selection.excluded == 1


def test_an_identity_run_twice_at_one_commit_is_reported_not_collapsed(tmp_path):
    path = _registry(tmp_path / "registry.csv", [_row(f1=0.1, run_id="first"),
                                                 _row(f1=0.2, run_id="second")])
    assert observed_rows(path, HEAD).retried == {("gcn", 0): ["first", "second"]}


# --------------------------------------------------------------------------- the verdict (main)


@pytest.fixture
def checker(tmp_path, monkeypatch):
    """Drive `main()` against a temporary manifest and registry, with the environment matching and
    HEAD pinned, so the verdict logic is tested end to end without a run."""
    import check_extract_regression as cer

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "git_commit": OLD, "env_pin": "unused.txt",
        "rows": [_ref("gcn", 0, illicit_f1=0.5), _ref("rf", 1, illicit_f1=0.7)],
    }), encoding="utf-8")
    registry = tmp_path / "registry.csv"
    monkeypatch.setattr(cer, "environment_drift", lambda pin: [])
    monkeypatch.setattr(cer, "default_registry_path", lambda root: registry)
    monkeypatch.setattr(cer, "resolve_commit", lambda ref: HEAD if ref == "HEAD" else OLD)

    def run(rows, *argv):
        _registry(registry, rows)
        monkeypatch.setattr(sys, "argv", ["check", "--compare-only", "--manifest", str(manifest),
                                          *argv])
        return cer.main()

    return run


def test_matching_rows_at_head_pass(checker):
    assert checker([_row("gcn", 0, 0.5), _row("rf", 1, 0.7)]) == 0


def test_a_missing_reference_row_fails(checker):
    """Mutation M4 (`ok` ignoring `missing`) left the suite green before this test."""
    assert checker([_row("gcn", 0, 0.5)]) == 1


def test_a_mismatch_fails(checker):
    assert checker([_row("gcn", 0, 0.5), _row("rf", 1, 0.7000000001)]) == 1


def test_rows_from_another_commit_cannot_pass_head(checker):
    assert checker([_row("gcn", 0, 0.5, commit=OLD), _row("rf", 1, 0.7, commit=OLD)]) == 1


def test_an_earlier_commit_can_be_certified_explicitly(checker):
    assert checker([_row("gcn", 0, 0.5, commit=OLD), _row("rf", 1, 0.7, commit=OLD)],
                   "--commit", "d614671") == 0


def test_nothing_to_compare_is_not_a_pass(checker):
    assert checker([]) == 1


def test_dirty_rows_cannot_certify(checker):
    assert checker([_row("gcn", 0, 0.5, dirty="true"), _row("rf", 1, 0.7, dirty="true")]) == 1


def test_a_retried_identity_fails_even_when_both_rows_match(checker):
    """ADR-008 clause 2: a mismatch is never retried. A second row at the same commit is a retry
    whatever it says, so the check refuses to choose between them."""
    rows = [_row("gcn", 0, 0.5, run_id="a"), _row("gcn", 0, 0.5, run_id="b"), _row("rf", 1, 0.7)]
    assert checker(rows) == 1


def test_run_with_commit_is_refused(monkeypatch):
    import check_extract_regression as cer

    monkeypatch.setattr(sys, "argv", ["check", "--run", "--commit", "abc"])
    with pytest.raises(SystemExit, match="refusing --run with --commit"):
        cer.main()


def test_real_manifest_identities_are_unique():
    """(arm, seed) is the pairing key — the 8 arm names must not collide across the three sources."""
    import json

    manifest = json.loads(
        (REPO_ROOT / "experiments" / "extract_reference_manifest.json").read_text(encoding="utf-8")
    )
    ids = [(e["arm"], e["seed"]) for e in manifest["rows"]]
    assert len(set(ids)) == len(ids) == 49
    assert len({e["arm"] for e in manifest["rows"]}) == 8
