"""Guards for the ADR-008 regression checker.

The checker is EXTRACT's acceptance test, so a checker that cannot *fail* is worse than no checker
at all — it would license the refactor while proving nothing. Each test below is written so that
breaking the corresponding behaviour makes it fail.
"""

from __future__ import annotations

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


def test_observed_rows_ignores_everything_but_the_check_tag(tmp_path):
    """The reference rows share (arm, seed) with the check rows, so selecting on the tag is what
    keeps the checker from comparing the reference against itself and always 'passing'."""
    import csv
    import json

    from gbe.run.registry import REGISTRY_COLUMNS

    path = tmp_path / "registry.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
        w.writeheader()
        for experiment, f1 in (("extract_reference", 0.111), ("extract_check", 0.222), (None, 0.333)):
            metrics = {"illicit_f1": f1, "arm": "gcn"}
            if experiment is not None:
                metrics["experiment"] = experiment
            w.writerow({c: "" for c in REGISTRY_COLUMNS} | {
                "run_id": f"r-{experiment}", "seed": "0",
                "metrics_json": json.dumps(metrics),
            })

    got = observed_rows(path)
    assert got == {("gcn", 0): {"illicit_f1": 0.222}}


def test_observed_rows_skips_rows_without_an_arm(tmp_path):
    """Untagged historical rows must not be mistaken for check rows."""
    import csv
    import json

    from gbe.run.registry import REGISTRY_COLUMNS

    path = tmp_path / "registry.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
        w.writeheader()
        w.writerow({c: "" for c in REGISTRY_COLUMNS} | {
            "run_id": "no-arm", "seed": "0",
            "metrics_json": json.dumps({"experiment": "extract_check", "illicit_f1": 0.9}),
        })
    assert observed_rows(path) == {}


def test_real_manifest_identities_are_unique():
    """(arm, seed) is the pairing key — the 8 arm names must not collide across the three sources."""
    import json

    manifest = json.loads(
        (REPO_ROOT / "experiments" / "extract_reference_manifest.json").read_text(encoding="utf-8")
    )
    ids = [(e["arm"], e["seed"]) for e in manifest["rows"]]
    assert len(set(ids)) == len(ids) == 49
    assert len({e["arm"] for e in manifest["rows"]}) == 8
