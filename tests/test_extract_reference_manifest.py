"""Guards for the frozen EXTRACT reference manifest (ADR-008).

The manifest is the bar the refactored core is measured against, and it is *data*, not code — so
nothing else in the suite would notice if it drifted, got truncated, or was re-frozen against
post-refactor numbers. These tests pin it to the append-only registry it was derived from.

They are deliberately strict about equality: ADR-008 clause 2 sets the bar at exact float equality,
so a manifest whose stored metrics merely round-trip to the registry's would silently weaken it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "experiments" / "extract_reference_manifest.json"
REGISTRY_PATH = REPO_ROOT / "experiments" / "registry.csv"

EXPECTED_COUNTS = {"gate0": 6, "gate3": 40, "gcn_ref": 3}
EXPECTED_TOTAL = 49
REQUIRED_METRICS = {"illicit_f1", "illicit_recall", "illicit_precision", "illicit_auc"}


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST_PATH.exists():
        pytest.fail(
            f"{MANIFEST_PATH.name} is missing. It is the frozen ADR-008 reference set and is "
            "tracked; regenerate with scripts/freeze_extract_reference.py only if it was never "
            "committed — never after the refactor has begun."
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry_by_run_id() -> dict[str, dict]:
    with REGISTRY_PATH.open(newline="", encoding="utf-8") as fh:
        return {r["run_id"]: r for r in csv.DictReader(fh)}


def test_manifest_has_the_expected_shape(manifest):
    """6 Gate-0 + 40 Gate-3 + 3 GCN rows. A truncated manifest is a weakened bar."""
    rows = manifest["rows"]
    assert len(rows) == EXPECTED_TOTAL
    assert manifest["counts"]["total"] == EXPECTED_TOTAL
    actual = {src: sum(1 for r in rows if r["source"] == src) for src in EXPECTED_COUNTS}
    assert actual == EXPECTED_COUNTS


def test_every_identity_is_unique(manifest):
    """(source, arm, seed) is the pairing key, so a collision would make a row unmatchable."""
    ids = [(r["source"], r["arm"], r["seed"]) for r in manifest["rows"]]
    assert len(set(ids)) == len(ids)


def test_every_reference_row_exists_in_the_registry(manifest, registry_by_run_id):
    missing = [r["run_id"] for r in manifest["rows"] if r["run_id"] not in registry_by_run_id]
    assert not missing, f"manifest cites run_ids absent from the registry: {missing}"


def test_stored_metrics_equal_the_registry_exactly(manifest, registry_by_run_id):
    """Bit-for-bit, not approximately — ADR-008's bar is exact float equality.

    This is the load-bearing test: it is what makes the manifest a faithful copy of the registry
    rather than a second, drifting source of truth.
    """
    mismatches = []
    for entry in manifest["rows"]:
        recorded = json.loads(registry_by_run_id[entry["run_id"]]["metrics_json"])
        for key, value in entry["metrics"].items():
            if recorded[key] != value:
                mismatches.append(
                    f"{entry['source']}/{entry['arm']}/seed{entry['seed']} {key}: "
                    f"manifest {value!r} != registry {recorded[key]!r}"
                )
    assert not mismatches, "manifest has drifted from the registry:\n" + "\n".join(mismatches)


def test_every_row_carries_the_four_required_metrics(manifest):
    for entry in manifest["rows"]:
        assert REQUIRED_METRICS <= set(entry["metrics"]), (
            f"{entry['source']}/{entry['arm']}/seed{entry['seed']} is missing required metrics"
        )


def test_only_post_adr007_rows_carry_auprc(manifest):
    """ADR-008: a new key is not a regression. Exactly the 3 GCN rows predate nothing and have it;
    the 46 older rows cannot, because AUPRC is not recoverable from a logged row."""
    with_auprc = {r["source"] for r in manifest["rows"] if "illicit_auprc" in r["metrics"]}
    assert with_auprc == {"gcn_ref"}
    assert sum(1 for r in manifest["rows"] if "illicit_auprc" in r["metrics"]) == 3


def test_gate3_and_gcn_rows_are_deterministic(manifest):
    """Equality is only enforceable against rows that ran under ADR-005 determinism."""
    for entry in manifest["rows"]:
        if entry["source"] in ("gate3", "gcn_ref"):
            assert entry["deterministic"] is True, (
                f"{entry['source']}/{entry['arm']}/seed{entry['seed']} is not deterministic"
            )


def test_config_hashes_match_the_registry(manifest, registry_by_run_id):
    """Stored for audit only (never for pairing), but a wrong hash means a mis-resolved row —
    which for Gate-3 would mean a mislabelled ablation arm."""
    for entry in manifest["rows"]:
        assert entry["config_hash"] == registry_by_run_id[entry["run_id"]]["config_hash"]


def test_manifest_points_at_the_environment_pin(manifest):
    """ADR-008 clause 4: equality is only meaningful against the pinned stack."""
    assert manifest["env_pin"] == "experiments/extract_reference_env.txt"
    assert (REPO_ROOT / manifest["env_pin"]).exists()
