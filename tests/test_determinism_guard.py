"""Guard: seeding actually pins deterministic kernels (ADR-005).

The failure this prevents is silent. `cudnn.deterministic` was set for months and did nothing
for a GNN — it governs only cuDNN's convolution/RNN algorithm choice, while neighbour
aggregation is a scatter-reduce whose CUDA atomics complete in scheduler-dependent order. The
result was a 0.034 illicit-F1 spread across identical repeats, wider than the effects the
Phase-3 ablations exist to detect, with nothing anywhere reporting a problem.

So the guarantee is asserted rather than assumed.
"""

from __future__ import annotations

import os

import pytest
import torch

from gbe.run.seeding import seed_everything


@pytest.fixture(autouse=True)
def _restore_determinism_flag():
    """Leave the process as we found it — this flag is global torch state."""
    before = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(before)


def test_seed_everything_enables_deterministic_algorithms():
    seed_everything(0)
    assert torch.are_deterministic_algorithms_enabled(), (
        "seed_everything left deterministic algorithms off; CUDA scatter would be "
        "nondeterministic and gate numbers irreproducible (ADR-005)"
    )


def test_cublas_workspace_config_is_set_at_import():
    """cuBLAS reads this when it builds its handle, so importing the module must set it —
    setting it inside seed_everything (which runs later) would be a silent no-op."""
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), (
        "CUBLAS_WORKSPACE_CONFIG is unset or not a reproducible setting; "
        f"got {os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}"
    )


def test_determinism_can_be_opted_out_explicitly():
    """Exploration may trade reproducibility for throughput — but only on purpose."""
    seed_everything(0, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    seed_everything(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()


def test_seed_is_returned_for_the_registry_row():
    assert seed_everything(7) == 7


def test_run_row_records_whether_it_was_deterministic(tmp_path):
    """ADR-005 clause 3: the escape hatch must not be a hole in provenance.

    A `deterministic=False` exploration run has to be distinguishable, after the fact, from a
    gate-grade one — otherwise the hatch quietly undoes the guarantee.
    """
    import csv
    import json

    from gbe.run.config import resolve_config
    from gbe.run.session import RunSession

    path = tmp_path / "registry.csv"
    cfg = resolve_config({"model": "toy", "seed": 0})
    with RunSession(cfg, registry_path=path) as run:
        run.log_metrics({"f1": 0.5})

    with path.open(newline="", encoding="utf-8") as fh:
        row = next(iter(csv.DictReader(fh)))
    metrics = json.loads(row["metrics_json"])
    assert "deterministic" in metrics, "run row does not record its determinism state"
    assert metrics["deterministic"] is True
    assert metrics["cublas_workspace_config"] in (":4096:8", ":16:8")
    assert metrics["f1"] == 0.5, "provenance clobbered the logged metrics"


def test_same_seed_reproduces_the_same_draws():
    """End-to-end: the point of all of the above."""
    seed_everything(3)
    a = torch.randn(64)
    seed_everything(3)
    b = torch.randn(64)
    assert torch.equal(a, b)
