"""Guards for the DGF-1 runners and their shared pre-registration module (ADR-012 clauses 3-5, 8;
ADR-013 clause 3's repro check).

No batch is executed here: what is pinned is the *policy* — which window a mode may touch, where
the seed count comes from, and what each mode is allowed to persist. The one property worth stating
plainly: **in gate mode the seed count is read from the pre-registration document, never from a
command-line flag**, so a batch cannot quietly run a different `n` than the one pre-registered.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path  # noqa: F401  (used by the repro-check fakes)

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from preregistration import require_accepted_preregistration, require_clean_tree  # noqa: E402
from run_dgf1_floor import resolve_floor_batch  # noqa: E402
from run_dgf1_gnn import (  # noqa: E402
    DELIBERATELY_UNCOMPARED,
    PILOT_SEEDS,
    REPRO_BATCH_SIZE,
    REPRO_LR,
    REPRO_METRIC_KEYS,
    REPRO_REFERENCE_RUN,
    REPRO_ROW_COLUMNS,
    compare_repro,
    device_problems,
    pin_runtime,
    read_row,
    rebuilt_reference_hash,
    repro_check_reference,
    repro_check_verdict,
    resolve_batch,
    score_file_problems,
)

from adapters.dgf1.eval import dgf1_hparams, dgf1_retune_grid  # noqa: E402


def _adr(tmp_path, status="accepted", seeds="12"):
    p = tmp_path / "ADR-012-x.md"
    line = f"**Stage-1 seeds:** {seeds}\n" if seeds is not None else ""
    p.write_text(f"# ADR\n\n**Status:** {status}\n{line}", encoding="utf-8")
    return p


def _args(**kw):
    return Namespace(**{"lr": None, "batch_size": None, "preregistration": None,
                        "device": None, "allow_dirty": False, **kw})


# -- which window each mode may touch -----------------------------------------------------------
def test_retune_and_pilot_stay_on_validation():
    configs, seeds, window, tag = resolve_batch("retune", _args())
    assert window == "val" and tag == "retune" and seeds == [0] and len(configs) == 9

    configs, seeds, window, tag = resolve_batch("pilot", _args(lr=1e-3, batch_size=512))
    assert window == "val" and tag == "pilot" and seeds == list(range(PILOT_SEEDS))
    assert configs == [(1e-3, 512)]


def test_gate_mode_takes_its_seed_count_from_the_adr_not_a_flag(tmp_path):
    args = _args(lr=1e-3, batch_size=512, preregistration=str(_adr(tmp_path, seeds="7")))
    configs, seeds, window, tag = resolve_batch("gate", args)
    assert window == "test" and tag == "gate"
    assert seeds == list(range(7)), "the pre-registered count must decide the batch size"
    assert configs == [(1e-3, 512)]


def test_gate_mode_is_refused_without_an_accepted_preregistration(tmp_path):
    base = dict(lr=1e-3, batch_size=512)
    with pytest.raises(SystemExit, match="no --preregistration"):
        resolve_batch("gate", _args(**base))
    with pytest.raises(SystemExit, match="not 'accepted'"):
        resolve_batch("gate", _args(**base, preregistration=str(_adr(tmp_path, status="proposed"))))
    with pytest.raises(SystemExit, match="no stage-1 seed count"):
        resolve_batch("gate", _args(**base, preregistration=str(_adr(tmp_path, seeds=None))))


def test_pilot_and_gate_require_the_retune_winner(tmp_path):
    for mode, extra in (("pilot", {}), ("gate", {"preregistration": str(_adr(tmp_path))})):
        with pytest.raises(SystemExit, match="needs --lr and --batch-size"):
            resolve_batch(mode, _args(**extra))


def test_the_real_adr_012_unlocks_the_test_window_with_exactly_the_derived_count():
    """Re-pinned 2026-09-14. Until the pilots ran, this test asserted the real ADR *refused* the
    test window, because its seed count was a placeholder. Clause 5's rule has since derived 8 and
    it is recorded in the header, so the state being pinned changed by design — not the guard.

    What stays pinned is the property that matters: the gate batch runs the number the document
    fixed, and a later edit to that line cannot silently change it without failing here."""
    adr = REPO_ROOT / "decisions" / "ADR-012-dgf1-gate1-preregistration.md"
    assert require_accepted_preregistration(adr) == 8


# -- ADR-013 clause 3: the repro check --------------------------------------------------------------
def test_repro_check_is_hard_wired_to_validation_seed_0_and_adr_012s_winner():
    configs, seeds, window, tag = resolve_batch("repro-check", _args())
    assert window == "val" and seeds == [0] and tag == "repro_check"
    # ADR-012's dated amendment of 2026-09-14, written out here too
    assert configs == [(3.318335548548489e-4, 2048)] == [(REPRO_LR, REPRO_BATCH_SIZE)]


def test_repro_check_refuses_every_flag_that_could_redirect_it(tmp_path):
    """Written out one flag at a time. `--device` is here because the device is in neither the
    hashed config nor the row, so a CPU run would clear the pre-run hash guard and then fail every
    float comparison — a false failure under a rule that forbids retrying (ADR-008 clause 2)."""
    for extra in ({"lr": 1e-3}, {"batch_size": 512}, {"preregistration": str(_adr(tmp_path))},
                  {"device": "cpu"}, {"allow_dirty": True}):
        with pytest.raises(SystemExit, match="hard-wired"):
            resolve_batch("repro-check", _args(**extra))
    # and the refusal names the flag, so the operator is not left guessing which one
    with pytest.raises(SystemExit, match="--device"):
        resolve_batch("repro-check", _args(device="cuda"))


def _repro_hp():
    from gbe.gnn import GNNHParams

    base_hp, _ = dgf1_hparams()
    return GNNHParams(**{**base_hp.as_dict(), "lr": REPRO_LR, "batch_size": REPRO_BATCH_SIZE,
                         "fan_out": tuple(base_hp.fan_out)})


def _repro_base_cfg():
    """What `main` hands the preflight in repro-check mode: the run's own tag, not the pilot's."""
    from adapters.dgf1.eval import dgf1_base_config

    return {**dgf1_base_config(), "phase": "P1", "experiment": "repro_check"}


def test_the_real_pilot_row_is_the_reference_and_the_preflight_accepts_it():
    """Calls the production preflight, so it pins the refusal and not just a hash rebuilt here.

    It catches drift in `config.yaml`, the hyperparameters or the split since the pilot — and also
    a preflight that forgot to rebuild under the *reference's* experiment tag, which would make the
    hashes differ forever and the check unrunnable."""
    from adapters.dgf1.eval import dgf1_splits
    from gbe.run.registry import default_registry_path

    registry = default_registry_path(REPO_ROOT)
    ref = read_row(registry, REPRO_REFERENCE_RUN)
    m = ref["metrics"]
    assert m["experiment"] == "pilot" and ref["seed"] == "0" and m["window"] == "370-481"
    assert (m["lr"], m["batch_size"]) == (REPRO_LR, REPRO_BATCH_SIZE)

    rebuilt = rebuilt_reference_hash(dgf1_splits()["val"], _repro_hp(), _repro_base_cfg())
    assert rebuilt == ref["config_hash"]
    assert repro_check_reference(registry, rebuilt)["run_id"] == REPRO_REFERENCE_RUN


def test_the_preflight_refuses_a_configuration_that_does_not_rebuild_the_reference(tmp_path):
    from gbe.run.registry import append_run

    registry = tmp_path / "registry.csv"
    append_run({"run_id": REPRO_REFERENCE_RUN, "config_hash": "a" * 64, "metrics_json": {}},
               path=registry)
    assert repro_check_reference(registry, "a" * 64)["config_hash"] == "a" * 64
    with pytest.raises(SystemExit, match="refusing repro-check"):
        repro_check_reference(registry, "b" * 64)


def test_the_hash_rebuild_depends_on_the_configuration_it_is_given():
    """A rebuild that ignored its arguments would accept any drifted configuration."""
    from gbe.gnn import GNNHParams
    from adapters.dgf1.eval import dgf1_splits

    splits = dgf1_splits()
    base = rebuilt_reference_hash(splits["val"], _repro_hp(), _repro_base_cfg())
    other_lr = GNNHParams(**{**_repro_hp().as_dict(), "lr": REPRO_LR * 2,
                             "fan_out": tuple(_repro_hp().fan_out)})
    assert rebuilt_reference_hash(splits["val"], other_lr, _repro_base_cfg()) != base
    assert rebuilt_reference_hash(splits["test"], _repro_hp(), _repro_base_cfg()) != base
    assert rebuilt_reference_hash(splits["val"], _repro_hp(),
                                  {**_repro_base_cfg(), "phase": "P9"}) != base


def _row(**metric_overrides):
    metrics = {"fraud_auc": 0.7810034416168538, "fraud_auprc": 0.04127410516906848,
               "fraud_f1": 0.05, "fraud_precision": 0.03, "fraud_recall": 0.8, "n_score": 100,
               "n_score_fraud": 2, "prevalence": 0.02, "n_train": 50, "scores_sha256": "a" * 64,
               "arm": "dgf1-parity", "window": "370-481", "features": "parity",
               "backbone": "graphsage", "lr": REPRO_LR, "batch_size": REPRO_BATCH_SIZE, "epochs": 40,
               "deterministic": True, "cublas_workspace_config": ":4096:8", "experiment": "pilot"}
    metrics.update(metric_overrides)
    metrics = {k: v for k, v in metrics.items() if v is not None or k == "scores_path"}
    metrics = {k: ("" if v is None else v) for k, v in metrics.items()}
    return {"run_id": "r", "model": "dgf1", "phase": "P1", "seed": "0",
            "data_snapshot_id": "snap", "git_dirty": "false", "notes": "", "metrics": metrics}


def test_compare_repro_passes_only_on_exact_equality():
    ref, obs = _row(), _row(experiment="repro_check")
    assert compare_repro(ref, obs) == []

    one_ulp = float(np.nextafter(ref["metrics"]["fraud_auprc"], 1.0))
    assert compare_repro(ref, _row(experiment="repro_check", fraud_auprc=one_ulp))
    assert compare_repro(ref, _row(experiment="repro_check", scores_sha256="b" * 64))
    assert any("missing" in p for p in compare_repro(ref, _row(experiment="repro_check", n_train=None)))


def test_compare_repro_rejects_a_row_that_cannot_certify():
    ref = _row()
    assert compare_repro(ref, _row())                                   # not tagged repro_check
    dirty = {**_row(experiment="repro_check"), "git_dirty": "true"}
    errored = {**_row(experiment="repro_check"), "notes": "dgf1 gnn on 370-481 | ERRORED"}
    assert compare_repro(ref, dirty) and compare_repro(ref, errored)
    other_seed = {**_row(experiment="repro_check"), "seed": "1"}
    assert compare_repro(ref, other_seed)


def test_withdrawn_keys_are_ignored_on_the_reference_and_refused_on_the_check():
    """The reference pilot row carries the withdrawn keys; that is history, not a mismatch. A check
    row carrying them means the reported-view scoring came back."""
    ref = _row(window_only_fraud_auprc=0.0369, first_appearance_fraud_auprc=0.0396)
    assert compare_repro(ref, _row(experiment="repro_check")) == []
    problems = compare_repro(ref, _row(experiment="repro_check", first_appearance_fraud_auprc=0.0396))
    assert any("withdrawn" in p for p in problems)


def test_the_compared_field_lists_are_pinned_literally():
    """A literal list defends against a key going missing from a *row*; nothing defends against a
    key going missing from the *list* — the same "assertion shares its source of truth with the
    code" hole ADR-013 clause 1 records. Both tuples are therefore written out here in full, so
    shrinking either one fails loudly."""
    assert set(REPRO_METRIC_KEYS) == {
        "fraud_auc", "fraud_auprc", "fraud_f1", "fraud_precision", "fraud_recall",
        "n_score", "n_score_fraud", "prevalence", "n_train", "scores_sha256",
        "arm", "window", "features", "backbone", "lr", "batch_size", "epochs",
        "deterministic", "cublas_workspace_config",
    }
    assert set(REPRO_ROW_COLUMNS) == {"model", "phase", "seed", "data_snapshot_id"}
    assert len(set(REPRO_METRIC_KEYS)) == len(REPRO_METRIC_KEYS)   # no duplicate padding the count


def test_changing_any_single_compared_field_is_reported():
    """Membership in the tuple is not the property; being *compared* is. Perturb each field on its
    own and require a problem, so a key cannot sit in the list while the comparison ignores it."""
    def perturbed(value):
        if isinstance(value, bool):
            return not value
        if isinstance(value, float):
            return float(np.nextafter(value, np.inf))
        if isinstance(value, int):
            return value + 1
        return f"{value}-changed"

    reference = _row()
    for key in REPRO_METRIC_KEYS:
        observed = _row(experiment="repro_check")
        observed["metrics"][key] = perturbed(observed["metrics"][key])
        assert compare_repro(reference, observed), f"{key} is listed but its change is not reported"
    for column in REPRO_ROW_COLUMNS:
        observed = _row(experiment="repro_check")
        observed[column] = f"{observed[column]}-changed"
        assert compare_repro(reference, observed), f"{column} is listed but its change is not reported"


def test_every_key_a_real_run_writes_is_either_compared_or_deliberately_skipped(tmp_path):
    """The drift guard: a metric added to the trainer later must be classified, not silently left
    out of the certification. Uses a real `run_dgf1` row on the synthetic fixture."""
    from adapters.dgf1.eval import dgf1_base_config
    from adapters.dgf1.train_gnn import run_dgf1
    from gbe.run.seeding import seed_everything
    from tests.test_dgf1_trainer import HP
    from tests.test_dgf1_floor import SPLIT_VAL, _synthetic

    registry = tmp_path / "registry.csv"
    seed_everything(0)
    run_id, _ = run_dgf1(0, _synthetic(), SPLIT_VAL, HP, dgf1_base_config(),
                         registry_path=registry, device="cpu", scores_dir=tmp_path / "scores")

    written = set(read_row(registry, run_id)["metrics"])
    # provenance the RunSession adds for every run, not DGF-1 results; the repro check pins the two
    # that bear on reproducibility (`deterministic`, `cublas_workspace_config`) and ignores the rest
    provenance = {"phase", "model", "data_snapshot_id", "split", "reverse_edges", "view_features"}
    unclassified = written - set(REPRO_METRIC_KEYS) - DELIBERATELY_UNCOMPARED - provenance
    assert not unclassified, (
        f"a DGF-1 row carries {sorted(unclassified)}, which the repro check neither compares nor "
        "records a reason for skipping"
    )


def test_score_file_problems_ties_the_row_to_the_file_on_disk(tmp_path):
    """The other half of ADR-013 clause 3's bar. A row's hash is taken from the arrays in memory,
    so nothing but this re-hash says the file still holds that vector."""
    from gbe.eval import save_scores

    path = tmp_path / "scores.npz"
    digest = save_scores(path, node_ids=np.arange(5), y_true=np.zeros(5, dtype=np.int64),
                         proba=np.linspace(0.1, 0.9, 5).astype(np.float32))
    good = _row(experiment="repro_check", scores_sha256=digest, scores_path=str(path))
    assert score_file_problems(good) == []

    stale = _row(experiment="repro_check", scores_sha256="c" * 64, scores_path=str(path))
    assert any("hashes to" in p for p in score_file_problems(stale))

    # the file replaced after the row was written — the case only a re-hash can catch
    save_scores(path, node_ids=np.arange(5), y_true=np.zeros(5, dtype=np.int64),
                proba=np.linspace(0.2, 0.9, 5).astype(np.float32))
    assert score_file_problems(good), "a rewritten score file must not pass"

    missing = _row(experiment="repro_check", scores_path=str(tmp_path / "absent.npz"))
    assert any("does not exist" in p for p in score_file_problems(missing))
    assert any("does not exist" in p for p in score_file_problems(_row(experiment="repro_check",
                                                                      scores_path=None)))


def test_the_verdict_requires_both_halves_of_the_bar(tmp_path):
    """`repro_check_verdict` is what `main` calls; it must fail if *either* half fails."""
    from gbe.eval import save_scores

    path = tmp_path / "scores.npz"
    digest = save_scores(path, node_ids=np.arange(3), y_true=np.zeros(3, dtype=np.int64),
                         proba=np.zeros(3, dtype=np.float32))
    reference = _row(scores_sha256=digest)
    observed = _row(experiment="repro_check", scores_sha256=digest, scores_path=str(path))
    assert repro_check_verdict(reference, observed) == []

    row_differs = _row(experiment="repro_check", scores_sha256=digest, scores_path=str(path),
                       fraud_auc=0.5)
    assert repro_check_verdict(reference, row_differs)
    file_gone = _row(experiment="repro_check", scores_sha256=digest,
                     scores_path=str(tmp_path / "gone.npz"))
    assert repro_check_verdict(reference, file_gone)


# -- the repro check end to end, with the training run faked ----------------------------------------
# Every guard that certifies the trainer lives in `main`, so testing only its helpers leaves the
# wiring unpinned: mutations that deleted the pre-run refusal, the dirty-tree refusal, or the call
# to the verdict itself all left the suite green. These tests drive `main` with `load_dgraph` and
# `run_dgf1` replaced, which is what makes that wiring visible without a 10-minute run.
def _fake_repro_run(monkeypatch, tmp_path, *, metrics_override=None, reference_hash=None):
    """Point `main` at a temp registry seeded with a reference row, and fake the training run."""
    import run_dgf1_gnn
    import preregistration
    from gbe.eval import save_scores
    from gbe.run import registry as registry_mod
    from gbe.run.registry import append_run
    from adapters.dgf1 import datasource_dgraph, train_gnn
    from adapters.dgf1.eval import dgf1_splits

    registry = tmp_path / "registry.csv"
    scores_dir = tmp_path / "scores"
    monkeypatch.setattr(registry_mod, "default_registry_path", lambda *a, **k: registry)
    monkeypatch.setattr(run_dgf1_gnn, "SCORES_DIR", scores_dir)
    monkeypatch.setattr(preregistration, "git_dirty", lambda *a, **k: False)

    digest = save_scores(scores_dir / "ref.npz", node_ids=np.arange(4),
                         y_true=np.zeros(4, dtype=np.int64), proba=np.zeros(4, dtype=np.float32))
    metrics = {**_row(scores_sha256=digest)["metrics"]}
    metrics.pop("scores_path", None)

    rebuilt = reference_hash or rebuilt_reference_hash(
        dgf1_splits()["val"], _repro_hp(), _repro_base_cfg())
    append_run({"run_id": REPRO_REFERENCE_RUN, "model": "dgf1", "phase": "P1", "seed": 0,
                "data_snapshot_id": "snap", "config_hash": rebuilt, "git_dirty": False,
                "metrics_json": metrics, "notes": ""}, path=registry)

    loaded: list[str] = []
    monkeypatch.setattr(datasource_dgraph, "load_dgraph",
                        lambda *a, **k: loaded.append("loaded") or object())

    def fake_run_dgf1(seed, data, split, hp, base_cfg, device=None, scores_dir=None, **kw):
        run_id = "dgf1-faked-repro"
        observed = {**metrics, "experiment": "repro_check", **(metrics_override or {})}
        path = Path(scores_dir) / f"{run_id}.npz"
        observed["scores_sha256"] = save_scores(
            path, node_ids=np.arange(4), y_true=np.zeros(4, dtype=np.int64),
            proba=np.zeros(4, dtype=np.float32) + observed.pop("_proba_shift", 0.0))
        observed["scores_path"] = str(path)
        append_run({"run_id": run_id, "model": "dgf1", "phase": "P1", "seed": seed,
                    "data_snapshot_id": "snap", "config_hash": "irrelevant", "git_dirty": False,
                    "metrics_json": observed, "notes": ""}, path=registry)
        return run_id, observed

    monkeypatch.setattr(train_gnn, "run_dgf1", fake_run_dgf1)
    monkeypatch.setattr(sys, "argv", ["run_dgf1_gnn.py", "--mode", "repro-check"])
    return run_dgf1_gnn.main, loaded


def test_main_certifies_a_run_that_reproduces_the_reference(monkeypatch, tmp_path):
    main, loaded = _fake_repro_run(monkeypatch, tmp_path)
    main()                      # no SystemExit: the run reproduced
    assert loaded == ["loaded"]


def test_main_exits_non_zero_when_the_run_does_not_reproduce(monkeypatch, tmp_path):
    """ADR-013 clause 3: exits non-zero on any mismatch. One ULP of AUPRC is a mismatch."""
    shifted = float(np.nextafter(0.04127410516906848, np.inf))
    main, _ = _fake_repro_run(monkeypatch, tmp_path, metrics_override={"fraud_auprc": shifted})
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_exits_non_zero_when_the_score_file_no_longer_matches(monkeypatch, tmp_path):
    """The other half of the bar: the row's floats agree, the vector on disk does not."""
    main, _ = _fake_repro_run(monkeypatch, tmp_path, metrics_override={"_proba_shift": 0.25})
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_refuses_before_loading_data_when_the_configuration_drifted(monkeypatch, tmp_path):
    """The pre-run refusal must fire *before* `load_dgraph`, or a drifted configuration costs a
    full training run that was never going to reproduce anything."""
    main, loaded = _fake_repro_run(monkeypatch, tmp_path, reference_hash="f" * 64)
    with pytest.raises(SystemExit, match="refusing repro-check"):
        main()
    assert loaded == [], "the dataset was loaded despite a refused configuration"


def test_main_refuses_a_dirty_tree_in_repro_check_mode(monkeypatch, tmp_path):
    import preregistration

    main, loaded = _fake_repro_run(monkeypatch, tmp_path)
    monkeypatch.setattr(preregistration, "git_dirty", lambda *a, **k: True)
    with pytest.raises(SystemExit, match="uncommitted"):
        main()
    assert loaded == []


# -- the CPU-fallback guard -------------------------------------------------------------------------
# The defect: `environment_drift` puts its GPU/CUDA comparisons behind `torch.cuda.is_available()`,
# and `resolve_device("auto")` falls back to CPU silently. Lose CUDA and the check reported no drift,
# trained on CPU, differed on every float, and recorded a sound trainer as a regression under a rule
# that forbids retrying. A second route reaches the same place with no driver failure: `device:` in
# `adapters/dgf1/config.yaml` is returned by `dgf1_hparams()`, not `dgf1_base_config()`, so it is in
# neither the hashed config nor the row.
REAL_PIN = REPO_ROOT / "experiments" / "extract_reference_env.txt"


def test_the_pin_runtime_block_parses_including_its_awkward_lines():
    """The pin's `[runtime]` block is CRLF, and `cublas_workspace_config` carries a trailing prose
    comment. A parser that ignores either refuses a stack that has not moved."""
    fields = pin_runtime(REAL_PIN)
    assert fields["gpu"] == "NVIDIA GeForce RTX 3050 Ti Laptop GPU"
    assert fields["cublas_workspace_config"] == ":4096:8", "the trailing comment was not stripped"
    assert fields["python"] == "3.13.5" and fields["cudnn"] == "91002"
    assert "torch" in fields and "[pip freeze]" not in fields
    # the pip-freeze section must not bleed in: it is `_pinned_versions`' territory
    assert "scikit-learn" not in fields


def test_a_cpu_device_is_refused_against_a_gpu_pin():
    assert device_problems(REAL_PIN, "cuda") == []
    assert device_problems(REAL_PIN, "cuda:0") == []
    for cpu in ("cpu", "cpu:0"):
        problems = device_problems(REAL_PIN, cpu)
        assert problems and "cannot reproduce a CUDA reference" in problems[0]


def test_a_pin_without_a_gpu_cannot_verify_the_device(tmp_path):
    """Refuse rather than pass silently: an unverifiable device is not a verified one."""
    pin = tmp_path / "pin.txt"
    pin.write_text("[runtime]\npython = 3.13.5\n\n[pip freeze]\ntorch==2.13.0\n", encoding="utf-8")
    assert any("records no gpu" in p for p in device_problems(pin, "cuda"))


def test_the_device_guard_is_not_itself_conditional_on_cuda(monkeypatch):
    """The mutation this guard exists to survive.

    A previous review demonstrated that a `torch.cuda.is_available()`-guarded implementation passes
    the CPU-fallback test vacuously — it skips exactly when it is needed. `device_problems` must
    report the problem with CUDA unavailable, which is the only state in which it matters."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from gbe.run import resolve_device

    assert str(resolve_device("auto")) == "cpu", "fixture did not reproduce the CUDA-less state"
    problems = device_problems(REAL_PIN, str(resolve_device("auto")))
    assert problems, "the guard skipped precisely when CUDA was unavailable — it is vacuous"


def test_main_refuses_a_cpu_run_before_loading_data(monkeypatch, tmp_path):
    """End to end: CUDA gone, so `auto` resolves to CPU. The run must be REFUSED before the dataset
    is touched and must write no row — not trained and then recorded as a regression."""
    import torch

    main, loaded = _fake_repro_run(monkeypatch, tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="cannot reproduce a CUDA reference"):
        main()
    assert loaded == [], "the dataset was loaded despite an unusable device"


def test_main_refuses_the_config_yaml_device_route(monkeypatch, tmp_path):
    """`device: cpu` in the adapter config bypasses the configuration-hash guard entirely, because
    the device is in neither the hashed config nor the row."""
    import adapters.dgf1.eval as dgf1_eval

    main, loaded = _fake_repro_run(monkeypatch, tmp_path)
    real_hparams = dgf1_eval.dgf1_hparams
    monkeypatch.setattr(dgf1_eval, "dgf1_hparams", lambda *a, **k: (real_hparams()[0], "cpu"))
    with pytest.raises(SystemExit, match="cannot reproduce a CUDA reference"):
        main()
    assert loaded == []


def test_read_row_requires_exactly_one_match(tmp_path):
    from gbe.run.registry import append_run

    registry = tmp_path / "registry.csv"
    append_run({"run_id": "a", "metrics_json": {"x": 1}}, path=registry)
    assert read_row(registry, "a")["metrics"] == {"x": 1}
    with pytest.raises(SystemExit, match="found 0"):
        read_row(registry, "b")
    append_run({"run_id": "a", "metrics_json": {"x": 2}}, path=registry)
    with pytest.raises(SystemExit, match="found 2"):
        read_row(registry, "a")


# -- the floor runner's modes mirror the GNN runner's -------------------------------------------
def _fargs(**kw):
    return Namespace(**{"max_depth": None, "subsample": None, "preregistration": None, **kw})


def test_floor_retune_is_the_nine_config_grid_on_validation():
    configs, seeds, window, tag = resolve_floor_batch("retune", _fargs())
    assert window == "val" and tag == "retune" and seeds == [0]
    assert len(configs) == 9 and len({tuple(sorted(c.items())) for c in configs}) == 9
    assert {k for c in configs for k in c} == {"max_depth", "subsample"}


def test_floor_pilot_seed_count_follows_whether_the_winner_subsamples():
    """Clause 2: a deterministic winner needs only enough seeds to evidence identical rows."""
    _, seeds, window, tag = resolve_floor_batch("pilot", _fargs(max_depth=6, subsample=1.0))
    assert seeds == [0, 1, 2] and window == "val" and tag == "pilot"
    _, seeds, _, _ = resolve_floor_batch("pilot", _fargs(max_depth=6, subsample=0.8))
    assert seeds == list(range(PILOT_SEEDS))


def test_floor_gate_takes_its_seed_count_from_the_adr(tmp_path):
    args = _fargs(max_depth=6, subsample=1.0, preregistration=str(_adr(tmp_path, seeds="9")))
    configs, seeds, window, tag = resolve_floor_batch("gate", args)
    assert window == "test" and tag == "gate" and seeds == list(range(9))
    assert configs == [{"max_depth": 6, "subsample": 1.0}]


def test_floor_gate_is_refused_without_an_accepted_preregistration(tmp_path):
    with pytest.raises(SystemExit, match="no --preregistration"):
        resolve_floor_batch("gate", _fargs(max_depth=6, subsample=1.0))
    with pytest.raises(SystemExit, match="no stage-1 seed count"):
        resolve_floor_batch("gate", _fargs(max_depth=6, subsample=1.0,
                                           preregistration=str(_adr(tmp_path, seeds=None))))


def test_floor_pilot_and_gate_require_the_retune_winner(tmp_path):
    for mode, extra in (("pilot", {}), ("gate", {"preregistration": str(_adr(tmp_path))})):
        with pytest.raises(SystemExit, match="needs --max-depth and --subsample"):
            resolve_floor_batch(mode, _fargs(**extra))


# -- the clean-tree guard -------------------------------------------------------------------------
def test_a_dirty_tree_is_refused_unless_overridden(monkeypatch):
    import preregistration

    monkeypatch.setattr(preregistration, "git_dirty", lambda: True)
    with pytest.raises(SystemExit, match="uncommitted"):
        require_clean_tree(allow_dirty=False)
    require_clean_tree(allow_dirty=True)          # explicit override, for throwaway smoke runs

    monkeypatch.setattr(preregistration, "git_dirty", lambda: False)
    require_clean_tree(allow_dirty=False)


# -- inherited hyperparameters and the grid ------------------------------------------------------
def test_dgf1_inherits_adr_003s_frozen_region():
    hp, device = dgf1_hparams()
    assert (hp.backbone, hp.num_layers, hp.hidden_dim, hp.aggr) == ("graphsage", 3, 128, "mean")
    assert hp.fan_out == (25, 10) and hp.epochs == 40 and hp.weight_decay == 5e-4
    assert hp.lr == pytest.approx(6.636671097096978e-4)
    assert device in {"auto", "cuda", "cpu"}


def test_the_retune_grid_is_nine_configs_around_the_inherited_lr():
    grid = dgf1_retune_grid()
    base_lr, _ = dgf1_hparams()
    assert len(grid) == 9 and len(set(grid)) == 9
    lrs = sorted({lr for lr, _ in grid})
    assert lrs == pytest.approx([base_lr.lr * 0.5, base_lr.lr, base_lr.lr * 2.0])
    assert sorted({b for _, b in grid}) == [512, 1024, 2048]
