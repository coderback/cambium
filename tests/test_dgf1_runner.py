"""Guards for the DGF-1 runners and their shared pre-registration module (ADR-012 clauses 3-5, 8).

No batch is executed here: what is pinned is the *policy* — which window a mode may touch, where
the seed count comes from, and what each mode is allowed to persist. The one property worth stating
plainly: **in gate mode the seed count is read from the pre-registration document, never from a
command-line flag**, so a batch cannot quietly run a different `n` than the one pre-registered.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from preregistration import require_accepted_preregistration, require_clean_tree  # noqa: E402
from run_dgf1_floor import resolve_floor_batch  # noqa: E402
from run_dgf1_gnn import PILOT_SEEDS, resolve_batch  # noqa: E402

from adapters.dgf1.eval import dgf1_hparams, dgf1_retune_grid  # noqa: E402


def _adr(tmp_path, status="accepted", seeds="12"):
    p = tmp_path / "ADR-012-x.md"
    line = f"**Stage-1 seeds:** {seeds}\n" if seeds is not None else ""
    p.write_text(f"# ADR\n\n**Status:** {status}\n{line}", encoding="utf-8")
    return p


def _args(**kw):
    return Namespace(**{"lr": None, "batch_size": None, "preregistration": None, **kw})


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


def test_the_real_adr_012_still_keeps_the_test_window_shut():
    """Its seed count is a placeholder until the pilot derives it — so gate mode must refuse."""
    adr = REPO_ROOT / "decisions" / "ADR-012-dgf1-gate1-preregistration.md"
    with pytest.raises(SystemExit, match="no stage-1 seed count"):
        require_accepted_preregistration(adr)


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
