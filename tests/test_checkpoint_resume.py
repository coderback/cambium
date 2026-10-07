"""Guard: a killed-and-resumed run is bit-for-bit identical to an uninterrupted one.

Spot instances are reclaimed mid-run, so resume must be exact. The toy training
loop draws a fresh noise tensor from the *global* torch RNG every step, so this
test genuinely exercises checkpoint RNG-state restoration — not just weights.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from gbe.run.checkpoint import load_checkpoint, save_checkpoint
from gbe.run.seeding import seed_everything

SEED = 7
N_TOTAL = 20
N_HALF = 8
IN_DIM, OUT_DIM, N = 4, 1, 32


def _fixed_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic synthetic data, independent of the global RNG."""
    gen = torch.Generator().manual_seed(1234)
    x = torch.randn(N, IN_DIM, generator=gen)
    y = torch.randn(N, OUT_DIM, generator=gen)
    return x, y


def _build():
    model = nn.Linear(IN_DIM, OUT_DIM)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    return model, opt


def _train_steps(model, opt, x, y, n_steps: int) -> None:
    loss_fn = nn.MSELoss()
    for _ in range(n_steps):
        noise = torch.randn_like(x) * 0.01  # consumes global RNG each step
        opt.zero_grad()
        loss = loss_fn(model(x + noise), y)
        loss.backward()
        opt.step()


def _params(model) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def test_resume_is_bitwise_identical(tmp_path):
    x, y = _fixed_data()

    # Run A: uninterrupted.
    seed_everything(SEED)
    model_a, opt_a = _build()
    _train_steps(model_a, opt_a, x, y, N_TOTAL)
    ref = _params(model_a)

    # Run B: train half, checkpoint, rebuild fresh, resume, finish.
    seed_everything(SEED)
    model_b, opt_b = _build()
    _train_steps(model_b, opt_b, x, y, N_HALF)
    ckpt = tmp_path / "toy.pt"
    save_checkpoint(ckpt, model_b, opt_b, config_hash="h", seed=SEED, step=N_HALF)

    # Fresh objects, as after a spot reclaim. Their random init is discarded by
    # load_checkpoint (weights + RNG state are restored from the bundle).
    model_r, opt_r = _build()
    meta = load_checkpoint(ckpt, model_r, opt_r)
    assert meta["step"] == N_HALF and meta["seed"] == SEED
    _train_steps(model_r, opt_r, x, y, N_TOTAL - N_HALF)
    got = _params(model_r)

    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert torch.equal(g, r), "resumed weights differ from the uninterrupted run"


def _build_adam():
    model = nn.Linear(IN_DIM, OUT_DIM)
    return model, torch.optim.Adam(model.parameters(), lr=0.05)


def _train_steps_every_rng(model, opt, x, y, n_steps: int) -> None:
    """Like `_train_steps`, but each step also draws from python's and numpy's RNGs."""
    loss_fn = nn.MSELoss()
    for _ in range(n_steps):
        scale = 0.01 * (1.0 + random.random()) * (1.0 + float(np.random.rand()))
        noise = torch.randn_like(x) * scale
        opt.zero_grad()
        loss = loss_fn(model(x + noise), y)
        loss.backward()
        opt.step()


def test_resume_restores_optimizer_state_and_every_rng(tmp_path):
    """The test above uses SGD without momentum, which keeps no state between steps, and draws only
    from torch's RNG. So dropping the optimizer restore, or the python or numpy RNG restore, left
    it green (audit of 2026-10-07, A2 E-N8). Adam carries moment estimates between steps, and this
    loop draws from all three generators. A fresh process is simulated by re-seeding differently
    before the resume."""
    x, y = _fixed_data()

    seed_everything(SEED)
    model_a, opt_a = _build_adam()
    _train_steps_every_rng(model_a, opt_a, x, y, N_TOTAL)
    ref = _params(model_a)

    seed_everything(SEED)
    model_b, opt_b = _build_adam()
    _train_steps_every_rng(model_b, opt_b, x, y, N_HALF)
    ckpt = tmp_path / "adam.pt"
    save_checkpoint(ckpt, model_b, opt_b, config_hash="h", seed=SEED, step=N_HALF)

    seed_everything(SEED + 1)   # the resumed process starts from unrelated RNG states
    model_r, opt_r = _build_adam()
    load_checkpoint(ckpt, model_r, opt_r)
    _train_steps_every_rng(model_r, opt_r, x, y, N_TOTAL - N_HALF)

    for g, r in zip(_params(model_r), ref):
        assert torch.equal(g, r), "resumed weights differ from the uninterrupted run"
