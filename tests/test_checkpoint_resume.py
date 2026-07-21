"""Guard: a killed-and-resumed run is bit-for-bit identical to an uninterrupted one.

Spot instances are reclaimed mid-run, so resume must be exact. The toy training
loop draws a fresh noise tensor from the *global* torch RNG every step, so this
test genuinely exercises checkpoint RNG-state restoration — not just weights.
"""

from __future__ import annotations

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
