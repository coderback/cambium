"""Throwaway demo: register one toy run through RunSession, proving the registry
gains exactly one well-formed row. Also demonstrates checkpoint/resume identity.

    python scripts/demo_toy_run.py

Writes to a temp registry by default so it never pollutes experiments/registry.csv.
Pass --real to append to the canonical registry instead.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import torch
from torch import nn

from gbe.run import RunSession, load_checkpoint, save_checkpoint, seed_everything
from gbe.run.config import resolve_config
from gbe.run.registry import REGISTRY_COLUMNS, default_registry_path


def _toy_train(seed: int, steps: int) -> nn.Linear:
    seed_everything(seed)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    x, y = torch.randn(16, 4), torch.randn(16, 1)
    loss_fn = nn.MSELoss()
    for _ in range(steps):
        noise = torch.randn_like(x) * 0.01
        opt.zero_grad()
        loss_fn(model(x + noise), y).backward()
        opt.step()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="append to experiments/registry.csv")
    args = ap.parse_args()

    cfg = resolve_config(
        {"model": "toy", "phase": "P0", "seed": 3, "data_snapshot_id": "toy-synth-v0", "lr": 0.05}
    )

    if args.real:
        reg_path = default_registry_path()
    else:
        reg_path = Path(tempfile.mkdtemp()) / "registry.csv"

    before = 0
    if reg_path.exists():
        with reg_path.open(newline="", encoding="utf-8") as fh:
            before = sum(1 for _ in csv.DictReader(fh))

    with RunSession(cfg, notes="demo toy run", registry_path=reg_path) as run:
        model = _toy_train(run.seed, steps=10)
        with torch.no_grad():
            final_loss = float(nn.MSELoss()(model(torch.randn(16, 4)), torch.randn(16, 1)))
        run.log_metrics({"final_loss": round(final_loss, 4), "steps": 10})

    with reg_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    added = len(rows) - before
    assert added == 1, f"expected exactly 1 new row, got {added}"
    assert list(rows[-1].keys()) == list(REGISTRY_COLUMNS)
    assert all(rows[-1][c] != "" for c in ("run_id", "config_hash", "seed", "metrics_json"))
    print(f"[registry] wrote 1 row to {reg_path}")
    print(f"[registry] last row: {rows[-1]}")

    # Checkpoint/resume identity: kill at step 6, resume, compare to uninterrupted.
    gen = torch.Generator().manual_seed(999)
    xt = torch.randn(16, 4, generator=gen)
    yt = torch.randn(16, 1, generator=gen)

    def train(model, opt, n):
        loss_fn = nn.MSELoss()
        for _ in range(n):
            noise = torch.randn_like(xt) * 0.01
            opt.zero_grad()
            loss_fn(model(xt + noise), yt).backward()
            opt.step()

    seed_everything(5)
    ref = nn.Linear(4, 1)
    train(ref, torch.optim.SGD(ref.parameters(), lr=0.05), 12)

    seed_everything(5)
    m = nn.Linear(4, 1)
    o = torch.optim.SGD(m.parameters(), lr=0.05)
    train(m, o, 6)
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "toy.pt"
        save_checkpoint(ck, m, o, cfg.config_hash, seed=5, step=6)
        r = nn.Linear(4, 1)
        ro = torch.optim.SGD(r.parameters(), lr=0.05)
        load_checkpoint(ck, r, ro)
        train(r, ro, 6)

    identical = all(torch.equal(a, b) for a, b in zip(r.parameters(), ref.parameters()))
    print(f"[checkpoint] resumed weights identical to uninterrupted run: {identical}")
    assert identical, "resume was not bitwise-identical"


if __name__ == "__main__":
    main()
