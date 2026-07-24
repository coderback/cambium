"""Deterministic seeding across python / numpy / torch (+ CUDA).

One function, called once at run start, so every stochastic component draws from a
known state. Determinism flags are set so a resumed run reproduces an uninterrupted
one bit-for-bit (see gbe.run.checkpoint).

**Why this module does more than seed (ADR-005).** Seeding alone does *not* make a GNN
reproducible on CUDA. Neighbour aggregation is a scatter-reduce — every edge atomically adds
into its destination's accumulator — and CUDA atomics complete in scheduler-dependent order.
Float addition is not associative, so each run sums the same values in a different order and
lands ~1e-7 apart; over ~1200 optimizer steps that compounds into a materially different model.
Measured on ELL-1: three identical repeats at one seed spanned **0.034 illicit-F1**, wider than
the effects the Phase-3 ablations exist to detect.

`cudnn.deterministic` does **not** fix this. It only governs cuDNN's algorithm choice for
convolutions and RNNs, and a GNN's hot path contains neither — it was inert here all along.
What fixes it is `torch.use_deterministic_algorithms(True)`, which selects PyTorch's
deterministic scatter kernel, plus `CUBLAS_WORKSPACE_CONFIG` for reproducible cuBLAS reductions.
Measured cost: ~5% wall-clock, i.e. free.
"""

from __future__ import annotations

import os
import random
import warnings

# cuBLAS reads this when it creates its handle — which happens on the first matmul — so it must
# be in the environment *before* any CUDA work. Setting it at import time (rather than inside
# seed_everything, which runs later at RunSession entry) is what makes that ordering hold.
# setdefault: an operator who has pinned :16:8 externally keeps their choice.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402  - must follow the env var above
import torch  # noqa: E402

# If CUDA was already up before this module was imported, the line above may have come too late
# to affect the cuBLAS handle. That failure is silent and produces irreproducible numbers, so
# say so loudly rather than let a run quietly lose its determinism guarantee.
if torch.cuda.is_available() and torch.cuda.is_initialized():  # pragma: no cover - env-specific
    warnings.warn(
        "gbe.run.seeding was imported after CUDA had already been initialised; "
        "CUBLAS_WORKSPACE_CONFIG may not have taken effect and CUDA results may not be "
        "reproducible. Import gbe.run (or set the variable in the environment) first.",
        RuntimeWarning,
        stacklevel=2,
    )


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed python, numpy, and torch (CPU + CUDA), and pin deterministic kernels.

    Args:
        seed: the seed to apply everywhere.
        deterministic: when True (default), require deterministic algorithms. This is
            **strict** — PyTorch raises if an op in the model has no deterministic
            implementation. That is deliberate: a loud failure on an unsupported op is worth
            far more than silently irreproducible gate numbers, which is the exact trap
            ADR-005 documents. Pass False only for throughput-bound exploration whose numbers
            will never be reported.

    Returns the seed so callers can record exactly what was used.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic)
    return seed
