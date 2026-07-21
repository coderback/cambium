"""Deterministic seeding across python / numpy / torch (+ CUDA).

One function, called once at run start, so every stochastic component draws from a
known state. Determinism flags are set so a resumed run reproduces an uninterrupted
one bit-for-bit (see gbe.run.checkpoint).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    """Seed python, numpy, and torch (CPU + CUDA); enable deterministic cuDNN.

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
    return seed
