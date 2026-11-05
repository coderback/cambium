"""Device selection — infra both models share (EXTRACT, doc-00 §9)."""

from __future__ import annotations

import torch


def resolve_device(device: str | None) -> torch.device:
    """``'cuda'``/``'cpu'``/``None`` -> a device; ``None`` or ``'auto'`` picks cuda when present."""
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
