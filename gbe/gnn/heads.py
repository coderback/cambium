"""Task heads — embedding -> task output (doc-00 §3.4).

The head is the only per-task piece on top of the shared embedding. ELL-1/DGF-1/EDR-1 use
node classification (this file); EDL-1 will register a link-prediction head, and a
contrastive head shapes the embedding space directly — all behind the same registry so the
backbone and trainer stay task-agnostic.

Heads emit logits only; the loss (class weighting, focal, InfoNCE) lives in the trainer so
thresholds and weights stay out of the core (gbe/CLAUDE.md rule 3).
"""

from __future__ import annotations

from torch import Tensor, nn


class NodeClassificationHead(nn.Module):
    """Linear map from a node embedding to class logits (illicit / licit for ELL-1)."""

    def __init__(self, in_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, embedding: Tensor) -> Tensor:
        return self.linear(embedding)


# Name -> class, mirroring BACKBONES. Link-pred / contrastive heads register here as their
# models arrive (doc-00 §3.4/§3.5) without changing the trainer contract.
HEADS: dict[str, type[nn.Module]] = {
    "node_clf": NodeClassificationHead,
}
