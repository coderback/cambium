"""NodeClassifier — the assembled encoder -> backbone -> head forward (doc-00 §2 diagram).

One forward pass yields **two** outputs (doc-00 §2): the *embedding* (the backbone readout,
the product that goes to a vector index or a future soft-token projector) and the *logits*
(the measurable, from the head). The embedding carries no head-specific assumptions
(readout contract, doc-00 §3.3), so the same model serves classification, retrieval, and
fusion consumers.
"""

from __future__ import annotations

from torch import Tensor, nn

from gbe.features.encoder import FeatureEncoder
from gbe.gnn.backbone import GNNBackbone


class NodeClassifier(nn.Module):
    """Compose a FeatureEncoder, a GNN backbone, and a head into one node model.

    Args:
        encoder: content ``[N, in]`` -> features ``[N, enc_out]``.
        backbone: features + edges -> node embeddings ``[N, emb]``.
        head: embedding -> logits (must accept ``backbone.out_dim``).
    """

    def __init__(
        self,
        encoder: FeatureEncoder,
        backbone: GNNBackbone,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.head = head

    def embed(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Node embeddings only — the readout, for indexing / retrieval."""
        return self.backbone(self.encoder(x), edge_index)

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(embedding, logits)`` from a single forward pass."""
        embedding = self.embed(x, edge_index)
        logits = self.head(embedding)
        return embedding, logits
