"""Per-node score vectors: saved with the ids they belong to, and content-hashed (ADR-012 cl.10).

ELL-1's regret drives this module. Its registry rows stored *summary metrics only*, so when ADR-007
added AUPRC it could not be computed for any past run — every arm would have had to be re-run. The
same thing happens to every later question (uncertainty intervals, calibration, precision-at-k) if
only summaries survive. A scored run therefore writes the vector it scored.

Two properties matter and are enforced here:

* **The node ids travel with the scores.** A paired comparison of two arms has to align them by id;
  aligning by position would be an undocumented contract that breaks silently the first time a mask
  or a sort order changes.
* **The hash is over the *content*, not the file.** `.npz` is a zip, whose bytes carry timestamps,
  so a file hash would differ on every rewrite of identical data. Hashing the arrays makes the
  digest reproducible: re-saving the same scores gives the same hash, and any changed value changes
  it.

Domain-agnostic: ids, labels and probabilities, no dataset knowledge (gbe/CLAUDE.md rule 1).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def content_hash(node_ids: np.ndarray, y_true: np.ndarray, proba: np.ndarray) -> str:
    """SHA-256 over the three arrays' raw bytes, in a fixed order and fixed dtypes."""
    h = hashlib.sha256()
    for arr, dtype in ((node_ids, np.int64), (y_true, np.int64), (proba, np.float32)):
        h.update(np.ascontiguousarray(arr, dtype=dtype).tobytes())
    return h.hexdigest()


def save_scores(
    path: str | Path, *, node_ids: np.ndarray, y_true: np.ndarray, proba: np.ndarray
) -> str:
    """Write ``(node_ids, y_true, proba)`` to ``path`` and return their content hash.

    The caller records the path and the returned hash in the run's registry row, which is what ties
    a score file to the run that produced it.
    """
    node_ids = np.ascontiguousarray(node_ids, dtype=np.int64)
    y_true = np.ascontiguousarray(y_true, dtype=np.int64)
    proba = np.ascontiguousarray(proba, dtype=np.float32)
    if not (node_ids.shape == y_true.shape == proba.shape) or node_ids.ndim != 1:
        raise ValueError(
            f"node_ids, y_true and proba must be 1-D and the same length; got "
            f"{node_ids.shape}, {y_true.shape}, {proba.shape}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, node_ids=node_ids, y_true=y_true, proba=proba)
    return content_hash(node_ids, y_true, proba)


def load_scores(path: str | Path) -> dict[str, np.ndarray]:
    """Read back ``{"node_ids", "y_true", "proba"}``."""
    with np.load(Path(path)) as f:
        return {k: f[k] for k in ("node_ids", "y_true", "proba")}


def assert_aligned(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    """Raise unless two score files cover exactly the same nodes, in the same order.

    The precondition of any paired comparison (ADR-012 clause 10): without it a bootstrap would
    silently pair arm A's user with arm B's neighbour.
    """
    if not np.array_equal(a["node_ids"], b["node_ids"]):
        raise AssertionError(
            "the two score sets cover different nodes (or the same nodes in a different order); "
            "a paired comparison would mis-align the arms."
        )
    if not np.array_equal(a["y_true"], b["y_true"]):
        raise AssertionError("the two score sets disagree about the labels of the same nodes.")
