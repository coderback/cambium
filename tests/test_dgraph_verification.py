"""Guards for the DGraph snapshot verifier — the parts testable before the data exists.

The shape assertions cannot be exercised until the registration-gated snapshot lands (by design:
ELL-1's loader was likewise written first and escalated on arrival). What *is* testable now is the
node-time proxy — the piece carrying a real methodological judgement — and that a discrepancy
escalates rather than being absorbed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_dgraph_snapshot import (  # noqa: E402
    EXPECTED_BACKGROUND,
    EXPECTED_FRAUD,
    EXPECTED_NODES,
    EXPECTED_NORMAL,
    _escalate,
    _node_time_proxy,
)


def test_doc_02_class_breakdown_sums_to_the_node_count():
    """doc-02 §2.1's own numbers must be internally consistent before we trust any of them:
    15,509 fraud + 1,210,092 normal + 2,474,949 background == 3,700,550 nodes."""
    assert EXPECTED_FRAUD + EXPECTED_NORMAL + EXPECTED_BACKGROUND == EXPECTED_NODES


def test_node_time_proxy_takes_the_earliest_incident_edge_and_credits_both_endpoints():
    #   0 -> 1 at t=5     1 -> 2 at t=3     3 -> 4 at t=9     node 5 isolated
    edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]])
    edge_time = torch.tensor([5, 3, 9])

    got = _node_time_proxy(edge_index, edge_time, n=6)

    # node 1 sits on both a t=5 and a t=3 edge -> 3, which is only correct if *both* the source
    # and destination endpoint of every edge are credited.
    assert got.tolist() == [5, 3, 3, 9, 9, -1]


def test_isolated_nodes_are_flagged_not_silently_zero():
    """A node with no edges has no time evidence. Returning 0 would place it at the very start of
    the timeline and quietly bias any temporal split built on this proxy."""
    got = _node_time_proxy(torch.tensor([[0], [1]]), torch.tensor([7]), n=4)
    assert got.tolist() == [7, 7, -1, -1]


def test_proxy_is_order_independent():
    """Edge ordering must not change the result — it is a min-reduction, not a scan."""
    ei = torch.tensor([[0, 1, 0], [1, 2, 2]])
    et = torch.tensor([9, 4, 6])
    perm = torch.tensor([2, 0, 1])
    assert _node_time_proxy(ei, et, 3).tolist() == _node_time_proxy(ei[:, perm], et[perm], 3).tolist()


def test_escalation_names_the_doc_and_forbids_editing_the_constant():
    """ADR-001 precedent: a doc-vs-data mismatch is reconciled in the doc, never by quietly
    changing the expected value here."""
    with pytest.raises(AssertionError) as exc:
        _escalate("node count", 123, EXPECTED_NODES, "doc-02 §2.1")
    message = str(exc.value)
    assert "doc-02 §2.1" in message
    assert "123" in message and str(EXPECTED_NODES) in message
    assert "Do NOT edit the constant" in message
    assert "doc-first" in message
