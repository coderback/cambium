"""Defending ablations — the controls that prove a gain is structural (doc-00 §8).

Domain-agnostic machinery, like the rest of `gbe.eval`: these take an edge set and a per-node
time tensor and return a counterfactual edge set. Which ablations a model runs, and what
counts as a pass, live in the adapter's config and its gate file — never here
(gbe/CLAUDE.md rule 3).

**Edge scramble** is the load-bearing one (doc-01 §4 Phase 3): retrain on a graph whose
topology is intact but whose correspondence to node content is destroyed. If performance does
not drop, the model was reading features, not structure.
"""

from __future__ import annotations

import torch
from torch import Tensor


def scramble_edges(edge_index: Tensor, time_step: Tensor, seed: int) -> Tensor:
    """Permute node identities **within each time step**, preserving the degree sequence.

    The permutation is applied per time value, never globally, and that is a correctness
    requirement rather than a stylistic one: a global permutation would map the two endpoints
    of a within-step edge onto nodes in *different* steps, manufacturing edges that cross the
    temporal cutoff. Those would either leak the future into training or be silently dropped by
    the induced-subgraph step — in both cases the ablation would no longer be comparable to the
    real-graph run it exists to be compared against.

    Permuting within a step preserves the exact degree sequence, the per-step edge count, and
    the temporal split, and destroys only the alignment between a node's position in the graph
    and its features/label. That isolates "is the structure informative?" from the separate
    question "does having neighbours at all help?" (the random-graph control, doc-00 §8).

    Args:
        edge_index: ``[2, E]`` edge tensor.
        time_step: ``[N]`` per-node time values.
        seed: seed for the permutation, so a scramble is reproducible from the registry row.

    Returns:
        A new ``[2, E]`` edge tensor. Symmetry is preserved: relabelling maps a reciprocal
        pair to a reciprocal pair, so a symmetrised input stays symmetrised.
    """
    generator = torch.Generator().manual_seed(int(seed))
    mapping = torch.arange(time_step.numel())
    for t in time_step.unique():
        idx = (time_step == t).nonzero(as_tuple=True)[0]
        mapping[idx] = idx[torch.randperm(idx.numel(), generator=generator)]
    return mapping[edge_index]


def _undirected_pairs(edge_index: Tensor) -> Tensor:
    """The ``u < v`` half of a symmetrised edge set, as a ``[P, 2]`` tensor of pairs."""
    mask = edge_index[0] < edge_index[1]
    return edge_index[:, mask].t()


def _symmetrise(pairs: Tensor) -> Tensor:
    """``[P, 2]`` pairs -> a symmetrised ``[2, 2P]`` edge_index."""
    src, dst = pairs[:, 0], pairs[:, 1]
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def random_graph_edges(edge_index: Tensor, time_step: Tensor, seed: int) -> Tensor:
    """Erdős–Rényi control: same per-step edge count, degree sequence destroyed (doc-00 §8).

    For each time step, draw fresh endpoint pairs uniformly among that step's nodes, matching
    the number of undirected edges the real graph has in that step. Drawing **within** the step
    is what keeps every edge inside one time window, so the temporal split survives — the same
    requirement that governs :func:`scramble_edges`, and for the same reason.

    Against :func:`scramble_edges` this changes three things at once (wiring, degree sequence,
    and the pairing between a node's degree and its own features), so it answers "does the graph
    matter?" but not "which property mattered?". :func:`configuration_model_edges` is the arm
    that separates those.

    Self-loops and duplicate pairs are removed, so the result may carry marginally fewer edges
    than the real graph; on a mean-degree-2.3 graph the collision rate is negligible.
    """
    generator = torch.Generator().manual_seed(int(seed))
    kept: list[Tensor] = []
    for t in time_step.unique():
        idx = (time_step == t).nonzero(as_tuple=True)[0]
        n = idx.numel()
        in_step = (time_step[edge_index[0]] == t) & (edge_index[0] < edge_index[1])
        n_pairs = int(in_step.sum())
        if n_pairs == 0 or n < 2:
            continue
        draw = torch.randint(0, n, (n_pairs, 2), generator=generator)
        pairs = idx[draw]
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]                       # drop self-loops
        pairs = torch.unique(pairs.sort(dim=1).values, dim=0)           # drop duplicates
        kept.append(pairs)
    if not kept:
        return edge_index.new_empty((2, 0))
    return _symmetrise(torch.cat(kept))


def configuration_model_edges(
    edge_index: Tensor, time_step: Tensor, seed: int, swaps_per_edge: int = 10
) -> Tensor:
    """Degree-preserving rewire by double-edge swaps — randomise *who*, keep *how many*.

    Repeatedly picks two undirected edges ``(a,b)`` and ``(c,d)`` within the same time step and
    rewrites them as ``(a,d)`` and ``(c,b)``. Every endpoint keeps its degree exactly, so each
    node retains its own connectivity *and* its own features; only the identity of its
    neighbours is randomised. Swaps that would create a self-loop or a duplicate edge are
    skipped, which is why this is a double-edge swap rather than naive stub-matching — the
    latter produces multigraphs.

    This is the arm that localises the structural signal (ADR-006 clause 5, reported not gated):
    ``real ≈ config`` says the gain is essentially a degree effect — pointed, because Elliptic's
    features 94-164 are hand-built one-hop aggregates that already encode degree-like statistics;
    ``real >> config`` says neighbour identity carries information beyond degree.

    Args:
        swaps_per_edge: swap attempts per edge. 10 is comfortably past the mixing point for a
            graph this sparse (mean degree 2.3).
    """
    generator = torch.Generator().manual_seed(int(seed))
    out: list[Tensor] = []
    for t in time_step.unique():
        in_step = (time_step[edge_index[0]] == t) & (edge_index[0] < edge_index[1])
        pairs = edge_index[:, in_step].t().clone()
        m = pairs.size(0)
        if m < 2:
            out.append(pairs)
            continue

        # Hot loop: ~10 attempts per edge, millions of iterations on a real graph. Scalar
        # indexing into a tensor costs ~100x a Python list lookup here, so the swap runs over
        # plain lists and ints and is written back to a tensor at the end. Edges are keyed as
        # a single packed int (u * n_nodes + v, canonical u < v) so membership is one hash.
        n_nodes = time_step.numel()
        us = pairs[:, 0].tolist()
        vs = pairs[:, 1].tolist()
        present = {u * n_nodes + v for u, v in zip(us, vs)}

        n_attempts = m * swaps_per_edge
        picks = torch.randint(0, m, (n_attempts, 2), generator=generator).tolist()
        for e1, e2 in picks:
            if e1 == e2:
                continue
            a, b = us[e1], vs[e1]
            c, d = us[e2], vs[e2]
            if a == c or a == d or b == c or b == d:   # shared endpoint -> would self-loop
                continue
            n1u, n1v = (a, d) if a < d else (d, a)
            n2u, n2v = (c, b) if c < b else (b, c)
            k1 = n1u * n_nodes + n1v
            k2 = n2u * n_nodes + n2v
            if k1 in present or k2 in present:
                continue                                # would duplicate an existing edge
            present.discard(a * n_nodes + b)
            present.discard(c * n_nodes + d)
            present.add(k1)
            present.add(k2)
            us[e1], vs[e1] = n1u, n1v
            us[e2], vs[e2] = n2u, n2v

        out.append(torch.tensor([us, vs], dtype=pairs.dtype).t())
    return _symmetrise(torch.cat(out))


def remove_edges(edge_index: Tensor) -> Tensor:
    """GNN-removed / param-matched control: an empty edge set (doc-01 §4 Phase 3).

    Message passing is disabled while the model, its parameter count, and its training budget
    stay identical — SAGEConv keeps its root weight and aggregates nothing, so this is the
    "GNN-removed" and "param/compute-matched" ablations in one run rather than a separate
    architecture that would introduce its own confound.
    """
    return edge_index.new_empty((2, 0))
