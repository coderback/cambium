"""Guards for the extracted shared trainer (gbe.gnn.train) — EXTRACT, doc-00 §9.

Written in the same session as the extraction. The properties that matter are the ones the two
models rely on identically: the hyperparameter record that lands in every config hash, balanced
class weights, fan-out extension, and a loop that touches only the graph it is handed.

Bit-for-bit equality of ELL-1's numbers is *not* tested here — it is ADR-008's checker's job, on
real data.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from gbe.gnn import GNNHParams, balanced_class_weights, build_model, fan_out_for, train_model
from gbe.run import resolve_device
from gbe.run.seeding import seed_everything


def _graph(n=60, f=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(n, generator=g) < 0.3).long()
    x = torch.randn(n, f, generator=g) + y[:, None].float()
    src = torch.arange(n)
    ei = torch.stack([torch.cat([src, (src + 1) % n]), torch.cat([(src + 1) % n, src])])
    return x, ei, y


# -- the hyperparameter record ----------------------------------------------------------------
def test_as_dict_shape_is_stable_because_config_hashes_depend_on_it():
    """Field names and the list-valued fan-out are what every registry row hashes."""
    d = GNNHParams().as_dict()
    assert set(d) == {"backbone", "num_layers", "hidden_dim", "aggr", "dropout", "lr", "fan_out",
                      "encoder_layers", "norm", "epochs", "batch_size", "weight_decay"}
    assert d["fan_out"] == [15, 10] and isinstance(d["fan_out"], list)


# -- class weights ------------------------------------------------------------------------------
def test_balanced_weights_follow_the_sklearn_scheme_and_average_to_one():
    y = torch.tensor([0] * 90 + [1] * 10)
    mask = torch.ones(100, dtype=torch.bool)
    w = balanced_class_weights(y, mask, torch.device("cpu"))
    assert torch.allclose(w, torch.tensor([100 / 180, 100 / 20]))
    assert torch.isclose((w * torch.tensor([0.9, 0.1])).sum(), torch.tensor(1.0))


def test_weights_use_only_the_seed_nodes():
    y = torch.tensor([0, 0, 1, 1, 1, 1])
    seeds = torch.tensor([True, True, True, False, False, False])
    w = balanced_class_weights(y, seeds, torch.device("cpu"))
    assert torch.allclose(w, torch.tensor([3 / 4, 3 / 2]))


def test_class_order_is_a_parameter_not_an_assumption():
    y = torch.tensor([0] * 90 + [1] * 10)
    mask = torch.ones(100, dtype=torch.bool)
    a = balanced_class_weights(y, mask, torch.device("cpu"), classes=(0, 1))
    b = balanced_class_weights(y, mask, torch.device("cpu"), classes=(1, 0))
    assert torch.allclose(a.flip(0), b)


def test_an_absent_class_does_not_divide_by_zero():
    y = torch.zeros(10, dtype=torch.long)
    w = balanced_class_weights(y, torch.ones(10, dtype=torch.bool), torch.device("cpu"))
    assert torch.isfinite(w).all()


# -- fan-out -------------------------------------------------------------------------------------
def test_fan_out_is_extended_to_the_layer_count_by_repeating_the_last_hop():
    assert fan_out_for(GNNHParams(num_layers=3, fan_out=(25, 10))) == [25, 10, 10]
    assert fan_out_for(GNNHParams(num_layers=2, fan_out=(25, 10))) == [25, 10]
    assert fan_out_for(GNNHParams(num_layers=1, fan_out=(25, 10))) == [25]


# -- the loop --------------------------------------------------------------------------------------
def test_training_changes_the_model_and_is_reproducible_from_the_seed():
    x, ei, y = _graph()
    hp = GNNHParams(num_layers=2, hidden_dim=16, fan_out=(5, 5), epochs=3, batch_size=16)
    dev = resolve_device("cpu")
    seeds = torch.ones(y.numel(), dtype=torch.bool)

    def trained():
        seed_everything(0)
        m = build_model(x.size(1), hp, dev)
        before = [p.detach().clone() for p in m.parameters()]
        train_model(m, x, ei, y, seeds, hp, dev)
        return before, [p.detach().clone() for p in m.parameters()]

    before, after = trained()
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "training changed nothing"
    _, again = trained()
    assert all(torch.equal(a, b) for a, b in zip(after, again))


def test_the_loop_only_sees_the_graph_it_is_handed():
    """The core takes the training graph as given and never inspects a cutoff, so a leakage
    decision cannot be smuggled into it (gbe/CLAUDE.md rule 1). With no edges, training still
    runs — the encoder and head learn, message passing has nothing to pass."""
    x, _, y = _graph()
    empty = torch.empty((2, 0), dtype=torch.long)
    hp = GNNHParams(num_layers=2, hidden_dim=8, fan_out=(5, 5), epochs=2, batch_size=16)
    seed_everything(0)
    dev = resolve_device("cpu")
    m = build_model(x.size(1), hp, dev)
    train_model(m, x, empty, y, torch.ones(y.numel(), dtype=torch.bool), hp, dev)


def test_the_epoch_hook_fires_once_per_epoch():
    x, ei, y = _graph()
    hp = GNNHParams(num_layers=2, hidden_dim=8, fan_out=(5, 5), epochs=4, batch_size=32)
    seed_everything(0)
    dev = resolve_device("cpu")
    m = build_model(x.size(1), hp, dev)
    seen: list[int] = []
    train_model(m, x, ei, y, torch.ones(y.numel(), dtype=torch.bool), hp, dev,
                on_epoch=lambda e, _m: seen.append(e))
    assert seen == [0, 1, 2, 3]


def test_build_model_emits_both_outputs_with_the_readout_width():
    x, ei, _ = _graph()
    hp = GNNHParams(num_layers=2, hidden_dim=16, fan_out=(5, 5))
    m = build_model(x.size(1), hp, resolve_device("cpu"))
    emb, logits = m(x, ei)
    assert emb.shape == (x.size(0), 16) and logits.shape == (x.size(0), 2)
