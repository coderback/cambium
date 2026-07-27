"""ELL-1 hyperparameter sweep — the one-time Optuna + ASHA study (doc-00 §6.4, doc-01 §4).

Tuned once, here, on the cheapest graph; the winner is frozen in an ADR and inherited by
DGF-1/EDR-1 (which then retune only lr + one graph-appropriate second knob, named per model —
ADR-009; DGF-1 names batch size). No NAS — a small *enumerated* space.

Integrity boundary (constitution: never tune against the held-out eval): the sweep uses an
**inner temporal split — train 1-29, validate 30-34** — and never touches the 35-49 test
window. Both are strict inductive, reusing the Stage-A trainer/eval unchanged: train message
passing over the <=29 induced subgraph, validation over the 30-34 induced subgraph. The
objective is validation illicit-F1 (the gate metric); ASHA prunes on per-epoch val-F1.

The search space is GraphSAGE-specific (doc §6.4): layers {2,3}, hidden {128,256}, aggregator
{mean,max}, dropout {0.2,0.5}, lr log-uniform, fan-out {[10,10],[15,10],[25,10]}. The GCN
comparator is *not* separately searched — it inherits the frozen architecture/optimisation so
the Gate-1 comparison is capacity- and compute-matched (doc-00 §8), only the convolution
differs.
"""

from __future__ import annotations

import optuna
import torch

from gbe.eval import TemporalSplit, induced_train_subgraph
from gbe.run.seeding import seed_everything
from adapters.ell1.train_gnn import (
    GNNHParams,
    build_model,
    evaluate,
    standardize_fit_on_train,
    train_model,
)

# The inner tuning split — the whole point of this module is that test_max is 34, never 49.
INNER_SPLIT = TemporalSplit(train_max=29, test_min=30, test_max=34)

# Fixed training budget shared by every trial and by the final runs (so selection reflects
# final training). Only the doc §6.4 dimensions vary per trial.
HPO_EPOCHS = 40
HPO_TRIALS = 50
TRIAL_SEED = 0  # same init seed every trial -> the sweep isolates the hyperparameter effect
STUDY_NAME = "ell1-graphsage-hpo"

# fan-out choices as strings (Optuna categoricals must be scalars), parsed back to tuples.
_FAN_OUTS = {"10,10": (10, 10), "15,10": (15, 10), "25,10": (25, 10)}


def suggest_hparams(trial: optuna.Trial) -> GNNHParams:
    """Draw one GraphSAGE config from the doc §6.4 enumerated space."""
    return GNNHParams(
        backbone="graphsage",
        num_layers=trial.suggest_categorical("num_layers", [2, 3]),
        hidden_dim=trial.suggest_categorical("hidden_dim", [128, 256]),
        aggr=trial.suggest_categorical("aggr", ["mean", "max"]),
        dropout=trial.suggest_categorical("dropout", [0.2, 0.5]),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        fan_out=_FAN_OUTS[trial.suggest_categorical("fan_out", list(_FAN_OUTS))],
        epochs=HPO_EPOCHS,
    )


def make_objective(data, device):
    """Build the Optuna objective bound to a loaded Elliptic graph + device.

    Standardisation is fit on the inner-train window (<=29) only — the 30-34 validation stats
    never touch the scaler, so tuning is leak-free end to end.
    """
    x = standardize_fit_on_train(data.x, data.time_step, INNER_SPLIT.train_max)
    edge_train, train_node_mask = induced_train_subgraph(
        data.edge_index, data.time_step, INNER_SPLIT
    )
    seed_mask = train_node_mask & data.labelled_mask

    def _val_f1(model) -> float:
        metrics, _ = evaluate(
            model, x, data.edge_index, data.time_step, data.y, INNER_SPLIT, device
        )
        return metrics["f1"]

    def objective(trial: optuna.Trial) -> float:
        seed_everything(TRIAL_SEED)
        hp = suggest_hparams(trial)
        model = build_model(x.size(1), hp, device)

        def on_epoch(epoch: int, m) -> None:
            trial.report(_val_f1(m), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            train_model(model, x, edge_train, data.y, seed_mask, hp, device, on_epoch=on_epoch)
            return _val_f1(model)
        finally:
            # free the trial's graph/activations so VRAM doesn't creep across a long sweep
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return objective


def build_study(storage: str | None = None, study_name: str = STUDY_NAME) -> optuna.Study:
    """A maximise study with a TPE sampler + ASHA (SuccessiveHalving) pruner.

    With ``storage`` (a SQLite URL) the study is **persistent and resumable**: a crash — e.g.
    an intermittent CUDA fault on the 4 GB laptop GPU — loses only the in-flight trial, and a
    restart continues from the completed trials in the DB (``load_if_exists=True``).
    """
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=8, reduction_factor=3),
    )
