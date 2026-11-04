"""The label-free cost of one matched-time training epoch, owed before ADR-023 is accepted.

What it measures, on steps up to 369 only:
  A. how many edges, and neighbours within two and three hops, users have at their own node time;
  B. which input columns a transform fitted at each user's own time would zero;
  C. one full training epoch under ADR-023 clause 2's rule: time, GPU memory and process memory.

What it never does: read a dataset label, read a step after 369, or compute a model metric. The epoch
trains on every user with node time <= 369, because choosing the labelled ones would use labels, so
it is an upper bound on a real epoch. Its targets are random, from a fixed generator, so the loss means
nothing and is never reported.

The statistics come from a binary search over each node's dated edges, so a batch shuffled across
times costs no per-time view. The run's Check section compares them with `view_node_features` before any
timing is trusted. This is a measurement, not row 6b's builder.

    python notebooks/measurements/2026-11-04-matched-time/measure_matched_epoch.py
    python notebooks/measurements/2026-11-04-matched-time/measure_matched_epoch.py --max-batches 20

Only a full run writes output.txt beside this script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gbe.run.seeding import seed_everything  # noqa: E402  (sets the CUBLAS variable before torch runs)

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from adapters.dgf1.datasource_dgraph import (  # noqa: E402
    EDGE_TYPE_MIN,
    NUM_EDGE_TYPES,
    VIEW_FEATURE_NAMES,
    GraphView,
    load_dgraph,
    view_node_features,
)
from adapters.dgf1.features import INPUT_FEATURE_NAMES  # noqa: E402
from adapters.dgf1.sampling import assert_temporal_batch, first_appearance_loader  # noqa: E402
from gbe.features.scaling import Standardizer  # noqa: E402
from gbe.gnn import GNNHParams, build_model  # noqa: E402
from gbe.gnn.train import fan_out_for  # noqa: E402

TRAIN_MAX = 369
NEIGHBOUR_SAMPLE = 50_000
CHECK_CUTOFFS = (50, 200, TRAIN_MAX)
GATE1_LR, GATE1_BATCH_SIZE = 3.318335548548489e-4, 2048  # the GNN's Gate-1 configuration (ADR-012:210-211)


class AsOfStatistics:
    """Each (node, time) pair's view statistics, read from sorted dated edges by binary search."""

    def __init__(self, fwd: Tensor, edge_time: Tensor, edge_type: Tensor, node_time: Tensor, device: torch.device):
        self.span = int(edge_time.max()) + 1
        nodes = torch.cat([fwd[0], fwd[1]])
        times = torch.cat([edge_time, edge_time])
        cols = torch.cat([edge_type, edge_type]) - EDGE_TYPE_MIN
        key = nodes * self.span + times
        order = torch.argsort(key)
        m = key.numel()
        cum = torch.zeros((m + 1, NUM_EDGE_TYPES), dtype=torch.int32)
        cum[torch.arange(1, m + 1), cols[order]] = 1
        self.cum = cum.cumsum(0, dtype=torch.int32).to(device)
        self.key = key[order].to(device)
        self.times = times[order].to(device)
        self.node_time = node_time.to(device)

    def __call__(self, nodes: Tensor, at: Tensor) -> Tensor:
        lo = torch.searchsorted(self.key, nodes * self.span)
        hi = torch.searchsorted(self.key, nodes * self.span + at, right=True)
        if not bool((hi > lo).all()):
            raise AssertionError("a node in a subgraph has no edge dated at or before its seed's time")
        hist = (self.cum[hi] - self.cum[lo]).float()
        since_last = (at - self.times[hi - 1]).float()
        since_first = (at - self.node_time[nodes]).float()
        return torch.cat([hist, since_last[:, None], since_first[:, None]], dim=1)


def quantiles(values: np.ndarray) -> str:
    p = np.percentile(values, [50, 90, 99])
    return f"min {values.min()}, median {p[0]:.0f}, p90 {p[1]:.0f}, p99 {p[2]:.0f}, max {values.max()}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-batches", type=int, default=None, help="stop the epoch early, as a smoke test")
    args = ap.parse_args()

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                            text=True).stdout.strip()
    say(f"commit {commit} | torch {torch.__version__} | device {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    t0 = time.perf_counter()
    data = load_dgraph(REPO_ROOT / "data" / "dgraph")
    x, node_time = data.x, data.node_time
    keep = data.edge_time <= TRAIN_MAX
    fwd, et, ety = data.edge_index[:, keep], data.edge_time[keep], data.edge_type[keep]
    del data, keep   # labels and masks go here, unread
    users = (node_time <= TRAIN_MAX).nonzero().view(-1)
    say(f"load: {time.perf_counter() - t0:.1f} s; users with node time <= {TRAIN_MAX}: {users.numel():,}; "
        f"forward edges dated <= {TRAIN_MAX}: {et.numel():,}")

    t0 = time.perf_counter()
    stats = AsOfStatistics(fwd, et, ety, node_time, device)
    say(f"statistics table built in {time.perf_counter() - t0:.1f} s; GPU memory held "
        f"{torch.cuda.memory_allocated() / 2**30 if device.type == 'cuda' else 0:.2f} GB")

    say("\nCheck: the binary search against view_node_features, every user in the view")
    view_369 = None
    for t in CHECK_CUTOFFS:
        k = et <= t
        t0 = time.perf_counter()
        feats, in_view = view_node_features(fwd[:, k], et[k], ety[k], node_time, t)
        cpu_s = time.perf_counter() - t0
        nodes = in_view.nonzero().view(-1)
        t0 = time.perf_counter()
        fast = torch.cat([stats(c.to(device), torch.full_like(c, t).to(device)).cpu()
                          for c in nodes.split(1_000_000)])
        fast_s = time.perf_counter() - t0
        if not torch.equal(fast, feats[nodes]):
            raise AssertionError(f"the binary search disagrees with view_node_features at t={t}")
        say(f"  t={t}: equal on {nodes.numel():,} users; view_node_features {cpu_s:.2f} s, search {fast_s:.2f} s")
        if t == TRAIN_MAX:
            view_369 = feats

    say("\nA. At each user's own node time")
    degree = torch.cat([stats(c.to(device), node_time[c].to(device))[:, :NUM_EDGE_TYPES].sum(1).cpu()
                        for c in users.split(1_000_000)]).long().numpy()
    say(f"  edges incident to the user: {quantiles(degree)}")
    say(f"  users with exactly one edge: {(degree == 1).mean():.4f}")

    both = torch.cat([fwd, fwd.flip(0)], dim=1)
    empty = torch.empty(0)
    view = GraphView(t_max=TRAIN_MAX, edge_index=both, edge_time=torch.cat([et, et]),
                     edge_type=torch.cat([ety, ety]), node_features=empty, in_view=empty)
    g = torch.Generator().manual_seed(0)
    sample = users[torch.randperm(users.numel(), generator=g)[:NEIGHBOUR_SAMPLE]]
    for hops, word in ((2, "two"), (3, "three")):   # the model has three layers (config.yaml, gnn.num_layers)
        t0 = time.perf_counter()
        counts = []
        for batch in first_appearance_loader(x, view, node_time, sample, [-1] * hops, 1024):
            assert_temporal_batch(batch)
            pairs = torch.unique(torch.stack([batch.batch, batch.n_id]), dim=1)
            counts.append(torch.bincount(pairs[0], minlength=batch.batch_size)[: batch.batch_size] - 1)
        within = torch.cat(counts).numpy()
        say(f"  other users within {word} hops, {sample.numel():,} sampled users, {time.perf_counter() - t0:.1f} s: {quantiles(within)}")
        say(f"  sampled users with at most one other user within {word} hops: {(within <= 1).mean():.4f}")

    say("\nB. Which input columns each transform would zero (Standardizer zeroes a constant column)")
    own = torch.cat([torch.cat([x[c], stats(c.to(device), node_time[c].to(device)).cpu()], dim=1)
                     for c in users.split(1_000_000)])
    own_fit = Standardizer.fit(own, torch.ones(own.size(0), dtype=torch.bool))
    del own
    gated = torch.cat([x[users], view_369[users]], dim=1)
    gated_fit = Standardizer.fit(gated, torch.ones(gated.size(0), dtype=torch.bool))
    del gated
    for name, fit in (("fitted as of each user's own time", own_fit), ("fitted on the view as of 369 (ADR-011 clause 3)", gated_fit)):
        zeroed = [n for n, c in zip(INPUT_FEATURE_NAMES, fit.constant.tolist()) if c]
        say(f"  {name}: {zeroed or 'none'}")
    del view_369, own_fit
    transform = Standardizer(mean=gated_fit.mean.to(device), std=gated_fit.std.to(device),
                             constant=gated_fit.constant.to(device))
    assert INPUT_FEATURE_NAMES[-len(VIEW_FEATURE_NAMES):] == VIEW_FEATURE_NAMES

    say("\nC. One training epoch under ADR-023 clause 2: shuffled across times, inputs as of each seed's time")
    seed_everything(0)
    cfg = yaml.safe_load((REPO_ROOT / "adapters" / "dgf1" / "config.yaml").read_text(encoding="utf-8"))["gnn"]
    hp = GNNHParams(**{**cfg, "fan_out": tuple(cfg["fan_out"]), "lr": GATE1_LR, "batch_size": GATE1_BATCH_SIZE})
    model = build_model(x.size(1) + len(VIEW_FEATURE_NAMES), hp, device)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    targets = torch.randint(0, 2, (x.size(0),), generator=torch.Generator().manual_seed(1))
    loader = first_appearance_loader(x, view, node_time, users, fan_out_for(hp), hp.batch_size, shuffle=True)
    process = psutil.Process()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def now() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    spent = {"sampling": 0.0, "inputs": 0.0, "training": 0.0}
    sizes, peak_rss, n_batches = [], 0, 0
    model.train()
    started = mark = now()
    for batch in loader:
        spent["sampling"] += now() - mark
        assert_temporal_batch(batch)

        mark = now()
        batch = batch.to(device)
        statistics = stats(batch.n_id, batch.seed_time[batch.batch])
        inputs = transform.transform(torch.cat([batch.x, statistics], dim=1))
        spent["inputs"] += now() - mark

        mark = now()
        opt.zero_grad()
        _, logits = model(inputs, batch.edge_index)
        seeds = batch.batch_size
        loss_fn(logits[:seeds], targets[batch.n_id[:seeds].cpu()].to(device)).backward()
        opt.step()
        spent["training"] += now() - mark

        sizes.append(batch.n_id.numel())
        n_batches += 1
        if n_batches % 50 == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)
        if n_batches % 500 == 0:
            say(f"  {n_batches:,} batches, {now() - started:.0f} s")
        if args.max_batches and n_batches >= args.max_batches:
            break
        mark = now()
    total = now() - started
    peak_rss = max(peak_rss, process.memory_info().rss)

    done = "full epoch" if not args.max_batches else f"PARTIAL, {n_batches} batches only"
    say(f"  {done}: {total:.1f} s for {n_batches:,} batches at batch size {hp.batch_size}, fan-out {fan_out_for(hp)}")
    say(f"  of which sampling {spent['sampling']:.1f} s, inputs {spent['inputs']:.1f} s, training {spent['training']:.1f} s")
    say(f"  nodes per batch: median {np.median(sizes):.0f}, max {max(sizes):,}")
    if device.type == "cuda":
        say(f"  GPU peak: allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB")
    say(f"  process memory peak (sampled every 50 batches): {peak_rss / 2**30:.2f} GB")
    say("  This epoch trains every user up to step 369, so a labelled-seed epoch is smaller. No metric was computed.")

    if not args.max_batches:
        (Path(__file__).parent / "output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
