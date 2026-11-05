"""Mutation check for matched-time's prototypes. It works on temporary copies and only reads the repo.

    python mutate.py rule
    python mutate.py guards
    python mutate.py view [start] [stop]

Each mutant is named for what it breaks, and a mutant that only stops the module importing does not count:
the unmutated copy must pass, and each kill must be a failed test, not an error at collection.
"""
import ctypes
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
HOME = "Path(__file__).resolve().parents[4]"
RULE, GUARDS, VIEW = "go_no_go_proto.py", "matched_guards_proto.py", "matched_view_proto.py"
PROTOS = (RULE, GUARDS, VIEW)
TESTS = {RULE: "test_go_no_go_proto.py", GUARDS: "test_matched_guards_proto.py", VIEW: "test_matched_view_proto.py"}

RULE_MUTANTS = [
    ("the rule receives the arms' gap", "    spreads: Mapping[str, tuple[float, float]]\n", "    spreads: Mapping[str, tuple[float, float]]\n    gap: float = 0.0\n"),
    ("the bar reads the lower mean", "    best = max(float(", "    best = min(float("),
    ("the bar reads DGF-1's mean only, so it sees direction", 'float(np.mean(list(floor["fraud_auc"].values()))))', "-1.0)"),
    ("the bar is strict", "sane = inputs.best_auc >= SANITY_BAR", "sane = inputs.best_auc > SANITY_BAR"),
    ("the bar is lower", "SANITY_BAR = 0.55 ", "SANITY_BAR = 0.50 "),
    ("the bar is ignored", 'return {"go": sane and bool(informative)', 'return {"go": bool(informative)'),
    ("precision is ignored", 'return {"go": sane and bool(informative)', 'return {"go": sane'),
    ("both metrics must be informative", "and bool(informative)", "and len(informative) == 2"),
    ("the power bar is lower", "if power[m] >= POWER_TARGET", "if power[m] >= 0.5"),
    ("the power uses the single-look level", "FINAL_N, m, alpha / 2)", "FINAL_N, m, alpha)"),
    ("the power is at fewer seeds", "FINAL_N = 20", "FINAL_N = 12"),
    ("the margin is the whole gap", "MARGIN_FACTOR = 0.5 ", "MARGIN_FACTOR = 1.0 "),
    ("no gated gain still has a margin", "if delta_gated > 0 else None", "if delta_gated >= 0 else None"),
    ("a negative gated gain has a margin", "return MARGIN_FACTOR * delta_gated if delta_gated > 0 else None", "return MARGIN_FACTOR * abs(delta_gated) or None"),
    ("neither arm varying crashes", "    m = margin(delta_gated)\n    if m is None or (s_dgf1 == 0 and s_floor == 0):\n        return 0.0\n    return welch_power(s_dgf1, s_floor, FINAL_N, m, alpha / 2)",
     "    m = margin(delta_gated)\n    if m is None:\n        return 0.0\n    return welch_power(s_dgf1, s_floor, FINAL_N, m, alpha / 2)"),
    ("one arm not varying fails precision", "    m = margin(delta_gated)\n    if m is None or (s_dgf1 == 0 and s_floor == 0):\n        return 0.0\n    return welch_power(s_dgf1, s_floor, FINAL_N, m, alpha / 2)",
     "    m = margin(delta_gated)\n    if m is None or (s_dgf1 == 0 or s_floor == 0):\n        return 0.0\n    return welch_power(s_dgf1, s_floor, FINAL_N, m, alpha / 2)"),
    ("the spreads are swapped with the wrong metric", "sample_var(list(floor[m].values()))", "sample_var(list(floor[METRICS[0]].values()))"),
    ("extra seeds are read", "set(values[metric]) != set(PILOT_SEEDS)", "not set(PILOT_SEEDS) <= set(values[metric])"),
    ("missing seeds are read", "set(values[metric]) != set(PILOT_SEEDS)", "not set(values[metric]) <= set(PILOT_SEEDS)"),
    ("a missing metric is not refused", "if metric not in values or set(", "if metric in values and set("),
    ("non-finite values are read", "if not all(math.isfinite(v) for v in values[metric].values()):", "if False:"),
    ("only one arm is checked", 'for arm, values in (("dgf1", dgf1), ("floor", floor)):', 'for arm, values in (("dgf1", dgf1),):'),
    ("the margin clause does not shift", "np.asarray(floor, dtype=float) + m, dgf1", "np.asarray(floor, dtype=float), dgf1"),
    ("the margin clause shifts the wrong way", "np.asarray(floor, dtype=float) + m, dgf1", "np.asarray(floor, dtype=float) - m, dgf1"),
    ("the margin clause does not swap the arms", "judge(np.asarray(floor, dtype=float) + m, dgf1,", "judge(dgf1, np.asarray(floor, dtype=float) + m,"),
    ("a refused gain clause still reads", '    if gain["refused"]:\n        return "refused"\n', ""),
    ("a refused margin clause counts", 'under = below is not None and not below["refused"] and bool(below["resolvable"])',
     'under = below is not None and (bool(below["refused"]) or bool(below["resolvable"]))'),
    ("a resolvable gain hides the margin", '    if resolved and under:\n        return "resolvable and below the margin"\n', ""),
    ("below the margin reads as nothing", 'return "below the margin" if under else "not resolvable"', 'return "not resolvable"'),
    ("a second look is never triggered", 'return stage2_triggered([r for r in results if r["clause"] == "gain"])', "return False"),
    ("a margin failure triggers a second look", 'return stage2_triggered([r for r in results if r["clause"] == "gain"])', "return stage2_triggered(results)"),
    ("power at a gain ignores the gain", "FINAL_N, m - gain, alpha / 2)", "FINAL_N, m, alpha / 2)"),
    ("power at a gain uses the single-look level", "FINAL_N, m - gain, alpha / 2)", "FINAL_N, m - gain, alpha)"),
    ("power at a gain has a value with no margin", "    m = margin(delta_gated)\n    if m is None or (s_dgf1 == 0 and s_floor == 0):\n        return 0.0\n    return welch_power(s_dgf1, s_floor, FINAL_N, m - gain, alpha / 2)",
     "    m = margin(delta_gated) or 0.02\n    if s_dgf1 == 0 and s_floor == 0:\n        return 0.0\n    return welch_power(s_dgf1, s_floor, FINAL_N, m - gain, alpha / 2)"),
    ("stage 1 stays final after stage 2", "last = stage2 if stage2 is not None else stage1", "last = stage1"),
    ("the plan keeps a metric where neither arm varies", "             if varies(list(dgf1[m].values())) or varies(list(floor[m].values()))}", "             }"),
    ("the plan drops a metric where one arm varies", "if varies(list(dgf1[m].values())) or varies(", "if varies(list(dgf1[m].values())) and varies("),
    ("an empty plan is not refused", "    if not pilot:\n", "    if False:\n"),
    ("the plan reads the wrong arm", 'PilotClause("DGF-1 above the floor", dgf1[m], floor[m])', 'PilotClause("DGF-1 above the floor", floor[m], dgf1[m])'),
    ("a failed metric is not labelled under-resolved", 'else f"under-resolved, {p:.3f}"', 'else f"{p:.3f}"'),
    ("the label's bar is lower", 'f"{p:.3f}" if p >= POWER_TARGET else', 'f"{p:.3f}" if p >= 0.4 else'),
    ("a metric with no margin prints a power", '    if margin(delta_gated[metric]) is None:\n        return "no margin clause"\n', ""),
]

_GO_SRC = r'_GO = re.compile(r"\*\*Go/no-go:\*\* go · \d{4}-\d{2}-\d{2} · \S")'
GUARD_MUTANTS = [
    ("the go line is not anchored", r'_GO_LINE_START = re.compile(r"^\*\*Go/no-go:\*\*.*$", re.MULTILINE)', r'_GO_LINE_START = re.compile(r"\*\*Go/no-go:\*\*.*$", re.MULTILINE)'),
    ("a no-go opens the window", _GO_SRC, r'_GO = re.compile(r"\*\*Go/no-go:\*\* (?:no-)?go · \d{4}-\d{2}-\d{2} · \S")'),
    ("go-ahead opens the window", _GO_SRC, r'_GO = re.compile(r"\*\*Go/no-go:\*\* go.* · \d{4}-\d{2}-\d{2} · \S")'),
    ("a go line needs no date or run ids", _GO_SRC, r'_GO = re.compile(r"\*\*Go/no-go:\*\* go\b")'),
    ("a go line needs no run ids", _GO_SRC, r'_GO = re.compile(r"\*\*Go/no-go:\*\* go · \d{4}-\d{2}-\d{2}")'),
    ("a go line's date is not checked", _GO_SRC, r'_GO = re.compile(r"\*\*Go/no-go:\*\* go · \S+ · \S")'),
    ("a capital Go opens the window", _GO_SRC, r'_GO = re.compile(r"\*\*Go/no-go:\*\* go · \d{4}-\d{2}-\d{2} · \S", re.IGNORECASE)'),
    ("a go beside a no-go opens the window", "return len(lines) == 1 and bool(_GO.match(lines[0]))", "return any(_GO.match(ln) for ln in lines)"),
    ("two go lines open the window", "return len(lines) == 1 and bool(_GO.match(lines[0]))", "return bool(lines) and all(_GO.match(ln) for ln in lines)"),
    ("a substring opens the window", "return len(lines) == 1 and bool(_GO.match(lines[0]))", 'return "**Go/no-go:** go · " in adr_text'),
    ("the opener runs before the check", "    if not go_recorded(adr_text):\n", "    opener()\n    if not go_recorded(adr_text):\n"),
    ("nothing is checked", "    if not go_recorded(adr_text):\n", "    if False:\n"),
    ("the working tree is read", "    return shown.stdout\n", '    return (repo / relative_path).read_text(encoding="utf-8")\n'),
    ("a document not in HEAD reads as empty", "    if shown.returncode != 0:\n", "    if False:\n"),
    ("every attempt's rows are read", "    own = tagged(rows, pilot_tag(k))\n", "    own = rows\n"),
    ("reproduction rows are read as pilot rows", "    own = tagged(rows, pilot_tag(k))\n", "    own = tagged(rows, pilot_tag(k)) + tagged(rows, repro_tag(k))\n"),
    ("the filter ignores the attempt", 'return [r for r in rows if _metrics(r).get("experiment") == tag]', 'return [r for r in rows if str(_metrics(r).get("experiment")).rsplit("_", 1)[0] == tag.rsplit("_", 1)[0]]'),
    ("a missing metric is not a void attempt", "    except (ValueError, KeyError) as why:\n", "    except ValueError as why:\n"),
    ("arms at two commits pass", "    _one_commit(rows, k)\n", ""),
    ("a reproduction on another commit passes", "(*tagged(rows, pilot_tag(k)), *tagged(rows, repro_tag(k)))}", "tagged(rows, pilot_tag(k))}"),
    ("reproduction is not checked", "    _check_reproduced(rows, k)\n", ""),
    ("reproduction checks only DGF-1", "    for arm in ARMS:\n        first =", "    for arm in ARMS[:1]:\n        first ="),
    ("reproduction checks only the floor", "    for arm in ARMS:\n        first =", "    for arm in ARMS[1:]:\n        first ="),
    ("the score hash is not compared", 'REPRODUCED_KEYS = (*METRICS, "scores_sha256")', "REPRODUCED_KEYS = (*METRICS,)"),
    ("the metrics are not compared", 'REPRODUCED_KEYS = (*METRICS, "scores_sha256")', 'REPRODUCED_KEYS = ("scores_sha256",)'),
    ("a reproduction that never ran passes", "        if len(first) != 1 or len(second) != 1:\n            raise ReproductionFailed(f\"attempt {k}, arm {arm}: seed 0 needs one pilot row and one reproduction row\")\n",
     "        if len(first) != 1 or len(second) != 1:\n            continue\n"),
    ("two missing hashes count as a match", "if first[0].get(key) is None or first[0].get(key) != second[0].get(key):", "if first[0].get(key) != second[0].get(key):"),
    ("the budget locks on a void attempt", "            continue\n    return True\n", "            return False\n    return True\n"),
    ("the budget never locks", "            clean_attempt(rows, k)\n            return False\n", "            clean_attempt(rows, k)\n            continue\n"),
    ("a void attempt reaches the rule in a decline", "result = decide(clean_attempt(rows, k), delta_gated)", "result = decide(rule_inputs(attempt_values(rows, k, DGF1_ARM), attempt_values(rows, k, FLOOR_ARM)), delta_gated)"),
    ("a void attempt's line hides its gaps", 'f"attempt {k} (void): {_gaps(rows, k)}; rule not computed: reproduction failed; "', 'f"attempt {k} (void): rule not computed: reproduction failed; "'),
    ("a void attempt's line does not say why", 'rule not computed: reproduction failed; "', 'rule not computed; "'),
    ("a broken attempt reads as a failed reproduction", 'rule not computed: broken pilot; "', 'rule not computed: reproduction failed; "'),
    ("a void attempt's root cause is not printed", """lines.append(f"attempt {k} (void): {_gaps(rows, k)}; rule not computed: reproduction failed; "
                         f"{why}; root cause: {root_causes.get(k, 'not found')}")""",
     """lines.append(f"attempt {k} (void): {_gaps(rows, k)}; rule not computed: reproduction failed; "
                         f"{why}")"""),
    ("the gap has the wrong sign", "means[0] - means[1]", "means[1] - means[0]"),
    ("a clean attempt's line hides the rule", """; rule: {'go' if result['go'] else 'no-go'}")""", '")'),
    ("attempts print out of order", "return sorted(found)", "return sorted(found, reverse=True)"),
    ("a decline with no rows prints nothing", 'return lines or ["no pilot rows exist; rule not computed"]', "return lines"),
]

AT = "at = batch.seed_time[batch.batch]"
FULL = "torch.full_like(batch.n_id, window_end)"
FIT_FEATS = "    feats, _ = view_node_features(fwd, et, ety, g.node_time, train_max)\n"
VIEW_MUTANTS = [
    ("static inputs on every path", AT, f"at = {FULL}"),
    ("window-end inputs at scoring only", AT, f"at = {FULL} if not shuffle else batch.seed_time[batch.batch]"),
    ("window-end inputs at training only", AT, f"at = {FULL} if shuffle else batch.seed_time[batch.batch]"),
    ("each neighbour is read at its own time", AT, "at = g.node_time[batch.n_id]"),
    ("the floor's training rows are static", "    X_train = own_time_rows(g, seeds, train_max)\n", "    X_train = torch.cat([g.x[seeds], statistics_table(g, train_max)(seeds, torch.full_like(seeds, train_max))], dim=1)\n"),
    ("the floor's scoring rows are static", "    X_score = own_time_rows(g, targets, window_end)\n", "    X_score = torch.cat([g.x[targets], statistics_table(g, window_end)(targets, torch.full_like(targets, window_end))], dim=1)\n"),
    ("the floor's training rows are transformed", "    X_train = own_time_rows(g, seeds, train_max)\n", "    X_train = fit_transform(g, train_max).transform(own_time_rows(g, seeds, train_max))\n"),
    ("the floor's scoring rows are transformed", "    X_score = own_time_rows(g, targets, window_end)\n", "    X_score = fit_transform(g, train_max).transform(own_time_rows(g, targets, window_end))\n"),
    ("the floor is handed the wrong users' labels", "y_score=g.y[targets].numpy()", "y_score=g.y[targets + 1].numpy()"),
    ("a reverse edge is misdated", "both, times, types = torch.cat([fwd, fwd.flip(0)], dim=1), torch.cat([et, et]), torch.cat([ety, ety])",
     "both, times, types = torch.cat([fwd, fwd.flip(0)], dim=1), torch.cat([et, torch.zeros_like(et)]), torch.cat([ety, ety])"),
    ("reverse edges are dropped", "both, times, types = torch.cat([fwd, fwd.flip(0)], dim=1), torch.cat([et, et]), torch.cat([ety, ety])", "both, times, types = fwd, et, ety"),
    ("the transform is refit on the scoring view", "gnn_pass(model, g, targets, window_end, transform, [-1] * layers, batch_size, shuffle=False)",
     "gnn_pass(model, g, targets, window_end, fit_transform(g, window_end), [-1] * layers, batch_size, shuffle=False)"),
    ("the transform is fit on every user", "Standardizer.fit(torch.cat([g.x, feats], dim=1), g.node_time <= train_max)", "Standardizer.fit(torch.cat([g.x, feats], dim=1), g.node_time >= -1)"),
    ("the transform is fit past the training view", "    fwd, et, ety = g.as_of(train_max)\n" + FIT_FEATS,
     "    fwd, et, ety = g.as_of(int(g.edge_time.max()))\n    feats, _ = view_node_features(fwd, et, ety, g.node_time, int(g.edge_time.max()))\n"),
    ("the transform is fit at each user's own time", FIT_FEATS,
     "    table = AsOfStatistics(fwd, et, ety, g.node_time)\n    feats = torch.cat([table(u[None], g.node_time[u][None]) if 0 <= g.node_time[u] <= train_max else torch.zeros(1, 13) for u in torch.arange(g.x.size(0))])\n"),
    ("the statistics count reverse edges", "    return AsOfStatistics(fwd, et, ety, g.node_time)\n", "    return AsOfStatistics(torch.cat([fwd, fwd.flip(0)], dim=1), torch.cat([et, et]), torch.cat([ety, ety]), g.node_time)\n"),
    ("the statistics table runs past the window", "    fwd, et, ety = g.as_of(window_end)\n    return AsOfStatistics", "    fwd, et, ety = g.as_of(int(g.edge_time.max()))\n    return AsOfStatistics"),
    ("the eligible edges run past the window", "    fwd, et, ety = g.as_of(window_end)\n    both, times", "    fwd, et, ety = g.as_of(int(g.edge_time.max()))\n    both, times"),
    ("raw features come from the wrong row", "torch.cat([batch.x, stats(batch.n_id, at)], dim=1)", "torch.cat([batch.x.roll(1, 0), stats(batch.n_id, at)], dim=1)"),
    ("the GNN's inputs skip the transform", "return transform.transform(torch.cat([batch.x, stats(batch.n_id, at)], dim=1))", "return torch.cat([batch.x, stats(batch.n_id, at)], dim=1)"),
    ("age is counted from the dataset's end", "since_first = (at - self.node_time[nodes]).float()", "since_first = (self.span - 1 - self.node_time[nodes]).float()"),
    ("recency reads the first edge, not the last", "since_last = (at - self.times[hi - 1]).float()", "since_last = (at - self.times[lo]).float()"),
    ("the histogram counts later edges", "hi = torch.searchsorted(self.key, nodes * self.span + at, right=True)", "hi = torch.searchsorted(self.key, nodes * self.span + self.span - 1, right=True)"),
    ("same-step edges are left out", "hi = torch.searchsorted(self.key, nodes * self.span + at, right=True)", "hi = torch.searchsorted(self.key, nodes * self.span + at, right=False)"),
    ("training batches are not shuffled", "gnn_pass(model, g, seeds, train_max, transform, fan_out, batch_size, shuffle=True)", "gnn_pass(model, g, seeds, train_max, transform, fan_out, batch_size, shuffle=False)"),
    ("scoring is shuffled", "[-1] * layers, batch_size, shuffle=False)", "[-1] * layers, batch_size, shuffle=True)"),
    ("scoring samples at a fan-out", "transform, [-1] * layers, batch_size", "transform, [1] * layers, batch_size"),
    ("the temporal check is removed", "        assert_temporal_batch(batch)\n", ""),
    ("the recency check is removed", "        check_seed_recency(batch, stats)\n", ""),
    ("the recency check reads nothing", "if bool(stats(seeds, batch.seed_time)[:, -2:].any()):", "if False:"),
    ("the model is handed flipped edges", "model(door_inputs(batch, stats, transform, at), batch.edge_index)", "model(door_inputs(batch, stats, transform, at), batch.edge_index.flip(0))"),
]

GROUPS = {"rule": (RULE, RULE_MUTANTS), "guards": (GUARDS, GUARD_MUTANTS), "view": (VIEW, VIEW_MUTANTS)}


def localise(text: str) -> str:
    return text.replace(HOME, f'Path(r"{REPO}")')


def run(d: Path, test: str) -> tuple[bool, bool, str]:
    """(failed, collection error, last line)."""
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", test], cwd=d, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    last = lines[-1] if lines else r.stderr[-200:]
    return r.returncode != 0, " error" in last and " failed" not in last, last


def main() -> None:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # below normal
    group = sys.argv[1]
    proto, mutants = GROUPS[group]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    stop = int(sys.argv[3]) if len(sys.argv) > 3 else len(mutants)
    source = localise((HERE / proto).read_text(encoding="utf-8"))
    for name, old, _ in mutants:
        assert source.count(old) == 1, f"{name}: anchor found {source.count(old)}x: {old!r}"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        d = Path(tmp)
        for f in (*PROTOS, *TESTS.values()):
            (d / f).write_text(localise((HERE / f).read_text(encoding="utf-8")), encoding="utf-8")
        failed, _, last = run(d, TESTS[proto])
        print(f"{group}, unmutated: {'FAILS' if failed else 'passes'} - {last}", flush=True)
        if failed:
            raise SystemExit(1)
        survived = errors = 0
        todo = mutants[start:stop]
        for name, old, new in todo:
            (d / proto).write_text(source.replace(old, new), encoding="utf-8")
            failed, error, last = run(d, TESTS[proto])
            survived += not failed
            errors += error
            word = "SURVIVED" if not failed else ("ERROR   " if error else "KILLED  ")
            print(f"{word} {name} - {last[:60]}", flush=True)
    print(f"\n{group}: {len(todo) - survived - errors}/{len(todo)} killed by a failing test, {errors} by an error, "
          f"{survived} survived ({start} to {stop} of {len(mutants)})", flush=True)


main()
