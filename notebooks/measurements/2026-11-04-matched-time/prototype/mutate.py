"""Mutation check for matched-time's prototypes. It works on temporary copies and only reads the repo.

    python mutate.py rule
    python mutate.py guards
    python mutate.py view [start] [stop]
    python mutate.py attribute

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
_GATE = "    if not go_earned(adr_text, rows, delta_gated):\n"
_BUDGET = 'return all(status(rows, k) == "void" for k in attempts_in(rows))'
_LIVE = 'live = [k for k in found if status(rows, k) != "void"]'
_PATHS = 'CODE_PATHS = ("adapters/", "gbe/", "scripts/", "configs/")'
_REPRO_LOOP = "    owed = []\n    for arm in ARMS:\n"
_BATCH_REPRO_LOOP = "    pilot, own = pilot_rows(rows, k), tagged(rows, BATCH_REPRO_TAG)\n    for arm in ARMS:\n"
_REFUSAL_LOOP = "        for arm in ARMS:\n            for metric in METRICS:\n"
_RERUN_OWN = 'own = [r for r in tagged(rows, BATCH_TAG) if _metrics(r).get("arm") == arm and r["seed"] == str(seed)]'
GUARD_MUTANTS = [
    # the go line's shape
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
    ("the working tree is read", "    return shown.stdout\n", '    return (repo / relative_path).read_text(encoding="utf-8")\n'),
    ("a document not in HEAD reads as empty", "    if shown.returncode != 0:\n", "    if False:\n"),
    # the go line is earned, and the check comes first
    ("the line's shape is not checked", "    if not go_recorded(adr_text) or k is None:\n", "    if k is None:\n"),
    ("the line need not name the attempt's rows", "return named == used and bool(decide", "return bool(decide"),
    ("the line may name part of the rows", "return named == used and", "return named <= used and"),
    ("the rule is not recomputed for the line", 'return named == used and bool(decide(clean_attempt(rows, k), delta_gated)["go"])', "return named == used"),
    ("the rule reads the last clean attempt", 'return next((k for k in attempts_in(rows) if status(rows, k) == "clean"), None)', 'return next((k for k in reversed(attempts_in(rows)) if status(rows, k) == "clean"), None)'),
    ("the opener runs before the check", _GATE, "    opener()\n" + _GATE),
    ("nothing is checked", _GATE, "    if False:\n"),
    ("the batch's reproduction is not checked", "    check_batch_reproduced(rows, first_clean(rows), head)\n", ""),
    ("the opener runs before the batch's reproduction", "    check_batch_reproduced(rows, first_clean(rows), head)\n    return opener()\n", "    result = opener()\n    check_batch_reproduced(rows, first_clean(rows), head)\n    return result\n"),
    # one attempt's rows
    ("every attempt's rows are read", "    own = pilot_rows(rows, k)\n", "    own = rows\n"),
    ("reproduction rows are read as pilot rows", "    own = pilot_rows(rows, k)\n", "    own = pilot_rows(rows, k) + repro_rows(rows, k)\n"),
    ("the filter ignores the attempt", 'return [r for r in rows if _metrics(r).get("experiment") == tag]', 'return [r for r in rows if str(_metrics(r).get("experiment")).rsplit("_", 1)[0] == tag.rsplit("_", 1)[0]]'),
    ("a missing metric is not a void attempt", "    except (ValueError, KeyError) as why:\n", "    except ValueError as why:\n"),
    ("arms at two commits pass", "    _one_commit(rows, k)\n", ""),
    ("a reproduction on another commit passes", 'commits = {r["git_commit"] for r in (*pilot_rows(rows, k), *repro_rows(rows, k))}', 'commits = {r["git_commit"] for r in pilot_rows(rows, k)}'),
    ("a later pilot row changes the status", "return tagged(rows, pilot_tag(k))[:PILOT_ROWS]", "return tagged(rows, pilot_tag(k))"),
    ("a later reproduction row changes the status", "return tagged(rows, repro_tag(k))[:REPRO_ROWS]", "return tagged(rows, repro_tag(k))"),
    ("the last pilot rows count, not the first", "return tagged(rows, pilot_tag(k))[:PILOT_ROWS]", "return tagged(rows, pilot_tag(k))[-PILOT_ROWS:]"),
    ("the last reproduction rows count, not the first", "return tagged(rows, repro_tag(k))[:REPRO_ROWS]", "return tagged(rows, repro_tag(k))[-REPRO_ROWS:]"),
    # reproduction
    ("reproduction is not checked", "    _check_reproduced(rows, k)\n", ""),
    ("reproduction checks only DGF-1", _REPRO_LOOP, "    owed = []\n    for arm in ARMS[:1]:\n"),
    ("reproduction checks only the floor", _REPRO_LOOP, "    owed = []\n    for arm in ARMS[1:]:\n"),
    ("a reproduction's seed is not checked", '    if any(r["seed"] != "0" for r in repro):\n', "    if False:\n"),
    ("two reproduction rows on an arm pass", "if len(first) != 1 or len(second) > 1:", "if len(first) != 1:"),
    ("a reproduction not yet run reads as failed", "            owed.append(arm)\n            continue\n", '            raise ReproductionFailed(f"attempt {k}, arm {arm}: never ran")\n'),
    ("a reproduction not yet run passes", "    if owed:\n", "    if False:\n"),
    ("the score hash is not compared", 'REPRODUCED_KEYS = (*METRICS, "scores_sha256")', "REPRODUCED_KEYS = (*METRICS,)"),
    ("the metrics are not compared", 'REPRODUCED_KEYS = (*METRICS, "scores_sha256")', 'REPRODUCED_KEYS = ("scores_sha256",)'),
    ("two missing hashes count as a match", "if first.get(key) is None or first.get(key) != _metrics(second).get(key):", "if first.get(key) != _metrics(second).get(key):"),
    ("a reproduction row need not be clean", "    if not _clean_row(second):\n", "    if False:\n"),
    ("a dirty row counts as clean", 'return row["git_dirty"] == "false" and _metrics(row)', "return _metrics(row)"),
    ("a non-deterministic row counts as clean", ' and _metrics(row).get("deterministic") is True and ', " and "),
    ("an errored row counts as clean", ' and "ERRORED" not in row["notes"]', ""),
    # what the runner will write
    ("a pilot may start after a clean attempt", _LIVE, 'live = [k for k in found if status(rows, k) == "owed"]'),
    ("a pilot may start with a reproduction owed", _LIVE, 'live = [k for k in found if status(rows, k) == "clean"]'),
    ("no root cause is needed", "    if missing:\n", "    if False:\n"),
    ("only the last attempt needs a root cause", "missing = [k for k in found if k not in root_causes]", "missing = [k for k in found[-1:] if k not in root_causes]"),
    ("a retry on the same commit is allowed", 'if head in {r["git_commit"] for r in pilot_rows(rows, last)} and last not in set(environmental):', "if False:"),
    ("an environmental cause does not help", " and last not in set(environmental):", ":"),
    ("the next attempt reuses a number", "    return last + 1\n", "    return last\n"),
    ("the first attempt is misnumbered", "    if not found:\n        return 1\n", "    if not found:\n        return 0\n"),
    ("a reproduction may run on another commit", '    if head != pilot_rows(rows, k)[0]["git_commit"]:\n', "    if False:\n"),
    ("a reproduction may run when none is owed", '    if status(rows, k) != "owed":\n', "    if False:\n"),
    # the budget, and what a decline prints
    ("the budget locks on a void attempt", _BUDGET, "return not attempts_in(rows)"),
    ("the budget never locks", _BUDGET, "return True"),
    ("the budget can decline with a reproduction owed", _BUDGET, 'return all(status(rows, k) != "clean" for k in attempts_in(rows))'),
    ("the budget reads only the first attempt", _BUDGET, 'return all(status(rows, k) == "void" for k in attempts_in(rows)[:1])'),
    ("a void attempt reaches the rule in a decline", "result = decide(clean_attempt(rows, k), delta_gated)", "result = decide(rule_inputs(attempt_values(rows, k, DGF1_ARM), attempt_values(rows, k, FLOOR_ARM)), delta_gated)"),
    ("an owed reproduction prints as a failure", "        except ReproductionFailed as why:\n", "        except (ReproductionFailed, ReproductionOwed) as why:\n"),
    ("a void attempt's line hides its gaps", 'f"attempt {k} (void): {_gaps(rows, k)}; rule not computed: reproduction failed; "', 'f"attempt {k} (void): rule not computed: reproduction failed; "'),
    ("a void attempt's line does not say why", 'rule not computed: reproduction failed; "', 'rule not computed; "'),
    ("a broken attempt reads as a failed reproduction", 'rule not computed: broken pilot; "', 'rule not computed: reproduction failed; "'),
    ("a void attempt's root cause is not printed", """lines.append(f"attempt {k} (void): {_gaps(rows, k)}; rule not computed: reproduction failed; "
                         f"{why}; root cause: {root_causes.get(k, 'not found')}")""",
     """lines.append(f"attempt {k} (void): {_gaps(rows, k)}; rule not computed: reproduction failed; "
                         f"{why}")"""),
    ("the gap has the wrong sign", "means[0] - means[1]", "means[1] - means[0]"),
    ("the gap reads rows that do not count", 'values = [_metrics(r)[m] for r in pilot_rows(rows, k) if', 'values = [_metrics(r)[m] for r in tagged(rows, pilot_tag(k)) if'),
    ("a clean attempt's line hides the rule", """; rule: {'go' if result['go'] else 'no-go'}")""", '")'),
    ("attempts print out of order", "return sorted(found)", "return sorted(found, reverse=True)"),
    ("a decline with no rows prints nothing", 'return lines or ["no pilot rows exist; rule not computed"]', "return lines"),
    # the test batch
    ("the batch's reproduction ignores the commit", ' and r["seed"] == "0" and r["git_commit"] == head]', ' and r["seed"] == "0"]'),
    ("the batch's reproduction may be missing or doubled", "        if len(second) != 1:\n", "        if len(second) < 1:\n"),
    ("the batch's reproduction is not compared", '        why = _matches(first[0], second[0], f"arm {arm}")\n        if why:\n            raise BrokenBatch(why)\n', ""),
    ("the batch's reproduction checks only DGF-1", _BATCH_REPRO_LOOP, _BATCH_REPRO_LOOP.replace("in ARMS:", "in ARMS[:1]:")),
    ("the batch's reproduction checks only the floor", _BATCH_REPRO_LOOP, _BATCH_REPRO_LOOP.replace("in ARMS:", "in ARMS[1:]:")),
    ("code may sit between the looks", "return not any(p.startswith(CODE_PATHS) for p in changed_paths)", "return True"),
    ("one clean path excuses the code", "return not any(p.startswith(CODE_PATHS)", "return not all(p.startswith(CODE_PATHS)"),
    ("adapters may change between the looks", _PATHS, 'CODE_PATHS = ("gbe/", "scripts/", "configs/")'),
    ("gbe may change between the looks", _PATHS, 'CODE_PATHS = ("adapters/", "scripts/", "configs/")'),
    ("scripts may change between the looks", _PATHS, 'CODE_PATHS = ("adapters/", "gbe/", "configs/")'),
    ("configs may change between the looks", _PATHS, 'CODE_PATHS = ("adapters/", "gbe/", "scripts/")'),
    ("an errored row with a metric is skipped", """            if any(key in m for key in METRICS):
                raise BrokenBatch(f"run {r['run_id']} errored after a metric was written")
""", ""),
    ("an errored row is not skipped", "        if _errored(r):\n            if any(key in m", "        if False:\n            if any(key in m"),
    ("a row that is not clean is read", """        if not _clean_row(r):
            raise BrokenBatch(f"run {r['run_id']} is not a clean, deterministic row")
""", ""),
    ("a completed row may run twice", """        if int(r["seed"]) in values:
            raise BrokenBatch(f"arm {arm} has seed {r['seed']} twice; a completed row is never re-run")
""", ""),
    ("a row with no finite metric is read", "        if metric not in m or not math.isfinite(float(m[metric])):\n", "        if metric not in m:\n"),
    ("a row with no metric crashes", "        if metric not in m or not math.isfinite(float(m[metric])):\n", "        if not math.isfinite(float(m[metric])):\n"),
    ("a short arm passes", "    if set(values) != set(range(n)):\n", "    if not set(values) <= set(range(n)):\n"),
    ("an extra seed passes", "    if set(values) != set(range(n)):\n", "    if not set(range(n)) <= set(values):\n"),
    ("another arm's rows are read", '        if m.get("arm") != arm:\n            continue\n        if _errored(r):', "        if _errored(r):"),
    ("another tag's rows are read", "    for r in tagged(rows, BATCH_TAG):\n        m = _metrics(r)\n", "    for r in rows:\n        m = _metrics(r)\n"),
    ("a re-run is allowed beside a completed row", "    if not errored or len(errored) != len(own):\n", "    if not errored:\n"),
    ("a re-run needs no errored row", "    if not errored or len(errored) != len(own):\n", "    if len(errored) != len(own):\n"),
    ("a re-run is allowed after a metric", "    if any(key in _metrics(r) for r in errored for key in METRICS):\n", "    if False:\n"),
    ("a re-run is allowed on another commit", '    if any(r["git_commit"] != head for r in errored):\n', "    if False:\n"),
    ("a re-run reads any seed", _RERUN_OWN, 'own = [r for r in tagged(rows, BATCH_TAG) if _metrics(r).get("arm") == arm]'),
    ("a re-run reads any arm", _RERUN_OWN, 'own = [r for r in tagged(rows, BATCH_TAG) if r["seed"] == str(seed)]'),
    ("a broken batch prints a reading", '        return f"refused: broken batch: {why}"\n', "        return None\n"),
    ("the refusal checks only DGF-1", _REFUSAL_LOOP, _REFUSAL_LOOP.replace("in ARMS:", "in ARMS[:1]:")),
    ("the refusal checks only the floor", _REFUSAL_LOOP, _REFUSAL_LOOP.replace("in ARMS:", "in ARMS[1:]:")),
    ("the refusal checks only ROC-AUC", _REFUSAL_LOOP, _REFUSAL_LOOP.replace("in METRICS:", "in METRICS[:1]:")),
    ("the refusal checks only AUPRC", _REFUSAL_LOOP, _REFUSAL_LOOP.replace("in METRICS:", "in METRICS[1:]:")),
    ("an exploratory hash mismatch keeps the rows", ' or _metrics(found[0]).get("scores_sha256") != expected:', ":"),
    ("a missing or doubled exploratory row keeps the rows", "        if len(found) != 1 or not _clean_row(found[0])", "        if not _clean_row(found[0])"),
    ("a dirty exploratory row keeps the rows", "if len(found) != 1 or not _clean_row(found[0]) or ", "if len(found) != 1 or "),
    ("the exploratory check reads only one arm", "for arm, expected in gated_pilot_hashes.items():", "for arm, expected in list(gated_pilot_hashes.items())[:1]:"),
    ("the exploratory check reads any tag", "    own = tagged(rows, EXPLORATORY_REPRO_TAG)\n", "    own = rows\n"),
]

AT = "at = row_times(batch)"
FULL = "torch.full_like(batch.n_id, window_end)"
FIT_FEATS = "    feats, _ = view_node_features(fwd, et, ety, g.node_time, train_max)\n"
REVERSE = "both, times, types = torch.cat([fwd, fwd.flip(0)], dim=1), torch.cat([et, et]), torch.cat([ety, ety])"
TO_MODEL = "_, logits = model(transform.transform(raw), batch.edge_index)"
RECENCY = "        check_seed_recency(batch, raw)\n"
WINDOW_FEATS = "    feats, _ = view_node_features(fwd, et, ety, node_time_earliest_edge(fwd, et, g.x.size(0)), last)\n"
WINDOW_MODEL = "model(transform.transform(torch.cat([batch.x, stats[batch.n_id]], dim=1)), batch.edge_index)"
WINDOW_SCORE = "X_score = torch.cat([g.x[targets], window_only_statistics(g, first, last)[targets]], dim=1)"
WINDOW_STATS = "    stats = window_only_statistics(g, first, last)\n    graph ="
VIEW_MUTANTS = [
    ("static inputs on every path", AT, f"at = {FULL}"),
    ("window-end inputs at scoring only", AT, f"at = {FULL} if not shuffle else row_times(batch)"),
    ("window-end inputs at training only", AT, f"at = {FULL} if shuffle else row_times(batch)"),
    ("each neighbour is read at its own time", AT, "at = g.node_time[batch.n_id]"),
    ("the floor's training rows are static", "    X_train = own_time_rows(g, seeds, train_max)\n", "    X_train = torch.cat([g.x[seeds], statistics_table(g, train_max)(seeds, torch.full_like(seeds, train_max))], dim=1)\n"),
    ("the floor's scoring rows are static", "    X_score = own_time_rows(g, targets, window_end)\n", "    X_score = torch.cat([g.x[targets], statistics_table(g, window_end)(targets, torch.full_like(targets, window_end))], dim=1)\n"),
    ("the floor's training rows are transformed", "    X_train = own_time_rows(g, seeds, train_max)\n", "    X_train = fit_transform(g, train_max).transform(own_time_rows(g, seeds, train_max))\n"),
    ("the floor's scoring rows are transformed", "    X_score = own_time_rows(g, targets, window_end)\n", "    X_score = fit_transform(g, train_max).transform(own_time_rows(g, targets, window_end))\n"),
    ("the floor is handed the wrong users' labels", "y_score=g.y[targets].numpy(), feature_names=INPUT_FEATURE_NAMES, meta={})\n    return fit_predict(\"xgboost\", design, seed)\n\n\n# ---", "y_score=g.y[targets + 1].numpy(), feature_names=INPUT_FEATURE_NAMES, meta={})\n    return fit_predict(\"xgboost\", design, seed)\n\n\n# ---"),
    ("a reverse edge is misdated", REVERSE, "both, times, types = torch.cat([fwd, fwd.flip(0)], dim=1), torch.cat([et, torch.zeros_like(et)]), torch.cat([ety, ety])"),
    ("a reverse edge is dated 3 steps early", REVERSE, "both, times, types = torch.cat([fwd, fwd.flip(0)], dim=1), torch.cat([et, et - 3]), torch.cat([ety, ety])"),
    ("reverse edges are dropped", REVERSE, "both, times, types = fwd, et, ety"),
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
    ("the GNN's inputs skip the transform", TO_MODEL, "_, logits = model(raw, batch.edge_index)"),
    ("age is counted from the dataset's end", "since_first = (at - self.node_time[nodes]).float()", "since_first = (self.span - 1 - self.node_time[nodes]).float()"),
    ("recency reads the first edge, not the last", "since_last = (at - self.times[hi - 1]).float()", "since_last = (at - self.times[lo]).float()"),
    ("the histogram counts later edges", "hi = torch.searchsorted(self.key, nodes * self.span + at, right=True)", "hi = torch.searchsorted(self.key, nodes * self.span + self.span - 1, right=True)"),
    ("same-step edges are left out", "hi = torch.searchsorted(self.key, nodes * self.span + at, right=True)", "hi = torch.searchsorted(self.key, nodes * self.span + at, right=False)"),
    ("training batches are not shuffled", "fan_out, batch_size, shuffle=True, loss=loss)", "fan_out, batch_size, shuffle=False, loss=loss)"),
    ("scoring is shuffled", "[-1] * layers, batch_size, shuffle=False)", "[-1] * layers, batch_size, shuffle=True)"),
    ("scoring samples at a fan-out", "transform, [-1] * layers, batch_size", "transform, [1] * layers, batch_size"),
    ("the temporal check is removed", "        assert_temporal_batch(batch)\n", ""),
    ("the recency check is removed", RECENCY, ""),
    ("the recency check reads nothing", "if bool(raw[: batch.batch_size, -2:].any()):", "if False:"),
    ("the recency check reads the first seed only", "if bool(raw[: batch.batch_size, -2:].any()):", "if bool(raw[:1, -2:].any()):"),
    ("the recency check reads neighbours too", "if bool(raw[: batch.batch_size, -2:].any()):", "if bool(raw[:, -2:].any()):"),
    ("the recency check reads one column", "if bool(raw[: batch.batch_size, -2:].any()):", "if bool(raw[: batch.batch_size, -1:].any()):"),
    ("the recency check recomputes instead of reading the rows", RECENCY, "        check_seed_recency(batch, raw_inputs(batch, stats, batch.seed_time[batch.batch]))\n"),
    ("the model is handed flipped edges", TO_MODEL, "_, logits = model(transform.transform(raw), batch.edge_index.flip(0))"),
    ("the loss reads labels in the unshuffled order", "loss(logits[: batch.batch_size], g.y[own])", "loss(logits[: batch.batch_size], g.y[seeds[: batch.batch_size]])"),
    ("the loss reads every row, not the seeds", "loss(logits[: batch.batch_size], g.y[own])", "loss(logits, g.y[batch.n_id])"),
    # the exploratory rows
    ("window-only statistics count reverse edges", WINDOW_FEATS, "    both = torch.cat([fwd, fwd.flip(0)], dim=1)\n    feats, _ = view_node_features(both, torch.cat([et, et]), torch.cat([ety, ety]), node_time_earliest_edge(fwd, et, g.x.size(0)), last)\n"),
    ("window-only keeps edges with one endpoint inside", "inside[g.fwd[0]] & inside[g.fwd[1]]", "(inside[g.fwd[0]] | inside[g.fwd[1]])"),
    ("window-only edges run past the window", "keep = (g.edge_time <= last) & inside", "keep = inside"),
    ("window-only drops reverse edges", "edge_index=torch.cat([fwd, fwd.flip(0)], dim=1), num_nodes", "edge_index=fwd, num_nodes"),
    ("window-only inputs read the gated view", WINDOW_STATS, "    stats = view_node_features(*g.as_of(last), g.node_time, last)[0]\n    graph ="),
    ("window-only inputs skip the transform", WINDOW_MODEL, "model(torch.cat([batch.x, stats[batch.n_id]], dim=1), batch.edge_index)"),
    ("window-only samples at a fan-out", "graph, num_neighbors=[-1] * layers, input_nodes=targets", "graph, num_neighbors=[1] * layers, input_nodes=targets"),
    ("the views are swapped for the GNN", '    if view == "first_appearance":\n        score_pass(', '    if view == "window_only":\n        score_pass('),
    ("the GNN's path takes any view", "    else:\n        _refuse(view)\n\n\ndef gated_rows", "    else:\n        score_pass(model, g, targets, last, transform, layers, batch_size)\n\n\ndef gated_rows"),
    ("the floor's path takes any view", "    else:\n        _refuse(view)\n    design", "    else:\n        X_score = gated_rows(g, targets, last)\n    design"),
    ("the exploratory floor trains on own-time rows", "X_train=gated_rows(g, seeds, train_max).numpy()", "X_train=own_time_rows(g, seeds, train_max).numpy()"),
    ("the exploratory floor trains at the window's end", "X_train=gated_rows(g, seeds, train_max).numpy()", "X_train=gated_rows(g, seeds, last).numpy()"),
    ("the exploratory floor's training rows are transformed", "X_train=gated_rows(g, seeds, train_max).numpy()", "X_train=fit_transform(g, train_max).transform(gated_rows(g, seeds, train_max)).numpy()"),
    ("the first-appearance floor reads the gated view", '        X_score = own_time_rows(g, targets, last)\n', "        X_score = gated_rows(g, targets, last)\n"),
    ("the window-only floor reads the gated view", WINDOW_SCORE, "X_score = gated_rows(g, targets, last)"),
    ("the window-only floor's rows are transformed", WINDOW_SCORE, f"X_score = fit_transform(g, train_max).transform({WINDOW_SCORE[10:]})"),
    ("the exploratory floor is handed the wrong labels", "y_train=g.y[seeds].numpy(),\n                         X_score", "y_train=g.y[seeds + 1].numpy(),\n                         X_score"),
]

# Which test catches what: ADR-023 clause 8 names these, and a kill under -x says only that something failed.
ATTRIBUTION = [
    ("a reverse edge is dated 3 steps early", "exact_edges"),
    ("a reverse edge is dated 3 steps early", "door_equals"),
    ("a reverse edge is misdated", "exact_edges"),
    ("the recency check recomputes instead of reading the rows", "run_time_check"),
    ("static inputs, with the recency check removed", "door_equals"),
    ("the transform is refit on the scoring view", "scoring_door"),
    ("window-only statistics count reverse edges", "window_only_gnn_door or window_only_floor_scores"),
    ("a reverse edge is dated 3 steps early, with the statistics' raise removed", "exact_edges"),
    ("a reverse edge is dated 3 steps early, with the statistics' raise removed", "door_equals"),
    ("the loss reads labels in the unshuffled order", "loss_reads"),
]
_NO_RAISE = ('''        if not bool((hi > lo).all()):
            raise AssertionError("a node in a subgraph has no edge dated at or before its seed's time")
''', "")
EXTRA_MUTANTS = {
    "static inputs, with the recency check removed": [(f"{AT}\n        raw = raw_inputs(batch, stats, at)\n{RECENCY}", f"at = {FULL}\n        raw = raw_inputs(batch, stats, at)\n")],
    "a reverse edge is dated 3 steps early, with the statistics' raise removed": [
        (REVERSE, "both, times, types = torch.cat([fwd, fwd.flip(0)], dim=1), torch.cat([et, et - 3]), torch.cat([ety, ety])"), _NO_RAISE],
}

GROUPS = {"rule": (RULE, RULE_MUTANTS), "guards": (GUARDS, GUARD_MUTANTS), "view": (VIEW, VIEW_MUTANTS)}


def localise(text: str) -> str:
    return text.replace(HOME, f'Path(r"{REPO}")')


def run(d: Path, test: str, select: str | None = None) -> tuple[bool, bool, str]:
    """(failed, collection error, last line). With ``select``, only those tests run, and all of them."""
    how = ["-k", select] if select else ["-x"]
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", *how, "-p", "no:cacheprovider", test], cwd=d, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    last = lines[-1] if lines else r.stderr[-200:]
    return r.returncode != 0, " error" in last and " failed" not in last, last


def attribute() -> None:
    """Each named mutant against the named tests alone: FAILED means those tests catch it, whatever else does."""
    source = localise((HERE / VIEW).read_text(encoding="utf-8"))
    named = {name: [(old, new)] for name, old, new in VIEW_MUTANTS} | EXTRA_MUTANTS
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        d = Path(tmp)
        for f in (*PROTOS, *TESTS.values()):
            (d / f).write_text(localise((HERE / f).read_text(encoding="utf-8")), encoding="utf-8")
        for name, select in ATTRIBUTION:
            mutated = source
            for old, new in named[name]:
                assert mutated.count(old) == 1, f"{name}: anchor found {mutated.count(old)}x"
                mutated = mutated.replace(old, new)
            (d / VIEW).write_text(source, encoding="utf-8")
            _, _, clean = run(d, TESTS[VIEW], select)
            (d / VIEW).write_text(mutated, encoding="utf-8")
            failed, error, last = run(d, TESTS[VIEW], select)
            word = "FAILED " if failed and not error else ("ERROR  " if error else "passed ")
            print(f"{word} {name} | -k {select} | unmutated: {clean[:24]} | mutated: {last[:44]}", flush=True)


def main() -> None:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # below normal
    group = sys.argv[1]
    if group == "attribute":
        attribute()
        return
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
