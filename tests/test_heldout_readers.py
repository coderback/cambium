"""The backstop over committed code, and the scripts ADR-021 retired (I11, clause 8, row 5a step 4).

I6 to I8, what the default load leaves out, are the guarantee. The backstop only catches raw routes. Its
limits are stated in ADR-021 clause 5: it is per file, and it cannot tell a held-out registry read from a
validation one.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import assemble_gate_dgf1_1 as assembler  # noqa: E402
import audit_dgf1_datasource as audit  # noqa: E402
import measure_dgf1_temporal_split as measure  # noqa: E402
import run_dgf1_floor  # noqa: E402
import run_dgf1_gnn  # noqa: E402
import verify_dgraph_snapshot as verifier  # noqa: E402

import adapters.dgf1.datasource_dgraph as DS  # noqa: E402
import adapters.dgf1.heldout as H  # noqa: E402

ACCESSOR_MODULES = ("adapters/dgf1/datasource_dgraph.py", "adapters/dgf1/eval.py", "adapters/dgf1/heldout.py")
HAND_PATH = "scripts/heldout_ledger.py"
SCOPE = ("scripts/", "adapters/dgf1/", "notebooks/")
RAW_CALLS = {"DGraphFin": "PyG loader", "TemporalSplit": "hand-built window",
             "_load_raw": "the raw loader"}
RAW_IMPORTS = {"DGraphFin": "PyG loader", "_load_raw": "the raw loader"}
OPENERS = {"open_test_window", "open_official_track"}
HAND_KEYWORDS = ("incidental", "backfill", "now")
# Where held-out data lives on disk: the snapshot, and the score files.
STORES = (("data", "dgraph"), ("experiments", "scores"))


def _name(node: ast.AST) -> str:
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


def _aliases(tree: ast.AST) -> dict[str, str]:
    """``import x as y`` and ``from m import x as y``: the local name, mapped to the real one."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.asname:
                    out[a.asname] = a.name.rsplit(".", 1)[-1]
    return out


def _store_path(node: ast.AST) -> bool:
    """A literal path into a held-out store: "data/dgraph/..." or ROOT / "experiments" / "scores"."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parts = tuple(p for p in re.split(r"[/\\]", node.value) if p)
        return any(parts[i:i + 2] == s for s in STORES for i in range(len(parts)))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        strs = [n for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        consts = [n.value for n in sorted(strs, key=lambda n: (n.lineno, n.col_offset))]   # source order
        return any(tuple(consts[i:i + 2]) == s for s in STORES for i in range(len(consts)))
    return False


def _mask_name(s: str) -> bool:
    return s.endswith("_mask") and s not in ("labelled_mask", "train_seed_mask", "window_target_mask")


def backstop(sources: dict[str, str]) -> list[str]:
    """Code that could reach held-out data without an accessor, or claims an incidental look.

    Parsed, not grepped: a pattern inside a string (a test fixture, this lint's own tables) is not a
    route to the data, and a call is.
    """
    out = []
    for name, src in sources.items():
        if (name in ACCESSOR_MODULES or not name.startswith(SCOPE)
                or re.match(r"notebooks/measurements/[^/]+/prototype/", name)):   # ADR evidence
            continue
        tree = ast.parse(src)
        alias = _aliases(tree)
        found, opens, reads_scores = set(), False, False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = _name(node.func)
                fn = alias.get(fn, fn)
                if fn in RAW_CALLS:
                    found.add(RAW_CALLS[fn])
                opens |= fn in OPENERS
                reads_scores |= fn == "load_scores"
                if name != HAND_PATH and any(k.arg in HAND_KEYWORDS for k in node.keywords):
                    found.add("sets a look's authority or time by hand")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    real = a.name.rsplit(".", 1)[-1]
                    if real in RAW_IMPORTS:
                        found.add(RAW_IMPORTS[real])
            elif _store_path(node):
                found.add("a held-out store path")
            elif isinstance(node, ast.Attribute) and _mask_name(node.attr):
                found.add("an official mask")
            elif (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                  and isinstance(node.slice.value, str) and _mask_name(node.slice.value)):
                found.add("an official mask")
        if reads_scores and not opens:
            found.add("score files without an accessor")
        out += [f"{name}: {f}" for f in sorted(found)]
    return sorted(out)


# ------------------------------------------------------------------ I11: the backstop


def test_the_backstop_finds_each_raw_route_and_ignores_strings():
    srcs = {
        "scripts/a.py": "data = DGraphFin(root=r)[0]\n",
        "scripts/b.py": "m = data.official_test_mask\n",
        "scripts/b2.py": "m = data.official_train_mask | data.official_val_mask\n",
        "scripts/c.py": "m = d['test_mask']\n",
        "scripts/d.py": "s = TemporalSplit(train_max=1, test_min=2, test_max=3)\n",
        "scripts/e.py": "g = load_scores(p)\n",
        "scripts/f.py": "record_look(p, incidental=flag)\n",
        "scripts/f2.py": "record_look(p, backfill=True)\n",
        "scripts/f3.py": "record_look(p, now=t)\n",
        "scripts/al.py": "from torch_geometric.datasets import DGraphFin as D\n",
        "scripts/al2.py": "from gbe.eval.scores import load_scores as ls\ng = ls(p)\n",
        "scripts/al3.py": "import x as tsplit\nfrom m import TemporalSplit as TS\ns = TS(1, 2, 3)\n",
        "scripts/raw.py": "from adapters.dgf1.heldout import _load_raw\n",
        "scripts/raw2.py": "import adapters.dgf1.heldout as h\nr = h._load_raw()\n",
        "scripts/sp.py": "a = np.load('data/dgraph/raw/dgraphfin.npz')\n",
        "scripts/sp2.py": "d = ROOT / 'experiments' / 'scores'\n",
        "scripts/sp3.py": "d = 'experiments\\\\scores\\\\run.npz'\n",
        "scripts/nsp.py": "d = ROOT / 'scores' / 'experiments'\nNEVER = ('data/', 'experiments/')\n",
        "scripts/ok.py": "t = data.labelled_mask & train_seed_mask(d, s) & window_target_mask(d, s)\n",
        "notebooks/g.py": "PAT = 'DGraphFin( and test_mask'\n",
        "notebooks/k.py": "data = DGraphFin(root=r)[0]\n",
        "scripts/h.py": "w = open_test_window(c, tier='scores')\ng = load_scores(p)\n",
        "tests/i.py": "data = DGraphFin(root=r)[0]\n",
        "scripts/test_j.py": "data = DGraphFin(root=r)[0]\n",
        "notebooks/measurements/x/prototype/p.py": "m = raw['test_mask']\n",
        "scripts/prototype/q.py": "m = raw['test_mask']\n",
        "adapters/dgf1/eval.py": "s = TemporalSplit(1, 2, 3)\n",
        HAND_PATH: "record_look(p, incidental=True, backfill=True)\n",
    }
    hand = "sets a look's authority or time by hand"
    assert backstop(srcs) == sorted([
        "notebooks/k.py: PyG loader", "scripts/a.py: PyG loader", "scripts/b.py: an official mask",
        "scripts/b2.py: an official mask", "scripts/c.py: an official mask",
        "scripts/d.py: hand-built window", "scripts/e.py: score files without an accessor",
        f"scripts/f.py: {hand}", f"scripts/f2.py: {hand}", f"scripts/f3.py: {hand}",
        "scripts/al.py: PyG loader", "scripts/al2.py: score files without an accessor",
        "scripts/al3.py: hand-built window",
        "scripts/raw.py: the raw loader", "scripts/raw2.py: the raw loader",
        "scripts/sp.py: a held-out store path", "scripts/sp2.py: a held-out store path",
        "scripts/sp3.py: a held-out store path",
        "scripts/test_j.py: PyG loader",
        "scripts/prototype/q.py: an official mask",
    ])


def test_the_code_in_this_repository_has_no_raw_route():
    """Committed and untracked code alike. A route found later is added here with a mutant (ADR-021)."""
    names = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "scripts/*.py", "adapters/dgf1/*.py",
                            "notebooks/*.py"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert len(names) > 20, "the scope found the repository's scripts"
    srcs = {n: (REPO_ROOT / n).read_text(encoding="utf-8") for n in names}
    assert backstop(srcs) == []


# ------------------------------------------------------------------ the scripts in clause 8


def test_the_gate_1_assembler_refuses_at_its_start():
    with pytest.raises(SystemExit, match="retired by ADR-021"):
        assembler.main()


def _synthetic():
    """Users 0 to 7. Two appear at 100, two at 400, two at 500, two at 700. The training users 0 and 1 have a
    second edge, dated 700, so a count that kept edges past the validation window would move."""
    y = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    edge_index = torch.tensor([[0, 2, 4, 6, 0], [1, 3, 5, 7, 1]])
    edge_time = torch.tensor([100, 400, 500, 700, 700])
    edge_type = torch.ones(5, dtype=torch.long)
    node_time = DS.node_time_earliest_edge(edge_index, edge_time, 8)
    return DS.prepare_dgraph(torch.zeros(8, DS.EXPECTED_FEATURES), edge_index, edge_time, edge_type,
                             DS.hide_later_labels(y, node_time, 481))


def test_the_audit_pins_no_test_window_figure():
    assert not [k for k in audit.PINNED if k.startswith("test") or "minus" in k]


def test_the_audit_builds_no_view_past_the_validation_window(monkeypatch, capsys):
    cutoffs = []
    real = audit.graph_view
    monkeypatch.setattr(audit, "graph_view", lambda d, t: cutoffs.append(t) or real(d, t))
    monkeypatch.setattr(audit, "load_dgraph", lambda **k: _synthetic())
    monkeypatch.setattr(sys, "argv", ["audit"])
    audit.main()
    out = capsys.readouterr().out
    assert cutoffs == [369, 481]
    assert "test" not in out.replace("clause-5 test", "")


def test_the_audit_reads_raw_features_only_up_to_the_validation_window(monkeypatch, capsys):
    data = _synthetic()
    data.x[4:] = -1          # users after the window: a figure that counted them would move
    monkeypatch.setattr(audit, "load_dgraph", lambda **k: data)
    monkeypatch.setattr(sys, "argv", ["audit"])
    audit.main()
    assert "-1, users up to 481: 0.00%" in capsys.readouterr().out


def test_the_measurement_counts_nothing_past_the_validation_window(monkeypatch, capsys):
    monkeypatch.setattr(DS, "load_dgraph", lambda **k: _synthetic())
    monkeypatch.setattr(measure, "elliptic_check", lambda: None)
    measure.main()
    out = capsys.readouterr().out
    assert "test" not in out and "official" not in out
    assert re.search(r"later edges between in-window nodes:\s+0\b", out), "the edge dated 700 is dropped first"
    assert re.search(r"edge-time filtered.*:\s+1\b", out)


def _verifier_data(**over):
    fields = dict(x=SimpleNamespace(shape=(verifier.EXPECTED_NODES, verifier.EXPECTED_FEATURES)),
                  edge_index=SimpleNamespace(size=lambda _: verifier.EXPECTED_EDGES),
                  edge_time=torch.tensor(verifier.EXPECTED_EDGE_TIME),
                  edge_type=torch.arange(1, verifier.EXPECTED_EDGE_TYPES + 1))
    return SimpleNamespace(**{**fields, **over})   # no y and no masks: reading either raises


def test_the_verifier_reads_labels_only_through_the_integrity_check(monkeypatch, capsys):
    monkeypatch.setattr(verifier, "_require_data", lambda root: None)
    monkeypatch.setattr(DS, "load_dgraph", lambda **k: _verifier_data())
    monkeypatch.setattr(H, "integrity_ok", lambda: True)
    monkeypatch.setattr(sys, "argv", ["verify"])
    assert verifier.main() == 0
    assert "pass or fail only" in capsys.readouterr().out


def test_the_verifier_escalates_on_a_failed_integrity_check(monkeypatch):
    monkeypatch.setattr(verifier, "_require_data", lambda root: None)
    monkeypatch.setattr(DS, "load_dgraph", lambda **k: _verifier_data())
    monkeypatch.setattr(H, "integrity_ok", lambda: False)
    monkeypatch.setattr(sys, "argv", ["verify"])
    with pytest.raises(AssertionError, match="label totals"):
        verifier.main()


# ------------------------------------------------------------------ the runners' window branch


@pytest.mark.parametrize("runner", [run_dgf1_floor, run_dgf1_gnn])
def test_validation_comes_from_the_default_path_and_records_nothing(runner, monkeypatch):
    def refuse(**kwargs):
        raise AssertionError("a validation run opened the test window")

    monkeypatch.setattr(H, "open_test_window", refuse)
    split, labels = runner.open_window("val", "pilot", None)
    assert (split.test_min, split.test_max) == (370, 481) and labels is None


@pytest.mark.parametrize("runner", [run_dgf1_floor, run_dgf1_gnn])
def test_a_test_window_mode_opens_the_window_through_the_accessor(runner, monkeypatch):
    seen = {}
    monkeypatch.setattr(H, "open_test_window", lambda **k: seen.update(k) or ("window", "labels"))
    assert runner.open_window("test", "gate3", "decisions/ADR-900.md") == ("window", "labels")
    assert seen == {"tier": "scores", "what": "gate3 batch scores", "by": "researcher",
                    "authority": "decisions/ADR-900.md", "batch": "gate3"}


@pytest.mark.parametrize("path", ["scripts/run_dgf1_floor.py", "scripts/run_dgf1_gnn.py"])
def test_each_runner_puts_the_labels_back_before_scoring(path):
    source = (REPO_ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n.func.id for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "open_window" in calls and "with_labels" in calls and "load_dgraph" in calls
    guards = [n for n in ast.walk(main) if isinstance(n, ast.If) and "labels is not None" in ast.get_source_segment(source, n.test)]
    assert len(guards) == 1 and "data = with_labels(data, labels)" in ast.get_source_segment(source, guards[0])
