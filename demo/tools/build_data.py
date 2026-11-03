"""Build demo/site-data/results.json from committed files only.

Reads experiments/registry.csv, gates/ and decisions/. Every arm is matched to registry rows by
the run ids its gate file lists, and each mean and std is checked against the printed gate row, so
the build fails if the page would disagree with a signed gate. Keys withdrawn by ERRATUM-DGF1-1
(window_only_*, first_appearance_*) are never copied.

    python demo/tools/build_data.py
"""

import csv
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REG = {r["run_id"]: r for r in csv.DictReader(open(ROOT / "experiments/registry.csv", encoding="utf8"))}
RUN = re.compile(r"(?:ell1|dgf1)-\d{8}T\d{6}Z-[0-9a-f]{8}")


def metrics(run_id):
    m = json.loads(REG[run_id]["metrics_json"])
    return {k: v for k, v in m.items() if not k.startswith(("window_only_", "first_appearance_"))}


def section(text, heading):
    start = text.index(heading)
    nxt = text.find("\n##", start + len(heading))
    return text[start:] if nxt < 0 else text[start:nxt]


def arm(gate_text, heading, keys, printed_cols):
    """keys: registry metric names. printed_cols: table column index of each key (seed is col 0)."""
    block = section(gate_text, heading)
    rows = []
    for line in block.splitlines():
        ids = RUN.findall(line)
        if not line.startswith("|") or len(ids) != 1:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        m = metrics(ids[0])
        row = {"seed": int(cells[0]), "run": ids[0], "commit": REG[ids[0]]["git_commit"][:7]}
        for key, col in zip(keys, printed_cols):
            assert abs(m[key] - float(cells[col])) < 6e-5, (ids[0], key, m[key], cells[col])
            row[key] = m[key]
        rows.append(row)
    assert rows, heading
    mean_line = next(l for l in block.splitlines() if "mean ± std" in l)
    cells = [c.strip().strip("*") for c in mean_line.strip("|").split("|")]
    out = {"rows": rows, "n": len(rows), "summary": {}}
    for key, col in zip(keys, printed_cols):
        vals = [r[key] for r in rows]
        mean = st.mean(vals)
        # ELL-1 Gates 0 and 1 print the population std; later gates pin ddof=1 (ADR-006). Keep whichever the gate printed.
        pm, ps = (float(x) for x in cells[col].split("±"))
        sd = next(v for v in (st.stdev(vals), st.pstdev(vals)) if abs(v - ps) < 6e-5)
        assert abs(mean - pm) < 6e-5, (heading, key, mean, sd, cells[col])
        out["summary"][key] = {"mean": mean, "std": sd, "ddof": 1 if sd == st.stdev(vals) else 0}
    return out


def gate(name):
    return (ROOT / "gates" / name).read_text(encoding="utf8")


ELL = ["illicit_f1", "illicit_recall", "illicit_auc"]
g0, g1, g3, d1 = gate("GATE-ELL1-0.md"), gate("GATE-ELL1-1.md"), gate("GATE-ELL1-3.md"), gate("GATE-DGF1-1.md")
data = {
    "ell1g0": {"rf": arm(g0, "### Random Forest", ELL, [1, 2, 3]), "lr": arm(g0, "### Logistic Regression", ELL, [1, 2, 3])},
    "ell1g1": {"gcn": arm(g1, "### GCN", ELL, [1, 2, 3]), "sage": arm(g1, "### GraphSAGE", ELL, [1, 2, 3])},
    "ell1g3": {a: arm(g3, f"### {a}", ELL, [1, 2, 3]) for a in ["real", "scrambled", "random", "no_edges", "config"]},
    "dgf1g1": {},
}
DK = ["fraud_auc", "fraud_auprc"]
sample = metrics(next(RUN.finditer(section(d1, "### DGF-1 — gated"))).group(0))
assert all(k in sample for k in DK), sorted(sample)
data["dgf1g1"]["gnn"] = arm(d1, "### DGF-1 — gated", DK, [1, 2])
data["dgf1g1"]["parity"] = arm(d1, "### Parity floor", DK, [1, 2])
data["dgf1g1"]["raw17"] = arm(d1, "### Raw-17 floor", DK, [1, 2])

boot = []
for line in section(d1, "## Paired bootstrap").splitlines():
    m = re.match(r"\| (\d) \| ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\] \| ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\] \|", line)
    if m:
        v = [float(x) for x in m.groups()]
        boot.append({"seed": int(v[0]), "roc": v[1:4], "auprc": v[4:7]})
assert len(boot) == 8
data["bootstrap"] = boot

# Local-94 diagnostic: reported, not gated, dirty tree. scripts/run_ell1_local94.py writes the arms in
# this order and its rows at c4bc923 echo no arm label (lab 2026-07-25), so they are split by run order.
rows94 = [r for r in REG.values() if r["git_commit"].startswith("c4bc923")]
gnn94 = [r for r in rows94 if "gnn" in r["notes"]]
rf94 = [r for r in rows94 if "floor: rf" in r["notes"]]
assert len(gnn94) == 16 and len(rf94) == 8
FIELDS = ("illicit_f1", "illicit_precision", "illicit_recall", "illicit_auc")
data["local94"] = {
    name: {f: st.mean(json.loads(r["metrics_json"])[f] for r in rs) for f in FIELDS} | {"n": len(rs)}
    for name, rs in (("gnn_local94", gnn94[:8]), ("gnn_all165", gnn94[8:]), ("rf_all165", rf94))
}

adrs = []
for p in sorted((ROOT / "decisions").glob("ADR-*.md")):
    if "template" in p.name:
        continue
    lines = p.read_text(encoding="utf8").splitlines()
    title = re.sub(r"^# ADR-\d+ — ", "", lines[0])
    status = next(l for l in lines if l.startswith("**Status:**")).replace("**Status:**", "").strip()
    state = status.split("·")[0].strip().split()[0]
    date_line = next((l for l in lines if l.startswith("**Date:**")), "")
    date = re.search(r"\d{4}-\d{2}-\d{2}", date_line)
    adrs.append({"id": p.name[:7], "file": p.name, "title": title, "state": state,
                 "erratum": "erratum" in status, "date": date.group(0) if date else None})
data["adrs"] = adrs

groups = Counter(f'{r["model"]}/{r["phase"]}' for r in REG.values())
data["registry"] = {"rows": len(REG), "groups": dict(sorted(groups.items()))}

out = ROOT / "demo" / "site-data"
out.mkdir(parents=True, exist_ok=True)
(out / "results.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf8")
print("registry rows", len(REG), "adrs", len(adrs), "local94", json.dumps(data["local94"], indent=0))
