"""Loaders for the round-2 report (reference audit, failure recovery, symbolic extraction). Read-only: every value
comes from a saved record or log; nothing is typed by hand."""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
LOGS = REP / "logs"

# (label, log file, design revision, decision) in chronological order
GATE_RUNS = [("Original design, attempt 1", "stageD_gate_attempt1.log", "rev. 2: functional linear front", "D33"),
             ("Original design, repair 1 (epochs)", "stageD_gate_attempt2.log", "rev. 2", "D33"),
             ("Original design, repair 2 (capacity)", "stageD_gate_attempt3.log", "rev. 2", "D33"),
             ("Revision 3: NFL front", "rev3_gate_attempt1.log", "rev. 3: pointwise-MLP front (A1)", "D38"),
             ("D41-selected configuration [SUPERSEDED, D53 cache defect]", "d41_gate_selected.log",
              "tensor-spline front (A3), r = 8", "D41/D45"),
             ("After D46 extension, same config [SUPERSEDED, D53]", "stageD_gate_ext.log", "A3 (extension kept it)",
              "D46/D47"),
             ("A3 configuration, recomputed with the fixed cache", "stageD_gate_D53.log", "A3 (search and extension "
              "recomputed: same selection)", "D53")]
GATE_RE = re.compile(r"R2: C = ([0-9.]+), oracle C = ([0-9.]+); probe R2 \(a,b,A,omega\) = \[([^\]]+)\], mean ([0-9.]+); "
                     r"epochs \[([0-9, ]+)\]")


def jsonl(path):
    p = REP / path
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def gate_history():
    out = []
    for label, f, rev, dec in GATE_RUNS:
        txt = (LOGS / f).read_text()
        m = GATE_RE.search(txt)
        r1 = re.search(r"R1: C = ([0-9.]+), oracle C = ([0-9.]+), sigma = ([0-9.]+)", txt)
        r3 = re.search(r"\|C - C_R2\| = ([0-9.]+)", txt)
        res = re.search(r"=+ (.*) in [0-9.]+s", txt)
        out.append(dict(label=label, log=f, revision=rev, decision=dec, C=float(m.group(1)), oracle=float(m.group(2)),
                        threshold=float(m.group(2)) - 0.03, probe=[float(x) for x in m.group(3).split()],
                        probe_mean=float(m.group(4)), epochs=m.group(5), R1C=float(r1.group(1)) if r1 else None,
                        R3dC=float(r3.group(1)) if r3 else None, summary=res.group(1) if res else ""))
    return out


def search_records():
    return [r for r in jsonl("learnability_search.jsonl")]


def confirmations():
    return {"D41 (seeds 101–105)": jsonl("learnability_confirmation_fae_mlp.jsonl"),
            "D46 (seeds 106–110)": jsonl("learnability_confirmation_fae_mlp_ext.jsonl")}


def symbolic_synth():
    return jsonl("kanaft_symbolic.jsonl")


def censoring_variants():
    recs = jsonl("censoring_variants.jsonl")
    beta = np.array([0.8, -0.5])
    out = {}
    for sc in ("CA2", "CA1", "MIS"):
        rs = [r for r in recs if r["scenario"] == sc]
        if not rs:
            continue
        out[sc] = {"censored": float(np.mean([r["censored_frac"] for r in rs])), "n_rep": len(rs), "est": {}}
        for k in rs[0]["slopes"]:
            b = np.array([r["slopes"][k] for r in rs])
            out[sc]["est"][k] = dict(bias=(b.mean(0) - beta).tolist(), rmse=np.sqrt(((b - beta) ** 2).mean(0)).tolist())
    return out


def kanaft_pbc():
    return jsonl("kanaft_pbc.jsonl")
