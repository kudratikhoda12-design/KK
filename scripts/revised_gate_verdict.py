"""Pre-registered verdicts, mechanically from the records.
  fa   (D61): CONFIRMED iff R1, R2, R3 pass on seed 41 and >= 4/5 of seeds 111-115 pass.
  fa4b (D71): CONFIRMED iff R1-A (amended, D68), R2, R3 pass on seed 61 and >= 4/5 of seeds 121-125 pass; the original
              early-stopped R1 is reported side by side and is not part of the rule.
Writes reports/revised_gate_verdict.json (fa) or reports/revised_gate_verdict_fa4b.json (fa4b)."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
SPEC = {"fa": dict(log="revised_gate_fa.log", test="test_learnability_fa.py", conf="learnability_confirmation_fae_mlp_fa.jsonl",
                   seeds=[111, 112, 113, 114, 115], gate_seed=41, rule_tests=("R1", "R2", "R3"),
                   out="revised_gate_verdict.json", rule="R1, R2, R3 pass on seed 41 and >= 4/5 of seeds 111-115 pass (D61)"),
        "fa4b": dict(log="revised_gate_fa4b.log", test="test_learnability_fa4b.py",
                     conf="learnability_confirmation_fae_mlp_fa4b.jsonl", seeds=[121, 122, 123, 124, 125], gate_seed=61,
                     rule_tests=("R1A", "R2", "R3"), out="revised_gate_verdict_fa4b.json",
                     rule="R1-A (D68), R2, R3 pass on seed 61 and >= 4/5 of seeds 121-125 pass (D71); original R1 "
                          "reported side by side, not part of the rule")}
SPEC["fa4c"] = dict(log="revised_gate_fa4c.log", test="test_learnability_fa4c.py",
                   conf="learnability_confirmation_fae_mlp_fa4c.jsonl", seeds=[131, 132, 133, 134, 135], gate_seed=81,
                   rule_tests=("R1A", "R2", "R3"), out="revised_gate_verdict_fa4c.json",
                   rule="R1-A, amended R2 and R3 pass on seed 81 and amended R2 passes on >= 4/5 of seeds 131-135 "
                        "(D80/D82); original R1 reported side by side, not part of the rule")
TESTNAME = {"R1": "test_R1_oracle_features_head", "R1A": "test_R1A_profile_sigma", "R2": "test_R2_fae_mlp_end_to_end",
            "R3": "test_R3_censoring_awareness"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("revision", nargs="?", default="fa", choices=list(SPEC))
    sp = SPEC[ap.parse_args().revision]
    log = (REP / "logs" / sp["log"]).read_text()
    outcome = {}
    for k, name in TESTNAME.items():
        m = re.search(rf"{sp['test']}::{name}\b.*?(PASSED|FAILED)", log, re.S)
        if m:
            outcome[k] = m.group(1) == "PASSED" and f"FAILED tests/{sp['test']}::{name}" not in log
    recs = [json.loads(l) for l in (REP / sp["conf"]).read_text().splitlines() if l.strip()]
    assert sorted(r["data_seed"] for r in recs) == sp["seeds"], sorted(r["data_seed"] for r in recs)
    n_pass = sum(r["pass"] for r in recs)
    m = re.search(r"R2: C = ([0-9.]+), oracle C = ([0-9.]+); probe R2 \(a,b,A,omega\) = \[([^\]]+)\], mean ([0-9.]+)", log)
    am = re.search(r"R2-amended: C_ceiling = ([0-9.]+); threshold = min\(([0-9.]+), ([0-9.]+)\) = ([0-9.]+)", log)
    thr = float(am.group(4)) if am else float(m.group(2)) - 0.03     # fa4c: the amended threshold (D82)
    v = dict(gate_seed=sp["gate_seed"], gate=outcome, gate_R2_C=float(m.group(1)), gate_R2_threshold=thr,
             gate_C_ceiling=float(am.group(1)) if am else None,
             gate_probe=[float(x) for x in m.group(3).split()], gate_probe_mean=float(m.group(4)),
             confirmations_passed=n_pass, confirmations_total=len(recs),
             verdict="CONFIRMED" if all(outcome.get(k, False) for k in sp["rule_tests"]) and n_pass >= 4 else "NOT CONFIRMED",
             rule=sp["rule"])
    (REP / sp["out"]).write_text(json.dumps(v, indent=1))
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
