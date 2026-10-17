"""D61 verdict, mechanically from the records: CONFIRMED iff R1, R2 and R3 of the revised gate (seed 41) passed
in the logged gate run AND >= 4 of 5 fresh confirmation seeds (111-115) passed. Writes reports/revised_gate_verdict.json."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"


def main():
    log = (REP / "logs" / "revised_gate_fa.log").read_text()
    outcome = {k: bool(re.search(rf"test_{k}_\S+ .*?PASSED", log, re.S) and not re.search(rf"FAILED \S+test_{k}_", log))
               for k in ("R1", "R2", "R3")}
    for k in outcome:                      # robust: read pytest's short summary lines
        outcome[k] = f"FAILED tests/test_learnability_fa.py::test_{k}" not in log and \
            re.search(rf"test_learnability_fa.py::test_{k}_\w+.*?(PASSED|FAILED)", log, re.S).group(1) == "PASSED"
    recs = [json.loads(l) for l in (REP / "learnability_confirmation_fae_mlp_fa.jsonl").read_text().splitlines()
            if l.strip()]
    seeds = sorted(r["data_seed"] for r in recs)
    assert seeds == [111, 112, 113, 114, 115], seeds
    n_pass = sum(r["pass"] for r in recs)
    m = re.search(r"R2: C = ([0-9.]+), oracle C = ([0-9.]+); probe R2 \(a,b,A,omega\) = \[([^\]]+)\], mean ([0-9.]+)", log)
    v = dict(gate_seed=41, gate=outcome, gate_R2_C=float(m.group(1)), gate_R2_threshold=float(m.group(2)) - 0.03,
             gate_probe=[float(x) for x in m.group(3).split()], gate_probe_mean=float(m.group(4)),
             confirmations_passed=n_pass, confirmations_total=len(recs),
             verdict="CONFIRMED" if all(outcome.values()) and n_pass >= 4 else "NOT CONFIRMED",
             rule="R1, R2, R3 pass on seed 41 and >= 4/5 of seeds 111-115 pass (D61)")
    (REP / "revised_gate_verdict.json").write_text(json.dumps(v, indent=1))
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
