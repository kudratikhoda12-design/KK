"""Round 12 lane worker (D126): 4 single-threaded workers pull (scenario, replicate) jobs from one shared queue in a fixed
priority order (file lock), so the work is balanced without changing any cell.

Order: (1) S-Add core replicates 0-19 (the H-R12 cells b2s_cox and afcm_v3 are the first cells of every replicate);
(2) S-Int, S-Sparse, S-Misspec log-normal and S-Int/P0 core replicates, interleaved by replicate; (3) phase-2 cells
in reverse D113 drop order: flat_ae (S-Sparse), S-Misspec log-logistic, B2 quantile sensitivity (S-Add).
D113 budget rule: once 12 h of wall clock have passed since the launch time (--start-utc), no further phase-2 job is
started; its cells are recorded as dropped_budget. Core jobs (including every H-R12 cell) always run.
Records: reports/round12/results_lane<k>.jsonl and reports/round12/p0_lane<k>.jsonl.
"""
import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "reports/round12/queue_state.json"


def jobs():
    J = [("S-Add", None, r, "core") for r in range(20)]
    for r in range(20):
        J += [("S-Int", None, r, "core"), ("S-Sparse", None, r, "core"), ("S-Misspec", "lognormal", r, "core"),
              ("S-Int/P0", None, r, "core")]
    J += [("S-Sparse", None, r, "phase2") for r in range(20)]
    J += [("S-Misspec", "loglogistic", r, "phase2") for r in range(20)]
    J += [("S-Add", None, r, "phase2") for r in range(20)]
    return J


def claim(lane):
    with open(str(QUEUE) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = json.loads(QUEUE.read_text()) if QUEUE.exists() else {"next": 0, "claims": []}
        J = jobs()
        if st["next"] >= len(J):
            return None
        j = J[st["next"]]
        st["claims"].append(dict(job=st["next"], lane=lane, scenario=j[0], sub=j[1], replicate=j[2], phase=j[3],
                                 utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        st["next"] += 1
        QUEUE.write_text(json.dumps(st, indent=1))
        return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", type=int, required=True)
    ap.add_argument("--start-utc", required=True)
    a = ap.parse_args()
    os.environ["R12_OUT"] = str(ROOT / f"reports/round12/results_lane{a.lane}.jsonl")
    os.environ["KANFAE_RESULTS"] = str(ROOT / f"reports/round12/p0_lane{a.lane}.jsonl")
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_round12 as R  # noqa: E402  (reads R12_OUT at import)
    deadline = (datetime.strptime(a.start_utc, "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
    while True:
        j = claim(a.lane)
        if j is None:
            break
        scen, sub, rep, phase = j
        t0 = time.time()
        if scen == "S-Int/P0":
            sys.argv = ["run_round12_p0.py", "--reps", f"{rep}-{rep}"]
            import run_round12_p0
            run_round12_p0.main()
        else:
            sys.argv = ["run_round12.py", "--scenario", scen, "--reps", f"{rep}-{rep}", "--phase", phase,
                        "--deadline-utc", deadline] + (["--sub", sub] if sub else [])
            R.main()
        print(f"[lane {a.lane}] job {scen}/{sub}/{rep}/{phase} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
