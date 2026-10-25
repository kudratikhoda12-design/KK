"""Round 15 lane worker: single-threaded workers pull jobs from one shared queue (file lock) in the fixed order of
run_round15_formal.jobs(); jobs already recorded as ok are skipped (resumable). Records: reports/round15/results_lane<k>.jsonl."""
import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "reports/round15/queue_state.json"


def claim(lane, J, skip):
    with open(str(QUEUE) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = json.loads(QUEUE.read_text()) if QUEUE.exists() else {"next": 0, "claims": []}
        while st["next"] < len(J) and J[st["next"]] in skip:
            st["next"] += 1
        if st["next"] >= len(J):
            QUEUE.write_text(json.dumps(st, indent=1))
            return None
        j = J[st["next"]]
        st["claims"].append(dict(job=j, lane=lane, utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        st["next"] += 1
        QUEUE.write_text(json.dumps(st, indent=1))
        return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", type=int, required=True)
    a = ap.parse_args()
    os.environ["R15_OUT"] = str(ROOT / f"reports/round15/results_lane{a.lane}.jsonl")
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_round15_formal as F  # noqa: E402  (reads R15_OUT at import)
    J, skip = F.jobs(), F.done()
    while True:
        j = claim(a.lane, J, skip)
        if j is None:
            break
        t0 = time.time()
        r = F.run_job(j)
        print(f"[lane {a.lane}] {j} {r['status']} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
