"""Round 20 lane worker: single-threaded workers pull jobs from one shared queue (file lock) in the fixed order of
run_round20.jobs(mode); jobs already recorded as ok are skipped (resumable). Records: reports/round20[/devA|/devB]/results_lane<k>.jsonl ($R20_MODE devA|devB|formal)."""
import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODE = __import__("os").environ.get("R20_MODE", "devA")
BASE = ROOT / ("reports/round20/" + MODE if MODE != "formal" else "reports/round20")
QUEUE = BASE / "queue_state.json"


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
    BASE.mkdir(parents=True, exist_ok=True)
    os.environ["R20_OUT"] = str(BASE / f"results_lane{a.lane}.jsonl")
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_round20 as F  # noqa: E402  (reads R20_OUT at import)
    J, skip = F.jobs(MODE), F.done(MODE)
    while True:
        j = claim(a.lane, J, skip)
        if j is None:
            break
        t0 = time.time()
        r = F.run_job(j, MODE)
        print(f"[lane {a.lane}] {j} {r['status']} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
