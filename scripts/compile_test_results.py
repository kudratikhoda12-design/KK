"""Compile reports/test_results.csv from the pytest records (every attempt kept; nothing filtered)."""
import csv
import json
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
recs = [json.loads(l) for l in (ROOT / "reports/logs/test_records.jsonl").read_text().splitlines() if l.strip()]
fields = ["run_id", "stage", "criterion", "kind", "nodeid", "outcome", "phase", "duration_s", "message"]
with open(ROOT / "reports/test_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for r in recs:
        w.writerow({k: r.get(k, "") for k in fields})
# latest outcome per test node (the run order in the file is chronological)
latest = OrderedDict()
for r in recs:
    latest[r["nodeid"]] = r
with open(ROOT / "reports/test_results_latest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for r in latest.values():
        w.writerow({k: r.get(k, "") for k in fields})
runs = OrderedDict()
for r in recs:
    d = runs.setdefault(r["run_id"], {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0})
    d[r["outcome"]] = d.get(r["outcome"], 0) + 1
print(json.dumps(runs, indent=1))
print("latest:", {o: sum(1 for r in latest.values() if r["outcome"] == o) for o in ("passed", "failed", "skipped", "xfailed")})
