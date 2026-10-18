"""Records every test outcome with its acceptance-criterion ID, DET/STAT kind and stage.

Records are appended to $KANFAE_TEST_RECORD (default reports/logs/test_records.jsonl) and later
compiled into reports/test_results.csv by scripts/compile_test_results.py.
"""
import json
import os
import time
from pathlib import Path

# Round 6 (owner housekeeping, D86): single-threaded numerics for the whole test session, set before numpy/scipy/torch
# are imported by any test module, so suite results match standalone single-threaded runs bit for bit (closes D85).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
_THREAD_LIMITS = None


def pytest_configure(config):
    global _THREAD_LIMITS
    import numpy  # noqa: F401  (load BLAS so the limit below applies to it)
    import scipy.linalg  # noqa: F401
    import torch
    from threadpoolctl import threadpool_limits
    _THREAD_LIMITS = threadpool_limits(1)        # kept alive for the whole session
    torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
RECORD = Path(os.environ.get("KANFAE_TEST_RECORD", ROOT / "reports" / "logs" / "test_records.jsonl"))
RUN_ID = os.environ.get("KANFAE_RUN_ID", time.strftime("%Y%m%dT%H%M%S"))


def _marker_arg(item, name):
    m = item.get_closest_marker(name)
    return m.args[0] if m and m.args else ""


def pytest_collection_modifyitems(items):
    for item in items:
        item.user_properties.append(("criterion", _marker_arg(item, "criterion")))
        item.user_properties.append(("kind", _marker_arg(item, "kind")))
        item.user_properties.append(("stage", _marker_arg(item, "stage")))


def pytest_runtest_logreport(report):
    # Record the call phase, plus setup failures/skips (which never reach the call phase).
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        props = dict(report.user_properties)
        msg = ""
        if report.outcome == "failed":
            msg = str(report.longrepr.reprcrash.message) if hasattr(report.longrepr, "reprcrash") else str(report.longrepr)
        elif report.outcome == "skipped":
            msg = str(report.longrepr[-1]) if isinstance(report.longrepr, tuple) else str(report.longrepr)
        outcome = report.outcome
        if hasattr(report, "wasxfail"):              # D64: expected failures are recorded as such, not as skips
            outcome = "xfailed" if report.outcome == "skipped" else "xpassed"
            msg = f"expected failure: {report.wasxfail}"
        rec = dict(run_id=RUN_ID, nodeid=report.nodeid, outcome=outcome, phase=report.when,
                   duration_s=round(report.duration, 4), criterion=props.get("criterion", ""),
                   kind=props.get("kind", ""), stage=props.get("stage", ""), message=msg[:500])
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        with open(RECORD, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
