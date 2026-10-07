#!/usr/bin/env python3
"""Run the complete project:  data -> processing -> features -> models -> evaluation -> SHAP -> inventory -> reports.

Usage
-----
    python run_project.py                    # everything (about 6-8 minutes on 4 cores; needs internet for the first download)
    python run_project.py --from forecast    # resume from a stage
    python run_project.py --only evaluate    # run a single stage
    python run_project.py --force-download   # re-download and re-verify the raw data

Stages (in order): env, acquire, inspect, prepare, eda, forecast, evaluate, explain, uncertainty, inventory, audit, tests, reports

External prerequisite: none beyond internet access for the first run (the official Kaggle files need credentials, so the
data are fetched from verified public Hugging Face mirrors, see reports/data_source.md).  If the raw files are already in
data/raw/ no download happens.
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config as C                      # noqa: E402
from src import pipeline as P                    # noqa: E402

ORDER = ["env", "acquire", "inspect", "prepare", "eda", "forecast", "evaluate", "explain", "uncertainty", "inventory", "audit", "tests", "reports"]


def stage_tests() -> None:
    """Run the unit-test suite and keep the log (a failing test aborts the run)."""
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=str(ROOT))
    (C.REPORTS / "test_results.txt").write_text(out.stdout + out.stderr)
    print(out.stdout[-1500:])
    if out.returncode != 0:
        raise RuntimeError("unit tests failed - see reports/test_results.txt")


def stage_reports() -> None:
    import pandas as pd
    if P.TIMINGS:                                                    # keep the runtime of the last full run for the README
        pd.DataFrame({"stage": list(P.TIMINGS), "seconds": [round(v, 1) for v in P.TIMINGS.values()]}).to_csv(C.OUT_TAB / "pipeline_runtime.csv", index=False)
    from src import docs
    docs.build_all()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", choices=ORDER, default=ORDER[0])
    ap.add_argument("--only", choices=ORDER, default=None)
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    random.seed(C.SEED)
    np.random.seed(C.SEED)
    C.ensure_dirs()
    stages = [args.only] if args.only else ORDER[ORDER.index(args.start):]
    t_all = time.time()
    for name in stages:
        with P.timed(name):
            if name == "acquire":
                P.stage_acquire(force=args.force_download)
            elif name == "tests":
                stage_tests()
            elif name == "reports":
                stage_reports()
            else:
                P.STAGES[name]()
    if P.FIGURE_ERRORS:
        print("\nFIGURE ERRORS (numerical results are unaffected):\n  " + "\n  ".join(P.FIGURE_ERRORS))
    print(f"\nAll requested stages finished in {(time.time() - t_all) / 60:.1f} min.")


if __name__ == "__main__":
    main()
