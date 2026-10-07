"""Pre-registered spot-fee check (pre-registration section 8), executed after the test.

Pre-registration section 8 lists a "spot alternative: 0.10% fee per side,
long/flat only, reported as a robustness check". It was not run in the original
pipeline (FEE_SPOT_TAKER was defined in config.py but unused). This script runs
exactly that one scenario and nothing else:

* the FROZEN test positions are re-used as saved in
  data/processed/BTCUSDT_test_backtest.parquet (no model is refitted, no
  threshold or selection is changed; the frozen mode is already long/flat);
* the fee component is replaced by the spot taker fee; half-spread and
  slippage stay as in the base cost model:
      cost per side = FEE_SPOT_TAKER + HALF_SPREAD + SLIPPAGE = 10 + 1 + 1 = 12 bp;
* no funding is applied, because a spot position pays no perpetual funding.

As a consistency check, the original base case (7 bp per side + actual funding)
is re-priced from the same saved positions and must reproduce the recorded test
net Sharpe. Outputs go to NEW files only; results.json, the experiment log and
every original table are left untouched.

    python scripts/run_spot_fee_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.backtest import run_backtest  # noqa: E402
from src.data import load_funding  # noqa: E402
from src.pipeline import periods_per_year  # noqa: E402
from src.risk import daily_returns, performance, sharpe_ci  # noqa: E402

OUT_CSV = config.TABLES_DIR / "robustness_spot_fee_check.csv"
OUT_JSON = config.REPORTS_DIR / "spot_fee_check.json"


def main() -> None:
    sel = json.loads((config.REPORTS_DIR / "selection.json").read_text())
    if sel["mode"] != "long_flat":
        raise SystemExit("The spot scenario is long/flat only; the frozen mode is not long_flat.")
    symbol, h = config.SYMBOL, sel["horizon"]
    events = pd.read_parquet(config.PROCESSED_DIR / f"{symbol}_dataset_h{h}_lag{sel['lag']}_off{sel['offset']}.parquet")
    frozen = pd.read_parquet(config.PROCESSED_DIR / f"{symbol}_test_backtest.parquet")
    pos = frozen["position"]
    ppy = periods_per_year(h)

    # 1) Consistency check: re-pricing the saved positions reproduces the recorded test result
    recorded = json.loads((config.REPORTS_DIR / "results.json").read_text())["test"]["performance"]
    base = run_backtest(events, pos, config.BASE_COST_PER_SIDE, funding=load_funding(symbol))
    base_perf = performance(base, ppy)
    if not np.isclose(base_perf["sharpe"], recorded["sharpe"], rtol=0, atol=1e-9):
        raise SystemExit(f"Re-priced base Sharpe {base_perf['sharpe']} != recorded {recorded['sharpe']}")

    # 2) The pre-registered spot scenario
    spot_cost = config.FEE_SPOT_TAKER + config.HALF_SPREAD + config.SLIPPAGE
    spot = run_backtest(events, pos, spot_cost, funding=None)
    rows = []
    for name, bt, cost, fund in [
        ("frozen test, perpetual taker fee (original, re-priced check)", base, config.BASE_COST_PER_SIDE, "actual funding"),
        ("pre-registered spot alternative (executed 2026-10-07)", spot, spot_cost, "none (spot)"),
    ]:
        p = performance(bt, ppy)
        lo, hi = sharpe_ci(bt["net_ret"].to_numpy(), ppy)
        d = daily_returns(bt)
        rows.append(dict(
            scenario=name, cost_per_side_bp=round(1e4 * cost, 2), funding=fund,
            net_sharpe=p["sharpe"], net_sharpe_ci_low=lo, net_sharpe_ci_high=hi,
            gross_sharpe=p["gross_sharpe"], cumulative_return=p["cumulative_return"],
            annualized_return=p["annualized_return"], max_drawdown=p["max_drawdown"],
            daily_VaR99_hist=float(-np.quantile(d, 0.01)), exposure=p["exposure"],
            round_trips_per_year=p["round_trips_per_year"], cost_sum=p["cost_sum"],
            gross_sum=p["gross_sum"], net_sum=p["net_sum"]))
    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        commit = "unknown"
    OUT_JSON.write_text(json.dumps(dict(
        label="Pre-registered robustness check (pre-registration section 8), executed after the test",
        executed_utc=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), code_commit=commit,
        definition="frozen test positions re-priced at FEE_SPOT_TAKER + HALF_SPREAD + SLIPPAGE per side, no funding",
        selection_unchanged=True, test_result_unchanged=True,
        base_reprice_matches_recorded_sharpe=True,
        results=table.to_dict("records")), indent=2, default=float))
    print(table.T.to_string())


if __name__ == "__main__":
    main()
