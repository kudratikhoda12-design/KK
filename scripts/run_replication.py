#!/usr/bin/env python3
"""Replication of the FROZEN methodology on another asset (default SOLUSDT).

Nothing is re-tuned: same features, same feature-set/model/hyper-parameters, same delta, same costs (in bps),
same split dates. Development walk-forward metrics are reported for context; the holdout part is delegated to
run_holdout.py (which goes through the same holdout guard).

    python scripts/build_dataset.py --symbol SOLUSDT --data-dir data/sol
    python scripts/run_replication.py --symbol SOLUSDT --data-dir data/sol
    python scripts/run_holdout.py --symbol SOLUSDT --data-dir data/sol --skip-secondary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ethof import backtest as B  # noqa: E402
from ethof import evaluation as EV  # noqa: E402
from ethof import statistics as S  # noqa: E402
from ethof import strategy as ST  # noqa: E402
from ethof import validation as V  # noqa: E402
from ethof.config import Paths, load_config  # noqa: E402
from ethof.features import FEATURE_SETS  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_holdout import make_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--data-dir", default="data/sol")
    a = ap.parse_args()
    os.environ["ETHOF_DATA_DIR"] = a.data_dir
    cfg = load_config()
    rs = cfg["research"]
    H = rs["primary_horizon"]
    paths = Paths(cfg["data_dir"])
    frozen = json.loads((ROOT / "reports" / "tables" / "frozen_parameters.json").read_text())
    TAB = ROOT / "reports" / "tables" / f"replication_{a.symbol}"
    TAB.mkdir(parents=True, exist_ok=True)
    dev = V.development_only(pl.read_parquet(paths.processed / "research" / f"{a.symbol}-research.parquet"),
                             rs["holdout"]["start"])
    wf = rs["walk_forward"]
    folds = V.walk_forward_folds(wf["start"], wf["first_validation"], wf["end_exclusive"], wf["block_months"],
                                 timedelta(days=wf["embargo_days"]))
    funding = pl.read_parquet(paths.processed / "funding" / f"{a.symbol}-funding.parquet")
    costs = ST.cost_scenarios(cfg)
    rows, summ = [], []
    for fset in ("A_price", "C_plus_flow", frozen["feature_set"]):
        preds = []
        for fo in folds:
            st_ = frozen["rf_train_stride"] if frozen["model"] == "rf" else wf["train_stride"]
            tr, va = V.split(dev, fo, H, stride=st_)
            tr = tr.filter(pl.col(f"fwd_ret_{H}").is_not_null()); va = va.filter(pl.col(f"fwd_ret_{H}").is_not_null())
            est = make_model(frozen["model"], frozen).fit(tr.select(FEATURE_SETS[fset]).to_numpy().astype(np.float32),
                                                          tr[f"up_{H}"].to_numpy())
            p = est.predict_proba(va.select(FEATURE_SETS[fset]).to_numpy().astype(np.float32))[:, 1]
            m = EV.classification(va[f"up_{H}"].to_numpy(), p, va[f"fwd_ret_{H}"].to_numpy())
            rows.append({"fset": fset, "fold": fo.name, **m})
            preds.append(pl.DataFrame({"ts": va["ts"], "p": p}))
        pr = pl.concat(preds)
        g = ST.grid_with_preds(dev, pr, V._d(wf["first_validation"]), V._d(wf["end_exclusive"]))
        s = B.prob_signal(np.nan_to_num(g["p"].to_numpy(), nan=0.5), frozen["delta"])
        r = {"fset": fset, "tradable_auc": ST.tradable_auc(g, H)}
        for cn in ("gross", "base"):
            st = B.summarize(B.run(g, s, H, costs[cn], funding), B.trade_returns(g, s, H, costs[cn]))
            r.update({f"sharpe_{cn}": st["sharpe"], f"total_return_{cn}": st["total_return"]})
        r["breakeven_bps"] = st["breakeven_cost_bps_per_side"]
        summ.append(r)
    f = pl.DataFrame(rows)
    f.write_csv(TAB / "dev_folds.csv")
    agg = f.group_by("fset").agg(auc_mean=pl.col("auc").mean(), auc_sd=pl.col("auc").std(),
                                 ll_improvement=(pl.col("log_loss_baseline") - pl.col("log_loss")).mean(),
                                 ic_mean=pl.col("ic_spearman").mean()).join(pl.DataFrame(summ), on="fset")
    agg.write_csv(TAB / "dev_summary.csv")
    st_rows = []
    for sig in ("ofi_1", "obi_1pct", "ofi_15"):
        for h in (5, 15):
            sub = dev.select(sig, f"fwd_ret_{h}").drop_nulls()
            r = S.hac_ols(sub[f"fwd_ret_{h}"].to_numpy() * 1e4, S.standardize(sub[sig].to_numpy())[:, None], ["x"], 2 * h)
            st_rows.append({"signal": sig, "h": h, "n": r["n"], "beta_bps_per_sd": r["b_x"], "t_hac": r["t_x"], "p_hac": r["p_x"]})
    pl.DataFrame(st_rows).write_csv(TAB / "dev_stat_tests.csv")
    with pl.Config(tbl_cols=12, tbl_width_chars=200):
        print(agg); print(pl.DataFrame(st_rows))


if __name__ == "__main__":
    main()
