#!/usr/bin/env python3
"""Stage 8 (development only): primary-model selection, signal-threshold selection, out-of-fold backtests,
ablation economics, stability and regime analysis. Uses only walk-forward validation predictions
(2024-03-01 .. 2025-09-30); the holdout is not touched.

Pre-declared selection rules (also in docs/final_methodology.md):
  * primary feature set   = E_all (the full information set; the ablation sets are reported, not selected)
  * primary model class   = lowest mean validation log loss on E_all, h = 15, among {logit, rf, lgbm}
  * signal threshold delta = highest out-of-fold net Sharpe at BASE cost among the configured deltas
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ethof import backtest as B  # noqa: E402
from ethof import strategy as ST  # noqa: E402
from ethof import validation as V  # noqa: E402
from ethof.config import Paths, load_config  # noqa: E402
from ethof.plotting import PALETTE, plt, save  # noqa: E402

TAB = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures" / "dev"
COLORS = {"ml": PALETTE["blue"], "momentum": PALETTE["orange"], "flow_rule": PALETTE["green"],
          "random": PALETTE["grey"], "buy_and_hold": PALETTE["purple"]}


def regime_cutoffs(research: pl.DataFrame, cfg: dict) -> tuple[float, float]:
    rg = cfg["research"]["regimes"]
    init = research.filter(pl.col("ts") < V._d(rg["cutoff_window_end"]))[rg["vol_feature"]].drop_nulls().to_numpy()
    return tuple(np.quantile(init, [1 / 3, 2 / 3]))


def main() -> None:
    cfg = load_config()
    rs = cfg["research"]
    H = rs["primary_horizon"]
    paths = Paths(cfg["data_dir"])
    research = V.development_only(pl.read_parquet(paths.processed / "research" / "ETHUSDT-research.parquet"),
                                  rs["holdout"]["start"])
    funding = pl.read_parquet(paths.processed / "funding" / "ETHUSDT-funding.parquet")
    costs = ST.cost_scenarios(cfg)
    summ = pl.read_csv(TAB / "model_summary.csv")

    # ------------------------------------------------------------------ primary model
    cand = summ.filter((pl.col("fset") == "E_all") & (pl.col("h") == H) & pl.col("config").str.contains("^(logit|lgbm|rf)_"))
    cand = cand.sort("log_loss_mean")
    primary = cand.row(0, named=True)
    sel = {"rule": "lowest mean validation log loss, feature set E_all, h=15",
           "candidates": cand.select("config", "log_loss_mean", "auc_mean", "auc_sd").to_dicts(),
           "selected": primary["config"]}
    vs, ve = V._d(rs["walk_forward"]["first_validation"]), V._d(rs["walk_forward"]["end_exclusive"])
    oof = pl.read_parquet(paths.processed / "oof" / f"{primary['config']}.parquet")
    g = ST.grid_with_preds(research, oof, vs, ve)

    # ------------------------------------------------------------------ delta selection (base cost)
    drows = []
    for d in rs["signal"]["prob_deltas"]:
        s = B.prob_signal(np.nan_to_num(g["p"].to_numpy(), nan=0.5), d)
        bt = B.run(g, s, H, costs["base"], funding)
        st = B.summarize(bt, B.trade_returns(g, s, H, costs["base"]))
        stg = B.summarize(B.run(g, s, H, costs["gross"], funding))
        drows.append({"delta": d, "active_share": float((s != 0).mean()), "sharpe_base": st["sharpe"],
                      "total_return_base": st["total_return"], "sharpe_gross": stg["sharpe"],
                      "sum_gross": stg["sum_gross"], "breakeven_bps": st["breakeven_cost_bps_per_side"],
                      "n_trades": st["n_trades"], "turnover_per_day": st["turnover_per_day"]})
    dsel = pl.DataFrame(drows)
    dsel.write_csv(TAB / "dev_delta_selection.csv")
    best = dsel.sort(["sharpe_base", "delta"], descending=[True, True]).row(0, named=True)
    delta = best["delta"]
    sel.update(delta_rule="highest out-of-fold net Sharpe at base cost", delta_selected=delta,
               delta_table=dsel.to_dicts())
    (TAB / "primary_model_selection.json").write_text(json.dumps(sel, indent=1, default=float))

    # ------------------------------------------------------------------ full OOF backtest at delta*
    res, frames = ST.evaluate(g, H, delta, rs["signal"]["flow_rule_z"], costs, funding, seed=1)
    res = res.with_columns(period=pl.lit("development OOF 2024-03-01..2025-09-30"))
    res.write_csv(TAB / "dev_backtest_summary.csv")
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for sname in ("ml", "momentum", "flow_rule", "random", "buy_and_hold"):
        bt = frames[(sname, "base")]
        eq = np.cumprod(1 + bt["net"].to_numpy())
        ax[0].plot(bt["ts"], eq, label=sname, color=COLORS[sname], lw=1)
        ax[1].plot(bt["ts"], eq / np.maximum.accumulate(eq) - 1, color=COLORS[sname], lw=0.8)
    btg = frames[("ml", "gross")]
    ax[0].plot(btg["ts"], np.cumprod(1 + btg["net"].to_numpy()), color=COLORS["ml"], ls="--", lw=1, label="ml (gross, no costs)")
    ax[0].set_yscale("log"); ax[0].set_ylabel("equity (log)"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"Development out-of-fold backtest, base costs ({costs['base'].per_side * 1e4:.2f} bps/side), H={H}m, delta={delta}")
    ax[1].set_ylabel("drawdown")
    save(fig, FIG / "dev_equity_drawdown.png")

    # ------------------------------------------------------------------ ablation economics + tradable AUC
    arows = []
    for cfgname in summ.filter(pl.col("h") == H)["config"].to_list():
        if cfgname.startswith("tune_"):
            continue
        o = pl.read_parquet(paths.processed / "oof" / f"{cfgname}.parquet")
        gg = ST.grid_with_preds(research, o, vs, ve)
        s = B.prob_signal(np.nan_to_num(gg["p"].to_numpy(), nan=0.5), delta)
        row = {"config": cfgname, "tradable_auc": ST.tradable_auc(gg, H),
               "auc": float(summ.filter(pl.col("config") == cfgname)["auc_mean"][0])}
        for cn in ("gross", "base"):
            st = B.summarize(B.run(gg, s, H, costs[cn], funding), B.trade_returns(gg, s, H, costs[cn]))
            row.update({f"sharpe_{cn}": st["sharpe"], f"total_return_{cn}": st["total_return"]})
        row["breakeven_bps"] = st["breakeven_cost_bps_per_side"]
        row["n_trades"] = st["n_trades"]
        row["avg_trade_gross_bps"] = float(B.trade_returns(gg, s, H, costs["gross"]).mean() * 1e4) if st["n_trades"] else float("nan")
        arows.append(row)
    abl = pl.DataFrame(arows).sort("config")
    abl.write_csv(TAB / "dev_ablation_economics.csv")

    # ------------------------------------------------------------------ monthly stability (primary model)
    o = oof.with_columns(month=pl.col("ts").dt.strftime("%Y-%m"))
    ml_bt = frames[("ml", "base")].with_columns(month=pl.col("ts").dt.strftime("%Y-%m"))
    mpnl = ml_bt.group_by("month").agg(net=pl.col("net").sum(), gross=pl.col("gross").sum())
    mrows = []
    for m_, sub in o.group_by("month"):
        y, p = sub["y"].to_numpy(), sub["p"].to_numpy()
        mrows.append({"month": m_[0], "n": len(y), "auc": float(roc_auc_score(y, p)),
                      "log_loss": float(log_loss(y, p, labels=[0, 1])),
                      "log_loss_prior": float(log_loss(y, np.full(len(y), y.mean()), labels=[0, 1])),
                      "hit_rate": float(((p > 0.5) == (y == 1)).mean())})
    mon = pl.DataFrame(mrows).join(mpnl, on="month").sort("month")
    mon.write_csv(TAB / "dev_monthly_stability.csv")
    fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    x = np.arange(mon.height)
    ax[0].plot(x, mon["auc"], marker="o", color=PALETTE["blue"]); ax[0].axhline(0.5, color="black", lw=0.8); ax[0].set_ylabel("AUC")
    ax[0].set_title("Monthly out-of-fold stability of the primary model (development)")
    ax[1].plot(x, mon["hit_rate"], marker="o", color=PALETTE["green"]); ax[1].axhline(0.5, color="black", lw=0.8); ax[1].set_ylabel("hit rate")
    ax[2].bar(x - 0.2, mon["gross"] * 100, width=0.4, color=PALETTE["grey"], label="gross")
    ax[2].bar(x + 0.2, mon["net"] * 100, width=0.4, color=PALETTE["red"], label="net (base cost)")
    ax[2].set_ylabel("ML strategy return (%)"); ax[2].legend()
    ax[2].set_xticks(x); ax[2].set_xticklabels(mon["month"], rotation=60, fontsize=8)
    save(fig, FIG / "dev_monthly_stability.png")

    # ------------------------------------------------------------------ volatility regimes
    lo, hi = regime_cutoffs(research, cfg)
    gr = g.with_columns(regime=pl.when(pl.col("vol_1440") <= lo).then(pl.lit("1_low"))
                        .when(pl.col("vol_1440") <= hi).then(pl.lit("2_mid")).otherwise(pl.lit("3_high")))
    sig = B.prob_signal(np.nan_to_num(g["p"].to_numpy(), nan=0.5), delta)
    rrows = []
    for cn in ("gross", "base"):
        bt = B.run(g, sig, H, costs[cn], funding).with_columns(regime=gr["regime"])
        for rname, sub in bt.group_by("regime"):
            ok = gr.filter(pl.col("regime") == rname[0]).join(oof.select("ts", "y"), on="ts").filter(pl.col("p").is_not_null())
            rrows.append({"regime": rname[0], "cost": cn, "minutes": sub.height,
                          "auc": float(roc_auc_score(ok["y"], ok["p"])) if ok.height else None,
                          "sum_net": float(sub["net"].sum()), "mean_net_bps_per_min": float(sub["net"].mean() * 1e4),
                          "turnover": float(sub["turnover"].sum())})
    reg = pl.DataFrame(rrows).sort("cost", "regime")
    reg.write_csv(TAB / "dev_regimes.csv")
    (TAB / "regime_cutoffs.json").write_text(json.dumps({"vol_1440_low_le": lo, "vol_1440_mid_le": hi,
                                                         "estimated_on": "2023-03-01..2024-02-29"}, indent=1))
    with pl.Config(tbl_rows=60, tbl_cols=16, tbl_width_chars=250):
        print(json.dumps({k: v for k, v in sel.items() if k != "delta_table"}, indent=1, default=float))
        print(dsel)
        print(res.select("strategy", "cost", "total_return", "sharpe", "max_drawdown", "n_trades", "win_rate",
                         "avg_trade_bps", "breakeven_cost_bps_per_side", "turnover_per_day"))
        print(abl); print(mon); print(reg)


if __name__ == "__main__":
    main()
