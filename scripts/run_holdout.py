#!/usr/bin/env python3
"""FINAL HOLDOUT EVALUATION (2025-10-01 .. 2026-09-30). Run once, after docs/final_methodology.md is frozen.

Frozen procedure (identical to development walk-forward):
  * holdout split into four 3-month blocks; before each block the model is re-fitted on all data from
    2023-03-01 up to (block start - 1 day embargo), labels purged, stride 5, frozen hyper-parameters.
    (Earlier holdout blocks therefore enter later training sets - the same expanding scheme as development;
    no parameter, threshold or feature choice is revisited.)
  * primary model + frozen delta -> signals -> cost-aware backtest (low/base/high), benchmarks.
  * pre-declared secondary evaluations: feature ablation A..E x {logit, lgbm, rf} at h = 15;
    horizons 5/30/60 for sets A, C, E x {logit, lgbm}; confirmatory HAC test of the main development
    finding (1-min OFI predicts reversal); volatility regimes; monthly stability.

    python scripts/run_holdout.py [--symbol ETHUSDT] [--tag final]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ethof import backtest as B  # noqa: E402
from ethof import evaluation as EV  # noqa: E402
from ethof import models as M  # noqa: E402
from ethof import statistics as S  # noqa: E402
from ethof import strategy as ST  # noqa: E402
from ethof import validation as V  # noqa: E402
from ethof.config import Paths, load_config  # noqa: E402
from ethof.features import FEATURE_SETS  # noqa: E402
from ethof.plotting import PALETTE, plt, save  # noqa: E402

METHODOLOGY = ROOT / "docs" / "final_methodology.md"
COLORS = {"ml": PALETTE["blue"], "momentum": PALETTE["orange"], "flow_rule": PALETTE["green"],
          "random": PALETTE["grey"], "buy_and_hold": PALETTE["purple"]}


def make_model(kind: str, frozen: dict):
    if kind == "lgbm":
        return M.lightgbm(**frozen["lgbm_params"])
    if kind == "rf":
        return M.random_forest(**frozen["rf_params"])
    return M.logistic(C=frozen["logit_C"])


def walk_holdout(full: pl.DataFrame, blocks, kind: str, fset: str, h: int, frozen: dict, stride: int) -> tuple[pl.DataFrame, list]:
    feats = FEATURE_SETS[fset]
    preds, rows = [], []
    for fo in blocks:
        tr, va = V.split(full, fo, h, stride=stride)
        tr = tr.filter(pl.col(f"fwd_ret_{h}").is_not_null()); va = va.filter(pl.col(f"fwd_ret_{h}").is_not_null())
        est = make_model(kind, frozen).fit(tr.select(feats).to_numpy().astype(np.float32), tr[f"up_{h}"].to_numpy())
        p = est.predict_proba(va.select(feats).to_numpy().astype(np.float32))[:, 1]
        y, fwd = va[f"up_{h}"].to_numpy(), va[f"fwd_ret_{h}"].to_numpy()
        rows.append({"block": fo.name, "valid_start": fo.valid_start.date().isoformat(), "n_train": tr.height,
                     **EV.classification(y, p, fwd)})
        preds.append(pl.DataFrame({"ts": va["ts"], "block": fo.name, "p": p, "y": y, "fwd": fwd}))
    return pl.concat(preds), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--data-dir")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--skip-secondary", action="store_true")
    a = ap.parse_args()
    import os
    if a.data_dir:
        os.environ["ETHOF_DATA_DIR"] = a.data_dir
    cfg = load_config()
    rs = cfg["research"]
    H = rs["primary_horizon"]
    paths = Paths(cfg["data_dir"])
    sym = a.symbol
    TAB = ROOT / "reports" / "tables" / ("holdout" if sym == "ETHUSDT" else f"replication_{sym}")
    FIG = ROOT / "reports" / "figures" / ("holdout" if sym == "ETHUSDT" else f"replication_{sym}")
    TAB.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / "reports" / "tables" / "frozen_parameters.json").read_text())

    full = pl.read_parquet(paths.processed / "research" / f"{sym}-research.parquet")
    hold = V.unlock_holdout(full, rs["holdout"]["start"], rs["holdout"]["end_exclusive"], METHODOLOGY,
                            ROOT / "reports" / "holdout_access_log.jsonl", purpose=f"{a.tag} evaluation {sym}")
    blocks = V.walk_forward_folds(rs["walk_forward"]["start"], rs["holdout"]["start"], rs["holdout"]["end_exclusive"],
                                  rs["walk_forward"]["block_months"], timedelta(days=rs["walk_forward"]["embargo_days"]))
    stride = rs["walk_forward"]["train_stride"]
    funding = pl.read_parquet(paths.processed / "funding" / f"{sym}-funding.parquet")
    costs = ST.cost_scenarios(cfg)
    hs, he = V._d(rs["holdout"]["start"]), V._d(rs["holdout"]["end_exclusive"])
    out = {"symbol": sym, "holdout_rows": hold.height, "blocks": [b.as_dict() for b in blocks],
           "frozen": frozen}

    # ------------------------------------------------------------------ primary model
    primary_stride = frozen["rf_train_stride"] if frozen["model"] == "rf" else stride
    preds, rows = walk_holdout(full, blocks, frozen["model"], frozen["feature_set"], H, frozen, primary_stride)
    preds.write_parquet(paths.processed / f"holdout_preds_{sym}_primary.parquet")
    blk = pl.DataFrame(rows)
    blk.write_csv(TAB / "primary_blocks.csv")
    y, p, fwd = preds["y"].to_numpy(), preds["p"].to_numpy(), preds["fwd"].to_numpy()
    out["primary_classification_pooled"] = EV.classification(y, p, fwd)
    pl.DataFrame(EV.calibration_table(y, p)).write_csv(TAB / "primary_calibration.csv")
    g = ST.grid_with_preds(full, preds, hs, he)
    out["primary_tradable_auc"] = ST.tradable_auc(g, H)

    res, frames = ST.evaluate(g, H, frozen["delta"], rs["signal"]["flow_rule_z"], costs, funding, seed=1)
    res.write_csv(TAB / "backtest_summary.csv")
    for (sname, cname), bt in frames.items():
        if cname == "base":
            bt.write_parquet(paths.processed / f"holdout_bt_{sym}_{sname}.parquet")

    # cost sensitivity curve for the ML strategy: net total return vs per-side cost
    s_ml = ST.signals(g, frozen["delta"], rs["signal"]["flow_rule_z"])["ml"]
    curve = []
    for c_bps in [0, 0.5, 1, 2, 3, 4, 5, 5.523, 6, 7, 8, 10]:
        st = B.summarize(B.run(g, s_ml, H, B.Costs(c_bps, 0, 0), funding))
        curve.append({"cost_bps_per_side": c_bps, "total_return": st["total_return"], "sharpe": st["sharpe"]})
    pl.DataFrame(curve).write_csv(TAB / "cost_sensitivity_ml.csv")

    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for sname in ("ml", "momentum", "flow_rule", "random", "buy_and_hold"):
        bt = frames[(sname, "base")]
        eq = np.cumprod(1 + bt["net"].to_numpy())
        ax[0].plot(bt["ts"], eq, color=COLORS[sname], lw=1, label=sname)
        ax[1].plot(bt["ts"], eq / np.maximum.accumulate(eq) - 1, color=COLORS[sname], lw=0.8)
    btg = frames[("ml", "gross")]
    ax[0].plot(btg["ts"], np.cumprod(1 + btg["net"].to_numpy()), color=COLORS["ml"], ls="--", lw=1, label="ml gross (no costs)")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=8); ax[0].set_ylabel("equity (log)")
    ax[0].set_title(f"{sym} FINAL HOLDOUT 2025-10..2026-09, base cost {costs['base'].per_side*1e4:.2f} bps/side")
    ax[1].set_ylabel("drawdown")
    pos = frames[("ml", "base")].group_by(pl.col("ts").dt.date().alias("d")).agg(
        pl.col("position").abs().mean().alias("gross_exposure"), pl.col("position").mean().alias("net_exposure")).sort("d")
    ax[2].plot(pos["d"], pos["gross_exposure"], color=PALETTE["blue"], label="mean |position| (ML)")
    ax[2].plot(pos["d"], pos["net_exposure"], color=PALETTE["red"], label="mean position (ML)")
    ax[2].legend(fontsize=8); ax[2].set_ylabel("exposure")
    save(fig, FIG / "equity_drawdown_exposure.png")

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for sname in ("ml", "momentum", "flow_rule", "buy_and_hold"):
        d = frames[(sname, "base")].group_by(pl.col("ts").dt.date().alias("d")).agg(pl.col("net").sum()).sort("d")
        r = d["net"].to_numpy()
        rs30 = pl.Series(r).rolling_mean(30) / pl.Series(r).rolling_std(30) * np.sqrt(365)
        ax[0].plot(d["d"], rs30, color=COLORS[sname], lw=1, label=sname)
        if sname in ("ml", "buy_and_hold"):
            ax[1].hist(r * 100, bins=60, alpha=0.6, color=COLORS[sname], label=sname)
    ax[0].axhline(0, color="black", lw=0.8); ax[0].set_title("rolling 30-day Sharpe (base cost)"); ax[0].legend(fontsize=7)
    ax[1].set_title("daily net return distribution (%)"); ax[1].legend(fontsize=7)
    cv = pl.DataFrame(curve)
    ax[2].plot(cv["cost_bps_per_side"], cv["total_return"] * 100, marker="o", color=PALETTE["blue"])
    ax[2].axhline(0, color="black", lw=0.8)
    for k in ("low", "base", "high"):
        ax[2].axvline(costs[k].per_side * 1e4, color=PALETTE["grey"], ls=":", lw=1)
        ax[2].text(costs[k].per_side * 1e4, ax[2].get_ylim()[1], k, fontsize=7, ha="center", va="bottom")
    ax[2].set_xlabel("per-side cost (bps)"); ax[2].set_ylabel("ML total net return (%)"); ax[2].set_title("cost sensitivity")
    save(fig, FIG / "rolling_sharpe_distribution_costs.png")

    # weekly returns
    wk = pl.concat([frames[(s, "base")].group_by_dynamic("ts", every="1w").agg(pl.col("net").sum()).with_columns(strategy=pl.lit(s))
                    for s in ("ml", "buy_and_hold", "momentum", "flow_rule")])
    wk.write_csv(TAB / "weekly_returns.csv")

    # ------------------------------------------------------------------ monthly stability + regimes
    mrows = []
    mp = frames[("ml", "base")].group_by(pl.col("ts").dt.strftime("%Y-%m").alias("month")).agg(net=pl.col("net").sum(), gross=pl.col("gross").sum())
    for m_, sub in preds.with_columns(month=pl.col("ts").dt.strftime("%Y-%m")).group_by("month"):
        yy, pp = sub["y"].to_numpy(), sub["p"].to_numpy()
        mrows.append({"month": m_[0], "auc": float(roc_auc_score(yy, pp)), "log_loss": float(log_loss(yy, pp, labels=[0, 1])),
                      "log_loss_prior": float(log_loss(yy, np.full(len(yy), yy.mean()), labels=[0, 1])),
                      "hit_rate": float(((pp > 0.5) == (yy == 1)).mean())})
    mon = pl.DataFrame(mrows).join(mp, on="month").sort("month")
    mon.write_csv(TAB / "monthly_stability.csv")
    cut = json.loads((ROOT / "reports" / "tables" / "regime_cutoffs.json").read_text())
    if sym != "ETHUSDT":   # same rule, estimated on the replication asset's own initial window
        init = full.filter(pl.col("ts") < V._d(rs["regimes"]["cutoff_window_end"]))["vol_1440"].drop_nulls().to_numpy()
        lo, hi = np.quantile(init, [1 / 3, 2 / 3])
    else:
        lo, hi = cut["vol_1440_low_le"], cut["vol_1440_mid_le"]
    reg = g.with_columns(regime=pl.when(pl.col("vol_1440") <= lo).then(pl.lit("1_low"))
                         .when(pl.col("vol_1440") <= hi).then(pl.lit("2_mid")).otherwise(pl.lit("3_high")))["regime"]
    rrows = []
    for cn in ("gross", "base"):
        bt = B.run(g, s_ml, H, costs[cn], funding).with_columns(regime=reg)
        for rname, sub in bt.group_by("regime"):
            ok = g.with_columns(regime=reg).filter((pl.col("regime") == rname[0]) & pl.col("p").is_not_null()).join(preds.select("ts", "y"), on="ts")
            rrows.append({"regime": rname[0], "cost": cn, "minutes": sub.height,
                          "auc": float(roc_auc_score(ok["y"], ok["p"])) if ok.height > 100 else None,
                          "sum_net": float(sub["net"].sum()), "turnover": float(sub["turnover"].sum())})
    pl.DataFrame(rrows).sort("cost", "regime").write_csv(TAB / "regimes.csv")

    # ------------------------------------------------------------------ confirmatory statistical test
    crow = []
    for sig_ in ("ofi_1", "obi_1pct", "ofi_15"):
        for h in (5, 15):
            sub = hold.select(sig_, f"fwd_ret_{h}").drop_nulls()
            r = S.hac_ols(sub[f"fwd_ret_{h}"].to_numpy() * 1e4, S.standardize(sub[sig_].to_numpy())[:, None], ["x"], 2 * h)
            crow.append({"signal": sig_, "h": h, "n": r["n"], "beta_bps_per_sd": r["b_x"], "t_hac": r["t_x"], "p_hac": r["p_x"]})
    C = pl.DataFrame(crow)
    C.with_columns(p_holm=pl.Series(S.holm(C["p_hac"].to_numpy()))).write_csv(TAB / "confirmatory_tests.csv")

    # ------------------------------------------------------------------ secondary: ablation + horizons
    if not a.skip_secondary:
        srows = []
        for fset in FEATURE_SETS:
            for kind in ("logit", "lgbm", "rf"):
                st_ = stride if kind != "rf" else frozen["rf_train_stride"]
                pr, _ = walk_holdout(full, blocks, kind, fset, H, frozen, st_)
                gg = ST.grid_with_preds(full, pr, hs, he)
                sg = B.prob_signal(np.nan_to_num(gg["p"].to_numpy(), nan=0.5), frozen["delta"])
                row = {"model": kind, "fset": fset, "h": H, **{k: v for k, v in EV.classification(pr["y"].to_numpy(), pr["p"].to_numpy(), pr["fwd"].to_numpy()).items()
                                                              if k in ("n", "auc", "log_loss", "log_loss_baseline", "ic_spearman")},
                       "tradable_auc": ST.tradable_auc(gg, H)}
                for cn in ("gross", "base"):
                    s2 = B.summarize(B.run(gg, sg, H, costs[cn], funding), B.trade_returns(gg, sg, H, costs[cn]))
                    row.update({f"sharpe_{cn}": s2["sharpe"], f"total_return_{cn}": s2["total_return"],
                                f"max_dd_{cn}": s2["max_drawdown"]})
                row["breakeven_bps"] = s2["breakeven_cost_bps_per_side"]
                srows.append(row)
                print(row, flush=True)
        pl.DataFrame(srows).write_csv(TAB / "ablation.csv")
        hrows = []
        for h in [x for x in rs["horizons"] if x != H]:
            for fset in ("A_price", "C_plus_flow", "E_all"):
                for kind in ("logit", "lgbm"):
                    pr, _ = walk_holdout(full, blocks, kind, fset, h, frozen, stride)  # same as development
                    m = EV.classification(pr["y"].to_numpy(), pr["p"].to_numpy(), pr["fwd"].to_numpy())
                    hrows.append({"model": kind, "fset": fset, "h": h, "auc": m["auc"], "log_loss": m["log_loss"],
                                  "log_loss_baseline": m["log_loss_baseline"], "ic_spearman": m["ic_spearman"]})
        pl.DataFrame(hrows).write_csv(TAB / "horizons.csv")

    (TAB / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    with pl.Config(tbl_rows=60, tbl_cols=16, tbl_width_chars=250):
        print(json.dumps({k: v for k, v in out.items() if k not in ("blocks",)}, indent=1, default=float))
        print(blk.select("block", "n", "auc", "log_loss", "log_loss_baseline", "ic_spearman"))
        print(res.select("strategy", "cost", "total_return", "ann_return", "sharpe", "sortino", "max_drawdown",
                         "n_trades", "win_rate", "profit_factor", "avg_trade_bps", "breakeven_cost_bps_per_side",
                         "turnover_per_day", "sum_funding"))
        print(mon); print(C)


if __name__ == "__main__":
    main()
