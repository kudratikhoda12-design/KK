#!/usr/bin/env python3
"""Stage 4: exploratory analysis + statistical alpha tests on the DEVELOPMENT period only.

Outputs: reports/figures/eda/*.png, reports/figures/stats/*.png, reports/tables/stats_*.csv, eda_*.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ethof import statistics as S  # noqa: E402
from ethof import validation as V  # noqa: E402
from ethof.config import Paths, load_config  # noqa: E402
from ethof.features import GROUPS  # noqa: E402
from ethof.plotting import PALETTE, plt, save  # noqa: E402

SYM = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = load_config()
    paths = Paths(cfg["data_dir"])
    rs = cfg["research"]
    H = rs["horizons"]
    df = pl.read_parquet(paths.processed / "research" / f"{SYM}-research.parquet")
    dev = V.development_only(df, rs["holdout"]["start"])
    # drop development rows whose 60-min label would reach into the holdout (affects only the last hour)
    dev = dev.filter(pl.col("label_end_60") <= V._d(rs["holdout"]["start"]))
    day = dev["ts"].dt.date().to_numpy()
    out: dict = {"rows_dev": dev.height}

    # ======================================================================== EDA
    E = FIG / "eda"
    d = dev.group_by(pl.col("ts").dt.date().alias("day")).agg(
        px=pl.col("px").last(), rv=(pl.col("ret_1").pow(2).sum()).sqrt() * np.sqrt(365),
        vol_usd=(pl.col("volume") * pl.col("px")).sum(), trades=pl.col("n_trades").sum(),
        fund=pl.col("funding_rate_last").last()).sort("day")
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(d["day"], d["px"], color=PALETTE["blue"]); ax[0].set_ylabel("ETH price (USDT)")
    ax[0].set_title("Q: how did price, volatility, activity and funding evolve over the development period?")
    ax[1].plot(d["day"], d["rv"], color=PALETTE["orange"]); ax[1].set_ylabel("daily realised vol\n(annualised)")
    ax[2].plot(d["day"], d["vol_usd"] / 1e9, color=PALETTE["green"]); ax[2].set_ylabel("volume (bn USDT/day)")
    ax[3].plot(d["day"], d["fund"] * 1e4, color=PALETTE["purple"]); ax[3].set_ylabel("funding rate (bps / 8h)")
    save(fig, E / "01_price_vol_volume_funding.png")

    # return distributions: heavy tails?
    r1 = dev["ret_1"].drop_nulls().to_numpy()
    r15 = dev["fwd_ret_15"].drop_nulls().to_numpy()
    from scipy import stats as st
    dist = []
    for nm, x in (("1m log return", r1), ("15m forward return", r15)):
        dist.append({"series": nm, "n": len(x), "mean_bps": x.mean() * 1e4, "std_bps": x.std() * 1e4,
                     "skew": float(st.skew(x)), "excess_kurtosis": float(st.kurtosis(x)),
                     "p01_bps": np.quantile(x, 0.01) * 1e4, "p99_bps": np.quantile(x, 0.99) * 1e4,
                     "share_zero": float((x == 0).mean())})
    pl.DataFrame(dist).write_csv(TAB / "eda_return_distribution.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, (nm, x) in zip(ax, (("1-minute log return", r1), ("15-minute forward return", r15))):
        z = (x - x.mean()) / x.std()
        a.hist(z, bins=200, range=(-10, 10), density=True, color=PALETTE["blue"], alpha=0.8, label="empirical")
        g = np.linspace(-10, 10, 400); a.plot(g, st.norm.pdf(g), color=PALETTE["red"], label="Normal")
        a.set_yscale("log"); a.set_title(f"{nm}: standardised, log density"); a.legend()
    save(fig, E / "02_return_tails.png")

    # autocorrelation of returns and absolute returns (sampled at 1 min)
    lags = np.arange(1, 61)
    def acf(x, L):
        x = x - x.mean(); v = (x * x).mean()
        return np.array([(x[l:] * x[:-l]).mean() / v for l in L])
    a_r, a_abs = acf(r1, lags), acf(np.abs(r1), lags)
    pl.DataFrame({"lag": lags, "acf_ret": a_r, "acf_absret": a_abs}).write_csv(TAB / "eda_acf.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ci = 1.96 / np.sqrt(len(r1))
    ax[0].bar(lags, a_r, color=PALETTE["blue"]); ax[0].axhspan(-ci, ci, color="grey", alpha=0.2)
    ax[0].set_title("ACF of 1-min returns (iid 95% band shaded)")
    ax[1].bar(lags, a_abs, color=PALETTE["orange"]); ax[1].set_title("ACF of |1-min returns| (volatility clustering)")
    for a in ax: a.set_xlabel("lag (minutes)")
    save(fig, E / "03_autocorrelation.png")

    # intraday seasonality
    hr = dev.group_by(pl.col("ts").dt.hour().alias("hour")).agg(
        vol=pl.col("volume").mean(), absret=pl.col("ret_1").abs().mean() * 1e4,
        ofi_sd=pl.col("ofi_1").std()).sort("hour")
    hr.write_csv(TAB / "eda_intraday.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].bar(hr["hour"], hr["vol"], color=PALETTE["green"]); ax[0].set_title("mean volume per minute by UTC hour (ETH)")
    ax[1].bar(hr["hour"], hr["absret"], color=PALETTE["orange"]); ax[1].set_title("mean |1-min return| by UTC hour (bps)")
    save(fig, E / "04_intraday_seasonality.png")

    # order-flow and book imbalance distributions
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, c, col in zip(ax, ("ofi_1", "ofi_15", "obi_1pct"), ("blue", "purple", "red")):
        x = dev[c].drop_nulls().to_numpy()
        a.hist(x, bins=100, color=PALETTE[col], alpha=0.85)
        a.set_title(f"{c}: mean {x.mean():+.3f}, sd {x.std():.3f}")
    save(fig, E / "05_imbalance_distributions.png")
    imb = []
    for c in ("ofi_1", "ofi_5", "ofi_15", "ofi_60", "trade_imb_15", "obi_1pct", "obi_2pct", "obi_3pct", "obi_5pct"):
        x = dev[c].drop_nulls().to_numpy()
        imb.append({"feature": c, "n": len(x), "mean": x.mean(), "sd": x.std(), "p05": np.quantile(x, .05),
                    "p50": np.median(x), "p95": np.quantile(x, .95), "acf1": float(np.corrcoef(x[1:], x[:-1])[0, 1])})
    pl.DataFrame(imb).write_csv(TAB / "eda_imbalance_summary.csv")

    # contemporaneous vs predictive: the order-flow / price-impact link
    cc = dev.gather_every(3).select("ofi_1", "ret_1", "obi_1pct", "fwd_ret_15").drop_nulls()
    out["corr_ofi1_ret1_contemporaneous"] = float(np.corrcoef(cc["ofi_1"], cc["ret_1"])[0, 1])
    out["corr_obi1_ret1_contemporaneous"] = float(np.corrcoef(cc["obi_1pct"], cc["ret_1"])[0, 1])

    # feature correlation heatmap
    feats = sum(GROUPS.values(), [])
    cm = dev.gather_every(10).select(feats).drop_nulls().to_pandas().corr(method="spearman")
    cm.to_csv(TAB / "eda_feature_spearman_corr.csv")
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=90, fontsize=6)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=6); ax.grid(False)
    fig.colorbar(im, shrink=0.7); ax.set_title("Spearman correlation between features (development, every 10th minute)")
    save(fig, E / "06_feature_correlation.png")

    # temporal stability of the flow/return link: quarterly rank IC of ofi_15 vs fwd_ret_15
    q = dev.with_columns(qtr=pl.col("ts").dt.year().cast(pl.String) + "Q" + pl.col("ts").dt.quarter().cast(pl.String))
    qic = q.group_by("qtr").agg(
        ic_ofi15=pl.corr("ofi_15", "fwd_ret_15", method="spearman"),
        ic_obi1=pl.corr("obi_1pct", "fwd_ret_15", method="spearman"),
        ic_ret15=pl.corr("ret_15", "fwd_ret_15", method="spearman"), n=pl.len()).sort("qtr")
    qic.write_csv(TAB / "stats_quarterly_rank_ic.csv")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(qic.height)
    for k, (c, col) in enumerate((("ic_ofi15", "blue"), ("ic_obi1", "red"), ("ic_ret15", "grey"))):
        ax.bar(x + (k - 1) * 0.27, qic[c], width=0.27, color=PALETTE[col], label=c)
    ax.set_xticks(x); ax.set_xticklabels(qic["qtr"]); ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Quarterly Spearman IC with the 15-min forward return (is the relationship stable?)"); ax.legend()
    save(fig, FIG / "stats" / "quarterly_ic.png")

    # ======================================================================== statistical alpha tests
    signals = ["ofi_1", "ofi_5", "ofi_15", "ofi_60", "trade_imb_15", "signed_flow_15", "large_ofi_15",
               "obi_1pct", "obi_5pct"]
    rows = []
    for s in signals:
        for h in H:
            sub = dev.select(s, f"fwd_ret_{h}").drop_nulls()
            x, y = sub[s].to_numpy(), sub[f"fwd_ret_{h}"].to_numpy()
            xs = S.standardize(x)
            hac = S.hac_ols(y * 1e4, xs[:, None], ["x"], maxlags=2 * h)
            rho = st.spearmanr(x[::5], y[::5]).statistic
            # non-overlapping check: one observation every h minutes, HC-robust errors
            idx = np.arange(0, len(y), h)
            no = S.hac_ols(y[idx] * 1e4, xs[idx][:, None], ["x"], maxlags=1)
            rows.append({"signal": s, "h": h, "n": hac["n"], "pearson": float(np.corrcoef(x, y)[0, 1]),
                         "spearman": float(rho), "beta_bps_per_sd": hac["b_x"], "t_hac": hac["t_x"],
                         "p_hac": hac["p_x"], "beta_nonoverlap": no["b_x"], "t_nonoverlap": no["t_x"],
                         "p_nonoverlap": no["p_x"], "r2": hac["r2"]})
    T1 = pl.DataFrame(rows)
    T1 = T1.with_columns(p_holm=pl.Series(S.holm(T1["p_hac"].to_numpy())),
                         p_bh=pl.Series(S.bh(T1["p_hac"].to_numpy())))
    T1.write_csv(TAB / "stats_univariate_predictive.csv")
    out["n_univariate_tests"] = T1.height

    # controls: does order flow survive momentum, volatility, volume?
    ctrl = ["ret_15", "ret_60", "vol_60", "volume_z_1440"]
    rows = []
    for s in ("ofi_15", "ofi_1", "obi_1pct", "trade_imb_15"):
        for h in H:
            sub = dev.select([s] + ctrl + [f"fwd_ret_{h}"]).drop_nulls()
            X = np.column_stack([S.standardize(sub[c].to_numpy()) for c in [s] + ctrl])
            r = S.hac_ols(sub[f"fwd_ret_{h}"].to_numpy() * 1e4, X, [s] + ctrl, maxlags=2 * h)
            rows.append({"signal": s, "h": h, "n": r["n"], "beta_signal_bps_per_sd": r[f"b_{s}"],
                         "t_signal": r[f"t_{s}"], "p_signal": r[f"p_{s}"],
                         **{f"t_{c}": r[f"t_{c}"] for c in ctrl}, "r2": r["r2"]})
    T2 = pl.DataFrame(rows)
    T2 = T2.with_columns(p_holm=pl.Series(S.holm(T2["p_signal"].to_numpy())))
    T2.write_csv(TAB / "stats_with_controls.csv")

    # deciles + block bootstrap of top-minus-bottom spread
    rows, dec_tabs = [], {}
    for s in ("ofi_15", "ofi_1", "obi_1pct"):
        for h in H:
            sub = dev.select("ts", s, f"fwd_ret_{h}").drop_nulls()
            x, y = sub[s].to_numpy(), sub[f"fwd_ret_{h}"].to_numpy()
            dt = S.decile_table(x, y)
            dec_tabs[(s, h)] = dt
            dt.with_columns(signal=pl.lit(s), h=pl.lit(h)).write_csv(TAB / f"stats_deciles_{s}_h{h}.csv")
            bb = S.block_bootstrap_spread(x, y, sub["ts"].dt.date().to_numpy(), n_boot=500)
            mono = st.spearmanr(dt["d"], dt["y_mean_bps"]).statistic
            rows.append({"signal": s, "h": h, **bb, "decile_monotonicity_spearman": float(mono)})
    T3 = pl.DataFrame(rows)
    T3 = T3.with_columns(p_holm=pl.Series(S.holm(T3["p_value"].to_numpy())))
    T3.write_csv(TAB / "stats_decile_spreads.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, s in zip(ax, ("ofi_15", "ofi_1", "obi_1pct")):
        for h, col in zip(H, ("blue", "green", "orange", "red")):
            dt = dec_tabs[(s, h)]
            a.plot(dt["d"], dt["y_mean_bps"], marker="o", color=PALETTE[col], label=f"h={h}m")
        a.axhline(0, color="black", lw=0.8); a.set_xlabel(f"{s} decile (1 = most selling)")
        a.set_title(f"{s} decile vs mean forward return (bps)")
    ax[0].legend()
    save(fig, FIG / "stats" / "decile_forward_returns.png")

    # ======================================================================== event study
    ev_rows = []
    lo, hi = np.nanquantile(dev["ofi_1"].to_numpy(), [0.05, 0.95])
    lo15, hi15 = np.nanquantile(dev["ofi_15"].to_numpy(), [0.05, 0.95])
    px = dev["px"].to_numpy(); r1n = dev["ret_1"].to_numpy()
    ks = [1, 5, 15, 30, 60]
    absr = np.abs(np.nan_to_num(r1n))
    base_abs = {k: np.nanmean(np.lib.stride_tricks.sliding_window_view(absr, k).sum(axis=1)) for k in ks}
    paths_plot = {}
    for nm, col, thr, side in (("ofi_1 > p95", "ofi_1", hi, 1), ("ofi_1 < p5", "ofi_1", lo, -1),
                               ("ofi_15 > p95", "ofi_15", hi15, 1), ("ofi_15 < p5", "ofi_15", lo15, -1)):
        x = dev[col].to_numpy()
        ev = np.where((x > thr) if side > 0 else (x < thr))[0]
        ev = ev[ev + 61 < len(px)]
        res = {"event": nm, "threshold": float(thr), "n_events": int(len(ev)),
               "event_bar_ret_bps": float(np.nanmean(r1n[ev]) * 1e4)}
        path = []
        for k in ks:
            fr = px[ev + k] / px[ev] - 1
            # day-block bootstrap CI for the mean post-event return
            evday = day[ev]
            ud, inv = np.unique(evday, return_inverse=True)
            sums = np.bincount(inv, weights=np.nan_to_num(fr)); cnt = np.bincount(inv)
            rng = np.random.default_rng(3)
            bs = [sums[i].sum() / cnt[i].sum() for i in (rng.integers(0, len(ud), len(ud)) for _ in range(500))]
            m = float(np.nanmean(fr))
            post_abs = np.lib.stride_tricks.sliding_window_view(absr, k).sum(axis=1)
            vr = float(np.nanmean(post_abs[ev + 1]) / base_abs[k]) if (ev + 1).max() < len(post_abs) else float("nan")
            res[f"ret_{k}m_bps"] = m * 1e4
            res[f"ci_lo_{k}m"] = float(np.quantile(bs, .025) * 1e4)
            res[f"ci_hi_{k}m"] = float(np.quantile(bs, .975) * 1e4)
            res[f"abs_move_ratio_{k}m"] = vr
            path.append(m * 1e4)
        paths_plot[nm] = path
        ev_rows.append(res)
    EV = pl.DataFrame(ev_rows)
    EV.write_csv(TAB / "stats_event_study.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, keys in zip(ax, (("ofi_1 > p95", "ofi_1 < p5"), ("ofi_15 > p95", "ofi_15 < p5"))):
        for k_, col in zip(keys, ("green", "red")):
            r = EV.filter(pl.col("event") == k_).row(0, named=True)
            a.plot([0] + ks, [0] + paths_plot[k_], marker="o", color=PALETTE[col], label=f"{k_} (n={r['n_events']:,})")
            a.fill_between(ks, [r[f"ci_lo_{k}m"] for k in ks], [r[f"ci_hi_{k}m"] for k in ks], color=PALETTE[col], alpha=0.15)
        a.axhline(0, color="black", lw=0.8); a.set_xlabel("minutes after event bar close")
        a.set_ylabel("mean cumulative return (bps)"); a.legend(); a.set_title("Event study: continuation or reversal?")
    save(fig, FIG / "stats" / "event_study.png")

    (TAB / "stats_summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=250):
        print(T1.select("signal", "h", "pearson", "beta_bps_per_sd", "t_hac", "p_holm", "t_nonoverlap"))
        print(T2.select("signal", "h", "beta_signal_bps_per_sd", "t_signal", "p_holm", "t_ret_15", "t_vol_60"))
        print(T3); print(EV)


if __name__ == "__main__":
    main()
