"""Run the market-module research pipeline end to end on REAL Binance data.

    python scripts/run_pipeline.py --stage all
    python scripts/run_pipeline.py --stage audit      # or eda / features / validation / test /
                                                      #    risk / stress / robustness / second_asset

Stages are ordered; each one saves tables to reports/tables/, figures to
figures/ and key numbers to reports/results.json. The validation stage writes
reports/selection.json (the frozen design); the test stage refuses to run
without it and never modifies it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, eda, plots, quality, risk, stress_test  # noqa: E402
from src.backtest import capacity_check  # noqa: E402
from src.data import load_funding, load_interim, load_processed  # noqa: E402
from src.features import FEATURES, build_dataset, minute_features, perturbation_leakage_check, timing_table  # noqa: E402
from src.pipeline import (Selection, periods_per_year, permutation_test, regime_ic,  # noqa: E402
                          robustness_stage, test_stage, univariate_tests, validation_stage)
from src.validation import ExperimentLog  # noqa: E402

T, F = config.TABLES_DIR, config.FIGURES_DIR
RESULTS = config.REPORTS_DIR / "results.json"
SELECTION = config.REPORTS_DIR / "selection.json"
LOG_PATH = T / "experiment_log.csv"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp, pd.Timedelta)):
        return str(o)
    if isinstance(o, (pd.DatetimeIndex, np.ndarray, list, tuple)):
        return [str(x) for x in o]
    return str(o)


def save_results(section: str, payload: dict) -> None:
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res[section] = payload
    RESULTS.write_text(json.dumps(res, indent=2, default=_json_default))


def load_log() -> ExperimentLog:
    log = ExperimentLog(LOG_PATH)
    if LOG_PATH.exists():
        log.rows = pd.read_csv(LOG_PATH).to_dict("records")
    return log


def dataset_path(symbol, h=config.PRIMARY_HORIZON_MIN, lag=config.EXECUTION_LAG_MIN, off=0) -> Path:
    return config.PROCESSED_DIR / f"{symbol}_dataset_h{h}_lag{lag}_off{off}.parquet"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
def stage_audit(symbol: str) -> None:
    raw = load_interim(symbol)
    a = quality.audit(raw)
    for name in ["dq_table", "file_summary", "missing_by_year", "monthly"]:
        a[name].to_csv(T / f"{symbol}_audit_{name}.csv", index=False)
    a["gaps"].head(25).to_csv(T / f"{symbol}_audit_gaps_top25.csv", index=False)
    a["outliers"].head(50).to_csv(T / f"{symbol}_audit_outliers_top50.csv")
    grid, treat = quality.build_processed(raw, a["outliers"])
    treat.to_csv(T / f"{symbol}_audit_treatments.csv", index=False)
    quality.save_processed(grid, symbol)
    manifest = config.RAW_DIR / "spot" / symbol / "1m" / "manifest.csv"
    man = pd.read_csv(manifest) if manifest.exists() else None
    save_results(f"{symbol}_audit", dict(
        schema={k: v for k, v in a["schema"].items() if k != "dtypes"}, timestamps=a["timestamps"],
        duplicates={k: (len(v) if hasattr(v, "__len__") else v) for k, v in a["duplicates"].items()},
        volume=a["volume"], stale=a["stale"], n_outliers=len(a["outliers"]),
        n_gaps=len(a["gaps"]), missing_minutes=int(a["gaps"]["missing_minutes"].sum()),
        grid_rows=len(grid), treatments=dict(zip(treat["treatment"], treat["count"])),
        files_checksum_ok=None if man is None else int(man["checksum_ok"].sum()),
        files_in_manifest=None if man is None else len(man),
        parquet_mb=round((config.INTERIM_DIR / f"{symbol}_1m_raw.parquet").stat().st_size / 1e6, 1),
        memory_mb=round(raw.memory_usage(deep=True).sum() / 1e6, 1)))
    print(a["dq_table"].to_string(index=False))


def stage_eda(symbol: str) -> None:
    g = load_processed(symbol)
    hourly, daily = eda.resample_bars(g, "1h"), eda.resample_bars(g, "1D")
    dev_h = hourly[hourly.index < config.TEST_START]
    r_dev = eda.log_returns(dev_h)
    eda.return_distribution_table(g).to_csv(T / "eda_return_distribution.csv", index=False)
    eda.dependence_tests(r_dev).to_csv(T / "eda_dependence_dev.csv", index=False)
    eda.volume_return_relation(dev_h).to_csv(T / "eda_volume_return_dev.csv", index=False)
    prof = eda.intraday_profile(dev_h)
    prof.to_csv(T / "eda_intraday_dev.csv")
    eda.extreme_moves(hourly).to_csv(T / "eda_extreme_hours.csv")
    reg = eda.regime_labels(hourly)
    reg_summary = reg.groupby(reg.index.year).agg(
        high_vol_share=("vol_regime", lambda s: (s == "high").mean()),
        trending_share=("trending", lambda s: (s == "trending").mean()),
        mean_hourly_vol_pct=("rv24_hourly", lambda s: 100 * s.mean()))
    reg_summary.to_csv(T / "eda_regimes_by_year.csv")
    eda.fig_price_and_volatility(daily, F / "eda_price_volatility.png")
    eda.fig_return_tails(eda.log_returns(hourly), F / "eda_hourly_return_qq.png")
    eda.fig_acf(r_dev, F / "eda_acf_dev.png")
    eda.fig_intraday(prof, F / "eda_intraday_dev.png")
    save_results("eda", dict(dependence=pd.read_csv(T / "eda_dependence_dev.csv").to_dict("records")))


def stage_features(symbol: str) -> None:
    g = load_processed(symbol)
    mf = minute_features(g)
    d = build_dataset(g, mfeat=mf)
    d.to_parquet(dataset_path(symbol))
    timing_table().to_csv(T / "feature_timing_table.csv", index=False)
    # Leakage audit on the REAL data at 24 random decision times spread over the sample
    rng = np.random.default_rng(config.RANDOM_SEED)
    ok = d.dropna(subset=FEATURES).index
    sample = pd.DatetimeIndex(np.sort(rng.choice(ok, 24, replace=False)))
    leak = perturbation_leakage_check(g, sample)
    leak.to_csv(T / "leakage_perturbation_real_data.csv", index=False)
    usable = d[FEATURES + ["y"]].notna().all(axis=1)
    uni = univariate_tests(d)
    uni.to_csv(T / "h1_h2_univariate_dev.csv", index=False)
    regime_ic(d).to_csv(T / "h3_regime_ic_dev.csv", index=False)
    d[FEATURES].describe().T.to_csv(T / "feature_summary.csv")
    save_results("features", dict(
        decisions=len(d), usable=int(usable.sum()), dropped_for_missing=int((~usable).sum()),
        up_rate_dev=float(d.loc[d.index < config.TEST_START, "y"].mean()),
        leakage_features_changed=int(leak["features_changed"].sum()),
        leakage_checks=len(leak), leakage_target_changed=int(leak["target_changed"].sum()),
        significant_features_holm=uni.loc[uni["reject_at_5pct"], "feature"].tolist()))


def stage_validation(symbol: str) -> None:
    d = pd.read_parquet(dataset_path(symbol))
    log = ExperimentLog(LOG_PATH)            # validation starts a fresh log
    v = validation_stage(d, log)
    v["folds"].to_csv(T / "validation_folds.csv", index=False)
    v["models"].to_csv(T / "validation_models.csv", index=False)
    v["signals"].to_csv(T / "validation_signal_grid.csv", index=False)
    v["baselines"].to_csv(T / "validation_baselines.csv", index=False)
    np.save(config.PROCESSED_DIR / "trial_sharpes.npy", v["trial_sharpes"])
    sel = v["selection"]
    SELECTION.write_text(json.dumps(dict(**asdict(sel), frozen_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                                         code_commit=git_commit()), indent=2, default=_json_default))
    log.save()
    save_results("validation", dict(selection=asdict(sel), n_trials=len(v["trial_sharpes"])))
    print(v["models"][["model", "auc", "logloss", "mean_fold_auc", "logloss_gain_p_one_sided"]].to_string())
    print(sel.reason)


def _selection() -> Selection:
    if not SELECTION.exists():
        raise SystemExit("reports/selection.json missing: run the validation stage first.")
    s = json.loads(SELECTION.read_text())
    return Selection(**{k: s[k] for k in Selection.__dataclass_fields__})


def stage_test(symbol: str) -> None:
    sel = _selection()
    d = pd.read_parquet(dataset_path(symbol))
    log = load_log()
    trials = np.load(config.PROCESSED_DIR / "trial_sharpes.npy")
    funding = load_funding(symbol)
    out = test_stage(d, sel, log, trials, funding=funding)
    log.save()
    out["backtest"].to_parquet(config.PROCESSED_DIR / f"{symbol}_test_backtest.parquet")
    out["buy_hold"].to_parquet(config.PROCESSED_DIR / f"{symbol}_test_buyhold.parquet")
    out["backtest"].head(48).to_csv(T / "test_backtest_sample_rows.csv")
    for k in ["per_block", "benchmarks", "calibration", "importance", "drift"]:
        out[k].to_csv(T / f"test_{k}.csv", index=False)
    out["confusion"].to_csv(T / "test_confusion.csv")
    bt, bh = out["backtest"], out["buy_hold"]
    plots.equity_curves({"strategy (net)": bt["net_ret"], "strategy (gross)": bt["gross_ret"],
                         "buy and hold": bh["net_ret"]}, F / "test_equity.png",
                        "Untouched test period: strategy vs buy-and-hold")
    plots.calibration(out["calibration"], F / "test_calibration.png", "Test-period calibration")
    plots.decile_returns(out["calibration"], F / "test_decile_returns.png",
                         "Mean forward return by predicted-probability decile (test)",
                         2e4 * config.BASE_COST_PER_SIDE)
    g = load_processed(symbol)
    cap = capacity_check(g, d.loc[bt.index], notional_usdt=100_000)
    save_results("test", dict(selection=asdict(sel), classification=out["classification"],
                              logloss_test=out["logloss_test"], performance=out["performance"],
                              dsr=out["dsr"], capacity=cap, funding_included=funding is not None,
                              per_block=out["per_block"].to_dict("records")))
    print(out["benchmarks"][["strategy", "sharpe", "annualized_return", "max_drawdown"]].to_string())


def stage_risk(symbol: str) -> None:
    bt = pd.read_parquet(config.PROCESSED_DIR / f"{symbol}_test_backtest.parquet")
    bh = pd.read_parquet(config.PROCESSED_DIR / f"{symbol}_test_buyhold.parquet")
    rows = []
    for name, b in [("strategy", bt), ("buy_and_hold", bh)]:
        dly = risk.daily_returns(b)
        t = risk.var_es(dly)
        t.insert(0, "series", name)
        rows.append(t)
    pd.concat(rows).to_csv(T / "risk_var_es_daily.csv", index=False)
    dly = risk.daily_returns(bt)
    kup = [risk.kupiec_pof(dly, lvl, window=250) for lvl in config.VAR_LEVELS]
    pd.DataFrame(kup).to_csv(T / "risk_kupiec.csv", index=False)
    paths = risk.bootstrap_paths(dly)
    mc = risk.summarize_paths(paths)
    mc.to_csv(T / "risk_bootstrap_mc.csv", index=False)
    plots.mc_histogram(paths, F / "risk_mc_one_year.png",
                       "Block-bootstrap Monte Carlo of one-year net return (strategy)")
    dd = stress_test.drawdown_table(bt)
    save_results("risk", dict(kupiec=kup, monte_carlo=mc.to_dict("records")[0], drawdown=dd))


def stage_stress(symbol: str) -> None:
    sel = _selection()
    d = pd.read_parquet(dataset_path(symbol))
    bt = pd.read_parquet(config.PROCESSED_DIR / f"{symbol}_test_backtest.parquet")
    bh = pd.read_parquet(config.PROCESSED_DIR / f"{symbol}_test_buyhold.parquet")
    pos = bt["position"]
    ppy = periods_per_year(sel.horizon)
    costs = stress_test.cost_scenarios(d, pos, ppy)
    vol = stress_test.volatility_shock(d, pos, ppy)
    deg = stress_test.signal_degradation(d, pos, ppy)
    worst = stress_test.historical_worst_days(bt, bh)
    gap = stress_test.gap_shock(bt)
    reg_v = stress_test.by_regime(bt, d["vol_regime"], ppy, bh)
    reg_t = stress_test.by_regime(bt, d["trend_regime"], ppy, bh)
    for n, t in [("costs", costs), ("volatility", vol), ("signal_degradation", deg), ("gap_shock", gap),
                 ("regime_vol", reg_v), ("regime_trend", reg_t)]:
        t.to_csv(T / f"stress_{n}.csv", index=False)
    worst.to_csv(T / "stress_worst_btc_days.csv")
    plots.sharpe_bars(costs, "scenario", "net_sharpe", F / "stress_costs.png",
                      "Net Sharpe under cost and slippage stress (test period)")
    save_results("stress", dict(costs=costs.to_dict("records"), signal_degradation=deg.to_dict("records")))


def stage_robustness(symbol: str) -> None:
    sel = _selection()
    g = load_processed(symbol)
    d = pd.read_parquet(dataset_path(symbol))
    log = load_log()
    rob = robustness_stage(g, d, sel, log, mfeat=minute_features(g))
    rob.to_csv(T / "robustness_variants.csv", index=False)
    perm = permutation_test(d, sel)
    log.save()
    plots.sharpe_bars(rob, "variant", "net_sharpe", F / "robustness_variants.png",
                      "Kill tests: net Sharpe of each variant (test period)",
                      highlight=f"threshold k={sel.k}, {sel.mode}")
    save_results("robustness", dict(variants=rob.to_dict("records"), permutation=perm,
                                    experiments_logged=len(log.rows)))


def stage_second_asset(symbol: str = config.SECOND_ASSET) -> None:
    """Frozen BTC design applied unchanged to a second asset (no re-selection)."""
    stage_audit(symbol)
    stage_features(symbol)
    sel = _selection()
    d = pd.read_parquet(dataset_path(symbol))
    log = load_log()
    trials = np.load(config.PROCESSED_DIR / "trial_sharpes.npy")
    out = test_stage(d, sel, log, trials)
    log.save()
    out["benchmarks"].to_csv(T / f"second_asset_{symbol}_benchmarks.csv", index=False)
    save_results(f"second_asset_{symbol}", dict(classification=out["classification"],
                                                performance=out["performance"]))


STAGES = ["audit", "eda", "features", "validation", "test", "risk", "stress", "robustness"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=STAGES + ["all", "second_asset"])
    ap.add_argument("--symbol", default=config.SYMBOL)
    a = ap.parse_args()
    config.ensure_dirs()
    todo = STAGES if a.stage == "all" else [a.stage]
    for s in todo:
        t0 = time.time()
        print(f"\n===== stage: {s} =====", flush=True)
        if s == "second_asset":
            stage_second_asset()
        else:
            globals()[f"stage_{s}"](a.symbol)
        print(f"stage {s} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
