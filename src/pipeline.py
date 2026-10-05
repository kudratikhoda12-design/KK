"""Research pipeline: univariate tests -> validation selection -> frozen test -> robustness.

Separation of evidence
----------------------
* ``validation_stage`` sees ONLY development data (< config.TEST_START). Every
  choice (model, C, LightGBM config, threshold k, long/short vs long/flat) is
  made here and returned as a frozen ``Selection``.
* ``test_stage`` applies the frozen selection once to the untouched test period.
* ``robustness_stage`` re-runs variants on the test period AFTER the result is
  recorded. Its outputs are reported, never used to change the selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config
from .backtest import buy_and_hold, positions_from_probs, positions_from_score, run_backtest
from .features import FEATURE_GROUPS, FEATURES, build_dataset
from .models import (auc_bootstrap_ci, calibration_deciles, classification_metrics, confusion,
                     feature_importance, fit_final, logloss_improvement_test, walk_forward_predict)
from .risk import deflated_sharpe, performance, probabilistic_sharpe, sharpe_ci
from .stats import holm, spearman_block_test
from .validation import ExperimentLog, development_folds, test_folds


def periods_per_year(horizon: int) -> float:
    return 365.25 * 24 * 60 / horizon


def model_grid() -> list[tuple[str, dict]]:
    grid = [("base_rate", {})]
    grid += [("logreg", {"C": c}) for c in config.LR_C_GRID]
    grid += [("lgbm", dict(p)) for p in config.LGBM_CONFIGS]
    return grid


def key(name: str, params: dict) -> str:
    return name if not params else f"{name}|" + ",".join(f"{k}={v}" for k, v in sorted(params.items()))


@dataclass
class Selection:
    model: str
    params: dict
    k: float
    mode: str
    features: list
    horizon: int = config.PRIMARY_HORIZON_MIN
    lag: int = config.EXECUTION_LAG_MIN
    offset: int = config.DECISION_OFFSET_MIN
    reason: str = ""


# --------------------------------------------------------------------------
# H1-H3: univariate evidence on development data only
# --------------------------------------------------------------------------
def univariate_tests(data: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    dev = data[data.index < config.TEST_START]
    rows = {}
    for f in FEATURES:
        rows[f] = spearman_block_test(dev[f].to_numpy(), dev["fwd_logret"].to_numpy(),
                                      config.BOOTSTRAP_BLOCK_HOURS, n_boot)
    out = pd.DataFrame(rows).T
    out = out.join(holm(out["p_block"].astype(float)))
    out.index.name = "feature"
    return out.reset_index()


def regime_ic(data: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    """H3: does the feature-return relationship differ across regimes?"""
    dev = data[data.index < config.TEST_START]
    rows = []
    for col in ["vol_regime", "trend_regime"]:
        for reg, sub in dev.groupby(col):
            for f in FEATURES:
                t = spearman_block_test(sub[f].to_numpy(), sub["fwd_logret"].to_numpy(),
                                        config.BOOTSTRAP_BLOCK_HOURS, n_boot)
                rows.append(dict(regime_type=col, regime=reg, feature=f, **t))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Validation (development only)
# --------------------------------------------------------------------------
def _fold_sharpes(bt: pd.DataFrame, fold_of: pd.Series, ppy: float) -> pd.Series:
    out = {}
    for f, idx in bt.groupby(fold_of.reindex(bt.index)).groups.items():
        s = bt.loc[idx, "net_ret"]
        out[f] = s.mean() / s.std() * np.sqrt(ppy) if s.std() > 0 else np.nan
    return pd.Series(out)


def validation_stage(data: pd.DataFrame, log: ExperimentLog, features=FEATURES,
                     horizon: int = config.PRIMARY_HORIZON_MIN) -> dict:
    ppy = periods_per_year(horizon)
    folds = development_folds(data)
    assert all(f.eval_end < config.TEST_START for f in folds), "validation touched the test period"
    fold_table = pd.DataFrame([f.describe() for f in folds])

    # 1) Predictions of every model on every validation fold
    preds, model_rows = {}, []
    for name, params in model_grid():
        pr = walk_forward_predict(data, folds, name, params, features)
        preds[key(name, params)] = pr
    base = preds["base_rate"]
    for k_, pr in preds.items():
        d = data.loc[pr.index]
        pooled = classification_metrics(d["y"], pr["p"], d["fwd_logret"])
        per_fold = []
        for fname, idx in pr.groupby("fold").groups.items():
            m = classification_metrics(d.loc[idx, "y"], pr.loc[idx, "p"], d.loc[idx, "fwd_logret"])
            bl = classification_metrics(d.loc[idx, "y"], base.loc[idx, "p"])
            per_fold.append(dict(fold=fname, auc=m["auc"], logloss=m["logloss"],
                                 logloss_better_than_base=m["logloss"] < bl["logloss"]))
        pf = pd.DataFrame(per_fold)
        llt = logloss_improvement_test(d["y"], pr["p"], base.loc[pr.index, "p"])
        row = dict(model=k_, **pooled, mean_fold_auc=pf["auc"].mean(),
                   share_folds_logloss_better=pf["logloss_better_than_base"].mean(),
                   logloss_gain_vs_base=llt["mean"], logloss_gain_p_one_sided=llt["p_value_one_sided"])
        model_rows.append(row)
        log.add("validation-model", k_, {}, features, horizon, config.EXECUTION_LAG_MIN,
                config.DECISION_OFFSET_MIN, f"expanding from {data.index.min():%Y-%m-%d}",
                f"{folds[0].eval_start:%Y-%m-%d}..{folds[-1].eval_end:%Y-%m-%d}", row)
    model_table = pd.DataFrame(model_rows)

    # 2) Pre-registered model choice: best LR and best LGBM by pooled log-loss, then simplicity rule
    def best(prefix):
        sub = model_table[model_table["model"].str.startswith(prefix)]
        return sub.sort_values("logloss").iloc[0]
    lr, gb = best("logreg"), best("lgbm")
    auc_gain = gb["mean_fold_auc"] - lr["mean_fold_auc"]
    lr_pf = preds[lr["model"]]
    gb_pf = preds[gb["model"]]
    wins = []
    for fname in lr_pf["fold"].unique():
        i = lr_pf.index[lr_pf["fold"] == fname]
        y = data.loc[i, "y"]
        wins.append(classification_metrics(y, gb_pf.loc[i, "p"])["logloss"]
                    < classification_metrics(y, lr_pf.loc[i, "p"])["logloss"])
    gb_win_share = float(np.mean(wins))
    use_gb = auc_gain > config.LGBM_MIN_AUC_GAIN and gb_win_share >= config.LGBM_MIN_FOLD_WIN_SHARE
    chosen = gb if use_gb else lr
    reason = (f"LightGBM AUC gain {auc_gain:+.4f} (needs >{config.LGBM_MIN_AUC_GAIN}), "
              f"log-loss better in {gb_win_share:.0%} of folds (needs >={config.LGBM_MIN_FOLD_WIN_SHARE:.0%})"
              f" -> {'LightGBM' if use_gb else 'logistic regression (simpler model kept)'}")
    chosen_name = chosen["model"].split("|")[0]
    chosen_params = dict(model_grid()[[key(n, p) for n, p in model_grid()].index(chosen["model"])][1])

    # 3) Signal grid on the chosen model's validation predictions (base costs)
    pr = preds[chosen["model"]]
    sig_rows, trial_sharpes = [], []
    for k in config.THRESHOLD_K_GRID:
        for mode in config.POSITION_MODES:
            pos = positions_from_probs(pr["p"], pr["train_pred_sd"], k, mode)
            bt = run_backtest(data, pos, config.BASE_COST_PER_SIDE)
            perf = performance(bt, ppy)
            fs = _fold_sharpes(bt, pr["fold"], ppy)
            r = dict(k=k, mode=mode, mean_fold_net_sharpe=fs.mean(), share_folds_positive=(fs > 0).mean(),
                     **{f"val_{a}": perf[a] for a in ["sharpe", "gross_sharpe", "annualized_return",
                                                       "max_drawdown", "exposure", "trades_per_year",
                                                       "win_rate", "cost_sum"]})
            sig_rows.append(r)
            trial_sharpes.append(bt["net_ret"].mean() / bt["net_ret"].std() if bt["net_ret"].std() > 0 else np.nan)
            log.add("validation-signal", chosen["model"], dict(k=k, mode=mode), features, horizon,
                    config.EXECUTION_LAG_MIN, config.DECISION_OFFSET_MIN, "walk-forward",
                    "development validation", r)
    sig_table = pd.DataFrame(sig_rows)
    eligible = sig_table[sig_table["val_trades_per_year"] >= config.MIN_TRADES_PER_YEAR]
    pick = (eligible if len(eligible) else sig_table).sort_values("mean_fold_net_sharpe",
                                                                   ascending=False).iloc[0]

    # 4) Baselines on the same validation periods
    vidx = pr.index
    base_rows = []
    for name, pos in [("buy_and_hold", pd.Series(1.0, index=vidx)),
                      ("momentum_1h_sign", positions_from_score(data.loc[vidx, "ret_60m"])),
                      ("momentum_24h_sign", positions_from_score(data.loc[vidx, "ret_24h"]))]:
        bt = run_backtest(data, pos, config.BASE_COST_PER_SIDE)
        perf = performance(bt, ppy)
        fs = _fold_sharpes(bt, pr["fold"], ppy)
        auc_score = data.loc[vidx, "ret_60m" if "1h" in name else "ret_24h"] if "momentum" in name else None
        base_rows.append(dict(strategy=name, mean_fold_net_sharpe=fs.mean(), **{f"val_{a}": perf[a] for a in
                              ["sharpe", "gross_sharpe", "annualized_return", "max_drawdown",
                               "trades_per_year"]},
                              auc=classification_metrics(data.loc[vidx, "y"],
                                                         _score_to_prob(auc_score))["auc"]
                              if auc_score is not None else np.nan))
        trial_sharpes.append(bt["net_ret"].mean() / bt["net_ret"].std())
        log.add("validation-baseline", name, {}, [], horizon, config.EXECUTION_LAG_MIN,
                config.DECISION_OFFSET_MIN, "-", "development validation", base_rows[-1])
    baseline_table = pd.DataFrame(base_rows)

    selection = Selection(model=chosen_name, params=chosen_params, k=float(pick["k"]),
                          mode=str(pick["mode"]), features=list(features), horizon=horizon,
                          reason=reason + f"; signal k={pick['k']}, mode={pick['mode']} "
                                          f"(best mean validation net Sharpe among eligible configs)")
    return dict(folds=fold_table, models=model_table, signals=sig_table, baselines=baseline_table,
                selection=selection, predictions=preds, trial_sharpes=np.asarray(trial_sharpes, float))


def _score_to_prob(score: pd.Series) -> pd.Series:
    """Rank-preserving map of a raw score to (0,1), only for computing AUC of a rule."""
    return 1 / (1 + np.exp(-score / (score.std() + 1e-12)))


# --------------------------------------------------------------------------
# Final test (run once with the frozen selection)
# --------------------------------------------------------------------------
def run_frozen(data: pd.DataFrame, sel: Selection, folds) -> tuple[pd.DataFrame, pd.DataFrame]:
    pr = walk_forward_predict(data, folds, sel.model, sel.params, sel.features)
    pos = positions_from_probs(pr["p"], pr["train_pred_sd"], sel.k, sel.mode)
    return pr, pos


def test_stage(data: pd.DataFrame, sel: Selection, log: ExperimentLog, trial_sharpes: np.ndarray,
               funding=None) -> dict:
    ppy = periods_per_year(sel.horizon)
    folds = test_folds(data)
    pr, pos = run_frozen(data, sel, folds)
    base_pr = walk_forward_predict(data, folds, "base_rate", {}, sel.features)
    d = data.loc[pr.index]
    bt = run_backtest(data, pos, config.BASE_COST_PER_SIDE, funding=funding)
    bh = buy_and_hold(data.loc[pr.index], config.BASE_COST_PER_SIDE)
    mom = run_backtest(data, positions_from_score(d["ret_60m"]), config.BASE_COST_PER_SIDE)

    cls = classification_metrics(d["y"], pr["p"], d["fwd_logret"])
    cls["auc_ci_low"], cls["auc_ci_high"] = auc_bootstrap_ci(d["y"], pr["p"])
    llt = logloss_improvement_test(d["y"], pr["p"], base_pr["p"])
    perf = performance(bt, ppy)
    perf["sharpe_ci_low"], perf["sharpe_ci_high"] = sharpe_ci(bt["net_ret"].to_numpy(), ppy)
    perf["psr_vs_0"] = probabilistic_sharpe(bt["net_ret"].to_numpy())
    dsr = deflated_sharpe(bt["net_ret"].to_numpy(), trial_sharpes)

    per_block = []
    for fname, idx in pr.groupby("fold").groups.items():
        sub = bt.loc[bt.index.intersection(idx)]
        m = classification_metrics(d.loc[idx, "y"], pr.loc[idx, "p"])
        p_ = performance(sub, ppy)
        per_block.append(dict(block=fname, auc=m["auc"], net_sharpe=p_["sharpe"],
                              net_cum_return=p_["cumulative_return"],
                              buy_hold_cum_return=performance(bh.loc[bh.index.intersection(idx)], ppy)["cumulative_return"],
                              exposure=p_["exposure"]))
    log.add("TEST", key(sel.model, sel.params), dict(k=sel.k, mode=sel.mode), sel.features, sel.horizon,
            sel.lag, sel.offset, "expanding, refit every 6 months", "2024-01-01..end",
            {**cls, **{f"test_{a}": v for a, v in perf.items() if isinstance(v, (int, float))},
             "dsr": dsr.get("dsr", np.nan)}, note="frozen selection, single run")

    # Importance of the model that actually traded in the last test block
    final_model = fit_final(data, folds[-1].train_mask, sel.model, sel.params, sel.features)
    return dict(predictions=pr, positions=pos, backtest=bt, buy_hold=bh, momentum=mom,
                classification=cls, logloss_test=llt, performance=perf, dsr=dsr,
                per_block=pd.DataFrame(per_block),
                benchmarks=pd.DataFrame([dict(strategy="model strategy", **perf),
                                         dict(strategy="buy and hold", **performance(bh, ppy)),
                                         dict(strategy="momentum 1h sign", **performance(mom, ppy))]),
                calibration=calibration_deciles(d["y"], pr["p"], d["fwd_logret"]),
                confusion=confusion(d["y"], pr["p"]),
                importance=feature_importance(final_model, sel.features))


# --------------------------------------------------------------------------
# Robustness / kill tests (post-hoc, reported, never used for selection)
# --------------------------------------------------------------------------
def _variant(data, sel: Selection, log: ExperimentLog, label: str, cost=config.BASE_COST_PER_SIDE,
             **changes) -> dict:
    s = Selection(**{**asdict(sel), **changes})
    ppy = periods_per_year(s.horizon)
    pr, pos = run_frozen(data, s, test_folds(data))
    bt = run_backtest(data, pos, cost)
    d = data.loc[pr.index]
    m = classification_metrics(d["y"], pr["p"], d["fwd_logret"])
    p = performance(bt, ppy)
    row = dict(variant=label, auc=m["auc"], ic=m["ic_spearman"], net_sharpe=p["sharpe"],
               gross_sharpe=p["gross_sharpe"], annualized_return=p["annualized_return"],
               max_drawdown=p["max_drawdown"], exposure=p["exposure"], trades_per_year=p["trades_per_year"])
    log.add("robustness", key(s.model, s.params), dict(k=s.k, mode=s.mode), s.features, s.horizon,
            s.lag, s.offset, "expanding", "test period", row, note=label)
    return row


def robustness_stage(grid: pd.DataFrame, data: pd.DataFrame, sel: Selection, log: ExperimentLog,
                     include_lgbm_swap: bool = True, mfeat: pd.DataFrame | None = None) -> pd.DataFrame:
    """mfeat: precomputed minute features (they do not depend on lag/offset/horizon)."""
    rows = []
    # Thresholds and modes around the chosen one (frozen model predictions)
    for k in config.THRESHOLD_K_GRID:
        for mode in config.POSITION_MODES:
            rows.append(_variant(data, sel, log, f"threshold k={k}, {mode}", k=k, mode=mode))
    # Feature-group removal
    for g, cols in FEATURE_GROUPS.items():
        feats = [f for f in sel.features if f not in cols]
        rows.append(_variant(data, sel, log, f"without {g} features", features=feats))
    # Model swap
    if include_lgbm_swap:
        other = ("lgbm", dict(config.LGBM_CONFIGS[0])) if sel.model == "logreg" else ("logreg", {"C": 0.1})
        rows.append(_variant(data, sel, log, f"model swapped to {other[0]}", model=other[0], params=other[1]))
    # Execution lag, decision offset, horizon: need rebuilt datasets
    for lag in config.ROBUSTNESS_LAGS_MIN:
        d2 = build_dataset(grid, sel.horizon, lag, sel.offset, mfeat)
        rows.append(_variant(d2, sel, log, f"execution lag {lag} min", lag=lag))
    for off in config.ROBUSTNESS_OFFSETS_MIN:
        d2 = build_dataset(grid, sel.horizon, sel.lag, off, mfeat)
        rows.append(_variant(d2, sel, log, f"decisions at :{off:02d}", offset=off))
    for h in config.ROBUSTNESS_HORIZONS_MIN:
        d2 = build_dataset(grid, h, sel.lag, 0, mfeat)
        rows.append(_variant(d2, sel, log, f"horizon {h} min", horizon=h))
    return pd.DataFrame(rows)


def permutation_test(data: pd.DataFrame, sel: Selection, n_perm: int = config.PERMUTATION_N,
                     seed: int = config.RANDOM_SEED) -> dict:
    """Shuffle training labels (destroying any real link), refit, record test AUC.

    Uses logistic regression for speed even if LightGBM was selected; the
    question is whether ANY real-label fit beats label noise.
    """
    rng = np.random.default_rng(seed)
    folds = test_folds(data)
    params = sel.params if sel.model == "logreg" else {"C": 0.1}
    real = walk_forward_predict(data, folds, "logreg", params, sel.features)
    y_real = data.loc[real.index, "y"]
    real_auc = classification_metrics(y_real, real["p"])["auc"]
    null = []
    for _ in range(n_perm):
        shuffled = data.copy()
        has_y = shuffled["y"].notna()
        shuffled.loc[has_y, "y"] = rng.permutation(shuffled.loc[has_y, "y"].to_numpy())
        pr = walk_forward_predict(shuffled, folds, "logreg", params, sel.features)
        null.append(classification_metrics(y_real.loc[pr.index], pr["p"])["auc"])
    null = np.asarray(null)
    return dict(real_auc=real_auc, null_mean=float(null.mean()), null_p95=float(np.quantile(null, 0.95)),
                p_value=float((np.sum(null >= real_auc) + 1) / (n_perm + 1)), n_perm=n_perm)
